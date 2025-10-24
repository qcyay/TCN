"""
人体运动数据可视化工具 (v3.3 最终简化版)
支持交互式Web界面和命令行两种模式
更新：
1. 简化文件选择为类型选择（exo 或 moment）
2. 自动匹配符合规则的文件
3. 支持在同一图中对比多人同一运动的同一参数
1. 新增时间对齐功能
5. 支持多选文件类型
6. 每个参数独立显示在一个子图中
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class MotionDataVisualizer:
    """运动数据可视化类"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root).resolve()

        self.file_types = {
            'exo': '传感器数据 (exo)',
            'moment': '力矩数据 (moment_filt)'
        }

        if not self.data_root.exists():
            raise ValueError(f"数据路径不存在: {self.data_root}")

        print(f"✓ 数据根目录: {self.data_root}")

    def get_all_subjects(self) -> List[str]:
        """获取所有人名"""
        subjects = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        subjects = sorted(subjects)
        print(f"✓ 找到 {len(subjects)} 个受试者: {subjects}")
        return subjects

    def get_motions_for_subjects(self, subjects: List[str]) -> List[str]:
        """获取多个人共有的运动类型"""
        if not subjects:
            return []

        print(f"\n检查共有运动类型...")
        motion_sets = []

        for subject in subjects:
            subject_path = self.data_root / subject
            if subject_path.exists():
                motions = [d.name for d in subject_path.iterdir() if d.is_dir()]
                motion_sets.append(set(motions))
                print(f"  {subject}: {sorted(motions)}")

        if motion_sets:
            common_motions = set.intersection(*motion_sets)
            result = sorted(list(common_motions))
            print(f"✓ 共有运动类型 ({len(result)}): {result}")
            return result

        print("✗ 未找到共有运动类型")
        return []

    def find_file_by_type(self, subject: str, motion: str, file_type: str) -> Optional[str]:
        """根据文件类型在指定目录下查找匹配的文件"""
        motion_path = self.data_root / subject / motion
        if not motion_path.exists():
            return None

        csv_files = list(motion_path.glob("*.csv"))

        for csv_file in csv_files:
            file_name = csv_file.name

            if file_type == 'exo':
                if file_name == 'exo.csv':
                    return file_name
                elif file_name.endswith('_exo.csv'):
                    prefix = file_name[:-8]
                    if prefix and not prefix[-1].isalpha():
                        return file_name
                    elif '_' in prefix:
                        return file_name

            elif file_type == 'moment':
                if file_name.endswith('moment_filt.csv') or file_name.endswith('moments_filt.csv'):
                    return file_name

        return None

    def check_file_type_availability(self, subjects: List[str], motions: List[str]) -> Dict[str, bool]:
        """检查每种文件类型是否可用"""
        print(f"\n检查可用文件类型...")
        availability = {'exo': False, 'moment': False}

        for file_type in ['exo', 'moment']:
            found = False
            for subject in subjects:
                for motion in motions:
                    if self.find_file_by_type(subject, motion, file_type):
                        found = True
                        break
                if found:
                    break
            availability[file_type] = found
            status = "✓" if found else "✗"
            print(f"  {status} {self.file_types[file_type]}: {'可用' if found else '不可用'}")

        return availability

    def get_available_columns(self, file_types: List[str], subjects: List[str],
                            motions: List[str]) -> Dict[str, str]:
        """获取可用的列名及其来源"""
        print(f"\n扫描可用的列...")
        columns_dict = {}

        for file_type in file_types:
            print(f"\n  扫描 {self.file_types[file_type]}...")
            success_count = 0

            for subject in subjects:
                for motion in motions:
                    file_name = self.find_file_by_type(subject, motion, file_type)
                    if not file_name:
                        continue

                    file_path = self.data_root / subject / motion / file_name
                    try:
                        if file_path.exists():
                            df = None
                            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                                try:
                                    df = pd.read_csv(str(file_path), nrows=1, encoding=encoding)
                                    break
                                except UnicodeDecodeError:
                                    continue

                            if df is not None:
                                clean_columns = [str(col).strip() for col in df.columns]
                                for col in clean_columns:
                                    if col.lower() not in ['time', 'frame']:
                                        columns_dict[col] = file_type
                                success_count += 1
                    except Exception as e:
                        pass

            print(f"  {self.file_types[file_type]} - 成功读取: {success_count} 个文件")

        print(f"\n✓ 总共找到 {len(columns_dict)} 个可用列")
        return columns_dict

    def load_data(self, subject: str, motion: str, file_type: str) -> pd.DataFrame:
        """加载单个CSV文件"""
        file_name = self.find_file_by_type(subject, motion, file_type)
        if not file_name:
            raise FileNotFoundError(f"未找到匹配的 {file_type} 文件: {subject}/{motion}")

        file_path = self.data_root / subject / motion / file_name
        file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        df = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(str(file_path), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法用常见编码读取文件: {file_path}")

        df.columns = df.columns.str.strip()
        return df

    def visualize(self, subjects: List[str], motions: List[str],
                 columns: List[str], columns_dict: Dict[str, str],
                 save_path: str = None, align_time: bool = False) -> go.Figure:
        """
        创建可视化图表
        每个参数独立显示在一个子图中
        """
        if not columns:
            raise ValueError("至少需要选择一个列进行可视化")

        n_cols = len(columns)

        # 计算子图布局
        n_rows = (n_cols + 1) // 2 if n_cols > 1 else 1
        n_plot_cols = 2 if n_cols > 1 else 1

        # 创建子图
        fig = make_subplots(
            rows=n_rows,
            cols=n_plot_cols,
            subplot_titles=columns,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # 为每个列创建子图
        for col_idx, column in enumerate(columns):
            row = col_idx // n_plot_cols + 1
            col = col_idx % n_plot_cols + 1

            file_type = columns_dict.get(column)
            if not file_type:
                continue

            color_idx = 0
            x_label = 'Time'

            for subject in subjects:
                for motion in motions:
                    try:
                        df = self.load_data(subject, motion, file_type)
                        if column not in df.columns:
                            print(f"警告: 列 '{column}' 不存在于 {subject}/{motion}")
                            continue

                        x_data, x_label = self._get_time_axis(df, align_time)
                        y_data = df[column]

                        trace_name = f"{subject}-{motion}"
                        color = colors[color_idx % len(colors)]

                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode='lines',
                                name=trace_name,
                                line=dict(color=color, width=2),
                                legendgroup=trace_name,
                                showlegend=(col_idx == 0)
                            ),
                            row=row,
                            col=col
                        )
                        color_idx += 1

                    except Exception as e:
                        print(f"警告: 加载 {subject}/{motion} 失败: {e}")

            # 设置坐标轴标签
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text=column, row=row, col=col)

        # 更新布局
        title_suffix = " (时间已对齐)" if align_time else ""

        fig.update_layout(
            height=400 * n_rows,
            showlegend=True,
            hovermode='x unified',
            title_text=f"运动数据可视化{title_suffix}",
            title_x=0.5
        )

        if save_path:
            fig.write_html(save_path)
            print(f"\n✓ 图表已保存到: {Path(save_path).resolve()}")

        return fig

    def _get_time_axis(self, df: pd.DataFrame, align_time: bool) -> Tuple[pd.Series, str]:
        """获取时间轴数据"""
        if 'time' in df.columns:
            x_data = df['time'].copy()
            x_label = 'Time (s)' if not align_time else 'Aligned Time (s)'
        elif 'Time' in df.columns:
            x_data = df['Time'].copy()
            x_label = 'Time (s)' if not align_time else 'Aligned Time (s)'
        else:
            x_data = pd.Series(df.index, dtype=float)
            x_label = 'Frame' if not align_time else 'Aligned Frame'

        if align_time and len(x_data) > 0:
            x_data = x_data - x_data.iloc[0]

        return x_data, x_label


def interactive_mode(data_root: str):
    """Streamlit交互式模式"""
    try:
        st.set_page_config(page_title="运动数据可视化", layout="wide")
    except Exception as e:
        print("\n错误: 交互式模式必须使用 streamlit 命令启动！")
        return

    st.title("🏃 人体运动数据可视化工具")
    st.markdown("*v3.3: 简化版 - 每个参数独立显示*")

    try:
        visualizer = MotionDataVisualizer(data_root)
    except ValueError as e:
        st.error(f"错误: {e}")
        return

    st.info(f"📂 数据路径: `{visualizer.data_root}`")

    st.sidebar.header("📁 数据选择")

    # 1. 选择人名
    all_subjects = visualizer.get_all_subjects()
    if not all_subjects:
        st.error(f"未找到数据")
        return

    selected_subjects = st.sidebar.multiselect(
        "选择人名 (可多选)",
        all_subjects,
        default=all_subjects[:1] if all_subjects else [],
        help="选择一个或多个受试者"
    )

    if not selected_subjects:
        st.info("👈 请选择至少一个人名")
        return

    st.sidebar.success(f"已选择 {len(selected_subjects)} 个受试者")

    # 2. 选择运动类型
    common_motions = visualizer.get_motions_for_subjects(selected_subjects)
    if not common_motions:
        st.warning("⚠️ 没有共同的运动类型")
        return

    selected_motions = st.sidebar.multiselect(
        "选择运动类型 (可多选)",
        common_motions,
        default=common_motions[:1] if common_motions else [],
        help="只显示所有选中受试者共有的运动类型"
    )

    if not selected_motions:
        st.info("👈 请选择至少一个运动类型")
        return

    st.sidebar.success(f"已选择 {len(selected_motions)} 个运动类型")

    # 3. 选择文件类型
    file_availability = visualizer.check_file_type_availability(selected_subjects, selected_motions)
    available_types = [ft for ft, available in file_availability.items() if available]

    if not available_types:
        st.warning("⚠️ 未找到匹配的数据文件")
        return

    file_type_options = {ft: visualizer.file_types[ft] for ft in available_types}

    selected_file_types = st.sidebar.multiselect(
        "选择数据类型 (可多选)",
        options=list(file_type_options.keys()),
        default=[available_types[0]],
        format_func=lambda x: file_type_options[x],
        help="可以同时选择传感器和力矩数据"
    )

    if not selected_file_types:
        st.info("👈 请选择至少一种数据类型")
        return

    # 1. 获取可用列
    with st.spinner("正在扫描文件列..."):
        columns_dict = visualizer.get_available_columns(
            selected_file_types, selected_subjects, selected_motions
        )

    if not columns_dict:
        st.warning("⚠️ 未找到可用的数据列")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 选择参数")

    # 为不同类型的列添加标签
    column_labels = {}
    for col, ftype in columns_dict.items():
        label = f"{col} [{ftype}]"
        column_labels[label] = col

    default_labels = list(column_labels.keys())[:4]

    selected_labels = st.sidebar.multiselect(
        f"可用参数 ({len(columns_dict)} 个)",
        list(column_labels.keys()),
        default=default_labels,
        help="每个参数将独立显示在一个子图中"
    )

    if not selected_labels:
        st.info("👈 请选择至少一个参数")
        return

    selected_columns = [column_labels[label] for label in selected_labels]

    # 5. 显示选项
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 显示选项")

    align_time = st.sidebar.checkbox(
        "🔄 对齐时间轴",
        value=False,
        help="将所有曲线的起始时间设为0"
    )

    # 显示当前选择总结
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 当前选择")
    st.sidebar.write(f"**人名:** {len(selected_subjects)} 个")
    st.sidebar.write(f"**运动:** {len(selected_motions)} 个")
    st.sidebar.write(f"**数据类型:** {', '.join([visualizer.file_types[ft] for ft in selected_file_types])}")
    st.sidebar.write(f"**参数:** {len(selected_columns)} 个")
    st.sidebar.write(f"**时间对齐:** {'是' if align_time else '否'}")

    total_combinations = len(selected_subjects) * len(selected_motions)
    st.sidebar.info(f"💡 将生成 {len(selected_columns)} 个子图\n每个子图包含 {total_combinations} 条曲线")

    # 6. 生成可视化
    with st.spinner('正在生成图表...'):
        try:
            fig = visualizer.visualize(
                selected_subjects,
                selected_motions,
                selected_columns,
                columns_dict,
                align_time=align_time
            )

            status_msg = f"✓ 成功生成 {len(selected_columns)} 个子图"
            if align_time:
                status_msg += " (时间已对齐)"

            st.success(status_msg)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 保存图表"):
                    suffix = "_aligned" if align_time else ""
                    save_path = f"visualization{'_'.join(selected_file_types)}{suffix}.html"
                    fig.write_html(save_path)
                    st.success(f"已保存: {save_path}")

        except Exception as e:
            st.error(f"❌ 生成图表出错: {e}")
            with st.expander("错误详情"):
                st.exception(e)


def command_line_mode(config: Dict):
    """命令行模式"""
    data_root = config['data_root']
    subjects = config['subjects']
    motions = config['motions']
    file_types = config.get('file_types', [config.get('file_type', 'moment')])
    if isinstance(file_types, str):
        file_types = [file_types]
    columns = config['columns']
    save_path = config.get('save_path', 'output.html')
    align_time = config.get('align_time', False)

    try:
        visualizer = MotionDataVisualizer(data_root)
    except ValueError as e:
        print(f"错误: {e}")
        return

    print(f"\n配置信息：")
    print(f"  人名: {subjects}")
    print(f"  运动: {motions}")
    print(f"  文件类型: {file_types}")
    print(f"  列: {columns}")
    print(f"  时间对齐: {'是' if align_time else '否'}")

    try:
        columns_dict = visualizer.get_available_columns(file_types, subjects, motions)

        fig = visualizer.visualize(subjects, motions, columns, columns_dict,
                                  save_path, align_time)
        print(f"\n✓ 可视化完成！")
        fig.show()
    except Exception as e:
        print(f"\n✗ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='人体运动数据可视化工具 (v3.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 交互式模式 (推荐):
     streamlit run motion_data_visualizer.py -- --data_root ./data --interactive
  
  2. 命令行模式:
     python motion_data_visualizer.py --data_root ./data \\
            --subjects subject1 subject2 --motions walk run \\
            --file_types exo moment --columns col1 col2 \\
            --align_time
  
  3. 配置文件模式:
     python motion_data_visualizer.py --config config.yaml

v3.3 更新:
  - 简化操作，去掉叠加显示功能
  - 每个参数独立显示在一个子图中
  - 支持多选文件类型
  - 支持时间对齐
        """
    )

    parser.add_argument('--data_root', type=str, help='数据根目录路径')
    parser.add_argument('--interactive', action='store_true', help='启动交互式界面')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--subjects', nargs='+', help='人名列表')
    parser.add_argument('--motions', nargs='+', help='运动类型列表')
    parser.add_argument('--file_types', nargs='+', choices=['exo', 'moment'],
                       help='文件类型')
    parser.add_argument('--columns', nargs='+', help='列名')
    parser.add_argument('--save_path', type=str, default='output.html')
    parser.add_argument('--align_time', action='store_true',
                       help='对齐时间轴')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        command_line_mode(config)
    elif args.interactive:
        if not args.data_root:
            parser.error("需要指定 --data_root")
        interactive_mode(args.data_root)
    elif all([args.data_root, args.subjects, args.motions, args.file_types, args.columns]):
        config = {
            'data_root': args.data_root,
            'subjects': args.subjects,
            'motions': args.motions,
            'file_types': args.file_types,
            'columns': args.columns,
            'save_path': args.save_path,
            'align_time': args.align_time
        }
        command_line_mode(config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()