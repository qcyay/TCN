# 人体运动数据可视化工具使用说明 (v3.3 简化版)

## 🆕 最新更新 (v3.3)

**简化操作 - 清晰易用 🎨**
- ✅ **去掉复杂的叠加功能**：每个参数独立显示，清晰明了
- ✅ **支持多选文件类型**：可同时选择传感器和力矩数据
- ✅ **每个参数一个子图**：便于详细查看每个参数的变化
- ✅ **多人对比**：在同一子图中显示多人同一参数的曲线
- ✅ **时间对齐**：统一起始时间，便于对比

## 📋 核心功能

- ✅ 多人多运动数据对比
- ✅ 时间对齐功能
- ✅ 多选文件类型（exo 和/或 moment）
- ✅ 每个参数独立显示
- ✅ 交互式Web界面
- ✅ 命令行批量处理

## 🔧 安装依赖

```bash
pip install pandas plotly streamlit pyyaml
```

## 🚀 快速开始

### 交互式模式（推荐）

```bash
streamlit run motion_data_visualizer.py -- --data_root ./data --interactive
```

**简单5步：**
1. 选择人名（可多选）
2. 选择运动类型（可多选）
3. 选择数据类型（exo 和/或 moment）
4. 选择参数（可多选）
5. 查看结果！

### 命令行模式

```bash
python motion_data_visualizer.py \
  --data_root ./data \
  --subjects subject1 subject2 \
  --motions walking \
  --file_types exo moment \
  --columns hip_flexion_l hip_flexion_l_moment knee_angle_l \
  --align_time
```

### 配置文件模式

```yaml
file_types: ["exo", "moment"]
columns:
  - hip_flexion_l
  - hip_flexion_l_moment
  - knee_angle_l
  - knee_angle_l_moment
align_time: true
```

```bash
python motion_data_visualizer.py --config config.yaml
```

## 📊 显示效果

### 每个参数独立显示

选择4个参数时，会生成4个子图：

```yaml
columns:
  - hip_flexion_l
  - hip_flexion_l_moment
  - knee_angle_l
  - knee_angle_l_moment
```

**结果：**
```
子图1: hip_flexion_l
  - subject1-walking
  - subject2-walking

子图2: hip_flexion_l_moment
  - subject1-walking
  - subject2-walking

子图3: knee_angle_l
  - subject1-walking
  - subject2-walking

子图4: knee_angle_l_moment
  - subject1-walking
  - subject2-walking
```

### 多人对比

选择3个人 + 2种运动时：

```yaml
subjects: [subject1, subject2, subject3]
motions: [walking, running]
columns: [hip_flexion_l]
```

**结果：**
```
子图: hip_flexion_l
  - subject1-walking
  - subject1-running
  - subject2-walking
  - subject2-running
  - subject3-walking
  - subject3-running

→ 6条曲线在同一子图中
```

## 💡 使用示例

### 示例1：查看传感器数据

```yaml
subjects: [subject1]
motions: [walking]
file_types: ["exo"]
columns:
  - hip_flexion_l
  - knee_angle_l
  - ankle_angle_l
```

**结果：** 3个子图，每个显示一个传感器数据

### 示例2：查看力矩数据

```yaml
subjects: [subject1, subject2]
motions: [walking]
file_types: ["moment"]
columns:
  - hip_flexion_l_moment
  - knee_angle_l_moment
```

**结果：** 2个子图，每个包含2条曲线（2个人）

### 示例3：同时查看传感器和力矩

```yaml
subjects: [subject1]
motions: [walking]
file_types: ["exo", "moment"]
columns:
  - hip_flexion_l
  - hip_flexion_l_moment
  - knee_angle_l
  - knee_angle_l_moment
```

**结果：** 4个子图，分别显示不同参数

### 示例4：多人多运动对比

```bash
streamlit run motion_data_visualizer.py -- --data_root ./data --interactive
```

操作：
- 选择3个人
- 选择2种运动
- 选择类型：exo
- 选择参数：hip_flexion_l, knee_angle_l

**结果：** 2个子图，每个包含6条曲线（3人×2运动）

## 📁 数据结构

```
data_root/
├── subject1/
│   ├── walking/
│   │   ├── exo.csv
│   │   └── joint_moments_filt.csv
│   └── running/
│       ├── exo.csv
│       └── moment_filt.csv
└── subject2/
    └── walking/
        ├── exo.csv
        └── joint_moments_filt.csv
```

### 文件命名规则

**传感器数据（exo）：**
- ✅ `exo.csv`
- ✅ `subject1_exo.csv`
- ✅ `walking_exo.csv`
- ❌ `power_exo.csv`（不符合规则）

**力矩数据（moment）：**
- ✅ `moment_filt.csv`
- ✅ `joint_moments_filt.csv`
- ✅ `walking_moment_filt.csv`
- ❌ `moment.csv`（缺少_filt后缀）

## 🎨 图表特点

- **清晰布局**：每个参数独立子图
- **颜色区分**：不同人/运动使用不同颜色
- **交互功能**：缩放、平移、悬停查看数值
- **图例控制**：点击显示/隐藏曲线
- **时间对齐**：可选统一起始时间

## 📊 图表交互

- **缩放**：鼠标滚轮或框选区域
- **平移**：按住鼠标左键拖动
- **重置**：双击图表
- **悬停**：查看具体数值
- **图例**：点击显示/隐藏对应曲线
- **保存**：工具栏相机图标保存为PNG

## 🐛 常见问题

### 问题1：未找到数据文件

**检查：**
1. 数据路径是否正确
2. 文件命名是否符合规则
3. 文件是否在正确的目录结构中

### 问题2：列名不存在

**检查：**
1. CSV文件是否包含该列
2. 列名是否完全匹配（区分大小写）
3. 列名中是否有多余空格

### 问题3：多人没有共同运动

**解决：**
- 只选择有共同运动的人
- 或在各人目录下添加相应运动数据

### 问题4：图表太多太密集

**解决：**
- 减少选择的参数数量
- 分批次查看不同参数
- 使用更大的显示器

## ⚙️ 配置选项

### 必需参数

- `data_root`: 数据根目录
- `subjects`: 人名列表
- `motions`: 运动类型列表
- `file_types`: 文件类型（exo/moment）
- `columns`: 要显示的列名

### 可选参数

- `align_time`: 时间对齐（true/false）
- `save_path`: 输出文件路径

## 📝 命令行示例

**基本用法：**
```bash
python motion_data_visualizer.py \
  --data_root ./data \
  --subjects subject1 \
  --motions walking \
  --file_types exo \
  --columns hip_flexion_l knee_angle_l
```

**时间对齐：**
```bash
python motion_data_visualizer.py \
  --data_root ./data \
  --subjects subject1 subject2 \
  --motions walking \
  --file_types moment \
  --columns hip_flexion_l_moment knee_angle_l_moment \
  --align_time
```

**多文件类型：**
```bash
python motion_data_visualizer.py \
  --data_root ./data \
  --subjects subject1 \
  --motions walking \
  --file_types exo moment \
  --columns hip_flexion_l hip_flexion_l_moment \
  --save_path output.html
```

## ⚠️ 注意事项

1. **文件命名**：严格遵守命名规则
2. **CSV格式**：需要包含time列作为时间轴
3. **列名匹配**：区分大小写，完全匹配
4. **共同运动**：多人选择时必须有共同运动类型
5. **编码格式**：建议使用UTF-8编码

## 🔄 版本历史

- **v3.3**: 简化操作，去掉叠加功能，每个参数独立显示
- v3.2: 支持多选文件类型，双轴叠加显示
- v3.1: 新增时间对齐功能
- v3.0: 简化文件选择，支持多人对比
- v1.0: 初始版本

## 📧 技术支持

遇到问题时：
1. 查看终端日志输出
2. 确认文件命名规则
3. 检查数据结构
4. 使用交互式模式逐步排查

---

**简单、清晰、易用！**