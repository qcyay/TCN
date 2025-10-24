import torch
import torch.nn as nn
from typing import Union
import sys
import os

# 添加父目录到路径以便导入positional_encoding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.positional_encoding import PositionalEncoding


class PredictorTransformer(nn.Module):
    '''
    基于Transformer Encoder的预测模型
    用于从传感器数据直接预测力矩值
    '''

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 sequence_length: int = 100,
                 use_positional_encoding: bool = True,
                 center: Union[float, torch.Tensor] = 0.,
                 scale: Union[float, torch.Tensor] = 1.):
        """
        参数:
            input_size: 输入特征数量
            output_size: 输出特征数量
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_encoder_layers: Encoder层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            sequence_length: 序列长度
            use_positional_encoding: 是否使用位置编码
            center: 输入数据的中心值（用于归一化）
            scale: 输入数据的缩放值（用于归一化）
        """
        super(PredictorTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.use_positional_encoding = use_positional_encoding

        # 将center和scale转换为tensor并注册为buffer
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)

        # 确保center和scale的形状正确 [C, 1] 用于广播
        if center.dim() == 0:  # 标量
            center = center.unsqueeze(0).unsqueeze(1)
        elif center.dim() == 1:  # [C]
            center = center.unsqueeze(1)

        if scale.dim() == 0:  # 标量
            scale = scale.unsqueeze(0).unsqueeze(1)
        elif scale.dim() == 1:  # [C]
            scale = scale.unsqueeze(1)

        # 注册为buffer
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

        # 输入投影层: 将输入特征投影到d_model维度
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=d_model,
                max_len=sequence_length * 2,  # 留出余量
                dropout=dropout
            )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 使用 [batch, seq, feature] 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        '''初始化模型权重'''
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, num_features, sequence_length]

        返回:
            output: 输出张量 [batch_size, num_outputs, sequence_length]
        """
        # x: [B, C, N]
        batch_size, num_features, seq_len = x.shape

        # 归一化输入特征
        # center, scale: [C, 1] -> 广播到 [B, C, N]
        x = (x - self.center) / self.scale

        # 转换维度: [B, C, N] -> [B, N, C]
        x = x.transpose(1, 2)

        # 输入投影: [B, N, C] -> [B, N, d_model]
        x = self.input_projection(x)

        # 添加位置编码
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        # Transformer Encoder: [B, N, d_model] -> [B, N, d_model]
        x = self.transformer_encoder(x)

        # 输出投影: [B, N, d_model] -> [B, N, output_size]
        x = self.output_projection(x)

        # 转换回原始格式: [B, N, output_size] -> [B, output_size, N]
        x = x.transpose(1, 2)

        return x

    def get_num_params(self):
        '''返回模型的参数数量'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 测试代码
if __name__ == "__main__":
    # 模型参数
    input_size = 25
    output_size = 2
    d_model = 128
    nhead = 8
    num_encoder_layers = 4
    dim_feedforward = 512
    dropout = 0.1
    sequence_length = 100
    batch_size = 32

    # 创建模型
    model = PredictorTransformer(
        input_size=input_size,
        output_size=output_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        sequence_length=sequence_length,
        use_positional_encoding=True
    )

    print(f"模型参数数量: {model.get_num_params():,}")

    # 创建随机输入
    x = torch.randn(batch_size, input_size, sequence_length)
    print(f"输入形状: {x.shape}")

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 测试模型可以正常训练
    criterion = nn.MSELoss()
    target = torch.randn(batch_size, output_size, sequence_length)
    loss = criterion(output, target)
    print(f"损失值: {loss.item():.6f}")

    # 反向传播测试
    loss.backward()
    print("反向传播成功!")