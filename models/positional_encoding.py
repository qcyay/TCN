import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    '''
    实现标准的正弦位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    '''

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # 应用sin和cos函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为buffer，不会被视为模型参数，但会随模型一起移动设备
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]

        返回:
            加上位置编码后的张量 [batch_size, seq_len, d_model]
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    '''
    可学习的位置编码
    '''

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建可学习的位置编码参数
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]

        返回:
            加上位置编码后的张量 [batch_size, seq_len, d_model]
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 测试代码
if __name__ == "__main__":
    # 测试标准位置编码
    d_model = 128
    seq_len = 50
    batch_size = 32

    # 创建位置编码层
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=1000, dropout=0.1)

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 应用位置编码
    output = pos_encoder(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码形状: {pos_encoder.pe.shape}")

    # 测试可学习位置编码
    learnable_pos_encoder = LearnablePositionalEncoding(d_model=d_model, max_len=1000, dropout=0.1)
    output2 = learnable_pos_encoder(x)
    print(f"\n可学习位置编码输出形状: {output2.shape}")
    print(f"可学习位置编码参数形状: {learnable_pos_encoder.pe.shape}")