import torch
import torch.nn as nn
from typing import Union, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.positional_encoding import PositionalEncoding


class GenerativeTransformer(nn.Module):
    '''
    生成式Transformer模型
    - 编码器：处理传感器数据特征（可选Transformer Encoder或Linear）
    - 解码器：自回归生成力矩预测
    '''

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 sequence_length: int = 100,
                 encoder_type: str = 'transformer',
                 use_positional_encoding: bool = True,
                 center: Union[float, torch.Tensor] = 0.,
                 scale: Union[float, torch.Tensor] = 1.):
        """
        参数:
            input_size: 输入特征数量（传感器数据维度）
            output_size: 输出特征数量（力矩维度）
            d_model: 模型隐藏维度
            nhead: 多头注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            sequence_length: 序列长度
            encoder_type: 编码器类型 ('transformer' 或 'linear')
            use_positional_encoding: 是否使用位置编码
            center: 输入归一化中心值
            scale: 输入归一化缩放值
        """
        super(GenerativeTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.encoder_type = encoder_type
        self.use_positional_encoding = use_positional_encoding

        # 归一化参数
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)

        if center.dim() == 0:
            center = center.unsqueeze(0).unsqueeze(1)
        elif center.dim() == 1:
            center = center.unsqueeze(1)

        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(1)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(1)

        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

        # 输入投影层（传感器数据）
        self.input_projection = nn.Linear(input_size, d_model)

        # 编码器
        if encoder_type == 'transformer':
            # 使用Transformer Encoder
            if use_positional_encoding:
                self.encoder_pos_encoder = PositionalEncoding(
                    d_model=d_model,
                    max_len=sequence_length * 2,
                    dropout=dropout
                )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers
            )
        else:
            # 使用简单的线性层
            self.encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

        # 解码器输入投影（力矩数据）
        self.decoder_input_projection = nn.Linear(output_size, d_model)

        # 位置编码（解码器）
        if use_positional_encoding:
            self.decoder_pos_encoder = PositionalEncoding(
                d_model=d_model,
                max_len=sequence_length * 2,
                dropout=dropout
            )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        '''初始化模型权重'''
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.decoder_input_projection.weight)
        nn.init.zeros_(self.decoder_input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        编码传感器数据

        参数:
            src: [batch_size, num_features, sequence_length]

        返回:
            memory: [batch_size, sequence_length, d_model]
        """
        # 归一化
        src = (src - self.center) / self.scale

        # [B, C, N] -> [B, N, C]
        src = src.transpose(1, 2)

        # 输入投影
        src = self.input_projection(src)  # [B, N, d_model]

        # 编码
        if self.encoder_type == 'transformer':
            if self.use_positional_encoding:
                src = self.encoder_pos_encoder(src)
            memory = self.encoder(src)  # [B, N, d_model]
        else:
            # 线性编码器，逐时间步处理
            memory = self.encoder(src)  # [B, N, d_model]

        return memory

    def decode(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码生成力矩预测

        参数:
            tgt: [batch_size, output_size, sequence_length] 解码器输入
            memory: [batch_size, sequence_length, d_model] 编码器输出
            tgt_mask: 因果掩码

        返回:
            output: [batch_size, output_size, sequence_length]
        """
        # [B, output_size, N] -> [B, N, output_size]
        tgt = tgt.transpose(1, 2)

        # 解码器输入投影
        tgt = self.decoder_input_projection(tgt)  # [B, N, d_model]

        # 添加位置编码
        if self.use_positional_encoding:
            tgt = self.decoder_pos_encoder(tgt)

        # 解码
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [B, N, d_model]

        # 输出投影
        output = self.output_projection(output)  # [B, N, output_size]

        # [B, N, output_size] -> [B, output_size, N]
        output = output.transpose(1, 2)

        return output

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（训练模式）

        参数:
            src: [batch_size, num_features, sequence_length] 传感器数据
            tgt: [batch_size, output_size, sequence_length] 目标力矩数据（shifted）
            tgt_mask: 因果掩码

        返回:
            output: [batch_size, output_size, sequence_length] 预测的力矩
        """
        # 编码
        memory = self.encode(src)

        # 解码
        output = self.decode(tgt, memory, tgt_mask)

        return output

    def generate(self,
                 src: torch.Tensor,
                 start_token: Optional[torch.Tensor] = None,
                 max_len: Optional[int] = None) -> torch.Tensor:
        """
        自回归生成（测试模式）

        参数:
            src: [batch_size, num_features, sequence_length] 传感器数据
            start_token: [batch_size, output_size, 1] 起始token（默认为0）
            max_len: 生成的最大长度（默认为sequence_length）

        返回:
            output: [batch_size, output_size, max_len] 生成的力矩序列
        """
        batch_size = src.size(0)
        if max_len is None:
            max_len = self.sequence_length

        # 编码
        memory = self.encode(src)  # [B, seq_len, d_model]

        # 初始化起始token
        if start_token is None:
            # 使用0作为起始token
            current_output = torch.zeros(
                batch_size, self.output_size, 1,
                device=src.device
            )
        else:
            current_output = start_token

        # 自回归生成
        for i in range(max_len - 1):
            # 创建因果掩码
            tgt_len = current_output.size(2)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(src.device)

            # 解码当前序列
            output = self.decode(current_output, memory, tgt_mask)

            # 取最后一个时间步的预测
            next_token = output[:, :, -1:]  # [B, output_size, 1]

            # 拼接到输出序列
            current_output = torch.cat([current_output, next_token], dim=2)

        return current_output

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        生成因果掩码（上三角掩码）

        参数:
            sz: 序列长度

        返回:
            mask: [sz, sz] 因果掩码
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_num_params(self):
        '''返回模型参数数量'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 测试代码
if __name__ == "__main__":
    # 模型参数
    input_size = 25
    output_size = 2
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    sequence_length = 100
    batch_size = 16

    print("=" * 60)
    print("测试生成式Transformer模型")
    print("=" * 60)

    # 测试Transformer编码器版本
    print("\n1. 测试Transformer Encoder版本")
    model_transformer = GenerativeTransformer(
        input_size=input_size,
        output_size=output_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        sequence_length=sequence_length,
        encoder_type='transformer',
        use_positional_encoding=True
    )

    print(f"模型参数数量: {model_transformer.get_num_params():,}")

    # 创建测试数据
    src = torch.randn(batch_size, input_size, sequence_length)
    tgt = torch.randn(batch_size, output_size, sequence_length)

    print(f"输入形状: {src.shape}")
    print(f"目标形状: {tgt.shape}")

    # 测试训练模式
    print("\n训练模式 (teacher forcing):")
    tgt_mask = GenerativeTransformer._generate_square_subsequent_mask(sequence_length)
    output_train = model_transformer(src, tgt, tgt_mask)
    print(f"训练输出形状: {output_train.shape}")

    # 测试生成模式
    print("\n生成模式 (自回归):")
    with torch.no_grad():
        output_gen = model_transformer.generate(src, max_len=50)
    print(f"生成输出形状: {output_gen.shape}")

    # 测试Linear编码器版本
    print("\n2. 测试Linear Encoder版本")
    model_linear = GenerativeTransformer(
        input_size=input_size,
        output_size=output_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        sequence_length=sequence_length,
        encoder_type='linear',
        use_positional_encoding=True
    )

    print(f"模型参数数量: {model_linear.get_num_params():,}")

    output_linear = model_linear(src, tgt, tgt_mask)
    print(f"训练输出形状: {output_linear.shape}")

    # 测试损失计算
    print("\n3. 测试损失计算")
    criterion = nn.MSELoss()
    target = torch.randn(batch_size, output_size, sequence_length)
    loss = criterion(output_train, target)
    print(f"损失值: {loss.item():.6f}")

    # 测试反向传播
    loss.backward()
    print("反向传播成功!")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)