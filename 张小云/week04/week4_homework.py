import torch
import torch.nn as nn
import math

# ==========================================
# 1. 缩放点积注意力 (Scaled Dot-Product Attention)
# ==========================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  #这里是给这个类加了一个dropout这个功能，防死记硬背的功能，随机关掉一部分权重
            #比如有一些模型被训练得比较好，会导致一些模型没有训练到，dropout的作用的把某些神经元暂时关闭像开关一样，让每一个神经元都能参与训练，不要偷懒。
    def forward(self, query, key, value, mask=None):  #query=查询；key：字典里的索引；value=字典里的具体内容
        # 获取 d_k，用于缩放
        d_k = query.size(-1)   #获取获取张量q的最后一个维度的大小，64维的就是64，128维的就是128。这里是为了之后的缩放做准备。
        
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # transpose(-2, -1) 用于转置最后两个维度
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #math.sqrt使用python自带的数学给d_k开数字平方根，如果上面d_k这里是128就是128开根号
                    #torch.matmul这是矩阵乘法它拿Query去和key做向量内积
        # 如果有掩码（例如填充位），将其设为负无穷，Softmax后概率为0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    #masked_fill如果在scores里面找到0(这里相当于mask=0)的数据把它改为-1e9，相当于改为无效值。
                                                        
        # 计算权重并应用 Dropout
        attn_weights = torch.softmax(scores, dim=-1) #我的scores里面的每一个向量过softmax后必须=1，就是dim=-1，用dim=-1是因为-1是向量里面的最后一个数，这样的话就是默认我这个向量从第一个开始到最后一个，加和为1.
        attn_weights = self.dropout(attn_weights) #dropout是把attn_weights过dropout然后得到attn_weights
        
        # 输出 = 权重 * V
        output = torch.matmul(attn_weights, value)  #这里是把attn_weights和value做矩阵乘法得到output。
        return output

# ==========================================
# 2. 多头注意力 (Multi-Head Attention)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
                                         #assert相当于代码里面的安检员，如果条件为真，什么都不做，条件为假会立刻停下来。
        self.d_model = d_model  #加了self是为了后面用起来方便，后面需要用到这两个参数，直接用就好了
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)     #q相当于我的择偶标准：如漂亮的性格好的
        self.w_k = nn.Linear(d_model, d_model)     #k是别人的特征标签，：漂亮的，凶的，有钱的
        self.w_v = nn.Linear(d_model, d_model)     #v是她的真实内涵，谈吐性格
        self.w_o = nn.Linear(d_model, d_model)     #q*k：算出谁和谁是真爱，生成一张配方表(分数)；v根据配方表的比例，把所有的v(食材)按比例调和在一起(看哪个分数最高就和q*k的调和在一起)
                                                   #w_o是把v(因为v1里面算出来的信息需要整合一下)得到的值整合在一起linear的作用是通过矩阵运算把原本的
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  #取出query里面的第一个数。
        
        # 1. 线性变换并分头
        # 形状变化: (batch, seq_len, d_model) -> (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #view是pytorch自带的一个功能，可以理解为捏橡皮泥或者重新包装，transpose是把前面的这个位置交换一下。
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)   #-1的位置是序列长度。
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 并行计算注意力
        attn_output = self.attention(q, k, v, mask)
        
        # 3. 拼接多头结果并投影
        # 形状变化: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) #transpose(1,2)表面上看我的数据变了，但只是电脑以为我的数据的位置变了，实际没有变。
        output = self.w_o(attn_output)                                                     #congiguous的作用是复制我原本的atten_output然后给到后面的view来计算。
        
        return self.dropout(output)  #把output带进dropout再过一遍，Dropout是在训练过程中随机把输出向量里的一部分数字变成0，为了防止过拟合。

# ==========================================
# 3. 前馈网络 (Position-wise Feed-Forward Network)
# ==========================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)   #dropout把一部分信息置为0，为了防止过拟合。
        self.relu = nn.ReLU()   #过滤掉负面的无关的信息，只留下精华。

    def forward(self, x):
        # 对每个位置独立进行变换
        return self.linear2(self.dropout(self.relu(self.linear1(x)))) #这里是从linear1先把向量扩大，然后过一遍rulu把负数过滤掉，
                                                               #然后dropout把一部分数字变成0，然后linear2再把维度压缩回原来的大小
                                                                
# ==========================================
# 4. Transformer Encoder Layer (完整层)
# ==========================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super().__init__()
        # 初始化子模块
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)  #LayerNorm是在前向传播的过程中边优化边训练，确保后面的数据能学的快。
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # --- 子层 1: 多头自注意力 + 残差 + 归一化 ---
        # Pre-Norm 结构 (x + Sublayer(Norm(x))) 或者 Post-Norm (Norm(x + Sublayer(x)))
        # 这里采用常见的 Post-Norm 写法：Norm(x + Sublayer(x))
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))  #残差连接，是先attn_output然后src加上attn_output再标准化一下norm1
        
        # --- 子层 2: 前馈网络 + 残差 + 归一化 ---
        ff_output = self.feed_forward(src)
        out = self.norm2(src + self.dropout(ff_output))
        
        return out

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 假设参数
    batch_size = 4
    seq_len = 10
    d_model = 512
    nhead = 8
    d_ff = 2048
    
    # 实例化模型
    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff)
    
    # 创建随机输入 (Batch, Seq_Len, Dim)
    x = torch.rand(batch_size, seq_len, d_model)  #torch.rand是随机数，在这里虽然参数都有写，但是为了防止模型偷懒，所以还是用的随机数
    
    # 前向传播
    output = encoder_layer(x)  #这里是把上面算出来的x带进encoder_layer算出来算出output
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
