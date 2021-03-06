import torch

emb_dim = 10
hidden_dim = 20
num_layers = 1
lstm = torch.nn.LSTM(emb_dim, hidden_dim, num_layers)

seq_len = 5  # 序列长度，即一句话又几个词
batch_size = 3  # 批处理大小
input_data = torch.randn(seq_len, batch_size, emb_dim)  # [5, 3, 10]

# 初始化的隐藏元和记忆元,通常它们的维度是一样的
# 2个LSTM层，batch_size=3,隐藏元维度20
h0 = torch.randn(num_layers, batch_size, hidden_dim)  # [2, 3, 20]
c0 = torch.randn(num_layers, batch_size, hidden_dim)  # [2, 3, 20]

output, (hn, cn) = lstm(input_data, (h0, c0))

print(output.size(), hn.size(), cn.size())
# torch.Size([5, 3, 20]) torch.Size([2, 3, 20]) torch.Size([2, 3, 20])
