import math
from parameters import *


# 批次处理
def batchify(data, batch_size):
    """
    从顺序数据开始，batchify将数据集排列成列。
    例如，使用字母表作为序列，批量大小为4，我们会得到
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    模型会把这些列视为独立的，这意味着不能学习如“g”对“f”的依赖性
    """
    device = torch.device("cpu")  # 创建一个CPU的设备对象
    nbatch = data.size(0) // batch_size  # 计算总批次数
    # 剔除多余的余数个元素
    data = data.narrow(0, 0, nbatch * batch_size)  # 窄化数据，去掉多余的元素
    # 在batch_size批次中平均划分数据
    data = data.view(batch_size, -1).t().contiguous()  # 将数据reshape为batch_size批次，并转置和连续化
    return data.to(device)  # 将数据移动到指定的设备上


###############################################################################
# Training code
###############################################################################
def repackage_hidden(h):
    """打包隐藏的状态，将它们从过去训练中分离出来。"""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    """
    get_batch将源数据细分为长度为args.seq_len的块。
    如果source等于batchify函数的输出，seq_len-limit为2，我们将获得以下两个i=0的变量：
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    因为这是由批处理函数处理的，数据的细分并不是沿着批次维度（即维度1）进行的
    块沿着维度0，对应于LSTM中的seq_len维度
    """
    seq_len = min(args.seq_len, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def cosine_similarity(list1, list2):
    # 计算分子（向量点积）
    dot_product = 0
    for i in range(len(list1)):
        dot_product += list1[i] * list2[i]

    # 计算分母（向量模长乘积）
    norm_list1 = 0
    norm_list2 = 0
    for i in range(len(list1)):
        norm_list1 += math.pow(list1[i], 2)
        norm_list2 += math.pow(list2[i], 2)
    norm_list1 = math.sqrt(norm_list1)
    norm_list2 = math.sqrt(norm_list2)

    # 计算余弦相似度
    similarity = dot_product / (norm_list1 * norm_list2)
    return similarity