import model
from spearman import get_spearman_cor
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
    模型会把这些列视为独立的，这意味着不能学习如“g”对“f”的依赖性，但允许更有效
    """
    # 将数据集划分为bsz部分
    device = torch.device("cpu")
    nbatch = data.size(0) // batch_size
    # 剔除多余的余数个元素
    data = data.narrow(0, 0, nbatch * batch_size)
    # 在batch_size批次中平均划分数据
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


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


def evaluate(data_source):
    # 模型评估.
    model.eval()
    total_loss = 0.
    vocab_size = len(corpus.dictionary)
    if args.model == 'FNN':
        vocab_size = vocab_size + 1
    if args.model not in ['Transformer', 'FNN']:
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            if args.model in ['Transformer', 'FNN']:
                output = model(data)
                output = output.view(-1, vocab_size)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
        get_spearman_cor(args.ws_txt, model, corpus.dictionary.word2idx, args.cuda)
    return total_loss / (len(data_source) - 1)
