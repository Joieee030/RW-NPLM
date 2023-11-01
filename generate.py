# This file generates new sentences sampled from the language model

import argparse
import torch
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/sentence',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--model', type=str, default=None,
                    help='Model Type')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
args = parser.parse_args()

# 手动设置随机种子以确保可重复性。
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

corpus = data.Corpus(args.data)
# n_tokens = len(corpus.dictionary)

# 初始化隐藏层
hidden = model.init_hidden(1)

"""这里的input改成自己想要生成的"""
# input = torch.randint(n_tokens, (1, 1), dtype=torch.long).to(device)
paper_name = "paper_1"
paper_index = torch.tensor(corpus.dictionary.word2idx[paper_name]).view(1, 1)

with open(args.outf, 'w') as output_file:
    # 无需计算梯度
    with torch.no_grad():
        # 对于每个word
        for i in range(args.words):
            # 输入paper_index和hidden状态，生成输出output和新的hidden状态
            output, hidden = model(paper_index, hidden)
            # 将output的维度减小到一维，并将张量转移到CPU上
            word_weights = output.squeeze().div(args.temperature).exp().cpu()

            # 从word_weights中使用多元采样方法选择一个word_idx
            word_idx = torch.multinomial(word_weights, 1)[0]
            # 将paper_index的每个元素设置为所选的word_idx
            paper_index.fill_(word_idx)

            # 将word_idx转换为对应的word
            word = corpus.dictionary.idx2word[word_idx]

            # 将生成的word写入output_file（换行符每19个word使用一次）
            output_file.write(word + ('\n' if i % 20 == 19 else ' '))

            # 每经过args.log_interval个iteration，打印生成的word数量
            if i % args.log_interval == 0:
                print('| 生成了 {} 个word'.format(i, args.words))