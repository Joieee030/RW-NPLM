import argparse
import torch
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/sentence', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, FNN)')
parser.add_argument('--embedding_dim', type=int, default=200, help='size of word embeddings')
parser.add_argument('--hidden_dim', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--num_of_layers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')

args = parser.parse_args()

# 选择设备
device = torch.device("cuda" if args.cuda else "cpu")

# 加载语料库数据
corpus = data.Corpus(args.data)

# 评估时每批大小
eval_batch_size = 10

# 语料库大小
vocab_size = len(corpus.dictionary)