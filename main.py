# coding: utf-8
import time
import model
import data2corpus
import torch.nn as nn
from utils import *


def train(train_data, epoch):
    """开始训练(可中断)"""
    model.train()
    total_loss = 0.
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i)
        # 从每一批开始，隐藏状态与以前的生产方式分离,否则模型会一直反向传播到数据集开始
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # 使用`clip_grad_norm防止RNN/LSTM中的爆炸梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.seq_len, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':

    # 加载语料库数据
    corpus = data2corpus.Corpus(args.data)
    # 评估时每批大小
    eval_batch_size = 10
    # 训练集，验证集，测试集
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    print(corpus.dictionary.word2idx)

    # 搭建模型
    # 语料库大小
    vocab_size = len(corpus.dictionary)
    print("Vocabulary size: {}".format(vocab_size))
    print("Project will run at: {}".format(device))
    model = model.RNNModel(
        args.model, vocab_size, args.embedding_dim, args.hidden_dim,
        args.num_of_layers, args.dropout, args.tied).to(device)

    # 学习率，当前最优损失
    lr = args.lr
    best_val_loss = None

    # 负对数损失函数(Negative Log Likelihood Loss)
    criterion = nn.NLLLoss()
    # 优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 开始训练(可以CTRL+C提前中止)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(train_data, epoch)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # 保存最优模型
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # 如果验证数据集中未发现任何改进，则减小学习率
                lr /= 4.0
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # 保存最优模型
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # 跑一下测试集
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
