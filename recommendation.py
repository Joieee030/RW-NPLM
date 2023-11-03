import numpy as np
import pandas as pd
from tqdm import tqdm
import data2corpus
from utils import cosine_similarity
from parameters import *

# 手动设置随机种子以确保可重复性。
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

corpus = data2corpus.Corpus(args.data)

# 初始化隐藏层
hidden = model.init_hidden(1)

df = pd.DataFrame(pd.read_excel(args.node_path))
paper_index_in_excel = [x for x in range(df.shape[0]) if df.iloc[x]["类型"] == "paper"]

# 无需计算梯度
with torch.no_grad():
    words = []
    for item in tqdm(paper_index_in_excel):
        paper_index = torch.tensor(corpus.dictionary.word2idx[str(item)]).view(1, 1)
        # 生成长度为words的句子
        sentence = []
        for i in range(args.words):
            # 输入paper_index和hidden状态，生成输出output和新的hidden状态
            output, hidden = model(paper_index, hidden)

            # 将output的维度减小到一维，并将张量转移到CPU上
            word_weights = output.squeeze().div(args.temperature).exp().cpu()

            # 从word_weights中使用多元采样方法选择一个word_idx
            word_idx = torch.multinomial(word_weights, 1)[0]

            # 将paper_index的每个元素设置为所选的word_idx
            paper_index.fill_(word_idx)

            # 将word_idx转换为对应的word并添加到words列表中
            sentence.append(int(corpus.dictionary.idx2word[word_idx]))
        words.append(sentence)
    # print(words)

    # 计算相似度
    scale = len(paper_index_in_excel)
    sim_metric = np.zeros((scale, scale))
    recommendation = {}
    for i in range(scale):
        for j in range(i + 1, scale):
            sim_metric[i, j] = cosine_similarity(words[i], words[j])
    print(sim_metric)

    # 输出每行最大相似度的论文名
    max_indexes = np.argmax(sim_metric, axis=1)
    for i in range(scale):
        recd_paper_index = paper_index_in_excel[max_indexes[i]]
        recommendation["论文" + str(i + 1) + "推荐"] = df.iloc[recd_paper_index]["值"]
    print(recommendation)
