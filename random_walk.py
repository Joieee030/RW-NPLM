import random
import pandas as pd
import numpy as np
from parameters import args


class MyWalker(object):

    def walker(self, walk_length, epoch, filename):
        df_edge = pd.DataFrame(pd.read_excel(args.edge_path))
        df_node = pd.DataFrame(pd.read_excel(args.node_path))
        start_nodes = [i for i in range(df_node.shape[0]) if df_node.iloc[i]["类型"] == "paper"]
        seqs = []

        for _ in range(epoch):
            seqs.append(self.random_walk(df_edge, start_nodes, walk_length=walk_length))
        seqs = np.vstack(seqs)
        np.savetxt(filename, seqs, delimiter=' ', fmt='%d', newline=' ')

    # 定义游走规则
    @staticmethod
    def random_walk(df, start_points, walk_length):
        # 参数：
        # df: 数据，包含点1和点2的列
        # start_points: 起始点列表
        # walk_length: 游走的长度
        # 返回值：游走序列列表

        for start_point in start_points:
            walk_sequence = [start_point]
            current_point = start_point
            step = 0

            while step < walk_length:
                # 从数据框中筛选与当前点相关的行
                filtered_df = df[df['点1'] == current_point]

                if len(filtered_df) == 0:
                    break

                # 随机选择下一个点
                next_index = random.randint(0, len(filtered_df) - 1)
                next_point = filtered_df.iloc[next_index]['点2']

                # 添加到游走序列中
                walk_sequence.append(next_point)
                current_point = next_point
                step += 1

            return walk_sequence


# test
if __name__ == '__main__':
    walker = MyWalker()
    walker.walker(walk_length=100,
                  epoch=100,
                  filename='./data/sentence/test.txt')
    walker.walker(walk_length=100,
                  epoch=1000,
                  filename='./data/sentence/train.txt')
    walker.walker(walk_length=100,
                  epoch=100,
                  filename='./data/sentence/valid.txt')
    print("Done!")