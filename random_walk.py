import torch
import pandas as pd
import numpy as np
from torch_cluster import random_walk


def excel2table(excel_path):
    df = pd.read_excel(excel_path)
    return df


class MyWalker(object):

    def walker(self, node_path, edge_path, walk_length, epoch, filename):
        row = []
        col = []
        walks = []
        node_table = excel2table(node_path)
        edge_table = excel2table(edge_path)
        for item in node_table.iloc[:, 0]:
            for node in edge_table.iloc[:, 1]:
                if item == node:
                    row.append(node)
            for i in range(len(edge_table)):
                if item == edge_table.iloc[i, 0]:
                    col.append(edge_table.iloc[i, 1])
        row = torch.tensor(row)
        col = torch.tensor(col)
        start_nodes = torch.tensor(
            [node_table.iloc[i, 0] for i in range(node_table.shape[0]) if node_table.iloc[i, 1] == "paper"])

        for _ in range(epoch):
            walks.append(random_walk(row, col, start_nodes, walk_length=walk_length))

        combined = np.vstack(walks)
        np.savetxt(filename, combined, delimiter=',', fmt='%d')

# test
# if __name__ == '__main__':
#     walker = MyWalker()
#     walker.walker(
#         node_path='./data/node_attr.xlsx',
#         edge_path='./data/relation.xlsx',
#         walk_length=5,
#         epoch=3,
#         filename='./data/walks.txt')