import torch
import pandas as pd
from torch_cluster import random_walk


class MyWalker(object):	
    
    def walker(nodeTable, edgeTable, walkLength, startNodes, epoch):
        row=[]
        col=[]
        nodeTable = pd.DataFrame(nodeTable)
        edgeTable = pd.DataFrame(edgeTable)
        for item in nodeTable.iloc[:,0]:
            for node in edgeTable.iloc[:,1]:
                if item == node:
                    row.append(node)
            for i in range(len(edgeTable)):
                if item == edgeTable.iloc[i,0]:
                    col.append(edgeTable.iloc[i,1])
        row = torch.tensor(row)
        col = torch.tensor(col)
        startNodes = torch.tensor(startNodes)
        walks = []

        for i in range(epoch):
            walks.append(random_walk(row, col, startNodes, walk_length=walkLength))

        return walks
