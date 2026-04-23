import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
from torch_geometric.nn import AGNNConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import ARMAConv
from torch.nn import Linear
import torch_geometric.transforms as T
import random
import numpy as np
import math


class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNClassifier:
    def __init__(self, gcn_type, rks, pN, number_neighbors=80, learning_rate):
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = learning_rate
        self.pNNeurons = 32
        self.pNEpochs = 200
        self.gcn_type = gcn_type
    
    #Separar as mascaras e labels
    def prepare(self, test_index, train_index, features, labels, matrix, limiar= 0.8, correlation='normal', dataset='flowers', cnn='resnet'):
        #print('Creating Masks...')
        self.train_mask = [False] * self.pN
        self.val_mask = [False] * self.pN
        self.test_mask = [False] * self.pN

        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True

        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)

        #print('Set Labels...')
        y = labels
        self.numbersOfClasses = max(y) + 1
        self.y = torch.tensor(y)

        self.x = torch.tensor(features)
        self.pNFeatures = len(features[0])

        self.create_graph(correlation, dataset, cnn, matrix, limiar=limiar)   


    def default(self, kgraph, limiar, matrix):
        edge_index = []

        for img1 in range(len(self.rks)):
            for pos in range(kgraph): 
                img2 = self.rks[img1][pos]

                if matrix[img1][img2] > limiar:
                    edge_index.append([img1, img2])

        edge_index= torch.tensor(edge_index)
        self.edge_index= edge_index.t().contiguous()

    def rec(self, top_k):
        refList = [[] for i in range(self.pN)]
        for img1 in range(len(self.rks)):
            for pos in range(top_k):
                img2 = self.rks[img1][pos]
                refList[img2].append(img1)
        edge_index = []
        for img1 in range(len(self.rks)):
            for pos in range(self.pK):
                img2 = self.rks[img1][pos]
                if img2 in refList[img1]:
                    edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        self.edge_index = edge_index.t().contiguous()

    def knn(self, top_k):
        edge_index = []
        for img1 in range(len(self.rks)):
            for pos in range(top_k):
                img2 = self.rks[img1][pos]
                edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        self.edge_index = edge_index.t().contiguous()

    def create_graph(self, correlation_measure, dataset, cnn, matrix, limiar):
        if correlation_measure == 'knn':
            self.knn(self.pK)
        if correlation_measure == 'rec':
            self.rec(self.pK)
        if correlation_measure == 'default':
            self.default(self.pK, limiar, matrix)

    def train_and_predict(self):
        data = Data(x=self.x.float(), edge_index=self.edge_index, y=self.y, 
                    test_mask=self.test_mask, train_mask=self.train_mask, val_mask=self.val_mask)
        model = SGC(self.pNFeatures, self.numbersOfClasses)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        model.train()
        for epoch in range(self.pNEpochs):
            optimizer.zero_grad()
            out = model(data)
            data.y = torch.tensor(data.y, dtype=torch.long)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.masked_select(pred, data.test_mask)
        embeddings = model(data)
        return embeddings, pred.tolist()
