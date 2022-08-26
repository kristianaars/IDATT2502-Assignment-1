import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt

dataset_link = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv"

dataset = pandas.read_csv(dataset_link,
                          usecols=['# day', 'length', 'weight'],
                          dtype={'# day': np.float32, 'length': np.float32, 'weight': np.float32})
dataset.columns = ['day', 'length', 'weight']

x_train = torch.tensor(dataset['day']).reshape(-1, 1)
y_train = torch.tensor(dataset['length']).reshape(-1, 1)
z_train = torch.tensor(dataset['weight']).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))
