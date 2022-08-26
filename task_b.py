import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt

dataset_link = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv"

dataset = pandas.read_csv(dataset_link,
                          usecols=['# day', 'length', 'weight'],
                          dtype={'# day': np.float32, 'length': np.float32, 'weight': np.float32})
dataset.columns = ['day', 'length', 'weight']

length_tensor = torch.tensor(dataset['length']).reshape(-1, 1)
weight_tensor = torch.tensor(dataset['weight']).reshape(-1, 1)

x_train = torch.cat((length_tensor, weight_tensor), 1)
y_train = torch.tensor(dataset['day']).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(80000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(), 'o', label='$(x^{(i)},y^{(i)})$')

grid_points = 10
x_grid, z_grid = np.meshgrid(np.linspace(1, 120, grid_points), np.linspace(1, 25, grid_points))
y_grid = np.empty([grid_points, grid_points])

for i in range(0, x_grid.shape[0]):
    for j in range(0, x_grid.shape[1]):
        tens = torch.tensor([np.float32(x_grid[i, j]), np.float32(z_grid[i, j])])
        tens.double()
        y_grid[i, j] = model.f(tens)

ax.plot_wireframe(x_grid, z_grid, y_grid, color='green')

plt.show()
