'''

Gradient Descent for f = 2x, X = 1,2,3,4 Y = 2, 4, 6, 8

Steps:

1) Forward pass
2) Cal loss (MSE) = J = 1/N * (wx -y)**2
3) Cal gradient dJ/dw = 1/N * 2x(wx-y)
4) Update weights w = w - lr*dw

'''

import numpy as np
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    # our model
    return w*x

def loss(y, y_pred):
    # MSE 1/N * 2x(wx-y)
    return ((y-y_pred)**2).mean()



# def gradient(x, y, y_pred):
#     # 1/N * 2x(wx-y)
#     return np.dot(2*x, (y_pred - y)).mean()

n_iters = 20
lr = 0.01

for epoch in range(n_iters):

    y_pred = forward(X)

    print(f'y_pred = {y_pred}')

    l = loss(Y,y_pred)

    print(f'loss = {l}  type = {type(l)}')

    #calculates gradient
    l.backward() #dl/dw

    # we dont want this to be part of our compuational graph
    #w -= lr*w.grad

    with torch.no_grad():
        w -= lr * w.grad

    w.grad.zero_()


    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

