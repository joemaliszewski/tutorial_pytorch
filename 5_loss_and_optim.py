'''

Designing model (input, output_size, forward_pass)
Construct loss and optimiser
Training loop
    - compute prediction
    - backward pass: grads
    - update weights
    
'''

import numpy as np
import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_iters = 20
lr = 0.01

def forward(x):
    # our model
    return w*x

# def loss(y, y_pred):
#     # MSE 1/N * 2x(wx-y)
#     return ((y-y_pred)**2).mean()

# callable function
loss = nn.MSELoss()

# def gradient(x, y, y_pred):
#     # 1/N * 2x(wx-y)
#     return np.dot(2*x, (y_pred - y)).mean()
optimizer = torch.optim.SGD([w], lr=lr)


for epoch in range(n_iters):

    y_pred = forward(X)

    print(f'y_pred = {y_pred}')

    l = loss(Y,y_pred)

    print(f'loss = {l}  type = {type(l)}')

    #calculates gradient
    l.backward() #dl/dw

    # we dont want this to be part of our compuational graph
    #w -= lr*w.grad

    # with torch.no_grad():
    #     w -= lr * w.grad

    optimizer.step()

    optimizer.zero_grad()

    #w.grad.zero_()


    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

