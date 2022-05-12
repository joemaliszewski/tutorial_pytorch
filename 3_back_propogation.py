'''
Simple example of backpropagation
'''

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#weight
w = torch.tensor(1.0, requires_grad=True)

#forward pass 
y_hat = w*x

#compute the loss
loss = (y_hat - y)**2
print(loss)

#backward pass (cal gradient dw/dloss)
loss.backward()
print(w.grad)

#update wieghts
#mext forward and backward pass



