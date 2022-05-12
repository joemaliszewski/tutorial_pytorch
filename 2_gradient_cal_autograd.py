'''
Basic functionality practice with the autograd package
'''

import torch

#if we want to calculate the gradient of a function with respect to x with requires_grad
x = torch.randn(3, requires_grad = True)

print(x)

#comnpuational graph is created 
y = x+2
print(y)

z = y*y*2
print(z)

# z.backward() #will cal gradient dz/dx
# print(x.grad)

#if not a scalar we must provide vector
v = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
z.backward(v)  #dv/dz


#To stop tracking of gradients
with torch.no_grad():
    y = x + 2
    print(y)


#training example

#fake training data 
weights = torch.ones(4, requires_grad= True)

for epoch in range(1):

    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    #this will empty the gradients
    weights.grad.zero_()

