"""
An overview of some basic and useful torch functions
"""

import torch
import numpy as np

#empty tensor (not initilized)
x = torch.empty(2,3)
print(x)

#tensor of ones
x = torch.ones(2,3)
print(x)

#tensor of zeros
x = torch.zeros(2,3)
print(x)

#tensor of random values
x = torch.rand(2,3)
print(x)

#dtype enforces the type of each element in a tensor 
x = torch.rand(2,3, dtype=torch.float16)
print(x.dtype)

#create tensor and print tensor and its size
x = torch.tensor([2.3, 6.9])
print(x)
print(x.size())

#maths operations two tensors
x = torch.tensor([3,4])
y = torch.tensor([5,6])
print(x, y)

#adding
z = x + y
print(z)
z = torch.add(x, y)
print(z)

#every function that has a trailing underscore does an inplace operation (modfies variable that func is applied on)
y.add_(x)

#dividing
z = x/y
z = torch.div(x, y)

#slicing operations

#getiing all the rows but the first column
x = torch.rand(6,2)
print(x[:, 0])

#getiing all the columns but the second row
x = torch.rand(3,5)
print(x[1, :])

# get elememt 
print(x[1,1].item())

#reshaping 

x = torch.rand(4,4)
print(x)
print(x.size())

#resize a tensor use -1 to automatically calculate the size of that dimension respective of the other provided
y = x.view([-1, 2])
print(y)
print(y.size())

#converting from torch tensor to numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b, type(b))

#if object is on CPU and noy GPU, they share the same memory location. If we change one, we change the other
a.add_(1)
print(b)

#converting from numpy to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(b)

#specifying a device

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)

    # to convert to numpy cannot be done on GPU has to be pushed back to CPU first
    z = z.to("cpu")
    z.numpy()

#requires grad default is false. if true, it tells torch that it will need to calculate the gradient later in optimzation step for that variable.
#as the veriable needs to be optimzed
x = torch.ones(5, requires_grad = True)








