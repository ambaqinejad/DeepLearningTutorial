import torch

# scalar tensor
scalar = torch.tensor(2)
print(scalar)

# check dimension of tensor
print(scalar.ndim)
print(scalar.shape)

# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())

# vector
vector = torch.tensor([1, 2, 3])
print(vector.ndim)
print(vector.type())
print(vector.shape)
print(vector.size())
print("------------------------")

# Matrix
MATRIX = torch.tensor([[1, 2], [3, 4]])
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX.size())
print("------------------------")

# Random Tensors
random_tensor = torch.rand(size=(3, 4, 1), dtype=torch.float32)
print(random_tensor)
print(random_tensor.ndim)
print("------------------------")

# Zeros and Ones
zeros = torch.zeros(size=(2, 4))
ones = torch.ones(size=(2, 4))
print(zeros)
print(ones)
print("------------------------")

# Range
tensor = torch.arange(0, 10, 1)
print(tensor)

# Sometimes you might want one tensor of a certain type with the same shape as another tensor.
zero_like = torch.zeros_like(tensor)
one_like = torch.ones_like(tensor)
print(zero_like)
print(one_like)
print("------------------------")

# Let's see how to create some tensors with specific datatypes.
float_64_tensor = torch.tensor([3, 2, 1],
                               dtype=torch.float64)
print(float_64_tensor.type())
print(float_64_tensor.dtype)
print(float_64_tensor.device)

print("------------------------")

# Basic Manipulation
tensor = torch.tensor([1, 2, 3])
print(tensor + 5)

tensor1 = torch.tensor([4, 5, 6])
tensor2 = torch.ones_like(tensor1)
print(tensor1 + tensor2)
print(tensor1 * 10)
print(tensor1 / 3)
print(tensor1 // 3)
print(tensor1 % 3)
print(tensor1 - 3)
print(tensor1 * tensor2)
print(tensor2 / tensor1)
print(torch.multiply(tensor1, 4))
print(torch.remainder(tensor1, tensor2 + 1))
print("------------------------")

# Tensor Matrix Multiply

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
print(tensor1 * tensor2)
print(tensor1 @ tensor2)
print(torch.matmul(tensor1, tensor2))

print("------------------------")

# Transpose

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)
print(tensor_A.T)
print(tensor_B @ tensor_A.T)

print("------------------------")

# Aggregation
x = torch.arange(0, 100, 10)
# x = x.to(torch.float32)
x = x.type(torch.float32)
print(torch.min(x))
print(x.min())
print(torch.max(x))
print(x.max())
print(torch.mean(x))
print(x.mean())

# Positional min/max
print(x.argmax())
print(x.argmin())
print("------------------------")

# Reshaping, stacking, squeezing and unsqueezing
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)
print(x.reshape(3, 2))
y = torch.tensor([1, 2, 3])
z = torch.stack([y, y, y, y], 0)
print(z)
z = torch.stack([y, y, y, y], 1)
print(z)

a = torch.rand(size=(1, 5, 1))
print(a)
print(torch.squeeze(a, 0))

b = torch.rand(size=[4])
print(b)
print(torch.unsqueeze(b, dim=0))
print(torch.unsqueeze(b, dim=1))