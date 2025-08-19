import torch

while True:
    x1 = torch.tensor([1, 2, 3]).to("cuda")
    x2 = torch.tensor([4, 5, 6]).to("cuda")
    y = x1 * x2
    print(y)