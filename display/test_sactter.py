import torch

src = torch.tensor([0, 0, 0, 0, 0])
stereo = torch.tensor([1, 1, 0, 0, 1])
index = torch.nonzero(stereo).squeeze()
#index = torch.tensor([0, 1])

mask = torch.tensor([1, 1, 1, 1, 1])

#mask.scatter_(0, index, src)

mask.scatter_(0, index, src)

print(index)
print(mask)