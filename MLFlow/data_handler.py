from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataset(batch_size=32):
    train_set = datasets.FashionMNIST(
      root="./data/FashionMNIST",
      train=True,
      download=True,
      transform=transforms.Compose([
        transforms.ToTensor()
      ])
    )

    test_set = datasets.FashionMNIST(
      root="./data/FashionMNIST",
      train=False,
      download=True,
      transform=transforms.Compose([
        transforms.ToTensor()
      ])
    )

    return DataLoader(train_set, batch_size=batch_size), DataLoader(test_set, batch_size=batch_size)
