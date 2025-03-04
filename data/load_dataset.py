from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_cifar(batch_size=64):
    # CIFAR-10专用归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    test_set = datasets.CIFAR10(
        root="./data/dataset",
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)