import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

def get_cifar10_data(batch_size=64, val_split=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 training dataset and split into training and validation sets
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    num_train = len(full_trainset)
    num_val = int(val_split * num_train)
    num_train = num_train - num_val
    trainset, valset = random_split(full_trainset, [num_train, num_val], generator = torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, persistent_workers=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=256,
                           shuffle=False, persistent_workers=True, num_workers=2)

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=256,
                            shuffle=False, persistent_workers=True, num_workers=2)
    
    return train_loader, val_loader, test_loader


def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainset, valset = random_split(trainset, [50000, 10000], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    )
    validation_loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    )

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    )

    return train_loader, validation_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_data()
    print(train_loader[0][0])
