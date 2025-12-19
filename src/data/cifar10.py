from __future__ import annotations

from dataclasses import dataclass

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class CIFAR10Loaders:
    trainloader: DataLoader
    testloader: DataLoader


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    test_batch_size: int = 100,
    num_workers: int = 2,
) -> CIFAR10Loaders:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return CIFAR10Loaders(trainloader=trainloader, testloader=testloader)
