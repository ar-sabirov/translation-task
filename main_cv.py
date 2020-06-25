import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer

from src.models import Net
from src.torch.lightning import LightningSystem

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    val_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform)

    model = Net()

    system = LightningSystem(model=model,
                             train_dataset=trainset,
                             num_classes=10,
                             val_dataset=val_dataset,
                             batch_size=16,
                             shuffle=True,
                             num_workers=2)

    trainer = Trainer()

    trainer.fit(system)
