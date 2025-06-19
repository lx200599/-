import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model.cnn_model import CNN
from model.train_utils import train_model, test_model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainval = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(trainval))
val_size = len(trainval) - train_size
train_dataset, val_dataset = random_split(trainval, [train_size, val_size])
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

train_model(model, train_loader, val_loader, device)
test_model(model, test_loader, device)
