from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device(
     "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride = 2
        )

        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.fc1 = nn.Linear(64*7*7, 50)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(50,10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = torch.flatten(x,1)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.drop1(x)
        x = self.fc2(x)

        return x

# Dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=str(PROJECT_ROOT / "outputs" / "mnist_torch"),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=str(PROJECT_ROOT / "outputs" / "mnist_torch"),
    train=False,
    download=True,
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False
)

model = MyConvNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

for epoch in range(epochs):

    # ===== Training =====
    model.train()

    running_loss = 0.0

    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        y = model(x)
        loss = criterion(y, t)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)


    # ===== Testing =====
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, t in test_loader:
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            predicted = torch.argmax(y, dim=1)

            total += t.size(0)

            correct += (predicted == t).sum().item()

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}, "
        f"Test Accuracy: {accuracy:.4f}"
    )
