import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os

from src.data.loader import get_dataloaders
from src.models.cnn import CNN


def train():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, _ = get_dataloaders(config)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "outputs/models/mnist_cnn.pth")


if __name__ == "__main__":
    train()