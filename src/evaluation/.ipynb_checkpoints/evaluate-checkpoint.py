import torch
import yaml

from src.data.loader import get_dataloaders
from src.models.cnn import CNN


def evaluate():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, testloader = get_dataloaders(config)

    model = CNN().to(device)
    model.load_state_dict(torch.load("outputs/models/mnist_cnn.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    evaluate()