import matplotlib.pyplot as plt
import torch


def show_predictions(model, dataloader, device):
    model.eval()

    images, labels = next(iter(dataloader))
    images = images.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, 6, figsize=(12, 2))

    for idx in range(6):
        ax = axes[idx]
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {preds[idx].item()}")
        ax.axis('off')

    plt.show()