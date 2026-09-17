import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.dog_cat_classifier.data.dataset import (
    create_dataloaders
)

from src.dog_cat_classifier.models.cnn import DogCatCNN

from src.dog_cat_classifier.utils.seed import set_seed


IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "models/dog_cat_cnn.pth"


def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(loader)

    accuracy = correct / total

    return average_loss, accuracy


def main():

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, validation_loader, test_loader = \
        create_dataloaders(
            IMAGE_SIZE,
            BATCH_SIZE
        )

    model = DogCatCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss / len(train_loader)
        )

        train_accuracy = correct / total

        validation_loss, validation_accuracy = \
            evaluate(
                model,
                validation_loader,
                criterion,
                device
            )

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        print("=" * 60)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print()
    print("FINAL TEST RESULT")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()