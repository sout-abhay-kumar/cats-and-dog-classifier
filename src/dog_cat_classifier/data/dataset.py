from datasets import load_dataset
from torch.utils.data import DataLoader

from .preprocessing import (
    get_train_transform,
    get_test_transform
)


def load_data():
    print("Loading Cats vs Dogs dataset...")

    dataset = load_dataset("microsoft/cats_vs_dogs")

    return dataset


def split_data(dataset):
    train_test = dataset["train"].train_test_split(
        test_size=0.2,
        seed=42
    )

    train_dataset = train_test["train"]
    remaining_dataset = train_test["test"]

    validation_test = remaining_dataset.train_test_split(
        test_size=0.5,
        seed=42
    )

    validation_dataset = validation_test["train"]
    test_dataset = validation_test["test"]

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


def create_dataloaders(image_size, batch_size):

    dataset = load_data()

    train_dataset, validation_dataset, test_dataset = \
        split_data(dataset)

    train_transform = get_train_transform(image_size)
    test_transform = get_test_transform(image_size)

    def transform_train(example):
        example["image"] = [
            train_transform(image.convert("RGB"))
            for image in example["image"]
        ]

        return example

    def transform_test(example):
        example["image"] = [
            test_transform(image.convert("RGB"))
            for image in example["image"]
        ]

        return example

    train_dataset.set_transform(transform_train)
    validation_dataset.set_transform(transform_test)
    test_dataset.set_transform(transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )