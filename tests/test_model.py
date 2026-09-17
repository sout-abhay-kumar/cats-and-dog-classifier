import torch

from src.dog_cat_classifier.models.cnn import DogCatCNN


def test_model_output():

    model = DogCatCNN()

    fake_images = torch.randn(
        4,
        3,
        128,
        128
    )

    output = model(fake_images)

    assert output.shape == (4, 2)