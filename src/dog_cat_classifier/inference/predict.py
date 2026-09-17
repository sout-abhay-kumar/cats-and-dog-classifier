import sys

import torch

from PIL import Image

from torchvision import transforms

from src.dog_cat_classifier.models.cnn import DogCatCNN


IMAGE_SIZE = 128

MODEL_PATH = "models/dog_cat_cnn.pth"

CLASS_NAMES = [
    "cat",
    "dog"
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = DogCatCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class = CLASS_NAMES[prediction]

    confidence = probabilities[
        0,
        prediction
    ].item()

    return predicted_class, confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python -m "
            "src.dog_cat_classifier.inference.predict "
            "image.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    prediction, confidence = predict(
        image_path
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )