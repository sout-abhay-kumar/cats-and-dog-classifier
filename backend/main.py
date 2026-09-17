import io
import os
import sys

import torch

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)


from src.dog_cat_classifier.models.cnn import DogCatCNN


app = FastAPI(
    title="Dog vs Cat Classifier API"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model_path = os.path.join(
    PROJECT_ROOT,
    "models",
    "dog_cat_cnn.pth"
)


model = DogCatCNN()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize(
        (128, 128)
    ),
    transforms.ToTensor()
])


@app.get("/")
def home():

    return {
        "message":
        "Dog vs Cat Classifier API is running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
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

    classes = [
        "cat",
        "dog"
    ]

    predicted_class = classes[prediction]

    confidence = probabilities[
        0,
        prediction
    ].item()

    return {
        "prediction": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        )
    }