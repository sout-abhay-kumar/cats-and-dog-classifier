# Dog-Cat Classifier

A hands-on machine-learning project that builds a dog-versus-cat image classifier from the ground up. The project uses a custom convolutional neural network (CNN), the Microsoft Cats vs Dogs dataset, PyTorch, a FastAPI backend, and a browser frontend.

The goal is to understand the complete machine-learning pipeline rather than begin with the user interface or use a pretrained model.

> **Project status:** The repository currently contains the project structure and documentation. The source files are scaffolded placeholders, so the commands below describe the implementation sequence and target workflow.

## Project Pipeline

Build the project in this order:

```text
1. Project setup
      |
2. Configuration
      |
3. Dataset loading
      |
4. Image preprocessing
      |
5. CNN architecture
      |
6. Training
      |
7. Evaluation
      |
8. Save model
      |
9. Prediction script
      |
10. FastAPI backend
      |
11. Frontend
      |
12. Tests
      |
13. GitHub
```

Do not start with FastAPI or the frontend. First train the CNN successfully and use it to predict a local image.

## 1. Project Setup

Open a terminal in the repository root:

```powershell
cd "C:\Development\AI Development\Projects\dog-cat-classifier"
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The intended dependencies are:

```text
torch
torchvision
datasets
fastapi
uvicorn
python-multipart
pillow
pyyaml
numpy
pytest
```

The dependency list should be kept in `requirements.txt` so the environment can be recreated consistently.

## 2. Configuration

Training settings belong in `configs/config.yaml` rather than being repeated throughout the source code. The intended configuration is:

```yaml
image_size: 128
batch_size: 32
epochs: 10
learning_rate: 0.001
num_classes: 2
class_names:
  - cat
  - dog
model_path: models/dog_cat_cnn.pth
```

This makes experiments easier to change. For example, the number of epochs can be changed from `10` to `20` without editing the training algorithm.

Use `.env.example` as a template for environment-specific values. Do not commit a real `.env` file or secrets.

## 3. Dataset Loading

The project uses the Hugging Face dataset `microsoft/cats_vs_dogs`. Dataset loading and splitting belong in `src/dog_cat_classifier/data/dataset.py`.

The intended loading sequence is:

1. Load the Cats vs Dogs dataset.
2. Split the original training data into 80% training and 20% temporary data.
3. Split the temporary data evenly into validation and test data.

This produces approximately:

```text
Dataset
|
|-- 80% training
|-- 10% validation
`-- 10% test
```

The repository also contains `data/raw/` and `data/processed/` for local datasets or generated data. Dataset-specific notes belong in `data/README.md`.

## 4. Image Preprocessing

Preprocessing belongs in `src/dog_cat_classifier/data/preprocessing.py`.

Training images should be:

1. Converted to RGB.
2. Resized to `128 x 128` pixels.
3. Randomly flipped horizontally.
4. Randomly rotated by up to 10 degrees.
5. Converted to PyTorch tensors.

Validation and test images should use the same resize and tensor conversion steps without random augmentation. This keeps evaluation repeatable.

The data flow is:

```text
Original image
      |
Resize to 128 x 128
      |
Training augmentation
      |
Tensor
      |
DataLoader
      |
CNN
```

`dataset.py` is responsible for applying the training transform to the training split and the test transform to the validation and test splits, then creating PyTorch `DataLoader` objects.

## 5. Build the Custom CNN

The model belongs in `src/dog_cat_classifier/models/cnn.py`.

This project intentionally builds its own CNN. It does not use a pretrained ResNet, EfficientNet, VGG, or another transfer-learning model. The model weights start randomly and are learned during training.

The planned architecture is:

```text
Input: 128 x 128 x 3
      |
Conv2d: 3 -> 32, ReLU, MaxPool2d
      |
Conv2d: 32 -> 64, ReLU, MaxPool2d
      |
Conv2d: 64 -> 128, ReLU, MaxPool2d
      |
Conv2d: 128 -> 256, ReLU, MaxPool2d
      |
Flatten
      |
Linear: 256 * 8 * 8 -> 512
      |
ReLU, Dropout(0.5)
      |
Linear: 512 -> 2
      |
Cat or dog logits
```

The four pooling layers reduce the spatial dimensions from `128 x 128` to `8 x 8`. The final layer returns two values, one for each class.

## 6. Test the CNN Before Training

Before downloading and training on the full dataset, verify the model shape with a fake batch. `tests/test_model.py` should create four random images with shape `4 x 3 x 128 x 128` and assert that the model output has shape `4 x 2`.

Run the test suite with:

```bash
pytest tests/test_model.py
```

This is a cheap check that can catch an incorrect flattened size or output layer before a long training run.

## 7. Train the Model

Training belongs in `src/dog_cat_classifier/training/train.py`. The script should:

1. Set the random seed.
2. Select CUDA when available, otherwise CPU.
3. Create the training, validation, and test data loaders.
4. Create `DogCatCNN`.
5. Use `CrossEntropyLoss`.
6. Use the Adam optimizer with the configured learning rate.
7. Run the forward pass, loss calculation, backpropagation, and weight update for each batch.
8. Evaluate on the validation split after each epoch.
9. Evaluate on the test split after training.
10. Save the learned weights to `models/dog_cat_cnn.pth`.

Run training from the project root with:

```bash
python -m src.dog_cat_classifier.training.train
```

The `scripts/train.py` file is the command-line entry-point location for the project and may delegate to the package training module.

### What Happens During Training?

For every batch, the learning loop is:

```text
Images
   |
CNN
   |
Predictions
   |
Compare with actual labels
   |
Loss
   |
Backpropagation
   |
Update weights
```

In code, the key operations are conceptually:

```python
outputs = model(images)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The model predicts, measures its error, calculates gradients, and updates its weights to reduce that error.

## 8. Evaluate the Model

Evaluation belongs in `src/dog_cat_classifier/training/evaluate.py`.

Evaluation should run without gradient updates and report at least loss and accuracy. Precision, recall, F1 score, and a confusion matrix are useful additions for understanding classification errors.

Run the evaluation entry point from the project root:

```bash
python scripts/evaluate.py
```

The evaluation code should load `models/dog_cat_cnn.pth` and use the test split that was not used to update the model weights.

## 9. Save the Model

After successful training, the expected output is:

```text
models/
`-- dog_cat_cnn.pth
```

The model file contains the learned PyTorch state dictionary. Keep generated model files out of Git unless the project intentionally shares a specific model artifact.

## 10. Predict a New Image

Single-image inference belongs in `src/dog_cat_classifier/inference/predict.py`.

The inference sequence is:

1. Load the saved model weights.
2. Open a local image with Pillow.
3. Convert it to RGB.
4. Resize it to `128 x 128` and convert it to a tensor.
5. Add a batch dimension.
6. Run the model without gradients.
7. Apply softmax to obtain probabilities.
8. Return `cat` or `dog` and the confidence score.

Run prediction with:

```bash
python -m src.dog_cat_classifier.inference.predict path/to/image.jpg
```

Example output:

```text
Prediction: dog
Confidence: 93.72%
```

Do not build the API until this local prediction flow works.

## 11. FastAPI Backend

The backend wraps the working inference flow in an HTTP API. Its files are:

```text
backend/
|-- __init__.py
|-- main.py
`-- schemas.py
```

The `POST /predict` endpoint should:

1. Accept an uploaded image.
2. Read the image bytes.
3. Preprocess the image using the same inference transform.
4. Run the trained CNN.
5. Return the predicted class and confidence as JSON.

The root endpoint should confirm that the API is running. Start the backend from the repository root with:

```bash
uvicorn backend.main:app --reload
```

Open the API at `http://127.0.0.1:8000`. FastAPI's interactive documentation is available at `http://127.0.0.1:8000/docs`.

The backend requires a trained model at `models/dog_cat_cnn.pth` before it can make predictions.

## 12. Frontend

The frontend is intentionally built after the ML pipeline and API:

```text
frontend/
|-- index.html
|-- style.css
`-- script.js
```

The browser workflow is:

1. Choose an image.
2. Show a local preview.
3. Send the image as multipart form data to `POST http://127.0.0.1:8000/predict`.
4. Display the returned class and confidence.

Serve the frontend from a second terminal while the backend is running:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500` in a browser. The backend must be running separately on port `8000`.

## 13. Tests

The test files are organized by responsibility:

```text
tests/
|-- __init__.py
|-- test_dataset.py       # Dataset splitting and DataLoader behavior
|-- test_model.py         # CNN input/output shape and model behavior
`-- test_api.py           # Backend endpoints and prediction responses
```

Run all tests from the repository root:

```bash
pytest
```

Run individual test groups when developing:

```bash
pytest tests/test_dataset.py
pytest tests/test_model.py
pytest tests/test_api.py
```

## 14. GitHub

After the project runs locally and the tests pass:

```bash
git add .
git commit -m "Build dog cat classifier pipeline"
git push -u origin main
```

Do not commit the following:

- The virtual environment directory
- `.env` files containing local settings or secrets
- Downloaded datasets
- Large generated model files, unless intentionally required

## Repository Structure

```text
dog-cat-classifier/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- .env.example
|
|-- configs/
|   `-- config.yaml
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- README.md
|
|-- models/
|   `-- .gitkeep
|
|-- notebooks/                         # Exploratory work
|
|-- src/
|   `-- dog_cat_classifier/
|       |-- __init__.py
|       |-- data/
|       |   |-- __init__.py
|       |   |-- dataset.py              # Dataset loading and splits
|       |   `-- preprocessing.py        # Image transforms
|       |-- models/
|       |   |-- __init__.py
|       |   `-- cnn.py                  # Custom CNN
|       |-- training/
|       |   |-- __init__.py
|       |   |-- train.py                # Training loop
|       |   `-- evaluate.py             # Metrics and evaluation
|       |-- inference/
|       |   |-- __init__.py
|       |   `-- predict.py              # Local image prediction
|       `-- utils/
|           |-- __init__.py
|           `-- seed.py                 # Reproducibility
|
|-- backend/
|   |-- __init__.py
|   |-- main.py                         # FastAPI application
|   `-- schemas.py                      # API schemas
|
|-- frontend/
|   |-- index.html                      # User interface
|   |-- style.css                       # Frontend styles
|   `-- script.js                       # Upload and API requests
|
|-- tests/
|   |-- __init__.py
|   |-- test_dataset.py
|   |-- test_model.py
|   `-- test_api.py
|
`-- scripts/
    |-- train.py                        # Training command entry point
    `-- evaluate.py                     # Evaluation command entry point
```

## Four Milestones

Think of the project as four milestones rather than as a collection of files:

### Milestone 1: Machine Learning

```text
Dataset -> CNN -> Training -> Evaluation
```

### Milestone 2: Inference

```text
Saved model -> New image -> Prediction
```

### Milestone 3: Backend

```text
HTTP request -> Model -> JSON response
```

### Milestone 4: Product

```text
Website -> API -> Model -> Prediction
```

## Complete Runtime Flow

```text
User uploads an image
          |
          v
Frontend: HTML and JavaScript
          |
          | HTTP POST /predict
          v
FastAPI backend
          |
          v
Pillow image loading
          |
          v
Image preprocessing
          |
          v
Custom CNN
          |
          v
Cat or dog plus confidence
          |
          v
JSON response to the frontend
```

## Training Flow

```text
Microsoft Cats vs Dogs dataset
          |
          v
Dataset loading and splitting
          |
          v
Image preprocessing
          |
          v
DataLoader
          |
          v
Custom CNN
          |
          v
Forward pass -> Loss -> Backpropagation -> Weight update
          |
          v
Evaluation
          |
          v
models/dog_cat_cnn.pth
```

The central principle is simple: build and understand the ML system first, then add inference, the backend, and the frontend around the working model.
