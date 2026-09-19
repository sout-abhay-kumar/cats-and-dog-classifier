const imageInput = document.getElementById("imageInput");

const preview = document.getElementById("preview");

const predictButton = document.getElementById("predictButton");

const result = document.getElementById("result");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (!file) {
    preview.style.display = "none";

    return;
  }

  preview.src = URL.createObjectURL(file);

  preview.style.display = "block";

  result.innerText = "";
});

predictButton.addEventListener("click", async () => {
  const file = imageInput.files[0];

  if (!file) {
    result.innerText = "Please select an image.";

    return;
  }

  const formData = new FormData();

  formData.append("file", file);

  predictButton.disabled = true;

  result.innerText = "🔍 Predicting...";

  try {
    const response = await fetch("https://cats-and-dog-classifier.onrender.com/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();

    if (data.prediction === "dog") {
      result.innerText = `🐶 DOG — ${data.confidence}%`;
    } else {
      result.innerText = `🐱 CAT — ${data.confidence}%`;
    }
  } catch (error) {
    console.error(error);

    result.innerText = "❌ Server error.";
  } finally {
    predictButton.disabled = false;
  }
});