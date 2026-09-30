import random
import string

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# -----------------------------
# Settings
# -----------------------------
CHARACTERS = string.ascii_uppercase + string.digits
CHAR_WIDTH = 32
CHAR_HEIGHT = 40
TRAINING_SAMPLES_PER_CLASS = 100
CAPTCHA_LENGTH = 4


# -----------------------------
# Find a font
# -----------------------------
def get_font(size=26):
    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arialbd.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()


FONT = get_font()


# -----------------------------
# Generate one character image
# -----------------------------
def generate_character(character):
    image = Image.new(
        "L",
        (CHAR_WIDTH, CHAR_HEIGHT),
        color=255
    )

    draw = ImageDraw.Draw(image)

    x = random.randint(3, 8)
    y = random.randint(4, 10)

    draw.text(
        (x, y),
        character,
        fill=0,
        font=FONT
    )

    # Slight rotation for variation
    angle = random.randint(-10, 10)

    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255
    )

    # Add a small amount of noise
    pixels = np.array(image)

    noise = np.random.randint(
        0,
        20,
        pixels.shape
    )

    pixels = np.clip(
        pixels - noise,
        0,
        255
    )

    return pixels.astype(np.uint8)


# -----------------------------
# Build training dataset
# -----------------------------
def create_dataset():
    X = []
    y = []

    print("Generating training data...")

    for character in CHARACTERS:

        for _ in range(TRAINING_SAMPLES_PER_CLASS):

            image = generate_character(character)

            # Flatten image into feature vector
            features = image.flatten()

            X.append(features)
            y.append(character)

    return np.array(X), np.array(y)


# -----------------------------
# Train SVM
# -----------------------------
def train_model():

    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining SVM model...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    return model


# -----------------------------
# Generate a CAPTCHA
# -----------------------------
def generate_captcha(model):

    captcha_text = "".join(
        random.choice(CHARACTERS)
        for _ in range(CAPTCHA_LENGTH)
    )

    captcha_image = Image.new(
        "L",
        (CHAR_WIDTH * CAPTCHA_LENGTH, CHAR_HEIGHT),
        color=255
    )

    predicted_text = ""

    for i, character in enumerate(captcha_text):

        char_image = generate_character(character)

        # Add character image to CAPTCHA
        image = Image.fromarray(char_image)

        captcha_image.paste(
            image,
            (i * CHAR_WIDTH, 0)
        )

        # Predict character
        features = char_image.flatten()

        prediction = model.predict(
            [features]
        )[0]

        predicted_text += prediction

    return captcha_text, predicted_text, captcha_image


# -----------------------------
# Main program
# -----------------------------
def main():

    print("=" * 55)
    print("MACHINE LEARNING CAPTCHA RECOGNITION")
    print("=" * 55)

    model = train_model()

    actual, predicted, image = generate_captcha(model)

    print("\nGenerated CAPTCHA :", actual)
    print("Predicted CAPTCHA :", predicted)

    if actual == predicted:
        print("Result: CAPTCHA recognized successfully.")
    else:
        print("Result: CAPTCHA recognition was incorrect.")

    plt.figure(figsize=(8, 2))

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"Actual: {actual} | Predicted: {predicted}"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()