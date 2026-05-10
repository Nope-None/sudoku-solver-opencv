"""
src/digit_recognizer.py
CNN model definition, MNIST training, and inference utilities.

Fixes applied:
- Crop 15% border from each cell edge to remove grid lines
- Empty-cell threshold raised 0.05 -> 0.12
- MNIST-style digit centering before inference
- Class 0 predictions treated as empty (0 never appears in Sudoku)
"""

import cv2
import numpy as np


def build_model(input_shape=(28, 28, 1), num_classes=10):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(save_path="models/digit_classifier.h5", epochs=10):
    from tensorflow.keras.datasets import mnist
    print("[INFO] Loading MNIST ...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=epochs, batch_size=64, verbose=1)
    model.save(save_path)
    print(f"[INFO] Model saved -> {save_path}")
    return model


def crop_cell(cell):
    """Strip 15% from each edge to remove grid border lines."""
    h, w = cell.shape[:2]
    mh, mw = int(h * 0.15), int(w * 0.15)
    return cell[mh:h - mh, mw:w - mw]


def is_empty_cell(cell, threshold=0.12):
    """
    True if cell has no digit.
    threshold=0.12 (was 0.05) prevents grid line artifacts being read as digits.
    """
    inner = crop_cell(cell)
    _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (np.sum(binary > 0) / binary.size) < threshold


def prepare_cell(cell):
    """Crop borders, centre digit bounding box (MNIST style), resize to 28x28, normalise."""
    inner = crop_cell(cell)
    _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = binary[y:y + h, x:x + w]
        roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    else:
        roi = binary
    resized = cv2.resize(roi, (28, 28))
    return resized.astype("float32") / 255.0


def read_board(cells, model):
    """Return a 9x9 board from 81 cell images (0 = empty)."""
    empty_flags = [is_empty_cell(c) for c in cells]
    indices = [i for i, e in enumerate(empty_flags) if not e]
    pred_map = {}
    if indices:
        batch = np.array([prepare_cell(cells[i]) for i in indices])[..., np.newaxis]
        preds = model.predict(batch, verbose=0)
        for bi, ci in enumerate(indices):
            d = int(np.argmax(preds[bi]))
            pred_map[ci] = d if d != 0 else 0  # 0 class = MNIST "0", never valid in Sudoku

    board = []
    for r in range(9):
        row = []
        for c in range(9):
            idx = r * 9 + c
            row.append(0 if empty_flags[idx] else pred_map.get(idx, 0))
        board.append(row)
    return board
