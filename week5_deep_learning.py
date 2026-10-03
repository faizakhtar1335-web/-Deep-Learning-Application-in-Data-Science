"""
Week 5: Deep Learning Application in Data Science
Project: Handwritten Digit Classification with a Neural Network
Dataset: MNIST (loaded through TensorFlow/Keras)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the publicly available MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Build the neural network
model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train the model with validation data
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.10,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ],
    verbose=1
)

# 6. Evaluate on the unseen test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 7. Predictions and additional classification metrics
from sklearn.metrics import classification_report, confusion_matrix

y_probability = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_probability, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=150)
plt.show()

# 9. Plot one example prediction
plt.figure(figsize=(4, 4))
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted: {y_pred[0]} | Actual: {y_test[0]}")
plt.axis("off")
plt.tight_layout()
plt.savefig("sample_prediction.png", dpi=150)
plt.show()
