
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Model settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Check dataset folders
if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        "Please create dataset/train and dataset/test "
        "and add labeled images to them."
    )

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

# Load testing images
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Read class names from training folders
class_names = train_data.class_names

print("Disease categories:", class_names)

# Check class names match
if class_names != test_data.class_names:
    raise ValueError(
        "Training and testing folders must contain "
        "the same category names."
    )

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model information
model.summary()

# Train model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# Create model directory
os.makedirs("model", exist_ok=True)

# Save trained model
model.save("model/skin_model.keras")

# Save category names in the same order as model outputs
with open("model/class_names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))

print("\nModel training completed!")
print("Model saved at: model/skin_model.keras")
print("Categories saved at: model/class_names.txt")
