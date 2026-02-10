import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import json


# Download and extract the flower dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(
    "flower_photos",
    origin=dataset_url,
    untar=True
)

data_dir = pathlib.Path(data_dir)


# Basic training parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 15


# Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)

print("Detected classes:", class_names)


# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Data augmentation to reduce overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# Load pretrained MobileNetV2 model
base_model = keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# Build the final model
inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)


# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# Save model and class labels
pathlib.Path("model").mkdir(exist_ok=True)

model.save("model/flower_model.keras")

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Model training completed and saved successfully.")
