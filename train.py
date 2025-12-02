# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # Paths
# TRAIN_DIR = "dataset/train"
# VAL_DIR = "dataset/val"
# MODEL_PATH = "model/face_mask_mobilenetv2.h5"

# # Parameters
# INIT_LR = 1e-4
# EPOCHS = 10
# BS = 32
# IMG_SIZE = (224, 224)

# # Data Generators
# train_aug = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest"
# )

# val_aug = ImageDataGenerator(rescale=1./255)

# print("Loading training data...")
# train_gen = train_aug.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BS,
#     class_mode="binary"
# )

# print("Loading validation data...")
# val_gen = val_aug.flow_from_directory(
#     VAL_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BS,
#     class_mode="binary"
# )

# print("Classes:", train_gen.class_indices)

# # Load MobileNetV2
# base_model = MobileNetV2(weights="imagenet", include_top=False,
#                          input_tensor=Input(shape=(224, 224, 3)))

# # Freeze base model
# base_model.trainable = False

# # Head of the model
# head = base_model.output
# head = AveragePooling2D(pool_size=(7, 7))(head)
# head = Flatten()(head)
# head = Dense(128, activation="relu")(head)
# head = Dropout(0.5)(head)
# head = Dense(1, activation="sigmoid")(head)

# model = Model(inputs=base_model.input, outputs=head)

# # Compile
# # opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# opt = Adam(learning_rate=INIT_LR)
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# # Train
# print("Training started...")
# history = model.fit(
#     train_gen,
#     steps_per_epoch=train_gen.samples // BS,
#     validation_data=val_gen,
#     validation_steps=val_gen.samples // BS,
#     epochs=EPOCHS
# )

# # Create model directory if not exists
# os.makedirs("model", exist_ok=True)

# print("Saving model...")
# model.save(MODEL_PATH)

# print(f"Training completed! Model saved at: {MODEL_PATH}")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "model/face_mask_mobilenetv2.h5"

# Image size
IMG_SIZE = 224
BATCH_SIZE = 4   # small batch for small dataset

print("Loading training data...")
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Loading validation data...")
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Classes:", train_data.class_indices)

# Avoid steps_per_epoch = 0
steps_per_epoch = max(1, train_data.samples // BATCH_SIZE)
val_steps = max(1, val_data.samples // BATCH_SIZE)

# Load MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\nTraining started...")
history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=val_steps,
    epochs=5
)

# Save model
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
