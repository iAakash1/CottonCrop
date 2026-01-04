import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from preprocessing.preprocess_training import preprocess_for_training

def load_dataset(dataset_dir):
    images, labels = [], []
    classes = sorted(os.listdir(dataset_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        for img_name in os.listdir(cls_path):
            img = cv2.imread(os.path.join(cls_path, img_name))
            if img is None:
                continue
            img = preprocess_for_training(img)
            images.append(img)
            labels.append(class_to_idx[cls])

    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
    return X, y, classes

def train_model(dataset_dir):
    X, y, classes = load_dataset(dataset_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(len(classes), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ModelCheckpoint("model/best_model.keras", save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        callbacks=callbacks
    )

    return model, classes
