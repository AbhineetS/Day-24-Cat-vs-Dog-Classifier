cat > run_cat_dog_cnn.py <<'PY'
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def download_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url, extract=False)
    base_dir = os.path.dirname(zip_path)

    extracted_dir = os.path.join(base_dir, "cats_and_dogs_filtered")
    if not os.path.exists(extracted_dir):
        import shutil
        shutil.unpack_archive(zip_path, base_dir)
    return extracted_dir

def make_datasets(base_dir, img_size=(150,150), batch_size=32):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    return train_ds, val_ds

def build_model(img_size=(150,150), n_classes=2):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(*img_size, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(history):
    try:
        plt.figure(figsize=(8,4))
        plt.plot(history.history["accuracy"], label="train_acc")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title("Training Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        plt.close()
        print("📊 Saved: training_history.png")
    except Exception as e:
        print("⚠️ Could not save plot:", e)

def main():
    print("📥 Downloading dataset...")
    data_root = download_dataset()

    print("📦 Preparing datasets...")
    train_ds, val_ds = make_datasets(data_root)

    class_names = train_ds.class_names

    print("🧠 Building model...")
    model = build_model(n_classes=len(class_names))
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_accuracy"
    )

    print("🚀 Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=[checkpoint],
        verbose=2
    )

    print("🔍 Evaluating...")
    loss, acc = model.evaluate(val_ds)
    print(f"🎯 Final Accuracy: {acc:.4f}")

    print("💾 Saving final model...")
    try:
        model.save("cat_dog_model.keras")
        print("Saved: cat_dog_model.keras")
    except Exception as e:
        print("Couldn't save model:", e)

    plot_history(history)

    print("✅ Done!")

if __name__ == "__main__":
    main()
PY