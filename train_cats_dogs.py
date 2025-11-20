import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def main():
    data_dir = "data"   # Folder must contain: /cats  and /dogs

    print("📦 Loading Dataset...")
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(180, 180),
        batch_size=32
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(180, 180),
        batch_size=32
    )

    class_names = train_ds.class_names
    print("✔ Classes:", class_names)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("🧠 Building CNN...")
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(180, 180, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("🚀 Training Model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    print("🔍 Evaluating...")
    loss, acc = model.evaluate(val_ds)
    print(f"Validation Accuracy = {acc:.4f}")

    print("💾 Saving model...")
    model.save("cat_dog_classifier.keras")
    print("✔ Model saved as cat_dog_classifier.keras")

    # Save training plot
    try:
        plt.figure()
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train", "val"])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=150)
        print("✔ Saved training_curve.png")
    except Exception as e:
        print("⚠️ Could not save training plot:", e)


if __name__ == "__main__":
    main()