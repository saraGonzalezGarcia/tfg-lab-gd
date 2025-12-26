import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    os.makedirs("models", exist_ok=True)

    # 1) Cargar MNIST (28x28) 0..9
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2) Normalizar a [0,1] y añadir canal (N, 28, 28, 1)
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    # 3) Split train/val
    x_val = x_train[-5000:]
    y_val = y_train[-5000:]
    x_train2 = x_train[:-5000]
    y_train2 = y_train[:-5000]

    # 4) Data augmentation (ayuda a generalizar a dibujos tipo Paint)
    aug = keras.Sequential([
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ])

    # 5) Modelo CNN
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        aug,

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 6) Guardar el mejor modelo según val_accuracy
    out_path = "models/cnn_mnist_best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(out_path, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    model.fit(
        x_train2, y_train2,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss={test_loss:.4f}  test_acc={test_acc:.4f}")
    print(f"Mejor modelo guardado en: {out_path}")


if __name__ == "__main__":
    main()
