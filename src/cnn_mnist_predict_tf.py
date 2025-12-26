import argparse
import numpy as np
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt


def preprocess_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      - x: (1, 28, 28, 1) float32 en [0,1] para el modelo
      - img28: (28, 28) float32 en [0,1] para visualizar
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)  # 0..255

    # Invertir si el fondo es blanco (Paint típico)
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Aumentar contraste (para que el trazo no se pierda)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    # Recorte (bounding box)
    thresh = 25.0
    mask = arr > thresh
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        arr = arr[y0:y1, x0:x1]

    # Hacer cuadrado y centrar
    h, w = arr.shape
    m = max(h, w)
    square = np.zeros((m, m), dtype=np.float32)
    yoff = (m - h) // 2
    xoff = (m - w) // 2
    square[yoff:yoff + h, xoff:xoff + w] = arr

    # Padding (para no tocar bordes)
    pad = max(2, m // 6)
    square = np.pad(square, pad_width=pad, mode="constant", constant_values=0)

    # Resize final a 28x28
    img2 = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8))
    img2 = img2.resize((28, 28), Image.Resampling.LANCZOS)

    img28 = np.asarray(img2).astype(np.float32) / 255.0  # (28,28) en [0,1]
    x = img28.reshape(1, 28, 28, 1)  # (1,28,28,1)

    return x, img28


def show_debug_plots(img28: np.ndarray, probs: np.ndarray, pred: int):
    # 1) Imagen preprocesada (lo que ve la CNN)
    plt.figure()
    plt.imshow(img28, cmap="gray")
    plt.title(f"Imagen preprocesada (28x28) - pred={pred}")
    plt.axis("off")

    # 2) Probabilidades por clase
    plt.figure()
    plt.bar(np.arange(10), probs)
    plt.title("Probabilidades por clase (0-9)")
    plt.xlabel("Clase")
    plt.ylabel("Probabilidad")
    plt.xticks(np.arange(10))
    plt.ylim(0, 1)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/cnn_mnist_best.keras")
    parser.add_argument("--image", required=True)
    parser.add_argument("--show", action="store_true", help="Muestra imagen preprocesada y barras de probas")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    x, img28 = preprocess_image(args.image)

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    top3 = np.argsort(-probs)[:3]

    print(f"PRED={pred}")
    print("Top-3:", [(int(c), float(probs[c])) for c in top3])

    if args.show:
        show_debug_plots(img28, probs, pred)


if __name__ == "__main__":
    main()
