import argparse
import numpy as np

from sklearn.datasets import load_digits


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    a1 = relu(X @ W1 + b1)
    logits = a1 @ W2 + b2
    return softmax(logits)


def preprocess_vector(x64: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = x64.astype(np.float32)
    x = (x - mean) / std
    return x.reshape(1, -1)


def image_to_8x8_vector(path: str) -> np.ndarray:
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)  # 0..255

    # Invertir si el fondo es blanco
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Aumentar contraste (para que el trazo no quede "flojo" al reducir)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    # Crear máscara del dígito y recortar bounding box
    thresh = 30.0
    mask = arr > thresh
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        arr = arr[y0:y1, x0:x1]

    # Hacer cuadrado (centrar)
    h, w = arr.shape
    m = max(h, w)
    square = np.zeros((m, m), dtype=np.float32)
    yoff = (m - h) // 2
    xoff = (m - w) // 2
    square[yoff:yoff + h, xoff:xoff + w] = arr

    # Padding alrededor para que no toque bordes
    pad = 10
    square = np.pad(square, pad_width=pad, mode="constant", constant_values=0)

    # Resize final a 8x8
    img2 = Image.fromarray(square.clip(0, 255).astype(np.uint8))
    img2 = img2.resize((8, 8), Image.Resampling.LANCZOS)

    out = np.asarray(img2).astype(np.float32)  # 0..255
    out = (out / 255.0) * 16.0                 # parecido a load_digits
    return out.reshape(-1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/mlp_digits_best.npz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--digits-index", type=int, help="Índice del dataset load_digits() para probar")
    group.add_argument("--image", type=str, help="Ruta a imagen PNG/JPG con un dígito centrado")
    args = parser.parse_args()

    data = np.load(args.model)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    mean, std = data["mean"], data["std"]

    if args.digits_index is not None:
        digits = load_digits()
        x64 = digits.data[args.digits_index]
        y = int(digits.target[args.digits_index])
        X = preprocess_vector(x64, mean, std)
        probs = forward(X, W1, b1, W2, b2)[0]
        pred = int(np.argmax(probs))

        top3 = np.argsort(-probs)[:3]
        print(f"REAL={y}  PRED={pred}")
        print("Top-3:", [(int(c), float(probs[c])) for c in top3])

    if args.image is not None:
        x64 = image_to_8x8_vector(args.image)
        X = preprocess_vector(x64, mean, std)
        probs = forward(X, W1, b1, W2, b2)[0]
        pred = int(np.argmax(probs))

        top3 = np.argsort(-probs)[:3]
        print(f"PRED={pred}")
        print("Top-3:", [(int(c), float(probs[c])) for c in top3])


if __name__ == "__main__":
    main()
