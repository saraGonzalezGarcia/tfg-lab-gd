import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y: np.ndarray, probs: np.ndarray, eps: float = 1e-9) -> float:
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.log(probs[np.arange(y.shape[0]), y])))


def accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == y))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    a1 = relu(X @ W1 + b1)
    logits = a1 @ W2 + b2
    return softmax(logits)


def main():
    os.makedirs("models", exist_ok=True)

    # 1) Cargar dataset
    digits = load_digits()
    X = digits.data.astype(np.float32)   # (n, 64) valores ~0..16
    y = digits.target.astype(np.int64)   # (n,)

    # 2) Split train/test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Split train/val (para seleccionar "mejor modelo")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 4) Normalización usando SOLO train (importante)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-6

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    n, d = X_train.shape
    K = 10
    H = 32

    rng = np.random.default_rng(42)
    W1 = (0.05 * rng.standard_normal((d, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (0.05 * rng.standard_normal((H, K))).astype(np.float32)
    b2 = np.zeros(K, dtype=np.float32)

    lr = 0.1
    epochs = 200
    batch_size = 64
    l2 = 1e-4

    best_val_acc = -1.0
    best_params = None

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        Xs = X_train[idx]
        ys = y_train[idx]

        for start in range(0, n, batch_size):
            end = start + batch_size
            Xb = Xs[start:end]
            yb = ys[start:end]
            nb = Xb.shape[0]

            # Forward
            z1 = Xb @ W1 + b1
            a1 = relu(z1)
            logits = a1 @ W2 + b2
            probs = softmax(logits)

            # Backward
            dlogits = probs.copy()
            dlogits[np.arange(nb), yb] -= 1.0
            dlogits /= nb

            dW2 = a1.T @ dlogits + l2 * W2
            db2 = np.sum(dlogits, axis=0)

            da1 = dlogits @ W2.T
            dz1 = da1 * (z1 > 0)

            dW1 = Xb.T @ dz1 + l2 * W1
            db1 = np.sum(dz1, axis=0)

            # GD update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Evaluación cada 20 epochs
        if epoch == 1 or epoch % 20 == 0:
            train_probs = forward(X_train, W1, b1, W2, b2)
            val_probs = forward(X_val, W1, b1, W2, b2)

            train_loss = cross_entropy(y_train, train_probs) + float(0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2)))
            val_loss = cross_entropy(y_val, val_probs) + float(0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2)))

            train_acc = accuracy(y_train, train_probs)
            val_acc = accuracy(y_val, val_probs)

            print(
                f"epoch={epoch:3d}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Guardar mejor según val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # Restaurar mejor modelo y evaluar en test
    W1, b1, W2, b2 = best_params
    test_probs = forward(X_test, W1, b1, W2, b2)
    test_acc = accuracy(y_test, test_probs)
    test_loss = cross_entropy(y_test, test_probs) + float(0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2)))

    print("\nMejor val_acc:", best_val_acc)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Guardar modelo + normalización
    out_path = "models/mlp_digits_best.npz"
    np.savez(
        out_path,
        W1=W1, b1=b1, W2=W2, b2=b2,
        mean=mean, std=std
    )
    print(f"\nModelo guardado en: {out_path}")


if __name__ == "__main__":
    main()
