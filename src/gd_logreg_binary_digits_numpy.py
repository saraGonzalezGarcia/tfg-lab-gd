import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Evita overflow en exp con valores grandes
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    # BCE = -mean(y*log(p) + (1-y)*log(1-p))
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    preds = (p >= threshold).astype(np.float32)
    return float(np.mean(preds == y))


def main():
    # 1) Dataset: dígitos (8x8) y nos quedamos con 0 vs 1
    digits = load_digits()
    X = digits.data.astype(np.float32)   # (n_samples, 64)
    y = digits.target

    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask].astype(np.float32)       # (n_samples,)

    # 2) Normalización (muy importante para GD)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n, d = X_train.shape

    # 3) Parámetros del modelo (lo que aprende)
    w = np.zeros(d, dtype=np.float32)
    b = 0.0

    # 4) Hiperparámetros
    lr = 0.2
    epochs = 300

    for epoch in range(1, epochs + 1):
        # Forward
        z = X_train @ w + b
        p = sigmoid(z)

        # Loss
        loss = binary_cross_entropy(y_train, p)

        # 5) Gradientes (derivados de BCE + sigmoid)
        # dw = (1/n) X^T (p - y)
        # db = mean(p - y)
        dw = (1.0 / n) * (X_train.T @ (p - y_train))
        db = float(np.mean(p - y_train))

        # 6) Update GD
        w -= lr * dw
        b -= lr * db

        if epoch == 1 or epoch % 50 == 0:
            train_acc = accuracy(y_train, p, threshold=0.5)
            print(f"epoch={epoch:3d}  loss={loss:.4f}  train_acc={train_acc:.4f}")

    # Test
    p_test = sigmoid(X_test @ w + b)
    test_loss = binary_cross_entropy(y_test, p_test)
    test_acc = accuracy(y_test, p_test, threshold=0.5)

    print("\nResultados en test:")
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")
    print("\nNota: el threshold=0.5 significa: si p>=0.5 => clase 1, si no => clase 0.")

    # Muestra algunas predicciones con probabilidad
    for i in range(5):
        prob = float(p_test[i])
        pred = 1 if prob >= 0.5 else 0
        print(f"Ejemplo {i}: prob(ser 1)={prob:.4f} -> pred={pred}  real={int(y_test[i])}")


if __name__ == "__main__":
    main()
