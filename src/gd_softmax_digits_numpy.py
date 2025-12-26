import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    logits: (n, K)
    devuelve probs: (n, K) con filas sumando 1
    """
    # Estabilidad numérica: restar el máximo por fila
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-9) -> float:
    """
    y_true: (n,) etiquetas enteras 0..K-1
    probs: (n, K) probabilidades
    """
    n = y_true.shape[0]
    probs = np.clip(probs, eps, 1 - eps)
    correct = probs[np.arange(n), y_true]
    return float(-np.mean(np.log(correct)))


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y_true))


def main():
    # 1) Cargar dataset
    digits = load_digits()
    X = digits.data.astype(np.float32)     # (n, 64)
    y = digits.target.astype(np.int64)     # (n,)

    # 2) Normalización (clave para GD)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n, d = X_train.shape
    K = 10  # clases 0..9

    # 4) Parámetros del modelo (lo que aprende)
    # W: (d, K), b: (K,)
    rng = np.random.default_rng(42)
    W = (0.01 * rng.standard_normal((d, K))).astype(np.float32)
    b = np.zeros(K, dtype=np.float32)

    # 5) Hiperparámetros
    lr = 0.5
    epochs = 500
    l2 = 1e-4  # regularización opcional (puedes poner 0.0 si quieres)

    for epoch in range(1, epochs + 1):
        # Forward
        logits = X_train @ W + b  # (n, K)
        probs = softmax(logits)   # (n, K)

        # Loss (cross-entropy + L2)
        loss = cross_entropy_loss(y_train, probs)
        loss += float(0.5 * l2 * np.sum(W * W))

        # Gradientes (derivada de softmax+crossentropy simplifica mucho)
        # dlogits = (probs - one_hot(y)) / n
        dlogits = probs.copy()
        dlogits[np.arange(n), y_train] -= 1.0
        dlogits /= n

        dW = X_train.T @ dlogits + l2 * W   # (d, K)
        db = np.sum(dlogits, axis=0)        # (K,)

        # Update GD
        W -= lr * dW
        b -= lr * db

        if epoch == 1 or epoch % 50 == 0:
            train_acc = accuracy(y_train, probs)
            print(f"epoch={epoch:3d}  loss={loss:.4f}  train_acc={train_acc:.4f}")

    # Evaluación en test
    test_probs = softmax(X_test @ W + b)
    test_loss = cross_entropy_loss(y_test, test_probs) + float(0.5 * l2 * np.sum(W * W))
    test_acc = accuracy(y_test, test_probs)

    print("\nResultados en test:")
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    # Mostrar algunas predicciones con confianza
    print("\nEjemplos (probabilidades top-3):")
    for i in range(5):
        p = test_probs[i]
        top3 = np.argsort(-p)[:3]
        print(f"real={y_test[i]}  pred={int(np.argmax(p))}  top3={[(int(c), float(p[c])) for c in top3]}")


    # --- Visualización con matplotlib ---
    preds_test = np.argmax(test_probs, axis=1)
    wrong_idx = np.where(preds_test != y_test)[0]

    if wrong_idx.size > 0:
        i = int(wrong_idx[0])  # primer fallo
        img = X_test[i].reshape(8, 8)

        print(f"\nMostrando primer fallo: real={y_test[i]} pred={preds_test[i]}")

        # 1) Imagen del dígito
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"Real: {y_test[i]}  Pred: {preds_test[i]}")
        plt.axis("off")

        # 2) Probabilidades (barras)
        plt.figure()
        plt.bar(np.arange(10), test_probs[i])
        plt.title("Probabilidades por clase (0-9)")
        plt.xlabel("Clase")
        plt.ylabel("Probabilidad")
        plt.xticks(np.arange(10))

        plt.show()
    else:
        print("\nNo hay fallos en test (raro pero posible). No hay nada que visualizar.")



if __name__ == "__main__":
    main()
