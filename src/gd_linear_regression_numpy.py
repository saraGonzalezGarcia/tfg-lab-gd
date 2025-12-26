import numpy as np

# 1) Loss: MSE (Mean Squared Error)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def main():
    rng = np.random.default_rng(42)

    # 2) Dataset sintético: y = 3x + 2 + ruido
    n = 200
    X = rng.uniform(-2, 2, size=(n, 1))
    y = 3.0 * X[:, 0] + 2.0 + rng.normal(0, 0.5, size=n)

    # 3) Parámetros del modelo (lo que aprende)
    w = 0.0
    b = 0.0

    # 4) Hiperparámetros del entrenamiento
    lr = 0.1          # learning rate
    epochs = 200      # iteraciones

    for epoch in range(1, epochs + 1):
        # Predicción del modelo lineal: y_hat = w*x + b
        y_pred = w * X[:, 0] + b

        # 5) Gradientes de la MSE respecto a w y b
        # dw = (2/n) * sum((y_pred - y) * x)
        # db = (2/n) * sum((y_pred - y))
        dw = (2.0 / n) * np.sum((y_pred - y) * X[:, 0])
        db = (2.0 / n) * np.sum(y_pred - y)

        # 6) Update de Gradient Descent
        w -= lr * dw
        b -= lr * db

        # 7) Logging
        if epoch == 1 or epoch % 20 == 0:
            loss = mse(y, y_pred)
            print(f"epoch={epoch:3d}  loss={loss:.4f}  w={w:.4f}  b={b:.4f}")

    print("\nResultado final:")
    print(f"w={w:.4f}  b={b:.4f}")
    print("Debería aproximarse a w≈3 y b≈2 (más o menos, por el ruido).")

if __name__ == "__main__":
    main()
