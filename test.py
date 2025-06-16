import numpy as np
import matplotlib.pyplot as plt

# 1. 適当な関数の設定
def f(x):
    return -100 * x * (x - 0.5)**2 * (x - 1)

# 2. データ生成
def generate_data(N, beta, func):
    x = np.random.uniform(0, 1, N)
    y_true = func(x)
    noise = np.random.normal(0, np.sqrt(1/beta), N)
    y = y_true + noise
    return x, y, y_true

# 3. 基底関数と計画行列
def radial_basis_functions(x, M, s):
    mus = np.linspace(0, 1, M, endpoint=False)
    Phi = np.exp(-((x[:, None] - mus[None, :])**2) / s)
    return Phi

# 4. 線形回帰の解
def fit_weights(Phi, y):
    w = np.linalg.pinv(Phi) @ y
    return w

# 4'. 線形回帰の解
def fit_weights2(Phi, y):
    w = np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ y)
    return w

# 5. 近似関数
def y_pred(x, w, M, s):
    Phi = radial_basis_functions(x, M, s)
    return Phi @ w

if __name__ == "__main__":
    # パラメータ設定
    N_list = [10, 30, 50, 100]    # データ数のリスト
    M_list = [3, 5, 10]          # 基底関数の数のリスト
    s = 0.01                      # 基底関数の幅
    beta = 100                    # ノイズの逆分散
    colors = ["red", "green", "orange", "purple", "brown", "cyan"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    y_min, y_max = -1, 4

    for idx, N in enumerate(N_list):
        x_train, y_train, y_true_train = generate_data(N, beta, f)
        x_test = np.linspace(0, 1, 200)
        y_test_true = f(x_test)

        ax = axes[idx]
        ax.plot(x_test, y_test_true, label="f(x) (True Function)", color="black", linewidth=2)
        ax.scatter(x_train, y_train, label="Training Data", color="blue", s=20, alpha=0.5)

        for i, M in enumerate(M_list):
            Phi = radial_basis_functions(x_train, M, s)
            w = fit_weights(Phi, y_train)
            y_test_pred = y_pred(x_test, w, M, s)
            ax.plot(x_test, y_test_pred, label=f"M={M}", color=colors[i % len(colors)])

        ax.set_title(f"N={N}, s={s}, beta={beta}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(y_min, y_max)
        ax.legend()
    
    print("a")
    plt.tight_layout()
    plt.show()
