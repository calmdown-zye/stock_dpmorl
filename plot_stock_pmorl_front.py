import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def main():
    exp_dir = "experiments/pmorl_stock_scalar"
    pattern = os.path.join(exp_dir, "policy_w*_returns.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No .npz files found in {exp_dir}")
        return

    Ws = []      # weight 벡터들 (w = [w1, w2])
    Js = []      # mean_return 벡터들 (J = [J_pm, J_trader])

    print("Loaded policies:")
    for f in files:
        data = np.load(f)
        w = data["weight"]            # shape (2,)
        mean_ret = data["mean_return"]  # shape (2,)

        Ws.append(w)
        Js.append(mean_ret)

        print(f"  {os.path.basename(f)}: w={w}, J={mean_ret}")

    Ws = np.stack(Ws, axis=0)  # (num_policies, 2)
    Js = np.stack(Js, axis=0)  # (num_policies, 2)

    # J_pm = 수익, J_trader = 실행 품질(더 덜 마이너스일수록 좋음)
    J_pm = Js[:, 0]
    J_trader = Js[:, 1]

    # 실행 품질은 - (TE + cost)이기 때문에,
    # 사람이 보기 쉽게 "집행 좋은 정도"를 + 방향으로 보고 싶다면 부호를 바꿔도 됨.
    # 여기서는 원래 값(J_trader)을 그대로 쓰고, y축 레이블에 설명만 달자.
    plt.figure(figsize=(7, 6))

    scatter = plt.scatter(J_pm, J_trader, c=Ws[:, 0], cmap="viridis", s=80)

    for i, (x, y) in enumerate(zip(J_pm, J_trader)):
        w = Ws[i]
        # 너무 복잡하지 않게, w1만 간단히 표시
        plt.text(x, y, f"w1={w[0]:.2f}", fontsize=8, ha="left", va="bottom")

    plt.colorbar(scatter, label="w1 (weight for portfolio return)")

    plt.xlabel("J_pm (portfolio return)")
    plt.ylabel("J_trader (sum of execution-quality reward)")
    plt.title("PMORL Scalarization: Pareto-style View of Policies")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(exp_dir, "pmorl_frontier.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
