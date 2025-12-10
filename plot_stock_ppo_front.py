# plot_stock_ppo_front.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_stock_cmp")
    parser.add_argument(
        "--objective",
        type=str,
        default="env_performance",
        choices=["env_performance", "execution_quality"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = os.path.join("experiments", args.exp_name)
    npz_path = os.path.join(exp_dir, f"ppo_{args.objective}_returns.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")

    data = np.load(npz_path)
    returns = data["returns"]          # (n_episodes, 2)
    mean_ret = data["mean_return"]     # (2,)
    print(f"mean J(pi) = {mean_ret}")

    # 단일 점만 찍는 버전
    plt.figure(figsize=(6, 5))
    plt.scatter(
        mean_ret[0],
        mean_ret[1],
        c="tab:green",
        marker="X",
        s=120,
        label=f"PPO ({args.objective})",
    )

    plt.xlabel("E[r_pm]  (portfolio return)")
    plt.ylabel("E[r_trader]  (execution quality)")
    plt.title("PPO J(pi) on Stock Portfolio Env")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(exp_dir, f"ppo_{args.objective}_front.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
