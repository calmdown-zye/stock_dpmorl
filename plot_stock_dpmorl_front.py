# plot_stock_dpmorl_front.py

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dpmorl_stock")
    parser.add_argument("--lamda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task_name = f"DPMORL.StockPortfolio.LossNormLamda_{args.lamda}"
    base_dir = os.path.join("experiments", args.exp_name, task_name)

    pattern = os.path.join(
        base_dir,
        f"MORL_StockPortfolio_PPO_policy*_seed{args.seed}_0.npz",
    )
    files = sorted(glob.glob(pattern))

    if not files:
        print("npz 파일을 찾지 못했어... 경로/exp-name/lamda가 맞는지 확인해줘.")
        print("찾은 경로 패턴:", pattern)
        return

    print(f"Found {len(files)} policies")
    all_means = []
    labels = []

    for path in files:
        data = np.load(path, allow_pickle=True)
        ep_returns = np.array(data["episode_vec_returns"])  # (num_episodes, reward_dim)

        # 혹시 1D로 되어 있으면 2D로 바꿔줌
        if ep_returns.ndim == 1:
            ep_returns = ep_returns.reshape(-1, 2)

        mean_ret = ep_returns.mean(axis=0)  # (2,)
        all_means.append(mean_ret)

        fname = os.path.basename(path)
        # 예: MORL_StockPortfolio_PPO_policyprogram-0_seed0_0.npz
        policy_name = fname.split("policy")[1].split("_seed")[0]
        labels.append(policy_name)

        print(f"{policy_name}: mean return = {mean_ret}")

    all_means = np.stack(all_means, axis=0)  # (num_policies, 2)

    # ---- 시각화 ----
    plt.figure(figsize=(6, 5))
    plt.scatter(all_means[:, 0], all_means[:, 1])

    for (x, y, label) in zip(all_means[:, 0], all_means[:, 1], labels):
        plt.text(x, y, label, fontsize=8, ha="center", va="bottom")

    plt.xlabel("Mean return dim 0")
    plt.ylabel("Mean return dim 1")
    plt.title(f"DPMORL StockPortfolio (lambda={args.lamda})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
