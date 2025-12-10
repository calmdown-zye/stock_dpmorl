"""
Vanilla PPO on a scalar-reward stock portfolio env.

FinRL_Hierarchical_Env가 벡터 보상(또는 dict 보상)을 내놓는다고 가정하고,
그 중 하나(예: env_performance)를 골라 scalar reward로 쓰는 래퍼를 씌운다.
학습 후에는 원래 벡터 보상 env 위에서 J(pi) (벡터 기대보상)를 추정해서 저장.
"""

import os
import argparse
from typing import List, Callable

import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ====== 0. 벡터 보상을 스칼라 보상으로 바꾸는 래퍼 ======

class SingleObjectiveEnv(gym.Wrapper):
    """
    FinRL_Hierarchical_Env처럼
    - reward가 벡터: [r_pm, r_trader]
    또는 dict로 나오는 환경을 받아서,

    그 중 하나만 골라 scalar reward로 변환해주는 래퍼.
    info["reward_vec"] 에는 원래 벡터 보상을 같이 넣어둔다.
    """

    def __init__(self, env: gym.Env, objective: str = "env_performance"):
        super().__init__(env)
        assert objective in ("env_performance", "execution_quality")
        self.objective = objective

    def step(self, action):
        obs, reward_raw, done, info = self.env.step(action)

        # 1) reward가 dict인 경우 (안 쓰고 있으면 무시)
        if isinstance(reward_raw, dict):
            r_pm = float(reward_raw.get("env_performance", 0.0))
            r_tr = float(reward_raw.get("execution_quality", 0.0))
            reward_vec = np.array([r_pm, r_tr], dtype=np.float32)

            if self.objective == "env_performance":
                scalar_reward = r_pm
            else:
                scalar_reward = r_tr

        # 2) reward가 벡터/배열인 경우 (지금 이 케이스)
        else:
            reward_vec = np.asarray(reward_raw, dtype=np.float32).reshape(-1)
            if reward_vec.size < 2:
                scalar_reward = float(reward_vec.item())
            else:
                if self.objective == "env_performance":
                    scalar_reward = float(reward_vec[0])
                else:
                    scalar_reward = float(reward_vec[1])

        info = dict(info)
        info["reward_vec"] = reward_vec
        info["objective"] = self.objective

        return obs, scalar_reward, done, info

    def close(self):
        """
        원본 env.close()에서 DataFrame을 함수처럼 호출하는 버그가 있어서,
        여기서는 그냥 no-op으로 둔다.
        """
        # self.env.close()  # <- 이거 부르면 다시 DataFrame close 버그 터질 수 있음
        return


# ====== 1. base env 생성 함수 ======

def make_base_stock_env(
    tickers: List[str],
    years: int,
    fee_rate: float = 0.0005,
    slip_rate: float = 0.0005,
    rebalance_L: int = 5,
) -> gym.Env:
    """
    벡터 보상을 내놓는 FinRL_Hierarchical_Env 생성 (PMORL 때 쓰던 버전 그대로).
    """

    # 네 프로젝트 구조에 맞춤 (PMORL에서 잘 돌던 버전)
    from loader_yf import load_market_frames  # type: ignore
    from cost import CostModel  # type: ignore
    from finrl_hier_env import FinRL_Hierarchical_Env  # type: ignore

    close, ret = load_market_frames(tickers=tickers, years=years)
    cost = CostModel(fee_rate=fee_rate, slip_rate=slip_rate)

    env = FinRL_Hierarchical_Env(close=close, ret=ret, cost=cost, L=rebalance_L)
    return env


def make_vec_env(
    tickers: List[str],
    years: int,
    num_envs: int,
    fee_rate: float,
    slip_rate: float,
    rebalance_L: int,
    objective: str,
):
    def _make_env():
        base_env = make_base_stock_env(
            tickers=tickers,
            years=years,
            fee_rate=fee_rate,
            slip_rate=slip_rate,
            rebalance_L=rebalance_L,
        )
        return SingleObjectiveEnv(base_env, objective=objective)

    # DummyVecEnv는 [callable, callable, ...] 형식 필요
    return DummyVecEnv([_make_env for _ in range(num_envs)])


# ====== 2. 학습된 정책 J(pi) (벡터) 평가 ======

def evaluate_policy_morl(
    model: PPO,
    make_env_fn: Callable[[], gym.Env],
    reward_dim: int,
    n_episodes: int = 50,
):
    """
    학습된 PPO policy를 원래 벡터 보상 env 위에서 평가해서 J(pi)를 추정.

    반환:
        returns: shape (n_episodes, reward_dim)
        mean_ret: shape (reward_dim,)
    """
    all_returns = []

    for ep in range(n_episodes):
        env = make_env_fn()
        obs = env.reset()
        done = False
        ret_vec = np.zeros(reward_dim, dtype=np.float32)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_vec, done, info = env.step(action)
            ret_vec += np.asarray(reward_vec, dtype=np.float32).reshape(-1)

        all_returns.append(ret_vec)

    returns = np.stack(all_returns, axis=0)
    mean_ret = returns.mean(axis=0)
    return returns, mean_ret


# ====== 3. 학습 스크립트 ======

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="NVDA,TSLA,XOM,GLD,TLT")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slip-rate", type=float, default=0.0005)
    parser.add_argument("--rebalance-L", type=int, default=5)
    parser.add_argument("--exp-name", type=str, default="ppo_stock_cmp")

    # 어떤 보상을 scalar로 쓸지 선택 (env_performance / execution_quality)
    parser.add_argument(
        "--objective",
        type=str,
        default="env_performance",
        choices=["env_performance", "execution_quality"],
        help="Which component of the vector reward to optimize as scalar.",
    )

    parser.add_argument("--eval-episodes", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [s.strip() for s in args.tickers.split(",")]

    env = make_vec_env(
        tickers=tickers,
        years=args.years,
        num_envs=args.num_envs,
        fee_rate=args.fee_rate,
        slip_rate=args.slip_rate,
        rebalance_L=args.rebalance_L,
        objective=args.objective,
    )

    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=exp_dir,
    )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    model_path = os.path.join(exp_dir, f"ppo_stock_policy_{args.objective}")
    model.save(model_path)
    print(f"[PPO] Saved model to {model_path}")

    # ===== 학습된 정책을 벡터 보상 env에서 평가해서 J(pi) 저장 =====

    def make_env_for_eval():
        return make_base_stock_env(
            tickers=tickers,
            years=args.years,
            fee_rate=args.fee_rate,
            slip_rate=args.slip_rate,
            rebalance_L=args.rebalance_L,
        )

    returns, mean_ret = evaluate_policy_morl(
        model=model,
        make_env_fn=make_env_for_eval,
        reward_dim=2,  # [r_pm, r_trader]
        n_episodes=args.eval_episodes,
    )

    print(f"[PPO] mean J(pi) ≈ {mean_ret} (objective={args.objective})")

    np.savez_compressed(
        os.path.join(exp_dir, f"ppo_{args.objective}_returns.npz"),
        returns=returns,
        mean_return=mean_ret,
    )

    env.close()


if __name__ == "__main__":
    main()
