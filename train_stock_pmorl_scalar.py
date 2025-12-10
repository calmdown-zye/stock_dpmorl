"""
train_stock_pmorl_scalar.py

Pareto Multi-Objective RL (PMORL) baseline:
여러 개의 가중치 w 에 대해, 각기 다른 스칼라 목적 w^T J(pi)를
PPO로 최적화하는 스크립트.

전제:
- base 주식 환경은 step()에서 벡터 보상 reward_vec (shape: [d])를 반환.
- 여기서는 그 위에 선형 스칼라화만 씌워서 w^T r 을 PPO의 reward로 사용.
"""

import os
import argparse
from typing import List, Callable

import numpy as np
import gym
from gym import spaces


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.monitor import Monitor


# =========================
#  1. Scalarization Wrapper
# =========================

class ScalarizedMORLEnv(gym.Wrapper):
    """
    Multi-objective env (reward_vec ∈ R^d)를 받아
    선형 스칼라화 reward = w^T reward_vec 로 바꿔주는 래퍼.

    PPO 입장에서는 그냥 scalar reward env처럼 보이지만,
    info["reward_vec"]에 원래 벡터 보상을 같이 넣어둠.
    """

    def __init__(self, env: gym.Env, weight: np.ndarray):
        super().__init__(env)
        self.weight = np.asarray(weight, dtype=np.float32).reshape(-1)

    def step(self, action):
        obs, reward_vec, done, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)

        # 선형 스칼라화: w^T r
        scalar_reward = float(np.dot(self.weight, reward_vec))

        # 벡터 보상은 나중에 분석할 수 있게 info에 저장
        info = dict(info)
        info["reward_vec"] = reward_vec
        info["weight"] = self.weight

        return obs, scalar_reward, done, info

    def close(self):
        """
        SB3에서 vec_env.close() 호출 시 들어오는 close.
        기본 gym.Wrapper.close()는 내부 env 체인을 따라가며 close를 호출하는데,
        지금 우리 환경 체인 어딘가에서 pandas DataFrame이 섞여 있어서
        TypeError가 나고 있음.

        파일럿 실험 수준에서는 특별히 정리할 리소스가 없으니
        close를 그냥 no-op으로 오버라이드해서 에러를 막는다.
        """
        # 굳이 super().close() 호출 안 하고 무시
        return
# =========================
#  2. Base Env 생성 함수 (TODO 부분 네 코드에 맞게 수정)
# =========================

def make_base_stock_env(
    tickers: List[str],
    years: int,
    fee_rate: float,
    slip_rate: float,
    rebalance_L: int,
) -> gym.Env:
    """
    다목적 주식 포트폴리오 환경을 생성하는 함수.

    ⚠️ 이 안은 네가 이미 만들어둔 환경 코드에 맞게 수정해야 한다.
       아래는 'loader_yf + CostModel + FinRL_Hierarchical_Env' 조합일 거라는
       가정 하에 작성한 예시(추측)라서, 실제 경로/클래스 이름이 다를 수 있음.
    """
    # TODO: 실제 모듈 경로/클래스 이름 확인해서 수정
    from loader_yf import load_market_frames      # type: ignore
    from cost import CostModel                     # type: ignore
    from finrl_hier_env import FinRL_Hierarchical_Env  # type: ignore

    close, ret = load_market_frames(tickers=tickers, years=years)
    cost = CostModel(fee_rate=fee_rate, slip_rate=slip_rate)

    # 여기서는 FinRL_Hierarchical_Env 가 reward_vec 을 반환하도록
    # 이미 수정되어 있다고 가정.
    env = FinRL_Hierarchical_Env(close=close, ret=ret, cost=cost, L=rebalance_L)

    return env


def make_scalarized_vec_env(
    tickers: List[str],
    years: int,
    weight: np.ndarray,
    fee_rate: float,
    slip_rate: float,
    rebalance_L: int,
    num_envs: int,
) -> DummyVecEnv:
    """
    특정 weight w 에 대해
    - base MORL env 생성
    - ScalarizedMORLEnv 로 감싸기
    - Monitor + DummyVecEnv 로 묶기
    """

    def make_one_env() -> gym.Env:
        base_env = make_base_stock_env(
            tickers=tickers,
            years=years,
            fee_rate=fee_rate,
            slip_rate=slip_rate,
            rebalance_L=rebalance_L,
        )
        env = ScalarizedMORLEnv(base_env, weight=weight)
        #env = Monitor(env)  # episode return/logging
        return env

    return DummyVecEnv([make_one_env for _ in range(num_envs)])


# =========================
#  3. 평가: J(pi) (벡터) 측정
# =========================
# =========================
#  3. 평가: J(pi) (벡터) 측정
# =========================

def evaluate_policy_morl(
    model: PPO,
    env: gym.Env,
    reward_dim: int,
    n_episodes: int = 50,
):
    """
    학습된 policy에 대해,
    base MORL env 위에서 (벡터 보상 기준) J(pi)를 추정.

    env는 FinRL_Hierarchical_Env 같은 "벡터 보상" 환경 하나라고 가정.
    (DummyVecEnv 아님!)
    """
    all_returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ret_vec = np.zeros(reward_dim, dtype=np.float32)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_vec, done, info = env.step(action)
            reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)
            ret_vec += reward_vec

        all_returns.append(ret_vec)

    returns = np.stack(all_returns, axis=0)
    mean_ret = returns.mean(axis=0)
    return returns, mean_ret


# =========================
#  4. Argument 파싱
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    # 데이터 관련
    parser.add_argument("--tickers", type=str, default="NVDA,TSLA,XOM,GLD,TLT")
    parser.add_argument("--years", type=int, default=5)

    # 보상 관련
    parser.add_argument("--reward-dim", type=int, default=2)  # r ∈ R^d

    # PMORL 관련
    parser.add_argument("--num-policies", type=int, default=5, help="학습할 w 개수 (정책 수)")
    parser.add_argument("--num-envs-per-policy", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=500_000)

    # 거래 비용 / 리밸런싱
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slip-rate", type=float, default=0.0005)
    parser.add_argument("--rebalance-L", type=int, default=5)

    # 기타
    parser.add_argument("--exp-name", type=str, default="pmorl_stock_scalar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)

    return parser.parse_args()


# =========================
#  5. 메인 루프: 여러 w 에 대해 PPO 학습
# =========================

def main():
    args = parse_args()

    tickers = [s.strip() for s in args.tickers.split(",")]
    os.makedirs("experiments", exist_ok=True)
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    np.random.seed(args.seed)

    # --- w 생성 ---
    # reward_dim=2일 때는 단순 grid, 그 외에는 Dirichlet 로 샘플 (둘 다 기본적인 방법)
    if args.reward_dim == 2:
        # 예: [1,0], [0.75,0.25], ..., [0,1]
        lambdas = np.linspace(0.0, 1.0, args.num_policies)
        weights = [np.array([lam, 1.0 - lam], dtype=np.float32) for lam in lambdas]
    else:
        alpha = np.ones(args.reward_dim, dtype=np.float32)
        weights = np.random.dirichlet(alpha, size=args.num_policies).astype(np.float32)

    print("사용할 weight 벡터들 (w):")
    for i, w in enumerate(weights):
        print(f"  policy {i}: w = {w}")

    # --- 각 w에 대해 별도 정책 학습 ---
    # --- 각 w에 대해 별도 정책 학습 ---
    for pid, w in enumerate(weights):
        print("=" * 80)
        print(f"[Policy {pid}] weight w = {w}")
        print("=" * 80)

        vec_env = make_scalarized_vec_env(
            tickers=tickers,
            years=args.years,
            weight=w,
            fee_rate=args.fee_rate,
            slip_rate=args.slip_rate,
            rebalance_L=args.rebalance_L,
            num_envs=args.num_envs_per_policy,
        )

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(exp_dir, "tb"),
            seed=args.seed,
        )

        model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

        # 모델 저장
        policy_name = f"policy_w{pid}"
        model_path = os.path.join(exp_dir, f"{policy_name}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # --- 벡터 보상 기준 J(pi) 평가 ---
        # 학습은 scalarized DummyVecEnv(vec_env)에서 하고,
        # 평가는 벡터 보상을 그대로 내보내는 base env(eval_env)에서 한다.
        eval_env = make_base_stock_env(
            tickers=tickers,
            years=args.years,
            fee_rate=args.fee_rate,
            slip_rate=args.slip_rate,
            rebalance_L=args.rebalance_L,
        )

        returns, mean_ret = evaluate_policy_morl(
            model=model,
            env=eval_env,
            reward_dim=args.reward_dim,
            n_episodes=args.eval_episodes,
        )

        print(f"[Policy {pid}] mean J(pi) ≈ {mean_ret} for w = {w}")

        # 결과 저장 (J(pi)와 w 둘 다)
        np.savez_compressed(
            os.path.join(exp_dir, f"{policy_name}_returns.npz"),
            weight=w,
            returns=returns,
            mean_return=mean_ret,
        )

        # ✅ 여기서는 eval_env를 '호출'하지 말고, 닫을 필요도 없으면 그냥 둬도 됨
        # eval_env.close() 같은 라인 통째로 제거
        vec_env.close()

    print("=== PMORL 학습 완료 ===")


if __name__ == "__main__":
    main()
