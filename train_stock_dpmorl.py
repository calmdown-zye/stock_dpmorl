"""
train_stock_dpmorl.py

DPMORL on stock portfolio environment.

원본: zpschang/DPMORL/main_policy.py (NeurIPS 2023)를 기반으로,
주식 포트폴리오용 FinRL_Hierarchical_Env 에 맞게
env 생성 부분과 경로만 최소 수정한 버전.
"""
from finrl_hier_env_dpmorl import FinRL_Hierarchical_Env_DPMORL

import os
import glob
import time
import argparse
from typing import List

import numpy as np
import torch
import gym
import sys

# =========================
#  경로 설정
# =========================

# 이 파일이 있는 MORL 루트 디렉토리
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# DPMORL 레포 디렉토리 (MORL/DPMORL)
DPMORL_DIR = os.path.join(ROOT_DIR, "DPMORL")

if DPMORL_DIR not in sys.path:
    sys.path.append(DPMORL_DIR)

# DPMORL 안쪽에 있는 유틸/모듈 경로들
NORM_DATA_PATH = os.path.join(DPMORL_DIR, "normalization_data", "data.pickle")
UTILITY_MODEL_ROOT = os.path.join(DPMORL_DIR, "utility-model-selected")

# stable-baselines3 & DPMORL utils
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# 원본 DPMORL utils.py 안의 DummyVecEnv
from utils import DummyVecEnv  # type: ignore

from MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import (
    MultiEnv_UtilityFunction,
    ObsInfoWrapper,
)
from MORL_stablebaselines3.utility_function.utility_function_parameterized import (
    Utility_Function_Parameterized,
)
from MORL_stablebaselines3.utility_function.utility_function_programmed import (
    Utility_Function_Programmed,
    Utility_Function_Linear,
)

from stable_baselines3.common.callbacks import BaseCallback


# ========= GPU 선택 (pynvml 없어도 죽지 않게 안전하게 수정) =========

def choose_gpu(args):
    """
    원본은 pynvml + 소켓으로 GPU 메모리 체크를 하는데,
    로컬 맥/CPU 환경에서도 바로 쓸 수 있도록 예외 처리 추가.
    """
    # total_timesteps == 0 이면 어차피 학습 안 하는 모드라 GPU 설정 스킵
    if args.total_timesteps == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    try:
        import socket
        import pynvml

        pynvml.nvmlInit()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        s.close()

        if args.gpu == "all":
            memory_gpu = []
            masks = np.ones(pynvml.nvmlDeviceGetCount())
            for gpu_id, mask in enumerate(masks):
                if mask == -1:
                    continue
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gpu.append(meminfo.free / 1024 / 1024)
            gpu1 = int(np.argmax(memory_gpu))
        else:
            gpu1 = int(args.gpu)

        print(f"****************************Chosen GPU : {gpu1}****************************")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu1)
        os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    except Exception as e:
        # pynvml 없거나 GPU가 없는 환경이면 그냥 CPU 사용
        print(f"[choose_gpu] GPU 선택을 건너뜁니다 (사유: {e}). CPU 사용.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ========= Argparse (원본 구조 유지) =========

def config_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Training Policies with DPMORL on stocks")

    parser.add_argument("--env", type=str, default="StockPortfolio")
    parser.add_argument("--exp_name", type=str, default="dpmorl_stock")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lamda", type=float, default=1e-2)
    parser.add_argument("--utility_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_test_episodes", type=int, default=100)
    parser.add_argument("--keep_scale", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--reward_two_dim", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--reward_dim_indices", type=str, default="")  # 예: "[0,1]"
    parser.add_argument("--linear_utility", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--augment_state", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--test_only", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_policies", type=int, default=1)
    parser.add_argument("--max_num_policies", type=int, default=20)
    parser.add_argument("--total_timesteps", type=float, default=1e6)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="all")

    # 포트폴리오 전용 옵션
    parser.add_argument("--tickers", type=str, default="NVDA,TSLA,XOM,GLD,TLT")
    parser.add_argument("--years", type=int, default=5)

    return parser.parse_args()


# ========= ReturnLogger (원본 코드 거의 그대로) =========

class ReturnLogger(BaseCallback):
    def __init__(self, save_dir, env_name, algo_name, policy_id, it, seed, verbose=0):
        super(ReturnLogger, self).__init__(verbose)
        self.episode_vec_returns = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.env_name = env_name
        self.algo_name = algo_name
        self.seed = seed
        self.iter = it
        self.policy_id = policy_id

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        if isinstance(infos, (tuple, list)):
            for info in infos:
                if "episode" in info:
                    self.episode_vec_returns.append(info["episode"]["r"])
        else:
            if self.locals.get("done") and "episode" in infos:
                self.episode_vec_returns.append(infos["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        file_name = (
            f"MORL_{self.env_name}_{self.algo_name}_policy{self.policy_id}"
            f"_seed{self.seed}_{self.iter}.npz"
        )
        file_path = os.path.join(self.save_dir, file_name)
        np.savez_compressed(file_path, episode_vec_returns=self.episode_vec_returns)


# ========= 포트폴리오용 env 생성 (핵심 수정 부분) =========

def make_stock_env(
    rank: int,
    utility_function,
    reward_dim: int,
    reward_dim_indices,
    tickers: List[str],
    years: int,
    seed: int = None,
):
    """
    DPMORL의 make_env 대체 버전.

    - loader_yf.load_market_frames 로 가격/수익률 가져옴
    - cost.CostModel, finrl_hier_env.FinRL_Hierarchical_Env 사용
    - FinRL_Hierarchical_Env 가 벡터 보상(reward_dim 차원)을
      반환한다고 가정하고, ObsInfoWrapper / MultiEnv_UtilityFunction 에
      그대로 연결.
    """

    from loader_yf import load_market_frames  # MORL 루트에 있는 파일
    from finrl_hier_env import CostModel
    from finrl_hier_env_dpmorl import FinRL_Hierarchical_Env_DPMORL

    def _init():
        if seed is not None:
            set_random_seed(seed + rank)

        close, ret = load_market_frames(tickers=tickers, years=years)
        cost = CostModel(fee_rate=0.0005, slip_rate=0.0005)

        # FinRL_Hierarchical_Env 자체가 multi-objective reward_vec 를
        # 반환한다고 가정 (이미 PMORL에서 그렇게 사용 중).
        env = FinRL_Hierarchical_Env_DPMORL(close=close, ret=ret, cost=cost, L=5)

        # DPMORL에서 사용하는 wrapper:
        # 관측에 보상/유틸리티 관련 정보 추가
        env = ObsInfoWrapper(
            env,
            reward_dim=reward_dim,
            reward_dim_indices=reward_dim_indices,
        )
        return env

    return _init


# ========= 평가 함수 (원본 단순화 버전) =========

def evaluate_policy(model, env, num_test_episodes):
    episode_returns = []
    obs = env.reset()
    while len(episode_returns) < num_test_episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = env.step(action)
        for info in infos:
            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                if len(episode_returns) >= num_test_episodes:
                    break
    return np.array(episode_returns)


# ========= main =========

def main():
    import pickle

    # ---- normalization data 로드 (경로 수정) ----
    with open(NORM_DATA_PATH, "rb") as file:
        normalization_data = pickle.load(file)

    args = config_args()
    choose_gpu(args)

    tickers = [s.strip() for s in args.tickers.split(",")]

    alg_name = "PPO"
    env_name = "StockPortfolio"

    # ---- 보상 차원 추정 ----
    if env_name in normalization_data:
        reward_shape = len(normalization_data[env_name]["min"][0])
    else:
        reward_shape = 2  # fallback

    if args.reward_two_dim:
        reward_shape = 2

    if args.reward_dim_indices == "":
        reward_dim_indices = list(range(reward_shape))
    else:
        reward_dim_indices = eval(args.reward_dim_indices)
        reward_shape = len(reward_dim_indices)

    print(f"{reward_dim_indices = }, {reward_shape = }")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if args.linear_utility:
        utility_class_programmed = Utility_Function_Linear
    else:
        utility_class_programmed = Utility_Function_Programmed

    norm = True

    utility_function_programmed = utility_class_programmed(
        reward_shape=reward_shape,
        norm=norm,
        lamda=args.lamda,
        function_choice=0,
        keep_scale=args.keep_scale,
    )
    num_utility_programmed = len(utility_function_programmed.utility_functions)

    # ---- 미리 학습된 유틸리티 로드 (경로 수정) ----
    util_model_dir = os.path.join(UTILITY_MODEL_ROOT, f"dim-{reward_shape}")
    assert os.path.isdir(util_model_dir), "There is no pretrained utility functions provided."

    num_pretrained_utility = len(glob.glob(os.path.join(util_model_dir, "*")))
    pretrained_utility_paths = [
        os.path.join(util_model_dir, f"utility-{i}.pt") for i in range(num_pretrained_utility)
    ]

    pretrained_utility_functions = []
    for path in pretrained_utility_paths:
        model = Utility_Function_Parameterized(
            reward_shape=reward_shape,
            norm=norm,
            lamda=args.lamda,
            max_weight=0.5,
            keep_scale=args.keep_scale,
            size_factor=1,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        model = model.to(DEVICE)
        pretrained_utility_functions.append(model)
    num_utility_pretrained = len(pretrained_utility_functions)

    if args.linear_utility:
        num_utility_pretrained = 0

    total_steps = int(args.total_timesteps)
    iterations = args.iters

    task_name = f"DPMORL.StockPortfolio.LossNormLamda_{args.lamda}"
    utility_dir = os.path.join(ROOT_DIR, "experiments", args.exp_name, task_name)
    os.makedirs(utility_dir, exist_ok=True)

    num_total_policies = min(
        num_utility_programmed + num_utility_pretrained,
        args.max_num_policies,
    )
    print(f"{num_total_policies = }")

    def get_utility(policy_idx):
        if policy_idx < num_utility_programmed:
            return utility_class_programmed(
                reward_shape=reward_shape,
                norm=norm,
                lamda=args.lamda,
                function_choice=policy_idx,
                keep_scale=args.keep_scale,
            )
        return pretrained_utility_functions[policy_idx - num_utility_programmed]

    # ===== 학습 =====
    if not args.test_only:
        for policy_idx in range(num_total_policies):
            utility_function = get_utility(policy_idx)

            if env_name in normalization_data:
                utility_function.min_val = normalization_data[env_name]["min"][0][
                    reward_dim_indices
                ]
                utility_function.max_val = normalization_data[env_name]["max"][0][
                    reward_dim_indices
                ]
                print("normalization data:", normalization_data[env_name])
            else:
                print("normalization data: None")

            env = DummyVecEnv(
                [
                    make_stock_env(
                        rank=i,
                        utility_function=utility_function,
                        reward_dim=reward_shape,
                        reward_dim_indices=reward_dim_indices,
                        tickers=tickers,
                        years=args.years,
                        seed=args.seed,
                    )
                    for i in range(args.num_envs)
                ]
            )

            env = MultiEnv_UtilityFunction(
                env,
                utility_function,
                reward_dim=reward_shape,
                augment_state=args.augment_state,
            )
            env.update_utility_function(utility_function)

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=DEVICE,
                n_epochs=5,
            )

            if policy_idx < num_utility_programmed:
                policy_name = f"program-{policy_idx}"
            else:
                policy_name = f"pretrain-{policy_idx - num_utility_programmed}"

            print(f"Training policy {policy_idx + 1} with {total_steps} steps...")
            curtime = time.time()

            return_logger = ReturnLogger(
                save_dir=utility_dir,
                env_name=env_name,
                algo_name=alg_name,
                policy_id=policy_name,
                it=0,
                seed=args.seed,
            )

            model.learn(total_timesteps=total_steps, callback=return_logger, progress_bar=True)
            print(f"Training one policy took {time.time() - curtime:.2f} seconds.")

            model.save(os.path.join(utility_dir, f"policy-{policy_name}"))

    # ===== 평가 =====
    if args.test_only:
        for policy_idx in range(num_total_policies):
            if policy_idx < num_utility_programmed:
                policy_name = f"program-{policy_idx}"
            else:
                policy_name = f"pretrain-{policy_idx - num_utility_programmed}"

            model_path = os.path.join(utility_dir, f"policy-{policy_name}.zip")
            if not os.path.exists(model_path):
                print(f"{policy_name} does not exist")
                continue

            model = PPO.load(model_path)
            env = DummyVecEnv(
                [
                    make_stock_env(
                        rank=i,
                        utility_function=get_utility(policy_idx),
                        reward_dim=reward_shape,
                        reward_dim_indices=reward_dim_indices,
                        tickers=tickers,
                        years=args.years,
                        seed=args.seed,
                    )
                    for i in range(10)
                ]
            )
            env = MultiEnv_UtilityFunction(
                env,
                get_utility(policy_idx),
                reward_dim=reward_shape,
                augment_state=args.augment_state,
            )

            print(
                f"Evaluating policy {policy_idx + 1} "
                f"with {args.num_test_episodes} episodes..."
            )
            returns = evaluate_policy(model, env, args.num_test_episodes)
            np.savez_compressed(
                os.path.join(utility_dir, f"test_returns_policy_{policy_name}.npz"),
                test_returns=returns,
            )


if __name__ == "__main__":
    main()
