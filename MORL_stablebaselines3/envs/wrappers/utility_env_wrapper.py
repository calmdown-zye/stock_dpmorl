import IPython
import numpy as np
import torch
from typing import Optional
from gym import Env
import gym
import gymnasium
from MORL_stablebaselines3.envs.utils import Array
import math
# from MORL_stablebaselines3.morl.utility_function_torch import Utility_Function
import sys
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

class ObsInfoWrapper(gym.Wrapper):
    def __init__(
            self,
            env, 
            reward_dim, 
            reward_dim_indices
    ):
        super().__init__(env)

        self.cur_timesteps = 0
        self.reward_dim = reward_dim
        self.reward_dim_indices = reward_dim_indices
        self.actual_reward_dim = self.env.reward_dim
        self.zt = np.zeros(self.actual_reward_dim)
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
                                           shape=self.action_space.shape, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(self.action_space.n)
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.obs_high = np.array(self.observation_space.high, dtype=np.float32)
            self.obs_low = np.array(self.observation_space.low, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high,
                                                   shape=self.observation_space.shape,
                                                   dtype=self.observation_space.dtype)
        elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
            # self.observation_space = gym.spaces.Discrete(self.observation_space.n + 2)
            self.observation_space = gym.spaces.Discrete(self.observation_space.n)


    def reset(self, *args, **kwargs):
        """
        Gym / Gymnasium 혼용 대응:
        - obs or (obs, info) 둘 다 들어올 수 있음.
        DPMORL DummyVecEnv는 obs만 쓰므로 obs만 반환.
        """
        out = super().reset(*args, **kwargs)

        if isinstance(out, (list, tuple)) and len(out) == 2:
            obs, info = out
        else:
            obs = out

        # 여기서 obs를 ObsInfoWrapper 목적에 맞게 가공
        # obs = self._augment_reset_obs(obs)  # 네 기존 코드 로직 유지

        return obs


    def _augment_state(self, state: np.ndarray, returns: np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, returns])
        return augmented_state

    def step(self, action):
        """
        Gym / Gymnasium 혼용 대응:
        - super().step(action)이 5개 (obs, reward, terminated, truncated, info)를 줄 수도 있고
        - 4개 (obs, reward, done, info)를 줄 수도 있음.
        여기서는 DPMORL DummyVecEnv가 4-return을 기대하므로,
        최종적으로는 (obs, reward, done, info) 형식으로 맞춰서 반환한다.
        """
        out = super().step(action)

        # 5-return (Gymnasium 스타일) 대응
        if isinstance(out, (list, tuple)) and len(out) == 5:
            next_obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        # 4-return (옛 Gym 스타일) 대응
        elif isinstance(out, (list, tuple)) and len(out) == 4:
            next_obs, reward, done, info = out
        else:
            raise RuntimeError(
                f"ObsInfoWrapper.step: 예상치 못한 반환값 형태: type={type(out)}, len={len(out) if hasattr(out, '__len__') else 'N/A'}"
            )

        # 여기서 next_obs 를 ObsInfoWrapper 목적에 맞게 가공
        # (기존 코드에 있던 관측 확장/보상 기록 로직 유지)
        # 예시: (네 원래 코드에 맞춰서 그대로 남겨 둬)
        # next_obs = self._augment_obs(next_obs, reward, info)

        return next_obs, reward, done, info

class MultiEnv_UtilityFunction(VecEnvWrapper):
    def __init__(
            self,
            venv: VecEnv,
            utility_function,
            discount_factor=0.99,
            reward_dim=2,
            augment_state=False,
            **kwargs
    ):
        super().__init__(venv)
        self.num_envs = venv.num_envs
        self.utility_function = utility_function
        self.reward_dim = reward_dim
        self.augment_state = augment_state

        self.zt = np.zeros([self.num_envs, self.reward_dim])
        self.gamma = discount_factor  # same to gamma for RL
        self.action_space = venv.action_space
        
        if self.augment_state:
            low = np.hstack([venv.observation_space.low, np.full((self.reward_dim, ), -np.inf)])
            high = np.hstack([venv.observation_space.high, np.full((self.reward_dim, ), np.inf)])
            
            self.observation_space = gym.spaces.Box(low=low, high=high,
                                                    shape=low.shape,
                                                    dtype=venv.observation_space.dtype)
            self.min_val = self.utility_function.min_val[np.newaxis, :]
            self.max_val = self.utility_function.max_val[np.newaxis, :]
        else:
            self.observation_space = venv.observation_space
        
            
    def step_wait(self):
        return self.venv.step_wait()

    def update_utility_function(self, func):
        self.utility_function = func
        self.utility_function.eval()

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        obs = self.venv.reset()
        self.zt = np.zeros([self.num_envs, self.reward_dim])
        self.total_reward = np.zeros([self.num_envs])
        if self.augment_state:
            normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
            obs = self._augment_state(obs, normalized_return)
        return obs

    def _augment_state(self, state: np.ndarray, returns: np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, returns])
        return augmented_state

    def step(self, action):
        """
        UtilityEnvWrapper step:
        - env로부터 scalar reward + info["reward_vec"]를 받음
        - z_t는 reward_vec으로 누적
        - PPO에는 utility 증가량(new_reward)만 전달
        """

        next_obs, reward, done, infos = super().step(action)

        # -------------------------------------------------
        # 1. reward_vec 수집 (VecEnv / 단일 env 모두 대응)
        # -------------------------------------------------
        if isinstance(infos, (list, tuple)):
            # VecEnv
            reward_vecs = []
            for info_i in infos:
                if "reward_vec" not in info_i:
                    raise RuntimeError("reward_vec not found in info. Env must provide vector reward.")
                reward_vecs.append(info_i["reward_vec"])

            reward_vecs = np.asarray(reward_vecs, dtype=float)  # (n_envs, reward_dim)

        else:
            # 단일 env
            if "reward_vec" not in infos:
                raise RuntimeError("reward_vec not found in info. Env must provide vector reward.")
            reward_vecs = np.asarray(infos["reward_vec"], dtype=float)[None, :]  # (1, reward_dim)

        # -------------------------------------------------
        # 2. z_t 업데이트 (Distributional return)
        # -------------------------------------------------
        zt_next = self.zt + reward_vecs

        # -------------------------------------------------
        # 3. PPO에 줄 scalar reward = utility 증가량
        # -------------------------------------------------
        with torch.no_grad():
            new_reward = self.utility_function(zt_next) - self.utility_function(self.zt)
            # new_reward shape: (n_envs,)

        self.total_reward += new_reward
        self.zt = zt_next

        # -------------------------------------------------
        # 4. 종료된 env 처리 (VecEnv / 단일 env 대응)
        # -------------------------------------------------
        if isinstance(done, np.ndarray):
            done_mask = done
        else:
            done_mask = np.array([done], dtype=bool)

        if done_mask.any():
            if self.augment_state:
                normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
                for idx, d in enumerate(done_mask):
                    if d and isinstance(infos, (list, tuple)):
                        info_env = infos[idx]
                        if 'terminal_observation' in info_env:
                            info_env['terminal_observation'] = np.concatenate(
                                [info_env['terminal_observation'], normalized_return[idx]],
                                axis=0
                            )

            # 종료된 env만 리셋
            self.total_reward[done_mask] = 0.0
            self.zt[done_mask] = 0.0

        # -------------------------------------------------
        # 5. 상태 augmentation
        # -------------------------------------------------
        if self.augment_state:
            normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
            next_obs = self._augment_state(next_obs, normalized_return)

        # -------------------------------------------------
        # 6. PPO 규약에 맞게 반환
        # -------------------------------------------------
        return next_obs, new_reward, done, infos


if __name__ == "__main__":
    pass