from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

# SB3 버전 호환용: dict_to_obs, obs_space_info 위치가 버전에 따라 다름
try:
    from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info
except ImportError:
    from stable_baselines3.common.vec_env.utils import dict_to_obs, obs_space_info

# gym → gymnasium 호환 래퍼
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def copy_obs_dict(obs):
    """
    예전 SB3 util.py 에 있던 copy_obs_dict 를 여기서 직접 정의.

    obs: dict(str -> np.ndarray) 혹은 np.ndarray
    반환: 값들이 np.copy 된 새 객체
    """
    if isinstance(obs, dict):
        return {k: np.copy(v) for k, v in obs.items()}
    else:
        return np.copy(obs)


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], reward_dim: int = 2):
        # 1) 원래 gym.Env 인스턴스들을 먼저 생성
        raw_envs = [fn() for fn in env_fns]

        # 2) SB3에서 제공하는 gym → gymnasium 호환 래퍼 적용
        #    이렇게 해야 action_space / observation_space 가 gymnasium.spaces.* 타입으로 변환됨
        self.envs = [_patch_env(env) for env in raw_envs]

        # 3) 같은 인스턴스를 재사용했는지 체크
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )

        # 4) VecEnv 초기화 (여기서 env.observation_space / env.action_space 는 이미 gymnasium.spaces.*)
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs, reward_dim), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata
        # 🔥 여기 추가: 에피소드 누적 버퍼
        self.reward_dim = reward_dim
        self.ep_rets = np.zeros((self.num_envs, reward_dim), dtype=np.float32)
        self.ep_lens = np.zeros(self.num_envs, dtype=int)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """
        Gym / Gymnasium 혼용 대응 + 에피소드 리턴 누적:
        - env.step(action)이 4개 (obs, reward, done, info)를 줄 수도 있고
        - 5개 (obs, reward, terminated, truncated, info)를 줄 수도 있음.

        내부적으로는 항상:
            obs, reward_vec, done, info
        형식으로 정규화해서 버퍼에 쌓고,
        done=True일 때 info["episode"]["r"]에 에피소드 누적 보상을 넣어준다.
        """
        for env_idx in range(self.num_envs):
            step_result = self.envs[env_idx].step(self.actions[env_idx])

            # 5-return (Gymnasium 스타일)
            if isinstance(step_result, (tuple, list)) and len(step_result) == 5:
                obs, rew, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            # 4-return (옛 Gym 스타일)
            elif isinstance(step_result, (tuple, list)) and len(step_result) == 4:
                obs, rew, done, info = step_result
            else:
                raise RuntimeError(
                    f"DummyVecEnv.step_wait: 예상치 못한 step 반환값: "
                    f"type={type(step_result)}, len={len(step_result) if hasattr(step_result, '__len__') else 'N/A'}"
                )

            # --- 보상 버퍼 업데이트 ---
            rew_arr = np.asarray(rew, dtype=np.float32).reshape(-1)
            self.buf_rews[env_idx] = rew_arr
            self.buf_dones[env_idx] = bool(done)

            # --- 에피소드 누적 (vector reward 기준) ---
            self.ep_rets[env_idx] += rew_arr
            self.ep_lens[env_idx] += 1

            info = dict(info)  # 수정 가능하도록 복사

            if done:
                # 이번 에피소드의 누적 벡터 리턴을 info["episode"]["r"]에 저장
                info["episode"] = {
                    "r": self.ep_rets[env_idx].copy(),       # shape = (reward_dim,)
                    "l": int(self.ep_lens[env_idx]),         # 에피소드 길이
                }

                # 다음 에피소드 준비
                self.ep_rets[env_idx][:] = 0.0
                self.ep_lens[env_idx] = 0

                # terminal obs 저장 후 reset
                info["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()

            self.buf_infos[env_idx] = info
            self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )



    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        """
        obs 가 np.ndarray, dict, list, tuple 등 다양한 형태로 올 수 있으므로
        SB3/VecEnv 버퍼 구조에 맞게 변환한다.
        """
        # (obs, info) 튜플로 오는 경우 -> obs만 취함
        if isinstance(obs, (list, tuple)):
            if len(obs) > 0:
                obs = obs[0]

        # dict 형태면 key별로 저장
        if isinstance(obs, dict):
            for key in self.keys:
                v = np.asarray(obs[key], dtype=np.float32)
                self.buf_obs[key][env_idx] = np.nan_to_num(v)
        else:
            # ndarray 형태로 강제 변환
            arr = np.asarray(obs, dtype=np.float32)

            # 혹시 차원이 맞지 않으면 flatten
            if arr.ndim > 1:
                arr = arr.flatten()

            key = self.keys[0]
            self.buf_obs[key][env_idx] = np.nan_to_num(arr)



    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: VecEnvIndices = None,
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        from stable_baselines3.common import env_util  # avoid circular import

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
