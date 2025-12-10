import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass
import pandas as pd


@dataclass
class CostModel:
    fee_rate: float = 0.0005
    slip_rate: float = 0.0005


class FinRL_Hierarchical_Env_DPMORL(gym.Env):
    """
    DPMORL용: 벡터 보상 r ∈ R^2 를 반환하는 주식 포트폴리오 환경.

    - observation: 1D 벡터 (pm_obs + trader_obs 비슷한 구조)
    - action: R^n (포트폴리오 비중 변경)
    - step: (obs, reward_vec, done, info)
    - reset: (obs, info)  ← ★ gymnasium 스타일로 맞춰줌 (DPMORL 유틸리티 래퍼 호환)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, close: pd.DataFrame, ret: pd.DataFrame, cost: CostModel, L: int = 5):
        super().__init__()

        # 인덱스/타입 정리
        close = close.sort_index()
        ret = ret.sort_index()
        assert close.index.equals(ret.index)

        self.close = close.astype(np.float32).ffill().bfill()
        self.ret = ret.astype(np.float32).fillna(0.0)

        self.tickers = list(self.close.columns)
        self.n = len(self.tickers)
        self.L = int(L)
        self.cost = cost

        self._t = 0
        self._T = len(self.close)
        self.w_exec = np.zeros(self.n, dtype=np.float32)

        # 관측/보상 차원 정의 (PMORL 버전이랑 거의 동일하게)
        self.pm_obs_dim = 6
        self.trader_obs_dim = self.n * 2
        self.obs_dim = self.pm_obs_dim + self.trader_obs_dim

        self.reward_dim = 2  # [r_pm, r_trader]

        # observation / action space (여기는 gym.spaces 써도 됨)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # DPMORL에서는 액션을 "포트폴리오 비중 변경" 정도로만 사용
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n,),
            dtype=np.float32,
        )

    # ---------- 내부 유틸 ----------

    def _pm_obs(self):
        idx = slice(max(0, self._t - 60), self._t)
        r = self.ret.iloc[idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if len(r) == 0:
            m20 = v20 = m60 = v60 = 0.0
        else:
            m20 = float(r.tail(20).mean().mean())
            v20 = float(r.tail(20).std().mean())
            m60 = float(r.tail(60).mean().mean())
            v60 = float(r.tail(60).std().mean())
        te_dummy = 0.0  # 여기서는 TE를 직접 쓰지 않음
        out = np.array(
            [m20, v20, m60, v60, te_dummy, float(self._t % self.L)],
            dtype=np.float32,
        )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _trader_obs(self):
        if self._t > 0:
            last_r = self.ret.iloc[self._t - 1]
        else:
            last_r = self.ret.iloc[0] * 0.0

        last_r = last_r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        obs = np.concatenate([self.w_exec, last_r.to_numpy(dtype=np.float32)], axis=0)
        obs = obs.astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_obs(self):
        obs = np.concatenate([self._pm_obs(), self._trader_obs()], axis=0)
        return np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_trade(self, target_change: np.ndarray) -> float:
        delta = np.clip(target_change, -1.0, 1.0).astype(np.float32)
        w_new = self.w_exec + delta
        w_new = np.clip(w_new, 0.0, 1.0)

        s = w_new.sum()
        if s > 0:
            w_new = w_new / s

        traded = np.abs(w_new - self.w_exec)
        cost = self.cost.fee_rate * traded.sum() + self.cost.slip_rate * traded.sum()
        self.w_exec = w_new
        return float(cost)

    # ---------- reset / step ----------
    def reset(self):
        self._t = 1
        self.w_exec = np.zeros(self.n, dtype=np.float32)
        obs = self._get_obs()
        assert isinstance(obs, np.ndarray), f"reset() must return ndarray, got {type(obs)}"
        return obs
    def step(self, action: np.ndarray):
        """
        DPMORL 쪽 유틸리티 래퍼는 reward_vec ∈ R^2 를 기대.
        여기서는:
          r_pm     = 포트폴리오 수익률
          r_trader = - (거래비용)
        정도로 단순하게 설정 (원하는 대로 바꿔도 됨)
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == self.n

        cost = self._apply_trade(action)

        r_t = self.ret.iloc[self._t].to_numpy(dtype=np.float32)
        r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)
        port_ret = float((self.w_exec * r_t).sum())

        # 보상 벡터: [환경 성과, 거래 품질(= 비용 패널티)]
        r_pm = port_ret
        r_trader = -cost
        reward_vec = np.array([r_pm, r_trader], dtype=np.float32)
        reward_vec = np.nan_to_num(reward_vec, nan=0.0, posinf=0.0, neginf=0.0)

        self._t += 1
        done = self._t >= self._T

        obs_next = self._get_obs()
        info = {
            "ret": float(port_ret),
            "cost": float(cost),
            "reward_vec": reward_vec,
        }

        # DPMORL / MORL_stablebaselines3 쪽은 (obs, reward_vec, done, info) 형식 기대
        return obs_next, reward_vec, done, info

    # 필요하면 seed, render 등 추가
    def seed(self, seed: int | None = None):
        np.random.seed(seed)
        return [seed]
