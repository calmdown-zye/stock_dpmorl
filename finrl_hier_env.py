import numpy as np
import pandas as pd
from dataclasses import dataclass
import gym
from gym import spaces
from gym.utils import seeding


@dataclass
class CostModel:
    fee_rate: float = 0.0005
    slip_rate: float = 0.0005

class FinRL_Hierarchical_Env(gym.Env):
    """
    Single-agent portfolio env (PM + Trader 통합 관점).

    - observation: concat(pm_obs, trader_obs) -> 1D vector
    - action: R^{2n}
        * first n: trader_action (weight change)
        * next n : w_star (target weights for TE 계산용)
    - reward: vector [r_pm, r_trader]
        * r_pm      = portfolio return
        * r_trader  = -(tracking_error + cost)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, close: pd.DataFrame, ret: pd.DataFrame, cost: CostModel, L: int = 5):
        super().__init__()
        
        
        close = close.sort_index()
        ret = ret.sort_index()
        
        self.close = close.astype(np.float32).ffill().bfill()
        self.ret = ret.astype(np.float32).fillna(0.0)
        
        

        assert close.index.equals(ret.index)
        self.tickers = list(self.close.columns)
        self.n = len(self.tickers)
        self.L = int(L)
        self.cost = cost

        self._t = 0
        self._T = len(self.close)
        self.w_exec = np.zeros(self.n, dtype=np.float32)
        self.mask = np.ones(self.n, dtype=bool)

        # 기존 정의 유지
        self.pm_obs_dim = 6
        self.trader_obs_dim = self.n * 2
        self.reward_dim = 2  # [r_pm, r_trader]
        self.np_random, _ = seeding.np_random(None)
        
        # 관측 공간: pm_obs + trader_obs
        obs_dim = self.pm_obs_dim + self.trader_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 액션 공간: trader_action(n) + w_star(n)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.n,),
            dtype=np.float32,
        )
        
    def seed(self, seed: int | None = None):
        """
        Stable-Baselines3 / gym 호환용 시드 함수.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    # ---------- 내부 관측 생성 함수들 ----------

    def _pm_obs(self):
        idx = slice(max(0, self._t - 60), self._t)
        r = self.ret.iloc[idx]
        
        r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        if len(r) == 0:
            m20 = v20 = m60 = v60 = 0.0
        else:
            m20 = float(r.tail(20).mean().mean())
            v20 = float(r.tail(20).std().mean())
            m60 = float(r.tail(60).mean().mean())
            v60 = float(r.tail(60).std().mean())
        te = 0.0
        out = np.array([m20, v20, m60, v60, te, float(self._t % self.L)], dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _trader_obs(self):
        if self._t > 0:
            last_r = self.ret.iloc[self._t - 1]
        else:
            last_r = self.ret.iloc[0] * 0
            
        last_r = last_r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        obs = np.concatenate([self.w_exec, last_r.to_numpy(dtype=np.float32)], axis=0)
        obs = obs.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_obs(self):
        """pm + trader 관측을 하나의 벡터로 합침."""
        out = np.concatenate([self._pm_obs(), self._trader_obs()], axis=0).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


    # ---------- reset / step ----------

    def reset(self):
        """옛날 gym 스타일: obs만 반환 (SB3가 patch해줌)."""
        self._t = 1
        self.w_exec = np.zeros(self.n, dtype=np.float32)
        return self._get_obs()

    def _apply_trade(self, target_change: np.ndarray):
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

    def step(self, action: np.ndarray):
        """
        action: shape (2n,)
            - action[:n]  : trader_action (weight change)
            - action[n:2n]: w_star (target weights for tracking error)
        return:
            obs_next: np.ndarray  (obs_dim,)
            reward_vec: np.ndarray shape (2,) = [r_pm, r_trader]
            done: bool
            info: dict
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        trader_action = action[:self.n]
        if action.shape[0] >= 2 * self.n:
            w_star = action[self.n:2 * self.n]
        else:
            # 액션이 n차원만 들어오면, w_star는 현 w_exec 기준으로 TE=0 으로 둠
            w_star = self.w_exec.copy()

        cost = self._apply_trade(trader_action)

        r_t = self.ret.iloc[self._t].to_numpy(dtype=np.float32)
        r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)
        port_ret = float((self.w_exec * r_t).sum())

        # tracking error: w_exec vs w_star
        te = float(np.sqrt(((self.w_exec - w_star) ** 2).sum()))

        r_trader = - (te + cost)
        r_pm = port_ret
        
        reward_vec = np.array([r_pm, r_trader], dtype=np.float32)
        reward_vec = np.nan_to_num(reward_vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        self._t += 1
        done = self._t >= self._T

        obs_next = self._get_obs()

        info = {
            "te": te,
            "cost": float(cost),
            "ret": float(port_ret),
            "reward_vec": reward_vec,
        }

        return obs_next, reward_vec, done, info

    # ---------- 기타 유틸 ----------

    def get_mask(self):
        return self.mask.copy()

    @property
    def pm_state_dim(self):
        return self.pm_obs_dim

    @property
    def trader_state_dim(self):
        return self.trader_obs_dim

    @property
    def trader_action_dim(self):
        return self.n

    @property
    def pm_action_dim(self):
        return 3  # momentum / value / lowvol

    @property
    def unwrapped(self):
        # SB3 / gym에서 base env 접근할 때 쓰는 속성
        return self
