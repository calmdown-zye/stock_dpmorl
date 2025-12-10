# src/envs/cost.py
from __future__ import annotations
import numpy as np

class CostModel:
    """
    간단한 체결 비용 모델
    - fee_rate: 체결 금액 대비 고정 수수료 비율 (왕복 기준이면 그대로 넣고, 편도면 2배 고려)
    - slip_rate: 체결 금액 대비 슬리피지 비율
    """
    def __init__(self, fee_rate: float = 0.0005, slip_rate: float = 0.0005):
        self.fee_rate = float(fee_rate)
        self.slip_rate = float(slip_rate)

    def transaction_cost(self, w_prev: np.ndarray, w_target: np.ndarray, price: np.ndarray) -> float:
        """
        포트폴리오 비중 변경에 따른 거래 비용(비율)을 근사 계산.
        - w_prev, w_target: 합이 1인 weight 벡터 (길이 N)
        - price: 현재 가격 벡터 (길이 N) — 단순 비율 계산에서는 직접 쓰지 않지만, 확장 여지 남김
        반환: 총 비용 비율 (e.g., 0.001 = 0.1%)
        """
        w_prev = np.asarray(w_prev, dtype=np.float32)
        w_target = np.asarray(w_target, dtype=np.float32)

        trade = np.abs(w_target - w_prev).sum()  # 총 체결 비중
        fee = self.fee_rate * trade
        slip = self.slip_rate * trade
        return float(fee + slip)

    def apply_after_trade(self, portfolio_ret: float, cost_rate: float) -> float:
        """
        거래 후 포트폴리오 수익률에서 비용 차감.
        """
        return float((1.0 + portfolio_ret) * (1.0 - cost_rate) - 1.0)
