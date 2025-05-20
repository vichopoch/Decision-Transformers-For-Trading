"""Simplified IPSA trading environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd


class IpsaTradingEnv(gym.Env):
    """Basic stock trading environment with transaction costs."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, cost: float = 0.001) -> None:
        super().__init__()
        self.data = data.sort_values("Date").reset_index(drop=True)
        self.tickers = sorted({t for t in data["Ticker"]})
        self.cost = cost
        self.current_step = 0
        self.weights = np.zeros(len(self.tickers), dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(self.tickers),))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(len(self.tickers) * 5 + 1,)
        )

    def _get_obs(self) -> np.ndarray:
        day = self.data[self.data["Date"] == self.dates[self.current_step]]
        features = day[["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        cash = 1.0 - self.weights.sum()
        return np.concatenate([features, [cash]]).astype(np.float32)

    @property
    def dates(self) -> np.ndarray:
        return self.data["Date"].unique()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.weights[:] = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)
        action /= action.sum() + 1e-8
        price_today = (
            self.data[self.data["Date"] == self.dates[self.current_step]]
            .set_index("Ticker")
            .loc[self.tickers, "Close"]
            .to_numpy()
        )
        if self.current_step + 1 < len(self.dates):
            price_next = (
                self.data[self.data["Date"] == self.dates[self.current_step + 1]]
                .set_index("Ticker")
                .loc[self.tickers, "Close"]
                .to_numpy()
            )
        else:
            price_next = price_today
        returns = price_next / price_today - 1
        # EVOLVE-START reward
        turnover = np.abs(action - self.weights).sum()
        reward = np.dot(returns, self.weights) - turnover * self.cost
        # EVOLVE-END reward
        self.weights = action
        self.current_step += 1
        terminated = self.current_step >= len(self.dates) - 1
        return self._get_obs(), reward, terminated, False, {}
