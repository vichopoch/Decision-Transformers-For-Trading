"""Train Elastic Decision Transformer."""

from __future__ import annotations

import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ..data.download_data import download_ohlcv
from ..envs.ipsa_env import IpsaTradingEnv
from ..features.feature_pipeline import add_indicators
from ..models.edt import ElasticDecisionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run quick test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_indicators(download_ohlcv())
    env = IpsaTradingEnv(df)
    obs, _ = env.reset()
    state_dim = obs.size
    act_dim = env.action_space.shape[0]
    model = ElasticDecisionTransformer(state_dim, act_dim)

    if args.verify:
        data = torch.zeros(1, 1, state_dim)
        acts = torch.zeros(1, 1, act_dim)
        out = model(data, acts)
        print("output", out.shape)
        return

    dataset = TensorDataset(torch.zeros(100, 1, state_dim), torch.zeros(100, 1, act_dim))
    loader = DataLoader(dataset, batch_size=16)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1):
        for states, acts in loader:
            pred = model(states, acts)
            loss = pred.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    main()
