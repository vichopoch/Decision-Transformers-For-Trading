"""AlphaEvolve mutation loop using o3-high."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

import openai

from ..data.download_data import download_ohlcv, DEFAULT_START
from ..envs.ipsa_env import IpsaTradingEnv
from ..features.feature_pipeline import add_indicators


def _run_patch(cmd: List[str], patch: str, cwd: Path) -> None:
    proc = subprocess.run(cmd, input=patch.encode(), cwd=cwd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def apply_patch(patch: str, repo: Path) -> None:
    """Apply a unified diff patch to ``repo``."""
    _run_patch(["patch", "-p1"], patch, repo)


def find_evolve_blocks(repo: Path) -> str:
    """Collect code between EVOLVE markers."""
    snippets: List[str] = []
    for path in repo.rglob("*.py"):
        lines = path.read_text().splitlines()
        capture = False
        buffer: List[str] = []
        for line in lines:
            if "EVOLVE-START" in line:
                capture = True
                continue
            if "EVOLVE-END" in line:
                capture = False
                snippets.append("\n".join(buffer))
                buffer = []
                continue
            if capture:
                buffer.append(line)
    return "\n\n".join(snippets)


def generate_patch(context: str) -> str:
    """Call the ``o3-high`` model to generate a diff."""
    messages = [
        {
            "role": "system",
            "content": "You improve a trading system by providing unified diffs only.",
        },
        {
            "role": "user",
            "content": f"Improve the following code blocks:\n{context}\nProvide a unified diff.",
        },
    ]
    resp = openai.ChatCompletion.create(model="o3-high", messages=messages)
    return resp.choices[0].message["content"]


def evaluate_repo(repo: Path) -> dict:
    """Simple evaluation returning cumulative reward."""
    df = add_indicators(download_ohlcv(tickers=["CHILE"], start=DEFAULT_START))
    env = IpsaTradingEnv(df)
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total += reward
    return {"reward": float(total)}


def run_alpha_loop(n: int, repo: Path = Path(".")) -> dict:
    """Run ``n`` mutation cycles."""
    repo = repo.resolve()
    artifacts = repo / "artifacts"
    artifacts.mkdir(exist_ok=True)
    best: dict | None = None
    best_patch = ""
    for _ in range(n):
        context = find_evolve_blocks(repo)
        patch = generate_patch(context)
        apply_patch(patch, repo)
        metrics = evaluate_repo(repo)
        if best is None or metrics["reward"] > best["reward"]:
            best = metrics
            best_patch = patch
    champion = {"patch": best_patch, "metrics": best}
    (artifacts / "champion.json").write_text(json.dumps(champion, indent=2))
    return best
