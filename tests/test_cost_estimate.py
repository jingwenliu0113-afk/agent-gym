"""experiments/cost_estimate.py 的契約測試（decision 012）。"""

from __future__ import annotations

import math

import pytest

from agents.baseline import PRICING
from env.hashing import canonical_json
from experiments.cost_estimate import EpisodeEstimate, estimate_episode, request_tokens, sweep_upper_bound

P = PRICING["claude-opus-5-5"]


def _request(chars: int) -> dict:
    """canonical_json({"messages", "system", "tools"}) 剛好 chars 個字元的 request（另含不計入的 key）。"""
    base = {"system": [], "tools": [], "messages": [""]}
    overhead = len(canonical_json(base))
    return {"model": "claude-opus-5-5", "max_tokens": 1, "system": [], "tools": [], "messages": ["x" * (chars - overhead)]}


def test_request_tokens_counts_only_prompt_parts() -> None:
    request = _request(700)
    assert request_tokens(request, 3.5) == 200
    assert request_tokens(request, 7.0) == 100
    assert request_tokens(_request(701), 3.5) == 201  # 無條件進位
    extra = dict(request, betas=["x" * 1000])
    assert request_tokens(extra, 3.5) == 200


def test_estimate_episode_arithmetic() -> None:
    requests = [_request(350), _request(700)]  # 100、200 tokens
    estimate = estimate_episode("t", requests, output_tokens=10, chars_per_token=3.5)
    assert isinstance(estimate, EpisodeEstimate)
    assert (estimate.task_id, estimate.calls) == ("t", 2)
    inputs = [100 + 0 * 10, 200 + 1 * 10]
    assert estimate.input_tokens == sum(inputs)
    assert estimate.output_tokens == 20
    expected = [(i * P["input"] + 10 * P["output"]) / 1e6 for i in inputs]
    assert estimate.call_costs == pytest.approx(tuple(expected))
    assert estimate.cost_usd == pytest.approx(sum(expected))


def test_turn_factor_extrapolates_with_last_request() -> None:
    requests = [_request(350), _request(700)]
    estimate = estimate_episode("t", requests, output_tokens=10, turn_factor=2.0, chars_per_token=3.5)
    assert estimate.calls == 4
    assert estimate.input_tokens == (100 + 0) + (200 + 10) + (200 + 20) + (200 + 30)
    assert estimate_episode("t", requests, output_tokens=10, turn_factor=1.25, chars_per_token=3.5).calls == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"turn_factor": 0.5}, {"turn_factor": math.inf}, {"output_tokens": -1}, {"output_tokens": 1.5}, {"model": "unknown-model"}],
    ids=repr,
)
def test_estimate_episode_rejects_bad_arguments(kwargs: dict) -> None:
    args = {"output_tokens": 10, **kwargs}
    with pytest.raises(ValueError):
        estimate_episode("t", [_request(350)], **args)


def test_estimate_episode_requires_requests() -> None:
    with pytest.raises(ValueError):
        estimate_episode("t", [], output_tokens=10)


def test_sweep_upper_bound() -> None:
    estimate = EpisodeEstimate("t", 3, 0, 0, 6.0, (1.0, 2.0, 3.0))
    # suffix sums: 6, 5, 3 → mean 14/3；× L=3 × n_fine
    assert sweep_upper_bound(estimate, 10) == pytest.approx(10 * 14.0)
    with pytest.raises(ValueError):
        sweep_upper_bound(estimate, 0)
