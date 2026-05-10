from __future__ import annotations

import synnetqos.generator as generator


def test_latency_preserves_network_order(monkeypatch) -> None:
    monkeypatch.setattr(generator.np.random, "normal", lambda *args, **kwargs: 0.0)

    signal = -90.0

    latency_4g = generator.generate_latency(signal, "Low", False, "4G", "Static")
    latency_5g_nsa = generator.generate_latency(signal, "Low", False, "5G NSA", "Static")
    latency_5g_sa = generator.generate_latency(signal, "Low", False, "5G SA", "Static")

    assert latency_5g_sa < latency_5g_nsa < latency_4g


def test_latency_preserves_congestion_order(monkeypatch) -> None:
    monkeypatch.setattr(generator.np.random, "normal", lambda *args, **kwargs: 0.0)

    signal = -90.0

    low = generator.generate_latency(signal, "Low", False, "5G SA", "Static")
    medium = generator.generate_latency(signal, "Medium", False, "5G SA", "Static")
    high = generator.generate_latency(signal, "High", False, "5G SA", "Static")

    assert low < medium < high


def test_latency_preserves_mobility_order(monkeypatch) -> None:
    monkeypatch.setattr(generator.np.random, "normal", lambda *args, **kwargs: 0.0)

    signal = -90.0

    static = generator.generate_latency(signal, "Low", False, "5G NSA", "Static")
    walking = generator.generate_latency(signal, "Low", False, "5G NSA", "Walking")
    driving = generator.generate_latency(signal, "Low", False, "5G NSA", "Driving")

    assert static < walking < driving


def test_jitter_preserves_congestion_and_mobility_order(monkeypatch) -> None:
    def deterministic_lognormal(mean, sigma):
        return float(generator.np.exp(mean))

    monkeypatch.setattr(generator.np.random, "lognormal", deterministic_lognormal)

    latency = 80.0

    low_static = generator.generate_jitter(latency, "Static", "Low")
    medium_static = generator.generate_jitter(latency, "Static", "Medium")
    medium_driving = generator.generate_jitter(latency, "Driving", "Medium")

    assert low_static < medium_static < medium_driving