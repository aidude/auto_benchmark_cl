from src.systems.sage.neuromod import NeuromodState
from src.systems.sage.lyapunov import ForgettingMonitor


def test_neuromod_defaults():
    state = NeuromodState()
    assert state.da == 0.0
    assert state.ne == 0.0


def test_forgetting_monitor_no_alarm_on_stable():
    monitor = ForgettingMonitor(window=5, threshold=0.5)
    for _ in range(5):
        monitor.record(1.0)
    assert not monitor.is_forgetting


def test_forgetting_monitor_detects_variance():
    monitor = ForgettingMonitor(window=4, threshold=0.1)
    for v in [1.0, 0.0, 1.0, 0.0]:
        monitor.record(v)
    assert monitor.is_forgetting
