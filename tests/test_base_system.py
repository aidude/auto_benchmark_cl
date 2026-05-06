import pytest
from src.systems.base import BaseSystem


class ConcreteSystem(BaseSystem):
    def act(self, observation):
        return "action"

    def update(self, observation, action, reward):
        pass

    def reset(self):
        pass


def test_name():
    s = ConcreteSystem()
    assert s.name == "ConcreteSystem"


def test_act_returns_value():
    s = ConcreteSystem()
    assert s.act({"text": "hello"}) == "action"


def test_reset_callable():
    s = ConcreteSystem()
    s.reset()  # should not raise
