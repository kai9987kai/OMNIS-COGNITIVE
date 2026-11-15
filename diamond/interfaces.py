"""Formal typed interface protocols for the DIAMOND world simulation and RL subsystem."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class WorldEnvironmentProtocol(Protocol):
    """Contract for persistent discrete/continuous simulation environments."""

    width: int
    height: int
    task_id: int

    def register_agent(self, agent_id: str, energy: float = 100.0) -> Tuple[int, int]: ...
    def observe(self, agent_id: str) -> Dict[str, Any]: ...
    def step_agent(self, agent_id: str, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: ...
    def step_world_dynamics(self) -> None: ...
    def set_task(self, task_id: int) -> None: ...
    def save_state(self) -> Dict[str, Any]: ...
    def load_state(self, state: Dict[str, Any]) -> None: ...
    def get_observatory_view(self) -> Dict[str, Any]: ...


@runtime_checkable
class RLLearnerProtocol(Protocol):
    """Contract shared by tabular Q-learning and PyTorch Deep Q-learning agents."""

    def get_q_values(self, state_key: str, features: Optional[List[float]] = None) -> List[float]: ...
    def select_action(
        self,
        state_key: str,
        neural_logits: Optional[List[float]] = None,
        neural_weight: float = 0.35,
        strategy: str = "epsilon_greedy",
        features: Optional[List[float]] = None,
    ) -> int: ...
    def update(
        self,
        state_key: str,
        action: int,
        reward: float,
        next_state_key: str,
        done: bool = False,
        ewc_lambda: float = 0.0,
        record_stats: bool = True,
        features: Optional[List[float]] = None,
        next_features: Optional[List[float]] = None,
    ) -> float: ...
    def consolidate_task(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class WorldModelProtocol(Protocol):
    """Contract for Dyna-style internal transition models used for model-based planning."""

    def observe_transition(self, state_key: str, action: int, next_state_key: str, reward: float) -> float: ...
    def plan_updates(self, learner: RLLearnerProtocol, planning_steps: int = 4) -> int: ...
