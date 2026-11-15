"""Reinforcement learning (Q-learning, TD error, Dyna world model, continual regularization) for DIAMOND."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class QAgentLearner:
    """Tabular Q-learning agent with exact TD error, exploration policies, and continual regularization."""

    def __init__(
        self,
        num_actions: int = 5,
        learning_rate: float = 0.15,
        discount_gamma: float = 0.92,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.04,
        seed: int = 42,
    ) -> None:
        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.discount_gamma = float(discount_gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.rng = random.Random(seed)

        self.q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * self.num_actions)
        self.ewc_anchors: Dict[str, List[float]] = {}
        self.ewc_importance: Dict[str, List[float]] = {}
        self.cumulative_reward: float = 0.0
        self.reward_history: List[float] = []
        self.td_error_history: List[float] = []

    def get_q_values(
        self,
        state_key: str,
        features: Optional[List[float]] = None,
    ) -> List[float]:
        """Return Q-values for a state key."""
        return list(self.q_table[state_key])

    def select_action(
        self,
        state_key: str,
        neural_logits: Optional[List[float]] = None,
        neural_weight: float = 0.35,
        strategy: str = "epsilon_greedy",
        features: Optional[List[float]] = None,
    ) -> int:
        """Select action combining Q-values and optional SONAPSE output logits."""
        q_vals = self.q_table[state_key]
        if neural_logits and len(neural_logits) == self.num_actions:
            combined = [
                (1.0 - neural_weight) * q + neural_weight * float(n)
                for q, n in zip(q_vals, neural_logits)
            ]
        else:
            combined = list(q_vals)

        if strategy == "boltzmann":
            temp = max(0.15, self.epsilon * 2.0)
            max_c = max(combined)
            exps = [math.exp(max(-10.0, min(10.0, (c - max_c) / temp))) for c in combined]
            s = sum(exps) or 1.0
            probs = [e / s for e in exps]
            return self.rng.choices(range(self.num_actions), weights=probs, k=1)[0]

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)

        max_val = max(combined)
        best_actions = [idx for idx, val in enumerate(combined) if abs(val - max_val) < 1e-9]
        return self.rng.choice(best_actions)

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
    ) -> float:
        """Compute TD error delta_t and apply Q(s,a) <- Q(s,a) + eta * delta_t."""
        current_q = self.q_table[state_key][action]
        max_next_q = 0.0 if done else max(self.q_table[next_state_key])
        delta_t = float(reward) + self.discount_gamma * max_next_q - current_q

        update_step = self.learning_rate * delta_t
        if ewc_lambda > 0.0 and state_key in self.ewc_anchors:
            anchor = self.ewc_anchors[state_key][action]
            imp = self.ewc_importance[state_key][action]
            update_step -= self.learning_rate * ewc_lambda * imp * (current_q - anchor)

        self.q_table[state_key][action] = current_q + update_step

        if record_stats:
            self.cumulative_reward += float(reward)
            self.reward_history.append(float(reward))
            self.td_error_history.append(float(delta_t))
            if len(self.reward_history) > 500:
                self.reward_history = self.reward_history[-500:]
                self.td_error_history = self.td_error_history[-500:]
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return float(delta_t)

    def consolidate_task(self) -> None:
        """Anchor Q-table parameters for continual learning retention across sequential tasks."""
        for s_key, q_vals in self.q_table.items():
            self.ewc_anchors[s_key] = list(q_vals)
            prev_imp = self.ewc_importance.get(s_key, [0.0] * self.num_actions)
            self.ewc_importance[s_key] = [
                min(5.0, p + 0.5 + 0.2 * abs(q)) for p, q in zip(prev_imp, q_vals)
            ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_gamma": self.discount_gamma,
            "epsilon": self.epsilon,
            "cumulative_reward": self.cumulative_reward,
            "q_table": {k: list(v) for k, v in self.q_table.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], seed: int = 42) -> "QAgentLearner":
        learner = cls(
            num_actions=data.get("num_actions", 5),
            learning_rate=data.get("learning_rate", 0.15),
            discount_gamma=data.get("discount_gamma", 0.92),
            epsilon=data.get("epsilon", 0.2),
            seed=seed,
        )
        learner.cumulative_reward = float(data.get("cumulative_reward", 0.0))
        for k, v in data.get("q_table", {}).items():
            learner.q_table[k] = list(v)
        return learner


class DynaWorldModel:
    """Tabular transition & reward model supporting model-based planning rollouts."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.transitions: Dict[Tuple[str, int], Tuple[str, float]] = {}
        self.prediction_errors: List[float] = []

    def observe_transition(
        self,
        state_key: str,
        action: int,
        next_state_key: str,
        reward: float,
    ) -> float:
        key = (state_key, int(action))
        pred_err = 0.0
        if key in self.transitions:
            _, prev_r = self.transitions[key]
            pred_err = abs(float(reward) - prev_r)
        else:
            pred_err = abs(float(reward))
        self.transitions[key] = (next_state_key, float(reward))
        self.prediction_errors.append(pred_err)
        if len(self.prediction_errors) > 300:
            self.prediction_errors = self.prediction_errors[-300:]
        return pred_err

    def plan_updates(self, learner: QAgentLearner, planning_steps: int = 4) -> int:
        if not self.transitions:
            return 0
        keys = list(self.transitions.keys())
        updates = 0
        for _ in range(min(planning_steps, len(keys))):
            s_key, act = self.rng.choice(keys)
            ns_key, rew = self.transitions[(s_key, act)]
            learner.update(s_key, act, rew, ns_key, done=False, record_stats=False)
            updates += 1
        return updates

    def mean_prediction_error(self) -> float:
        if not self.prediction_errors:
            return 0.0
        return float(sum(self.prediction_errors) / len(self.prediction_errors))
