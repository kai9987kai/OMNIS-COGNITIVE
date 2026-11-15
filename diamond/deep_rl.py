"""PyTorch Deep Q-Network (DQN) reinforcement learner with replay buffer and EWC consolidation."""

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """3-layer MLP mapping 8-D DIAMOND sensory features to 5 action Q-values."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Bounded experience replay buffer for Deep Q-learning."""

    def __init__(self, capacity: int = 1000, seed: int = 42) -> None:
        self.buffer: Deque[Tuple[List[float], int, float, List[float], bool]] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        features: List[float],
        action: int,
        reward: float,
        next_features: List[float],
        done: bool,
    ) -> None:
        self.buffer.append((list(features), int(action), float(reward), list(next_features), bool(done)))

    def sample(self, batch_size: int) -> List[Tuple[List[float], int, float, List[float], bool]]:
        k = min(batch_size, len(self.buffer))
        return self.rng.sample(list(self.buffer), k=k)

    def __len__(self) -> int:
        return len(self.buffer)


class DeepQAgentLearner:
    """PyTorch DQN learner matching QAgentLearner's interface with EWC continual regularization."""

    def __init__(
        self,
        input_dim: int = 8,
        num_actions: int = 5,
        learning_rate: float = 0.008,
        discount_gamma: float = 0.92,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.992,
        min_epsilon: float = 0.04,
        batch_size: int = 16,
        target_sync_interval: int = 16,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        torch.manual_seed(seed)
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.discount_gamma = float(discount_gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.batch_size = int(batch_size)
        self.target_sync_interval = int(target_sync_interval)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = DeepQNetwork(self.input_dim, 32, self.num_actions).to(self.device)
        self.target_net = DeepQNetwork(self.input_dim, 32, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=1500, seed=seed)

        # Compatibility attributes with QAgentLearner / Morpheus / Pipeline
        self.q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * self.num_actions)
        self.ewc_anchors: Dict[str, torch.Tensor] = {}
        self.ewc_importance: Dict[str, torch.Tensor] = {}
        self.cumulative_reward: float = 0.0
        self.reward_history: List[float] = []
        self.td_error_history: List[float] = []
        self.update_steps: int = 0

    def _to_tensor(self, features: Optional[List[float]]) -> torch.Tensor:
        vec = list(features or [0.0] * self.input_dim)
        if len(vec) < self.input_dim:
            vec = vec + [0.0] * (self.input_dim - len(vec))
        return torch.tensor(vec[: self.input_dim], dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_q_values(
        self,
        state_key: str,
        features: Optional[List[float]] = None,
    ) -> List[float]:
        """Evaluate online DQN Q-values from features or cached table."""
        if features is not None:
            with torch.no_grad():
                q_vals = self.online_net(self._to_tensor(features)).squeeze(0).cpu().tolist()
            self.q_table[state_key] = q_vals
            return list(q_vals)
        return list(self.q_table[state_key])

    def select_action(
        self,
        state_key: str,
        neural_logits: Optional[List[float]] = None,
        neural_weight: float = 0.35,
        strategy: str = "epsilon_greedy",
        features: Optional[List[float]] = None,
    ) -> int:
        if features is not None:
            with torch.no_grad():
                q_vals = self.online_net(self._to_tensor(features)).squeeze(0).cpu().tolist()
            self.q_table[state_key] = q_vals
        else:
            q_vals = self.q_table[state_key]

        if neural_logits and len(neural_logits) == self.num_actions:
            combined = [
                (1.0 - neural_weight) * q + neural_weight * float(n)
                for q, n in zip(q_vals, neural_logits)
            ]
        else:
            combined = list(q_vals)

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
        s_feat = list(features or [0.0] * self.input_dim)
        ns_feat = list(next_features or s_feat)

        with torch.no_grad():
            s_t = self._to_tensor(s_feat)
            ns_t = self._to_tensor(ns_feat)
            cur_q = float(self.online_net(s_t)[0, action].item())
            max_next_q = 0.0 if done else float(torch.max(self.target_net(ns_t)).item())
            td_error = float(reward) + self.discount_gamma * max_next_q - cur_q
            self.q_table[state_key] = self.online_net(s_t).squeeze(0).cpu().tolist()

        if features is not None:
            self.replay_buffer.push(s_feat, action, reward, ns_feat, done)

        if len(self.replay_buffer) >= min(8, self.batch_size):
            batch = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
            states_b = torch.tensor([b[0][: self.input_dim] for b in batch], dtype=torch.float32, device=self.device)
            actions_b = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
            next_states_b = torch.tensor([b[3][: self.input_dim] for b in batch], dtype=torch.float32, device=self.device)
            dones_b = torch.tensor([1.0 if b[4] else 0.0 for b in batch], dtype=torch.float32, device=self.device)

            q_pred = self.online_net(states_b).gather(1, actions_b).squeeze(1)
            with torch.no_grad():
                q_next = self.target_net(next_states_b).max(dim=1).values
                q_target = rewards_b + self.discount_gamma * (1.0 - dones_b) * q_next

            loss = F.smooth_l1_loss(q_pred, q_target)
            if ewc_lambda > 0.0 and self.ewc_anchors:
                ewc_penalty = torch.tensor(0.0, device=self.device)
                for name, param in self.online_net.named_parameters():
                    if name in self.ewc_anchors:
                        ewc_penalty = ewc_penalty + torch.sum(
                            self.ewc_importance[name] * (param - self.ewc_anchors[name]) ** 2
                        )
                loss = loss + float(ewc_lambda) * ewc_penalty

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 2.0)
            self.optimizer.step()

            self.update_steps += 1
            if self.update_steps % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

        if record_stats:
            self.cumulative_reward += float(reward)
            self.reward_history.append(float(reward))
            self.td_error_history.append(float(td_error))
            if len(self.reward_history) > 500:
                self.reward_history = self.reward_history[-500:]
                self.td_error_history = self.td_error_history[-500:]
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return float(td_error)

    def consolidate_task(self) -> None:
        """Anchor DeepQNetwork weights and accumulate Fisher importance proxy."""
        for name, param in self.online_net.named_parameters():
            self.ewc_anchors[name] = param.detach().clone()
            grad_sq = (param.grad.detach() ** 2) if param.grad is not None else torch.ones_like(param) * 0.1
            prev = self.ewc_importance.get(name, torch.zeros_like(param))
            self.ewc_importance[name] = torch.clamp(prev + grad_sq + 0.05, max=10.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_type": "deep_q_network",
            "input_dim": self.input_dim,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_gamma": self.discount_gamma,
            "epsilon": self.epsilon,
            "cumulative_reward": self.cumulative_reward,
            "state_dict": {k: v.cpu().tolist() for k, v in self.online_net.state_dict().items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], seed: int = 42) -> "DeepQAgentLearner":
        learner = cls(
            input_dim=data.get("input_dim", 8),
            num_actions=data.get("num_actions", 5),
            learning_rate=data.get("learning_rate", 0.008),
            discount_gamma=data.get("discount_gamma", 0.92),
            epsilon=data.get("epsilon", 0.2),
            seed=seed,
        )
        learner.cumulative_reward = float(data.get("cumulative_reward", 0.0))
        if "state_dict" in data:
            sd = {k: torch.tensor(v, dtype=torch.float32, device=learner.device) for k, v in data["state_dict"].items()}
            learner.online_net.load_state_dict(sd)
            learner.target_net.load_state_dict(sd)
        return learner
