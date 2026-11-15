"""Standalone runnable demonstration of DIAMOND world dynamics, tabular Q-learning, and Deep RL."""

from __future__ import annotations

import json

from diamond.deep_rl import DeepQAgentLearner
from diamond.environment import ACTION_NAMES, DiamondWorld
from diamond.interfaces import RLLearnerProtocol, WorldEnvironmentProtocol, WorldModelProtocol
from diamond.rl import DynaWorldModel, QAgentLearner


def main() -> None:
    world = DiamondWorld(width=10, height=10, seed=42)
    q_learner = QAgentLearner(seed=42)
    dqn_learner = DeepQAgentLearner(seed=42)
    world_model = DynaWorldModel(seed=42)

    assert isinstance(world, WorldEnvironmentProtocol)
    assert isinstance(q_learner, RLLearnerProtocol)
    assert isinstance(dqn_learner, RLLearnerProtocol)
    assert isinstance(world_model, WorldModelProtocol)

    world.register_agent("tab_agent", energy=100.0)
    world.register_agent("dqn_agent", energy=100.0)

    tab_total = 0.0
    dqn_total = 0.0
    for _ in range(10):
        world.step_world_dynamics()
        obs_t = world.observe("tab_agent")
        act_t = q_learner.select_action(obs_t["state_key"], features=obs_t["features"])
        next_t, rew_t, done_t, _ = world.step_agent("tab_agent", act_t)
        q_learner.update(
            obs_t["state_key"],
            act_t,
            rew_t,
            next_t["state_key"],
            done=done_t,
            features=obs_t["features"],
            next_features=next_t["features"],
        )
        world_model.observe_transition(obs_t["state_key"], act_t, next_t["state_key"], rew_t)
        world_model.plan_updates(q_learner, planning_steps=2)
        tab_total += rew_t

        obs_d = world.observe("dqn_agent")
        act_d = dqn_learner.select_action(obs_d["state_key"], features=obs_d["features"])
        next_d, rew_d, done_d, _ = world.step_agent("dqn_agent", act_d)
        dqn_learner.update(
            obs_d["state_key"],
            act_d,
            rew_d,
            next_d["state_key"],
            done=done_d,
            features=obs_d["features"],
            next_features=next_d["features"],
        )
        dqn_total += rew_d

    print("=== DIAMOND WORLD & RL EXAMPLE ===")
    print(
        json.dumps(
            {
                "actions": len(ACTION_NAMES),
                "tabular_total_reward": round(tab_total, 4),
                "dqn_total_reward": round(dqn_total, 4),
                "world_resources": len(world.resources),
                "world_hazards": len(world.hazards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
