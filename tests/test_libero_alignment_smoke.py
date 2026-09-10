"""Optional one-task/one-episode LIBERO state-capture smoke test."""

import numpy as np
import pytest

pytest.importorskip("libero", reason="LIBERO is not installed")

from libero.libero import benchmark

from experiments.robot.libero.alignment_evaluator import (
    EvaluatorConfig,
    ReasoningClaims,
    evaluate_alignment,
)
from experiments.robot.libero.libero_utils import (
    capture_alignment_state,
    get_libero_dummy_action,
    get_libero_env,
)


def test_one_task_one_episode_state_capture():
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(0)
    env, _ = get_libero_env(task, "prismatic", resolution=64)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        before = capture_alignment_state(env, obs)
        obs, _, _, _ = env.step(get_libero_dummy_action("prismatic"))
        after = capture_alignment_state(env, obs)
        result = evaluate_alignment(
            ReasoningClaims(raw_text=""),
            np.zeros((1, 7)),
            before,
            after,
            config=EvaluatorConfig(),
        )
        assert before.ee_position.shape == (3,)
        assert after.ee_position.shape == (3,)
        assert before.gripper_openness is not None
        assert 0.0 <= before.gripper_openness <= 1.0
        assert len(before.gripper_joint_limits) == len(before.gripper_positions)
        top_drawer = before.objects["wooden_cabinet_1_top_region"]
        assert len(top_drawer.joint_positions) == 1
        assert top_drawer.open_state is not None
        assert top_drawer.closed_state is not None
        assert result.aggregate_score is None
    finally:
        env.close()
