"""
IK Solver Environment for Reinforcement Learning

A 20-joint humanoid character IK solver using PufferLib Ocean framework.
The agent learns to find joint angles that achieve end-effector targets
while following a reference pose.

State:
    - Joint angles (20 values, normalized to [-1, 1])
    - Joint velocities (20 values, normalized)
    - End-effector position errors in root frame (4 end-effectors x 3 = 12 values)
    - End-effector priorities (4 values)
    - Center of mass error in root frame (3 values)
    - CoM priority (1 value)
    - Reference pose error (20 values, normalized)
    - Reference pose priority (1 value)
    Total: 81 observations

Action:
    - Delta joint angles (20 values in [-1, 1], scaled by action_scale)

Reward:
    - Improvement in end-effector position error (weighted by priority)
    - Improvement in center of mass error (weighted by priority)
    - Improvement in pose error (weighted by priority)
    - Smoothness penalty for large actions
    - Bonus for achieving target (total error < threshold)
"""

import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.ik_solver import binding

NUM_JOINTS = 20
NUM_END_EFFECTORS = 4
OBS_SIZE = 81


class IKSolver(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        report_interval=1,
        max_steps=64,
        reward_ee_weight=1.0,
        reward_com_weight=0.5,
        reward_pose_weight=0.1,
        reward_smoothness_weight=0.01,
        action_scale=0.1,
        buf=None,
        seed=0,
    ):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        super().__init__(buf)
        self.actions = np.zeros((num_envs, NUM_JOINTS), dtype=np.float32)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_steps=max_steps,
            reward_ee_weight=reward_ee_weight,
            reward_com_weight=reward_com_weight,
            reward_pose_weight=reward_pose_weight,
            reward_smoothness_weight=reward_smoothness_weight,
            action_scale=action_scale,
        )

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = np.clip(actions.reshape(-1, NUM_JOINTS), -1.0, 1.0)
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_state(self, env_idx=0):
        """Get the current state of the environment for visualization.
        
        Returns a dict containing:
            - joint_positions: (NUM_JOINTS * 3,) world positions of all joints
            - joint_rotations: (NUM_JOINTS * 4,) world rotations (quaternions) of all joints
            - ee_positions: (NUM_END_EFFECTORS * 3,) current end-effector positions
            - target_ee_positions: (NUM_END_EFFECTORS * 3,) target end-effector positions
            - com_position: (3,) current center of mass position
            - target_com_position: (3,) target center of mass position
            - joint_parents: (NUM_JOINTS,) parent index for each joint (-1 for root)
            - ee_joint_indices: (NUM_END_EFFECTORS,) joint index for each end-effector
            - num_joints: int
            - num_end_effectors: int
        """
        return binding.vec_get(self.c_envs, env_idx)

    def set_targets(
        self,
        env_idx=0,
        target_ee_positions=None,
        target_ee_priorities=None,
        target_com=None,
        target_com_priority=None,
        reference_pose=None,
        reference_pose_priority=None,
    ):
        """Set targets for the IK solver.
        
        Args:
            env_idx: Index of the environment to set targets for
            target_ee_positions: (NUM_END_EFFECTORS * 3,) target positions for end-effectors
            target_ee_priorities: (NUM_END_EFFECTORS,) priorities for each end-effector
            target_com: (3,) target center of mass position
            target_com_priority: float, priority for center of mass target
            reference_pose: (NUM_JOINTS,) reference joint angles
            reference_pose_priority: float, priority for reference pose
        """
        kwargs = {}
        if target_ee_positions is not None:
            kwargs['target_ee_positions'] = np.asarray(target_ee_positions, dtype=np.float32)
        if target_ee_priorities is not None:
            kwargs['target_ee_priorities'] = np.asarray(target_ee_priorities, dtype=np.float32)
        if target_com is not None:
            kwargs['target_com'] = np.asarray(target_com, dtype=np.float32)
        if target_com_priority is not None:
            kwargs['target_com_priority'] = float(target_com_priority)
        if reference_pose is not None:
            kwargs['reference_pose'] = np.asarray(reference_pose, dtype=np.float32)
        if reference_pose_priority is not None:
            kwargs['reference_pose_priority'] = float(reference_pose_priority)
        
        binding.vec_put(self.c_envs, env_idx, **kwargs)


def test_performance(timeout=10, atn_cache=1024):
    """Benchmark environment performance."""
    num_envs = 1024
    env = IKSolver(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.uniform(-1, 1, (atn_cache, num_envs, NUM_JOINTS)).astype(
        np.float32
    )

    import time

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f"SPS: {sps:,}")


if __name__ == "__main__":
    test_performance()
