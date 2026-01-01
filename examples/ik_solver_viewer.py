"""
IK Solver 3D Viewer

A real-time 3D visualization tool for the IK Solver environment.
Supports viewing the humanoid skeleton, end-effector targets, and center of mass.
Can be used with trained policies for inference or manual target control.

Controls:
    Mouse: Rotate camera (drag)
    Scroll: Zoom in/out
    WASD: Move camera
    Space: Reset camera
    1-4: Select end-effector to move (1=left hand, 2=right hand, 3=left foot, 4=right foot)
    Arrow keys: Move selected end-effector target
    R: Reset targets to random
    P: Toggle policy inference (if model loaded)
    ESC: Exit

Usage:
    python examples/ik_solver_viewer.py
    python examples/ik_solver_viewer.py --model path/to/checkpoint.pt
"""

import argparse
import numpy as np

try:
    from raylib import rl, colors
    import pyray
    RAYLIB_AVAILABLE = True
except ImportError:
    RAYLIB_AVAILABLE = False
    print("Warning: raylib not available. Install with: pip install raylib")

from pufferlib.ocean.ik_solver.ik_solver import IKSolver, NUM_JOINTS, NUM_END_EFFECTORS


JOINT_NAMES = [
    "Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe"
]

EE_NAMES = ["Left Hand", "Right Hand", "Left Foot", "Right Foot"]
EE_COLORS = [
    (255, 100, 100, 255),
    (100, 255, 100, 255),
    (100, 100, 255, 255),
    (255, 255, 100, 255),
]


class IKSolverViewer:
    def __init__(self, width=1280, height=720, model_path=None):
        self.width = width
        self.height = height
        self.model_path = model_path
        self.policy = None
        self.use_policy = False
        
        self.env = IKSolver(num_envs=1)
        self.env.reset()
        
        self.selected_ee = 0
        self.target_move_speed = 0.02
        
        self.camera_distance = 3.0
        self.camera_yaw = 0.0
        self.camera_pitch = 0.3
        self.camera_target = [0.0, 1.0, 0.0]
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        try:
            import torch
            from pufferlib.ocean.torch import IKSolverPolicy
            
            self.policy = IKSolverPolicy(self.env, hidden_size=256)
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.policy.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.policy.load_state_dict(checkpoint)
            self.policy.eval()
            self.use_policy = True
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.policy = None
            self.use_policy = False
    
    def _get_camera_position(self):
        x = self.camera_target[0] + self.camera_distance * np.cos(self.camera_pitch) * np.sin(self.camera_yaw)
        y = self.camera_target[1] + self.camera_distance * np.sin(self.camera_pitch)
        z = self.camera_target[2] + self.camera_distance * np.cos(self.camera_pitch) * np.cos(self.camera_yaw)
        return (x, y, z)
    
    def _draw_skeleton(self, joint_positions, joint_parents):
        positions = joint_positions.reshape(-1, 3)
        
        for i in range(NUM_JOINTS):
            parent = joint_parents[i]
            pos = positions[i]
            
            joint_color = colors.WHITE
            if i in [8, 11, 14, 18]:
                ee_idx = [8, 11, 14, 18].index(i)
                joint_color = EE_COLORS[ee_idx]
            
            rl.DrawSphere((pos[0], pos[1], pos[2]), 0.02, joint_color)
            
            if parent >= 0:
                parent_pos = positions[parent]
                rl.DrawLine3D(
                    (parent_pos[0], parent_pos[1], parent_pos[2]),
                    (pos[0], pos[1], pos[2]),
                    colors.LIGHTGRAY
                )
    
    def _draw_targets(self, target_ee_positions, ee_positions):
        targets = target_ee_positions.reshape(-1, 3)
        current = ee_positions.reshape(-1, 3)
        
        for i in range(NUM_END_EFFECTORS):
            target = targets[i]
            curr = current[i]
            color = EE_COLORS[i]
            
            if i == self.selected_ee:
                rl.DrawSphereWires((target[0], target[1], target[2]), 0.05, 8, 8, color)
            else:
                rl.DrawSphere((target[0], target[1], target[2]), 0.03, color)
            
            rl.DrawLine3D(
                (curr[0], curr[1], curr[2]),
                (target[0], target[1], target[2]),
                (color[0], color[1], color[2], 128)
            )
    
    def _draw_com(self, com_position, target_com_position):
        rl.DrawSphere((com_position[0], com_position[1], com_position[2]), 0.04, colors.ORANGE)
        rl.DrawSphereWires((target_com_position[0], target_com_position[1], target_com_position[2]), 0.04, 8, 8, colors.YELLOW)
    
    def _draw_ground(self):
        rl.DrawGrid(10, 0.5)
    
    def _draw_ui(self, state):
        rl.DrawText(b"IK Solver Viewer", 10, 10, 20, colors.WHITE)
        rl.DrawText(f"Selected EE: {EE_NAMES[self.selected_ee]} (1-4 to change)".encode(), 10, 35, 16, EE_COLORS[self.selected_ee])
        rl.DrawText(b"Arrow keys: Move target | R: Random targets", 10, 55, 16, colors.LIGHTGRAY)
        
        if self.policy:
            status = "ON" if self.use_policy else "OFF"
            rl.DrawText(f"Policy: {status} (P to toggle)".encode(), 10, 75, 16, colors.GREEN if self.use_policy else colors.RED)
        
        ee_error = np.linalg.norm(state['ee_positions'] - state['target_ee_positions'])
        com_error = np.linalg.norm(state['com_position'] - state['target_com_position'])
        rl.DrawText(f"EE Error: {ee_error:.4f}".encode(), 10, self.height - 50, 16, colors.WHITE)
        rl.DrawText(f"CoM Error: {com_error:.4f}".encode(), 10, self.height - 30, 16, colors.WHITE)
    
    def _handle_input(self, state):
        if rl.IsKeyPressed(rl.KEY_ESCAPE):
            return False
        
        if rl.IsKeyPressed(rl.KEY_ONE):
            self.selected_ee = 0
        elif rl.IsKeyPressed(rl.KEY_TWO):
            self.selected_ee = 1
        elif rl.IsKeyPressed(rl.KEY_THREE):
            self.selected_ee = 2
        elif rl.IsKeyPressed(rl.KEY_FOUR):
            self.selected_ee = 3
        
        target_ee = state['target_ee_positions'].reshape(-1, 3).copy()
        moved = False
        
        if rl.IsKeyDown(rl.KEY_UP):
            target_ee[self.selected_ee, 1] += self.target_move_speed
            moved = True
        if rl.IsKeyDown(rl.KEY_DOWN):
            target_ee[self.selected_ee, 1] -= self.target_move_speed
            moved = True
        if rl.IsKeyDown(rl.KEY_LEFT):
            target_ee[self.selected_ee, 0] -= self.target_move_speed
            moved = True
        if rl.IsKeyDown(rl.KEY_RIGHT):
            target_ee[self.selected_ee, 0] += self.target_move_speed
            moved = True
        
        if moved:
            self.env.set_targets(env_idx=0, target_ee_positions=target_ee.flatten())
        
        if rl.IsKeyPressed(rl.KEY_R):
            self.env.reset()
        
        if rl.IsKeyPressed(rl.KEY_P) and self.policy:
            self.use_policy = not self.use_policy
        
        if rl.IsKeyPressed(rl.KEY_SPACE):
            self.camera_distance = 3.0
            self.camera_yaw = 0.0
            self.camera_pitch = 0.3
            self.camera_target = [0.0, 1.0, 0.0]
        
        wheel = rl.GetMouseWheelMove()
        if wheel != 0:
            self.camera_distance = max(1.0, min(10.0, self.camera_distance - wheel * 0.3))
        
        if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_LEFT):
            delta = rl.GetMouseDelta()
            self.camera_yaw += delta.x * 0.005
            self.camera_pitch = max(-1.5, min(1.5, self.camera_pitch + delta.y * 0.005))
        
        if rl.IsKeyDown(rl.KEY_W):
            self.camera_target[2] -= 0.05
        if rl.IsKeyDown(rl.KEY_S):
            self.camera_target[2] += 0.05
        if rl.IsKeyDown(rl.KEY_A):
            self.camera_target[0] -= 0.05
        if rl.IsKeyDown(rl.KEY_D):
            self.camera_target[0] += 0.05
        
        return True
    
    def _step_policy(self):
        if self.policy and self.use_policy:
            import torch
            obs = torch.from_numpy(self.env.observations).float()
            with torch.no_grad():
                actions, _ = self.policy(obs)
                if hasattr(actions, 'mean'):
                    action = actions.mean.numpy()
                else:
                    action = actions.numpy()
            self.env.step(action)
        else:
            action = np.zeros((1, NUM_JOINTS), dtype=np.float32)
            self.env.step(action)
    
    def run(self):
        if not RAYLIB_AVAILABLE:
            print("Cannot run viewer: raylib not available")
            return
        
        rl.InitWindow(self.width, self.height, b"IK Solver 3D Viewer")
        rl.SetTargetFPS(60)
        rl.DisableCursor()
        
        camera = pyray.Camera3D()
        camera.up = (0.0, 1.0, 0.0)
        camera.fovy = 45.0
        camera.projection = pyray.CAMERA_PERSPECTIVE
        
        while not rl.WindowShouldClose():
            state = self.env.get_state(env_idx=0)
            
            if not self._handle_input(state):
                break
            
            self._step_policy()
            state = self.env.get_state(env_idx=0)
            
            cam_pos = self._get_camera_position()
            camera.position = cam_pos
            camera.target = tuple(self.camera_target)
            
            rl.BeginDrawing()
            rl.ClearBackground((20, 20, 30, 255))
            
            rl.BeginMode3D(camera)
            self._draw_ground()
            self._draw_skeleton(state['joint_positions'], state['joint_parents'])
            self._draw_targets(state['target_ee_positions'], state['ee_positions'])
            self._draw_com(state['com_position'], state['target_com_position'])
            rl.EndMode3D()
            
            self._draw_ui(state)
            
            rl.EndDrawing()
        
        rl.EnableCursor()
        rl.CloseWindow()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="IK Solver 3D Viewer")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    args = parser.parse_args()
    
    viewer = IKSolverViewer(
        width=args.width,
        height=args.height,
        model_path=args.model
    )
    viewer.run()


if __name__ == "__main__":
    main()
