# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is the **BeyondMimic Motion Tracking Inference** repository - a C++ implementation of motion tracking policies for legged robots using ONNX runtime. The system is built on the `legged_control2` framework and implements a custom ROS 2 controller that tracks reference motions from trained neural network policies.

**Key Papers & Resources:**
- Website: https://beyondmimic.github.io/
- Paper: https://arxiv.org/abs/2508.08241
- Training code: https://github.com/HybridRobotics/whole_body_tracking

## Build Commands

This is a ROS 2 Humble package. Always ensure you're in a workspace root (e.g., `~/colcon_ws`) before building.

### Initial Setup
```bash
# Install dependencies (run from workspace root)
rosdep install --from-paths src --ignore-src -r -y

# Build the package with debug info
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-up-to motion_tracking_controller

# Source the workspace
source install/setup.bash
```

### Development Build
```bash
# Quick rebuild after code changes (from workspace root)
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-select motion_tracking_controller
source install/setup.bash
```

## Running the System

### Simulation (MuJoCo)

**Load policy from WandB:**
```bash
ros2 launch motion_tracking_controller mujoco.launch.py wandb_path:=<your_wandb_run_path>
```

**Load local ONNX file:**
```bash
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=<absolute_or_tilde_path>/policy.onnx
```

**Additional launch arguments:**
- `robot_type:=g1` (default)
- `start_step:=<int>` - Starting timestep for motion playback (default: 0)
- `ext_pos_corr:=true/false` - Enable external position correction via `/mid360` topic

### Real Robot

**Requirements:**
1. Connect via ethernet cable
2. Set ethernet adapter to static IP: `192.168.123.11`
3. Find network interface with `ifconfig` (e.g., `eth0`, `enp3s0`)

**Launch command:**
```bash
ros2 launch motion_tracking_controller real.launch.py network_interface:=<interface> wandb_path:=<wandb_run_path>
# OR
ros2 launch motion_tracking_controller real.launch.py network_interface:=<interface> policy_path:=<path>/policy.onnx
```

**Joystick Controls (Unitree remote):**
- Standby controller (joint position): `L1 + A`
- Motion tracking controller (policy): `R1 + A`
- E-stop (damping): `B`

**Note:** Real robot launch automatically records rosbag with topic filtering (excludes Unitree internal topics).

## Architecture

### Core Components

The system follows the `legged_control2` controller pattern with three main C++ classes:

1. **MotionTrackingController** (`MotionTrackingController.h/cpp`)
   - Inherits from `RlController` (legged_control2)
   - Manages the RL environment observation loop
   - Registers custom command and observation terms
   - Lifecycle: `on_init()` → `on_configure()` → `on_activate()`

2. **MotionOnnxPolicy** (`MotionOnnxPolicy.h/cpp`)
   - Inherits from `OnnxPolicy` (legged_control2)
   - Wraps ONNX neural network inference
   - Extracts reference motion from network outputs:
     - Joint positions/velocities
     - Body positions/orientations (multiple bodies)
   - Parses metadata from ONNX file (anchor body, body names, impedance gains)
   - Maintains timestep counter for temporal policies

3. **MotionCommand** (`MotionCommand.h/cpp`)
   - Defines observation terms for the policy
   - Computes robot state relative to reference motion in local frame
   - Handles coordinate transforms (world → robot base frame)

### Observation System

Custom observation terms registered via `parserObservation()`:
- `motion_anchor_pos_b` / `motion_ref_pos_b` - Anchor body position in base frame
- `motion_anchor_ori_b` / `motion_ref_ori_b` - Anchor body orientation in base frame  
- `robot_body_pos` - Multiple body positions for tracking
- `robot_body_ori` - Multiple body orientations (6D rotation representation)

All observations are computed in the robot's local coordinate frame to enable position-invariant tracking.

### Policy Data Flow

1. ONNX model loaded from file or downloaded from WandB
2. Metadata parsed: joint order, anchor body, body names, impedance gains
3. Each control step:
   - Observations collected via `ObservationManager`
   - Policy forward pass with timestep input
   - Outputs: joint positions, velocities, body poses (world frame)
   - Controller tracks reference motion with impedance control

### Configuration

**Controllers Config:** `config/g1/controllers.yaml`
- Controller manager settings (500 Hz update rate)
- State estimator configuration (base link, contact points)
- Standby controller (23 DoF joint positions with PD gains)
- Walking controller (50 Hz update rate)

**Launch Files:**
- `mujoco.launch.py` - Simulation launch with dynamic config override
- `real.launch.py` - Real robot launch with rosbag recording
- `wandb.launch.py` - Utility to download ONNX from WandB runs

**Robot Support:**
- Primary: Unitree G1 (23 DoF configuration)
- URDF/MuJoCo models in `urdf/g1/` and `mjcf/g1/`

## ONNX Model Requirements

Models must be exported with specific metadata and I/O structure:

**Required Inputs:**
- Observations (standard RL observation vector)
- `time_step` (1x1 tensor for temporal policies)

**Required Outputs:**
- `joint_pos` - Target joint positions
- `joint_vel` - Target joint velocities  
- `body_pos_w` - Body positions in world frame
- `body_quat_w` - Body orientations as quaternions (w,x,y,z)

**Required Metadata:**
- `anchor_body_name` - Reference body for tracking
- `body_names` - CSV list of bodies to track
- Additional: joint order, impedance gains, etc.

**Inspection Script:** `scripts/onnx_metadata.py` - View model metadata and I/O shapes

## Development Notes

### Controller Plugin System

This controller is registered as a `controller_interface` plugin via:
- `motion_tracking_controller.xml` - Plugin description
- `PLUGINLIB_EXPORT_CLASS` macro in `MotionTrackingController.cpp`
- CMakeLists.txt: `pluginlib_export_plugin_description_file()`

### Reference Implementations

For starting a new controller from scratch, see:
- Template: https://github.com/qiayuanl/legged_template_controller
- Full docs: https://qiayuanl.github.io/legged_control2_doc/

### Coordinate Frames

The system uses several coordinate frames:
- **World frame**: Fixed inertial frame
- **Robot base frame**: Pelvis link, estimated by `state_estimator`
- **Local frame**: Yaw-aligned frame for position-invariant tracking
- Transforms handled by Pinocchio (see `worldToInit_` in `MotionCommand`)

### Common Patterns

**Adding new observation terms:**
1. Create class inheriting from `ObservationTerm` 
2. Implement `getSize()` and `evaluate()`
3. Register in `parserObservation()` method

**Modifying policy behavior:**
- Override `forward()` in `MotionOnnxPolicy` for custom inference
- Override `parseMetadata()` for custom ONNX metadata fields
- Adjust `startStep_` in constructor to skip initial frames

### Dependencies

- **legged_control2**: Core framework for legged robot control
- **legged_rl_controllers**: RL-specific controllers and policies
- **unitree_description**: Robot URDF models
- **unitree_systems**: Hardware interface for real robot
- **unitree_bringup**: Launch utilities and teleop
- **mujoco_sim_ros2**: MuJoCo simulator integration
- **Eigen3**: Linear algebra
- **ONNX Runtime**: Neural network inference (via legged_control2)

## Troubleshooting

**Build fails with "legged_control2 not found":**
- Install from debian source: https://qiayuanl.github.io/legged_control2_doc/installation.html#debian-source-recommended
- Add Unitree debian source as shown in README

**Policy path not found:**
- Use absolute paths or paths starting with `~`
- Check WandB authentication: `wandb login`

**Robot doesn't track motion:**
- Verify ONNX metadata with `scripts/onnx_metadata.py`
- Check joint order matches robot configuration (23 DoF vs 29 DoF)
- Ensure state estimator is running and publishing

**Controller not switching:**
- Controllers spawn in specific states (active/inactive)
- Use joystick buttons or controller_manager CLI to switch
- Check `ros2 control list_controllers` for status
