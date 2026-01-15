# G1 Motion Tracking Controller: 29 DOF to 23 DOF Conversion

This document describes the changes made to adapt the `motion_tracking_controller` package from the original 29-DOF G1 robot configuration to the 23-DOF configuration for use with Isaac Lab trained models.

## Overview

The original controller was designed for a 29-DOF G1 humanoid robot but needed to be adapted for a 23-DOF configuration that matches the training environment. The 23-DOF version removes:
- Waist roll and pitch joints (keeping only waist yaw)
- Wrist pitch and yaw joints on both arms (keeping only wrist roll)

This reduces the DOF from 29 to 23 actuated joints while maintaining the 6-DOF floating base.

## Files Created/Modified

### 1. URDF Files

#### Created: `urdf/g1/g1_23dof.urdf`
- New URDF file with 23 DOF joint configuration
- **Critical fix**: Does NOT include `floating_base_joint` definition
- Reason: Pinocchio automatically creates `root_joint` (6 DOF) for floating-base robots
- Including both `floating_base_joint` and Pinocchio's auto-created `root_joint` caused:
  - Duplicate joint conflicts
  - NaN values in kinematic chain (`oMf[]` transforms)
  - Memory corruption ("double free or corruption" errors)

#### Created: `urdf/g1/robot_23dof.xacro`
- Top-level xacro file that includes:
  - `g1_23dof.urdf` - Robot structure
  - `mujoco_23dof.xacro` - MuJoCo-specific configuration

#### Created: `urdf/g1/mujoco_23dof.xacro`
- Includes `ros2_control_23dof.xacro` for hardware interface

#### Created: `urdf/g1/ros2_control_23dof.xacro`
- Defines ros2_control hardware interfaces for 23 DOF configuration
- **Key insight**: Lines 38-41 show `floating_base_joint` COMMENTED OUT
```xml
<!-- ADD THIS FLOATING BASE JOINT -->
<!-- <joint name="floating_base_joint">
    <state_interface name="position"/>
    <state_interface name="velocity"/>
</joint> -->
```
- This was the crucial fix: removing explicit `floating_base_joint` resolved all NaN/crash issues
- MuJoCo's ros2_control plugin can write directly to Pinocchio's auto-created `root_joint`

### 2. MuJoCo XML Files

#### Created: `mjcf/g1/g1_23dof.xml`
- MuJoCo model file with 23 DOF configuration
- Matches the joint structure used in Isaac Lab training

#### Reference: `mjcf/g1/g1_29dof.xml`
- Original 29 DOF version kept for reference

### 3. Controller Configuration

#### Modified: `config/g1/controllers.yaml`
- Updated `joint_names` list to include only 23 joints
- Commented out 29-DOF joints with markers like `# 29DoF version`
- Updated `default_position`, `kp`, and `kd` arrays to match 23 DOF
- Sections updated:
  - `standby_controller/joint_names` (lines 23-53)
  - `standby_controller/default_position` (lines 72-75)
  - `standby_controller/kp` and `kd` (lines 76-81)
  - `walking_controller/joint_names` (similar updates)

### 4. Launch Files

#### Modified: `launch/mujoco.launch.py`
- Line 147: Changed to load `robot_23dof.xacro` instead of `robot.xacro`
```python
"robot_23dof.xacro"
```

### 5. Source Code Changes

#### Modified: `src/MotionCommand.cpp`

**Key changes to handle initialization timing and prevent NaN propagation:**

1. **Lazy Initialization Pattern** (lines 117-170):
   - Added `ensureWorldToInitComputed()` method
   - Defers `worldToInit_` transform computation until first use
   - Prevents accessing uninitialized Pinocchio frame transforms during controller activation
   - Includes safety checks for valid kinematics data:
     ```cpp
     if (!q.allFinite() || !v.allFinite()) {
       return;  // Will retry on next observation computation
     }
     ```
   - Verifies `worldToAnchor` transform is finite before using it

2. **Safe Observation Returns** (lines 190-240):
   - `getAnchorPositionLocal()`: Returns `Zero()` if transform not ready
   - `getAnchorOrientationLocal()`: Returns identity rotation if not ready
   - Prevents NaN from propagating into RL controller observations

3. **Removed Problematic q.allFinite() Check**:
   - Original code checked if all joint positions were finite
   - With dual joints (root_joint + floating_base_joint), q[0:3] contained NaN
   - After removing floating_base_joint, this issue was resolved

#### Modified: `include/motion_tracking_controller/MotionCommand.h`

**Added lazy initialization support** (lines 82-86):
```cpp
// Coordinate transformation (lazy initialization to avoid NaN from uninitialized kinematics)
mutable pinocchio::SE3 worldToInit_;
mutable bool worldToInitComputed_ = false;

void ensureWorldToInitComputed() const;
```

## Root Cause and Solution

### The Problem
Initial implementation crashed with:
- `[ERROR] [mujoco_sim-3]: process has died [pid XXXXX, exit code -6]`
- Error message: `double free or corruption (out)`
- NaN values appearing in joint accelerations (QACC) causing endless simulation resets

### The Investigation Trail
1. Initially thought it was a race condition (worldToInit_ computed before hardware state ready)
2. Implemented lazy initialization to defer transform computation
3. Discovered q[0:3] (base position) contained NaN while q[3:6] (quaternion) was valid
4. Traced to duplicate floating base joints:
   - **Joint 1**: `root_joint` (auto-created by Pinocchio) - had NaN positions
   - **Joint 2**: `floating_base_joint` (explicit in URDF) - had valid MuJoCo data
5. Attempted to skip NaN checks, but `worldToAnchor` transform itself was NaN

### The Solution
**Remove `floating_base_joint` from URDF/ros2_control xacro**

Key insights:
- Pinocchio's `legged_model` library (closed source, in `/opt/ros/humble`) automatically creates `root_joint` for floating-base robots
- MuJoCo's `mujoco_ros2_control` plugin can write state directly to Pinocchio's `root_joint`
- No need for explicit `floating_base_joint` in URDF
- The explicit joint was added by mistake (likely by an AI assistant unfamiliar with Pinocchio's behavior)

## Pinocchio Joint Structure (23 DOF)
After fix, the joint structure is:
```
njoints=25, nv=29, nq=30
  Joint 0: universe (nv=1, nq=1) - Pinocchio internal
  Joint 1: root_joint (nv=6, nq=7) - Auto-created floating base
  Joint 2-24: Actuated joints (23 joints, nv=23, nq=23)
```

Command size calculation:
```cpp
commandSize_ = 2 * (pinModel.nv - 6)  // 2 * (29 - 6) = 46
// Accounts for position + velocity for 23 actuated joints
// Excludes universe (1 DOF) and root_joint (6 DOF)
```

## Testing and Validation

### Working Configuration
```bash
ros2 launch motion_tracking_controller mujoco.launch.py \
  wandb_path:=rodrigo55-tc/23dof_dance2/vrzmb9u3
```

### Expected Behavior
- Simulation starts without crashes
- No "double free or corruption" errors
- Controller loads successfully
- Robot executes trained dance motions from ONNX model
- No NaN values in observations or commands

### Debug Output (for reference)
When working correctly, you should see:
```
[mujoco_sim-3] [DEBUG] Pinocchio joints (njoints=25, nv=29, nq=30):
[mujoco_sim-3]   Command size: 46 (2 * (29 - 6))  [Expected: 46]
[mujoco_sim-3]   Joint 0: universe (nv=1, nq=1)
[mujoco_sim-3]   Joint 1: root_joint (nv=6, nq=7)
[mujoco_sim-3]   Joint 2: left_hip_pitch_joint (nv=1, nq=1)
...
```

## Key Takeaways

1. **Don't define floating_base_joint explicitly when using Pinocchio** - it auto-creates root_joint
2. **Lazy initialization prevents NaN propagation** - wait for valid kinematics data before computing transforms
3. **Safety returns are essential** - return neutral observations when data isn't ready yet
4. **Debug logging helped identify the issue** - print joint structure and data validity during initialization
5. **Trust original URDF structure** - the unitree_description package had `floating_base_joint` commented out with "CAUTION: uncomment for mujoco" note, which was misleading

## Files to Review

If adapting for other DOF configurations or robots:
- Check if robot uses Pinocchio for kinematics → don't add explicit floating_base_joint
- Update joint lists in `controllers.yaml` to match your DOF count
- Update `action_scale` and PD gains in controller metadata
- Create corresponding MJCF file for MuJoCo simulation
- Ensure ONNX model expects matching observation dimensions (command size, body counts, etc.)

---

**Date**: January 14, 2026  
**Status**: ✅ Working - Robot successfully dancing with 23 DOF configuration
