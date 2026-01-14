//
// MotionTrackingController.h - ROS2 controller for humanoid motion tracking
//
// This controller uses reinforcement learning policies to track reference motions
// on humanoid robots. It extends the base RlController with motion-specific
// command and observation terms.
//
// Architecture:
//  1. Loads ONNX neural network policy containing reference motion
//  2. Extracts joint and body pose references from the policy
//  3. Computes observations (current state vs reference)
//  4. Runs policy inference to get control actions
//  5. Sends joint commands via ros2_control
//
// Integrates with the legged_control2 framework and ros2_control.

#pragma once

#include <legged_rl_controllers/RlController.h>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {

/**
 * @brief ROS2 controller for whole-body motion tracking on humanoid robots
 * 
 * This controller uses a neural network policy (loaded from ONNX) to track
 * reference motions. The policy outputs joint position/velocity references
 * and body pose trajectories, which the controller uses to compute observations
 * and control actions.
 * 
 * Control flow:
 *  1. Policy outputs reference motion (joint pos/vel, body poses)
 *  2. Controller computes observations (current state vs reference)
 *  3. Policy inference produces control actions
 *  4. Actions are sent as joint torque commands via ros2_control
 * 
 * Custom command terms:
 *  - motion: Reference joint positions and velocities from policy
 * 
 * Custom observation terms:
 *  - motion_anchor_pos_b: Reference anchor position in local frame
 *  - motion_anchor_ori_b: Reference anchor orientation in local frame
 *  - robot_body_pos: Current body positions in local frame
 *  - robot_body_ori: Current body orientations in local frame
 */
class MotionTrackingController : public RlController {
 public:
  // ROS2 lifecycle callbacks
  
  /// Initialize controller (declare parameters)
  controller_interface::CallbackReturn on_init() override;

  /// Configure controller (load ONNX policy, extract metadata)
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  /// Activate controller (start control loop)
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  /// Deactivate controller (stop control loop)
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  /// Parse and register motion-specific command terms
  bool parserCommand(const std::string& name) override;
  
  /// Parse and register motion-specific observation terms
  bool parserObservation(const std::string& name) override;

  MotionCommandCfg cfg_;                      ///< Configuration (anchor body, tracked bodies)
  MotionCommandTerm::SharedPtr commandTerm_;  ///< Motion command term instance
};

}  // namespace legged