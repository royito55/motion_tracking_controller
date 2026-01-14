//
// Created by qiayuanl on 3/7/25.
//
// MotionCommand.h - Command term for motion tracking
//
// Provides reference joint positions and velocities from the ONNX policy as command
// terms to the RL controller. Also handles coordinate frame transformations between
// the motion reference frame and the robot's current frame.

#pragma once

#include "motion_tracking_controller/MotionOnnxPolicy.h"
#include "motion_tracking_controller/common.h"

#include <legged_rl_controllers/CommandManager.h>

#include <utility>

namespace legged {

/**
 * @brief Command term that provides reference motion from the ONNX policy
 * 
 * This class extracts joint position and velocity references from the motion tracking
 * policy and provides them as command terms. It also manages coordinate frame
 * transformations to align the reference motion with the robot's current pose.
 * 
 * Key responsibilities:
 *  - Extract joint references from policy (positions and velocities)
 *  - Compute transformation from motion frame to world frame
 *  - Provide anchor body pose in local coordinates
 *  - Provide tracked body poses in local coordinates (for observations)
 */
class MotionCommandTerm : public CommandTerm {
 public:
  using SharedPtr = std::shared_ptr<MotionCommandTerm>;

  /**
   * @brief Constructor
   * @param cfg Configuration containing anchor body and tracked body names
   * @param motionPolicy Shared pointer to the ONNX motion tracking policy
   */
  MotionCommandTerm(MotionCommandCfg cfg, MotionOnnxPolicy::SharedPtr motionPolicy)
      : cfg_(std::move(cfg)), motionPolicy_(std::move(motionPolicy)), anchorRobotIndex_(0), anchorMotionIndex_(0),
        commandSize_(2 * motionPolicy_->getActionSize()) {}

  /// Get command value (reference joint positions and velocities)
  vector_t getValue() override;
  
  /// Initialize command term and compute coordinate transformations
  void reset() override;

  /// Get configuration (anchor body and tracked body names)
  MotionCommandCfg getCfg() const { return cfg_; }
  
  /// Get reference anchor position in robot's local frame (3D)
  vector3_t getAnchorPositionLocal() const;
  
  /// Get reference anchor orientation in robot's local frame (6D rotation representation)
  vector_t getAnchorOrientationLocal() const;
  
  /// Get current robot body positions in anchor's local frame (3D per body)
  vector_t getRobotBodyPositionLocal() const;
  
  /// Get current robot body orientations in anchor's local frame (6D per body)
  vector_t getRobotBodyOrientationLocal() const;

 protected:
  /// Get command size (2 * number of actuated joints: positions + velocities)
  size_t getSize() const override { return commandSize_; }

  // Configuration and policy
  MotionCommandCfg cfg_;                     ///< Anchor and tracked body names
  MotionOnnxPolicy::SharedPtr motionPolicy_; ///< Pointer to ONNX policy
  size_t commandSize_ = 0;                   ///< Command vector size (2*num_joints)

  // Frame indices
  size_t anchorRobotIndex_;    ///< Index of anchor body in robot model
  size_t anchorMotionIndex_;   ///< Index of anchor body in motion data
  std::vector<size_t> bodyIndices_{};  ///< Indices of tracked bodies in robot model
  
  // Coordinate transformation (lazy initialization to avoid NaN from uninitialized kinematics)
  mutable pinocchio::SE3 worldToInit_;  ///< Transform from motion initial frame to world frame
  mutable bool worldToInitComputed_ = false;  ///< Flag for lazy initialization
  
  /// Compute worldToInit_ transform on first use (lazy initialization)
  void ensureWorldToInitComputed() const;
};

/**
 * @brief Extract yaw-only rotation from a quaternion
 * 
 * Converts a full 3D rotation quaternion to a 2D rotation (yaw only) by:
 * 1. Computing the yaw angle from the quaternion
 * 2. Constructing a new quaternion with only the yaw component
 * 3. Setting pitch and roll to zero
 * 
 * This is useful for aligning motions in the horizontal plane while ignoring
 * pitch and roll differences between the reference motion and the robot.
 * 
 * @tparam Scalar Scalar type (float or double)
 * @param q Input quaternion with full 3D rotation
 * @return Quaternion representing only yaw rotation around the Z-axis
 */
template <typename Scalar>
Eigen::Quaternion<Scalar> yawQuaternion(const Eigen::Quaternion<Scalar>& q) {
  // Extract yaw angle using atan2 formula for quaternions
  Scalar yaw = std::atan2(Scalar(2) * (q.w() * q.z() + q.x() * q.y()), 
                          Scalar(1) - Scalar(2) * (q.y() * q.y() + q.z() * q.z()));
  Scalar half_yaw = yaw * Scalar(0.5);
  
  // Construct quaternion with only yaw rotation (around Z-axis)
  // Format: (w, x, y, z) where x=0, y=0 for pure yaw rotation
  Eigen::Quaternion<Scalar> ret(std::cos(half_yaw), Scalar(0), Scalar(0), std::sin(half_yaw));
  return ret.normalized();
}

}  // namespace legged