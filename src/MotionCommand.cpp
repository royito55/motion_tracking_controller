//
// Created by qiayuanl on 3/7/25.
//
// MotionCommand.cpp - Provides reference motion commands from the ONNX policy
//
// This file implements the MotionCommandTerm class which extracts reference joint
// positions and velocities from the motion tracking policy and provides them as
// command terms to the RL controller. It also handles coordinate frame transformations
// between the motion reference frame and the robot's current frame.

#include "motion_tracking_controller/MotionCommand.h"

namespace legged {

/**
 * @brief Get the command value (reference joint positions and velocities)
 * 
 * Returns a vector containing the reference joint positions followed by reference
 * joint velocities from the motion policy. This serves as the "command" that the
 * controller tries to track.
 * 
 * @return Vector of size (2 * num_actuated_joints) containing [joint_pos, joint_vel]
 */
vector_t MotionCommandTerm::getValue() {
  // Concatenate joint positions and velocities into a single vector
  // The comma operator in Eigen allows elegant vector construction
  vector_t cmd = (vector_t(getSize()) << motionPolicy_->getJointPosition(), motionPolicy_->getJointVelocity()).finished();
  
  // Safety check: ensure command contains valid values
  if (!cmd.allFinite()) {
    std::cerr << "[MotionCommand] WARNING: getValue() returning non-finite command values!" << std::endl;
    std::cerr << "  joint_pos finite: " << motionPolicy_->getJointPosition().allFinite() << std::endl;
    std::cerr << "  joint_vel finite: " << motionPolicy_->getJointVelocity().allFinite() << std::endl;
  }
  
  return cmd;
}

/**
 * @brief Initialize the motion command term and compute coordinate frame transformations
 * 
 * This function is called when the controller is activated. It performs several key tasks:
 * 1. Calculates the command size based on the number of actuated joints
 * 2. Identifies the anchor body frame in both the robot and motion models
 * 3. Computes the transformation between the world frame and the motion initial frame
 *    to align the motion with the robot's current pose
 * 
 * The coordinate frame transformation ensures that the reference motion starts from
 * the robot's current position and yaw orientation, making the motion tracking seamless.
 */
void MotionCommandTerm::reset() {
  const auto& pinModel = model_->getPinModel();
  
  // Calculate command size: 2 * (actuated joints only)
  // Pinocchio's nv (velocity dimension) includes:
  //   - 1 DOF for universe joint
  //   - 6 DOF for root_joint (floating base)
  //   - 23 DOF for actuated joints (for G1 23-DoF model)
  // Total nv = 35, so actuated joints = nv - 6 = 23
  // Command size = 2 * 23 = 46 (position + velocity for each joint)
  commandSize_ = 2 * (pinModel.nv - 6);
  
  // Debug output to verify joint structure (can be removed in production)
  std::cout << "[DEBUG] Pinocchio joints (njoints=" << pinModel.njoints << ", nv=" << pinModel.nv << ", nq=" << pinModel.nq << "):" << std::endl;
  std::cout << "  Command size: " << commandSize_ << " (2 * (" << pinModel.nv << " - 6))  [Expected: 46]" << std::endl;
  for (size_t i = 0; i < pinModel.joints.size(); ++i) {
    std::cout << "  Joint " << i << ": " << pinModel.names[i] << " (nv=" << pinModel.joints[i].nv() << ", nq=" << pinModel.joints[i].nq() << ")" << std::endl;
  }
  
  // Find the anchor body (typically torso_link) in the robot model
  // This is the reference frame for motion tracking
  anchorRobotIndex_ = pinModel.getFrameId(cfg_.anchorBody);
  if (anchorRobotIndex_ >= pinModel.nframes) {
    throw std::runtime_error("Anchor body " + cfg_.anchorBody + " not found.");
  }
  
  // Find all tracked body frames in the robot model
  // These bodies' positions and orientations are used for observation
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }
  
  // Find which index in the motion body list corresponds to the anchor body
  // This is needed to extract the anchor's pose from the policy outputs
  anchorMotionIndex_ = cfg_.bodyNames.size();
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    if (cfg_.bodyNames[i] == cfg_.anchorBody) {
      anchorMotionIndex_ = i;
      break;
    }
  }
  if (anchorMotionIndex_ == cfg_.bodyNames.size()) {
    throw std::runtime_error("Anchor body " + cfg_.anchorBody + " not found in body names.");
  }

  // Defer worldToInit_ computation until first use (lazy initialization)
  // This prevents NaN propagation from uninitialized Pinocchio frame transforms
  // during controller activation when kinematics data might not be ready yet.
  worldToInitComputed_ = false;
  worldToInit_ = pinocchio::SE3::Identity();  // Safe fallback
}

/**
 * @brief Compute worldToInit_ transformation (lazy initialization)
 * 
 * This function is called on first use of worldToInit_ to ensure that:
 * 1. Pinocchio kinematics has been computed
 * 2. Robot state has been read from hardware/simulation
 * 3. Motion policy has been initialized with valid body poses
 * 
 * The transformation aligns the reference motion with the robot's current pose.
 */
void MotionCommandTerm::ensureWorldToInitComputed() const {
  static int call_count = 0;
  call_count++;
  
  // std::cerr << "\n[MotionCommand] ensureWorldToInitComputed() call #" << call_count << std::endl;
  // std::cerr << "  worldToInitComputed_: " << worldToInitComputed_ << std::endl;
  
  // if (worldToInitComputed_) {
  //   std::cerr << "  -> Already computed, returning early" << std::endl;
  //   return;  // Already computed
  // }

  // Safety check: ensure policy has been initialized with valid data
  if (motionPolicy_->getBodyPositions().empty() || 
      motionPolicy_->getBodyOrientations().empty()) {
    std::cerr << "  -> Policy not initialized yet (body positions/orientations empty)" << std::endl;
    return;  // Will retry on next call
  }
  // std::cerr << "  Policy has " << motionPolicy_->getBodyPositions().size() << " body positions" << std::endl;

  // Check if model state (q, v) contains valid data
  const auto& q = model_->getGeneralizedPosition();
  const auto& v = model_->getGeneralizedVelocity();
  
  // std::cerr << "  q size: " << q.size() << ", finite: " << q.allFinite() << std::endl;
  // std::cerr << "  q[0:7]: " << q.head(7).transpose() << std::endl;
  // std::cerr << "  v size: " << v.size() << ", finite: " << v.allFinite() << std::endl;
  
  if (!q.allFinite() || !v.allFinite()) {
    std::cerr << "  -> q or v contains NaN/Inf, returning early" << std::endl;
    return;  // Will retry on next observation computation
  }
  
  // Get anchor poses from motion policy and robot model
  pinocchio::SE3 initToAnchor(motionPolicy_->getBodyOrientations()[anchorMotionIndex_], 
                               motionPolicy_->getBodyPositions()[anchorMotionIndex_]);
  pinocchio::SE3 worldToAnchor = model_->getPinData().oMf[anchorRobotIndex_];
  
  // std::cerr << "  initToAnchor pos: " << initToAnchor.translation().transpose() << std::endl;
  // std::cerr << "  worldToAnchor pos: " << worldToAnchor.translation().transpose() 
  //           << ", finite: " << worldToAnchor.translation().allFinite() << std::endl;
  
  // Verify the computed transform is valid
  if (!worldToAnchor.translation().allFinite()) {
    std::cerr << "  -> worldToAnchor contains NaN, returning early" << std::endl;
    return;  // Retry later
  }

  // Extract only yaw rotation (project to horizontal plane)
  // This prevents pitch/roll misalignment between motion and robot
  initToAnchor.rotation() = yawQuaternion(quaternion_t(initToAnchor.rotation()));
  worldToAnchor.rotation() = yawQuaternion(quaternion_t(worldToAnchor.rotation()));

  // Compute the transformation: worldToInit transforms points from motion frame to world frame
  worldToInit_ = worldToAnchor * initToAnchor.inverse();
  worldToInitComputed_ = true;

  // Debug output for transformation verification
  // std::cerr << "[MotionCommand] Computed worldToInit transformation:" << std::endl;
  // std::cerr << "  initToAnchor:" << std::endl << initToAnchor << std::endl;
  // std::cerr << "  worldToAnchor:" << std::endl << worldToAnchor << std::endl;
  // std::cerr << "  worldToInit_:" << std::endl << worldToInit_ << std::endl;
}

/**
 * @brief Get the reference anchor position in the robot's local frame
 * 
 * Transforms the anchor body's reference position from the motion frame to the
 * robot's current anchor frame. This provides the target position for the anchor
 * body relative to its current position.
 * 
 * @return 3D vector representing the target position in local coordinates [x, y, z]
 */
vector3_t MotionCommandTerm::getAnchorPositionLocal() const {
  ensureWorldToInitComputed();  // Lazy initialization
  
  // If worldToInit_ not ready yet, return zero position (neutral observation)
  if (!worldToInitComputed_) {
    std::cerr << "[MotionCommand] getAnchorPositionLocal: returning ZERO (not computed yet)" << std::endl;
    return vector3_t::Zero();
  }
  
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];  // Current anchor pose in world

  const auto& anchorPos = motionPolicy_->getBodyPositions()[anchorMotionIndex_];  // Reference in motion frame
  
  // Transform: motion_frame -> world_frame -> anchor_local_frame
  return anchorPoseReal.actInv(worldToInit_.act(anchorPos));
}

/**
 * @brief Get the reference anchor orientation in the robot's local frame
 * 
 * Transforms the anchor body's reference orientation from the motion frame to the
 * robot's current anchor frame. Uses 6D rotation representation (first two columns
 * of the rotation matrix) for continuity in neural network inputs.
 * 
 * @return 6D vector representing rotation matrix: [R00, R01, R10, R11, R20, R21]
 */
vector_t MotionCommandTerm::getAnchorOrientationLocal() const {
  ensureWorldToInitComputed();  // Lazy initialization
  
  // If worldToInit_ not ready yet, return identity rotation (neutral observation)
  // Identity = [1, 0, 0, 1, 0, 0] in 6D representation
  if (!worldToInitComputed_) {
    std::cerr << "[MotionCommand] getAnchorOrientationLocal: returning IDENTITY (not computed yet)" << std::endl;
    vector_t rot6(6);
    rot6 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    return rot6;
  }
  
  const auto& anchorPoseReal = model_->getPinData().oMf[anchorRobotIndex_];
  const pinocchio::SE3 anchorOri(motionPolicy_->getBodyOrientations()[anchorMotionIndex_], vector3_t::Zero());
  
  // Transform orientation: motion_frame -> world_frame -> anchor_local_frame
  const auto rot = anchorPoseReal.actInv(worldToInit_.act(anchorOri)).rotation();
  
  // Convert to 6D representation (first two columns of rotation matrix)
  // This representation is continuous and works well with neural networks
  vector_t rot6(6);
  rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
  return rot6;
}

/**
 * @brief Get positions of all tracked bodies in the anchor's local frame
 * 
 * Returns the current positions of all tracked body frames relative to the anchor
 * body frame. This provides spatial awareness of the robot's configuration.
 * 
 * @return Vector of size (3 * num_bodies) containing [x, y, z] for each body
 */
vector_t MotionCommandTerm::getRobotBodyPositionLocal() const {
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];
  vector_t value(3 * cfg_.bodyNames.size());
  
  // Transform each body position from world frame to anchor's local frame
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }
  return value;
}

/**
 * @brief Get orientations of all tracked bodies in the anchor's local frame
 * 
 * Returns the current orientations of all tracked body frames relative to the anchor
 * body frame using 6D rotation representation. This provides orientation awareness
 * of the robot's configuration.
 * 
 * @return Vector of size (6 * num_bodies) containing 6D rotation for each body
 */
vector_t MotionCommandTerm::getRobotBodyOrientationLocal() const {
  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];
  vector_t value(6 * cfg_.bodyNames.size());
  
  // Transform each body orientation from world frame to anchor's local frame
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& rot = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]).rotation();
    
    // Convert to 6D representation (first two columns of rotation matrix)
    vector_t rot6(6);
    rot6 << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    value.segment(i * 6, 6) = rot6;
  }
  return value;
}

}  // namespace legged