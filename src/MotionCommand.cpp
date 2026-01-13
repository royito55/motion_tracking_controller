//
// Created by qiayuanl on 3/7/25.
//
// MotionCommand.cpp
//
// PURPOSE (high level):
// --------------------
// This file acts as the *bridge* between a learned motion policy
// and the MPC controller.
//
// It:
//   1. Reads reference data from the policy (joint positions, velocities, body poses)
//   2. Aligns the policy motion frame with the robot’s real world frame
//   3. Packages everything into vectors that MPC cost functions can consume
//

#include "motion_tracking_controller/MotionCommand.h"

namespace legged {


// ============================================================================
// getValue()
// ============================================================================
// This function is called EVERY MPC timestep.
//
// It returns a single vector containing the **reference command**
// that MPC should track.
//
// For your setup (23 DoF), the reference vector layout is:
//
//   [ q_ref(23),
//     dq_ref(23),
//     anchor_x,
//     anchor_y ]
//
// Total size = 48
// The anchor orientation and other body poses are handled in a separate getAnchorPositionLocal()
vector_t MotionCommandTerm::getValue() {
  // Safety check: if policy data not ready, return zeros
  if (motionPolicy_->getJointPosition().size() == 0) {
    return vector_t::Zero(getSize());
  }
  
  // Get anchor position in local frame (3D)
  const auto anchorPosLocal = getAnchorPositionLocal();
  
  // Return: joint_pos(23) + joint_vel(23) + anchor_xy(2) = 48 elements
  return (vector_t(getSize())
          << motionPolicy_->getJointPosition(),
             motionPolicy_->getJointVelocity(),
             anchorPosLocal.head<2>())  // Only x, y (no z)
      .finished();
}


// ============================================================================
// reset()
// ============================================================================
// Called when the controller starts or resets.
//
// This function:
//   - Finds important frames in the Pinocchio model
//   - Identifies the anchor body
//   - Aligns the motion policy’s frame with the robot’s current pose
//
void MotionCommandTerm::reset() {

  // --------------------------------------------------------------------------
  // 1) Get the Pinocchio model
  // --------------------------------------------------------------------------
  const auto& pinModel = model_->getPinModel();

  // --------------------------------------------------------------------------
  // 2) Find the anchor body frame index in the robot model
  // --------------------------------------------------------------------------
  // cfg_.anchorBody is a string, e.g. "pelvis"
  //
  anchorRobotIndex_ = pinModel.getFrameId(cfg_.anchorBody);

  // --------------------------------------------------------------------------
  // 3) Debug: print all joint names
  // --------------------------------------------------------------------------
  // This is VERY useful when verifying:
  //   - joint order
  //   - floating base joints
  //   - fixed joints
  //
  std::cerr << "DEBUG: Total joints from model: "
            << model_->getNumJoints() << std::endl;
  std::cerr << "DEBUG: Joint names from Pinocchio model:" << std::endl;

  for (size_t i = 0; i < pinModel.joints.size(); ++i) {
    std::cerr << "  Joint " << i << ": "
              << pinModel.names[i] << std::endl;
  }

  // --------------------------------------------------------------------------
  // 4) Safety check: anchor body must exist
  // --------------------------------------------------------------------------
  if (anchorRobotIndex_ >= pinModel.nframes) {
    throw std::runtime_error(
        "Anchor body " + cfg_.anchorBody + " not found.");
  }

  // --------------------------------------------------------------------------
  // 5) Convert tracked body names into frame indices
  // --------------------------------------------------------------------------
  // These bodies will be tracked by body position/orientation costs.
  //
  for (const auto& bodyName : cfg_.bodyNames) {
    bodyIndices_.push_back(pinModel.getFrameId(bodyName));
    if (bodyIndices_.back() >= pinModel.nframes) {
      throw std::runtime_error("Frame " + bodyName + " not found.");
    }
  }

  // --------------------------------------------------------------------------
  // 6) Find anchor index in the policy’s body list
  // --------------------------------------------------------------------------
  // motionPolicy_->getBodyPositions() is ordered according to cfg_.bodyNames
  //
  anchorMotionIndex_ = cfg_.bodyNames.size();
  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    if (cfg_.bodyNames[i] == cfg_.anchorBody) {
      anchorMotionIndex_ = i;
      break;
    }
  }

  if (anchorMotionIndex_ == cfg_.bodyNames.size()) {
    throw std::runtime_error(
        "Anchor body " + cfg_.anchorBody +
        " not found in body names.");
  }

  // --------------------------------------------------------------------------
  // 7) Align motion policy frame with robot frame
  // --------------------------------------------------------------------------
  // The policy motion starts at an arbitrary pose.
  // The robot is already somewhere in the world.
  //
  // We compute a transform so:
  //   policy_frame → world_frame
  //
  
  // Safety check: ensure policy has been initialized with valid data
  if (motionPolicy_->getBodyPositions().empty() || 
      motionPolicy_->getBodyOrientations().empty()) {
    std::cerr << "WARNING: Policy body data not initialized yet. "
              << "Skipping frame alignment in reset()." << std::endl;
    // Set identity transform as fallback
    worldToInit_ = pinocchio::SE3::Identity();
    return;
  }
  
  pinocchio::SE3 initToAnchor(
      motionPolicy_->getBodyOrientations()[anchorMotionIndex_],
      motionPolicy_->getBodyPositions()[anchorMotionIndex_]);

  pinocchio::SE3 worldToAnchor =
      model_->getPinData().oMf[anchorRobotIndex_];

  // Remove roll/pitch (keep yaw only)
  initToAnchor.rotation() =
      yawQuaternion(quaternion_t(initToAnchor.rotation()));
  worldToAnchor.rotation() =
      yawQuaternion(quaternion_t(worldToAnchor.rotation()));

  // Final transform used everywhere else
  worldToInit_ = worldToAnchor * initToAnchor.inverse();

  // Debug transforms
  std::cerr << "initToAnchor:" << std::endl << initToAnchor << std::endl;
  std::cerr << "worldToAnchor:" << std::endl << worldToAnchor << std::endl;
  std::cerr << "worldToInit_:" << std::endl << worldToInit_ << std::endl;
}


// ============================================================================
// getAnchorPositionLocal()
// ============================================================================
// Returns the anchor position error in the policy-local frame.
//
vector3_t MotionCommandTerm::getAnchorPositionLocal() const {
  // Safety check: if policy data not ready, return zero
  if (motionPolicy_->getBodyPositions().empty() ||
      anchorMotionIndex_ >= motionPolicy_->getBodyPositions().size()) {
    return vector3_t::Zero();
  }

  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];

  const auto& anchorPos =
      motionPolicy_->getBodyPositions()[anchorMotionIndex_];

  // Transform policy anchor position into robot-local coordinates
  return anchorPoseReal.actInv(
      worldToInit_.act(anchorPos));
}


// ============================================================================
// getAnchorOrientationLocal()
// ============================================================================
// Returns the anchor orientation error (as quaternion)
// in the policy-local frame.
//
vector_t MotionCommandTerm::getAnchorOrientationLocal() const {
  // Safety check: if policy data not ready, return identity quaternion
  if (motionPolicy_->getBodyOrientations().empty() ||
      anchorMotionIndex_ >= motionPolicy_->getBodyOrientations().size()) {
    vector_t quat4(4);
    quat4 << 0, 0, 0, 1;  // Identity quaternion (x, y, z, w)
    return quat4;
  }

  const auto& anchorPoseReal =
      model_->getPinData().oMf[anchorRobotIndex_];

  pinocchio::SE3 anchorOri(
      motionPolicy_->getBodyOrientations()[anchorMotionIndex_],
      vector3_t::Zero());

  const auto rot =
      anchorPoseReal.actInv(
          worldToInit_.act(anchorOri)).rotation();

  quaternion_t quat(rot);

  vector_t quat4(4);
  quat4 << quat.x(), quat.y(), quat.z(), quat.w();

  return quat4;
}


// ============================================================================
// getRobotBodyPositionLocal()
// ============================================================================
// Returns all tracked body positions in anchor-local coordinates.
//
vector_t MotionCommandTerm::getRobotBodyPositionLocal() const {
  vector_t value(3 * cfg_.bodyNames.size());
  
  // Safety check: if body indices not initialized, return zeros
  if (bodyIndices_.empty()) {
    value.setZero();
    return value;
  }

  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];

  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& bodyPoseLocal =
        anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]);
    value.segment(3 * i, 3) = bodyPoseLocal.translation();
  }

  return value;
}


// ============================================================================
// getRobotBodyOrientationLocal()
// ============================================================================
// Returns all tracked body orientations in anchor-local coordinates,
// encoded as a 6D rotation representation.
//
vector_t MotionCommandTerm::getRobotBodyOrientationLocal() const {

  const auto& data = model_->getPinData();
  const auto& anchorPoseReal = data.oMf[anchorRobotIndex_];

  vector_t value(6 * cfg_.bodyNames.size());

  for (size_t i = 0; i < cfg_.bodyNames.size(); ++i) {
    const auto& rot =
        anchorPoseReal.actInv(
            data.oMf[bodyIndices_[i]]).rotation();

    vector_t rot6(6);
    rot6 << rot(0, 0), rot(0, 1),
            rot(1, 0), rot(1, 1),
            rot(2, 0), rot(2, 1);

    value.segment(i * 6, 6) = rot6;
  }

  return value;
}

}  // namespace legged
