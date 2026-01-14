//
// Created by qiayuanl on 5/14/25.
//
// MotionOnnxPolicy.cpp - Neural network policy for motion tracking
//
// This file implements the MotionOnnxPolicy class which wraps an ONNX neural network
// model for motion tracking. The policy takes robot observations as input and outputs:
//  - Joint position/velocity references
//  - Body position/orientation trajectories for multiple tracked bodies
//  - Control actions
//
// The policy is temporal (stateful) and uses a time_step input to track progression
// through the reference motion sequence.

#include "motion_tracking_controller/MotionOnnxPolicy.h"

#include <iostream>

namespace legged {

/**
 * @brief Reset the policy to its initial state
 * 
 * This function resets the temporal state of the policy, setting the time step
 * to the configured start step. It also performs an initial forward pass with
 * zero observations to initialize all internal outputs (joint positions, velocities,
 * body poses, etc.).
 * 
 * The start step allows skipping initial frames of the motion, which is useful
 * for testing different portions of a reference motion sequence.
 */
void MotionOnnxPolicy::reset() {
  OnnxPolicy::reset();  // Reset parent class state
  timeStep_ = startStep_;  // Initialize to start step (default 0)
  
  // Perform initial forward pass to populate outputs
  forward(vector_t::Zero(getObservationSize()));
}

/**
 * @brief Run neural network inference to get motion references
 * 
 * This function performs a forward pass through the ONNX neural network model.
 * It takes the current robot observations and time step as inputs, and produces:
 *  - Joint position references (23-DoF for G1)
 *  - Joint velocity references
 *  - Body positions in world frame (14 bodies for G1)
 *  - Body orientations as quaternions (w, x, y, z)
 *  - Control actions
 * 
 * The time step is automatically incremented with each call, allowing the policy
 * to progress through the reference motion sequence.
 * 
 * @param observations Current robot state observations (130-dim for G1 23-DoF)
 * @return Control actions from the policy
 */
vector_t MotionOnnxPolicy::forward(const vector_t& observations) {
  // Prepare time step input tensor
  tensor2d_t timeStep(1, 1);
  timeStep(0, 0) = static_cast<tensor_element_t>(timeStep_++);
  inputTensors_[name2Index_.at("time_step")] = timeStep;
  
  // Run parent class forward pass (ONNX inference)
  OnnxPolicy::forward(observations);

  // Extract joint position and velocity references from network outputs
  // These are used as targets for the low-level joint controller
  jointPosition_ = outputTensors_[name2Index_.at("joint_pos")].row(0).cast<scalar_t>();
  jointVelocity_ = outputTensors_[name2Index_.at("joint_vel")].row(0).cast<scalar_t>();
  
  // Clear previous body pose data
  bodyPositions_.clear();
  bodyOrientations_.clear();

  // Extract body poses in world frame
  // These describe the reference trajectory for multiple bodies (e.g., pelvis, torso, feet, hands)
  auto body_pos_w = outputTensors_[name2Index_.at("body_pos_w")].cast<scalar_t>();
  auto body_quat_w = outputTensors_[name2Index_.at("body_quat_w")].cast<scalar_t>();

  // Convert body poses to internal format
  for (size_t i = 0; i < body_pos_w.rows(); ++i) {
    // Extract position for body i
    vector3_t pos = body_pos_w.row(i);
    
    // Extract and convert quaternion (w, x, y, z) for body i
    vector_t quat = body_quat_w.row(i);
    quaternion_t ori;
    ori.w() = quat(0);  // Quaternion scalar part
    ori.coeffs().head(3) = quat.tail(3);  // Quaternion vector part (x, y, z)
    
    // Store body pose
    bodyPositions_.push_back(pos);
    bodyOrientations_.push_back(ori);
  }
  
  // Return control actions for the robot
  return getLastAction();
}

/**
 * @brief Parse metadata embedded in the ONNX model
 * 
 * Extracts configuration information stored in the ONNX model's metadata:
 *  - anchor_body_name: The reference frame for motion tracking (e.g., "torso_link")
 *  - body_names: List of all tracked body frames (e.g., pelvis, feet, hands)
 * 
 * This metadata comes from the training process and ensures the controller
 * configuration matches what the policy was trained with.
 */
void MotionOnnxPolicy::parseMetadata() {
  // Parse parent class metadata (joint names, impedance gains, etc.)
  OnnxPolicy::parseMetadata();
  
  // Extract anchor body name from ONNX metadata
  anchorBodyName_ = getMetadataStr("anchor_body_name");
  std::cout << '\t' << "anchor_body_name: " << anchorBodyName_ << '\n';
  
  // Extract list of tracked body names from ONNX metadata
  bodyNames_ = parseCsv<std::string>(getMetadataStr("body_names"));
  std::cout << '\t' << "body_names: " << bodyNames_ << '\n';
}

}  // namespace legged