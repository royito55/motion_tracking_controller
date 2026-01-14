//
// Created by qiayuanl on 5/14/25.
//
// MotionOnnxPolicy.h - Neural network policy for motion tracking
//
// Extends the base OnnxPolicy class to handle motion tracking specific outputs:
//  - Reference joint positions and velocities
//  - Reference body positions and orientations (multiple tracked bodies)
//  - Temporal progression through motion sequences via time_step input
//
// The policy is trained to output reference trajectories that the robot should follow.

#pragma once

#include <legged_rl_controllers/OnnxPolicy.h>

namespace legged {

/**
 * @brief ONNX neural network policy for motion tracking
 * 
 * This class wraps an ONNX model that takes robot observations and outputs:
 *  - Control actions for the robot
 *  - Reference joint positions and velocities
 *  - Reference body poses (positions and orientations) for multiple tracked bodies
 * 
 * The policy is temporal (stateful) - it uses a time_step input that increments
 * with each forward pass, allowing it to progress through a motion sequence.
 * 
 * Key features:
 *  - Loads motion tracking specific outputs (joint refs, body poses)
 *  - Extracts metadata (anchor body, tracked body list) from ONNX file
 *  - Supports starting from arbitrary time steps in the motion sequence
 */
class MotionOnnxPolicy : public OnnxPolicy {
 public:
  using SharedPtr = std::shared_ptr<MotionOnnxPolicy>;
  
  /**
   * @brief Constructor
   * @param modelPath Path to the ONNX model file
   * @param startStep Initial time step (0 = start from beginning, >0 = skip initial frames)
   */
  MotionOnnxPolicy(const std::string& modelPath, size_t startStep) : OnnxPolicy(modelPath), startStep_(startStep) {}

  /// Reset policy to initial state and set time step to startStep
  void reset() override;
  
  /// Run neural network inference with current observations and time step
  vector_t forward(const vector_t& observations) override;

  // Metadata accessors (extracted from ONNX file)
  
  /// Get name of anchor body (reference frame for tracking, e.g., "torso_link")
  std::string getAnchorBodyName() const { return anchorBodyName_; }
  
  /// Get list of all tracked body names (e.g., ["pelvis", "left_foot", "right_foot", ...])
  std::vector<std::string> getBodyNames() const { return bodyNames_; }

  // Reference motion getters (updated after each forward pass)
  
  /// Get reference joint positions from last forward pass (23-DoF for G1)
  vector_t getJointPosition() const { return jointPosition_; }
  
  /// Get reference joint velocities from last forward pass
  vector_t getJointVelocity() const { return jointVelocity_; }
  
  /// Get reference body positions in world frame (one per tracked body)
  std::vector<vector3_t> getBodyPositions() const { return bodyPositions_; }
  
  /// Get reference body orientations as quaternions (one per tracked body)
  std::vector<quaternion_t> getBodyOrientations() const { return bodyOrientations_; }

  /// Parse motion-specific metadata from ONNX model
  void parseMetadata() override;

 protected:
  // Temporal state
  size_t timeStep_ = 0;   ///< Current time step in motion sequence (auto-increments)
  size_t startStep_ = 0;  ///< Initial time step (allows skipping initial frames)
  
  // Reference motion outputs (updated by forward())
  vector_t jointPosition_;                      ///< Reference joint positions (23-DoF for G1)
  vector_t jointVelocity_;                      ///< Reference joint velocities
  std::vector<vector3_t> bodyPositions_;        ///< Reference body positions in world frame
  std::vector<quaternion_t> bodyOrientations_;  ///< Reference body orientations (quaternions)
  
  // Configuration (loaded from ONNX metadata)
  std::string anchorBodyName_;           ///< Anchor body name (e.g., "torso_link")
  std::vector<std::string> bodyNames_;   ///< List of tracked body names
};

}  // namespace legged