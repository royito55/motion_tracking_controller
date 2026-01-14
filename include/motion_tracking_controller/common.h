//
// Created by qiayuanl on 3/7/25.
//
// common.h - Common data structures and utility functions for motion tracking
//
// This file contains shared types and helper functions used across the motion
// tracking controller components.

#pragma once

#include <legged_model/common.h>
#include <string>
#include <vector>

namespace legged {

/**
 * @brief Configuration for motion command term
 * 
 * Specifies which bodies are tracked during motion tracking:
 *  - anchorBody: The reference frame for tracking (typically "torso_link")
 *  - bodyNames: List of all tracked body frames (e.g., pelvis, feet, hands)
 * 
 * This configuration is extracted from the ONNX policy metadata during
 * controller initialization.
 */
struct MotionCommandCfg {
  std::string anchorBody;               ///< Name of anchor body (reference frame)
  std::vector<std::string> bodyNames;   ///< Names of all tracked bodies
};

/**
 * @brief Convert quaternion to 4D vector in (w, x, y, z) format
 * 
 * Converts a quaternion orientation to a 4-element vector with the scalar
 * part (w) first, followed by the vector part (x, y, z).
 * 
 * @param ori Input quaternion
 * @return 4D vector [w, x, y, z]
 */
inline vector_t rotationToVectorWxyz(const quaternion_t& ori) {
  vector_t vec(4);
  vec(0) = ori.w();                   // Scalar part (w)
  vec.segment(1, 3) = ori.coeffs().head(3);  // Vector part (x, y, z)
  return vec;
}

/**
 * @brief Convert rotation matrix to 4D vector in (w, x, y, z) format
 * 
 * Converts a 3x3 rotation matrix to a 4-element quaternion vector by first
 * converting to a quaternion, then to the vector format.
 * 
 * @param ori Input rotation matrix
 * @return 4D vector [w, x, y, z]
 */
inline vector_t rotationToVectorWxyz(const matrix3_t& ori) {
  return rotationToVectorWxyz(quaternion_t(ori));
}

}  // namespace legged