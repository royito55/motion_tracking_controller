//
// Created by qiayuanl on 3/7/25.
//
// MotionObservation.h - Observation terms for motion tracking
//
// Defines observation terms that compute state information fed to the neural
// network policy. These observations capture the difference between the robot's
// current state and the reference motion.
//
// Observation types:
//  - MotionAnchorPosition: Reference anchor position in local frame (3D)
//  - MotionAnchorOrientation: Reference anchor orientation in local frame (6D)
//  - RobotBodyPosition: Current body positions in local frame (3D per body)
//  - RobotBodyOrientation: Current body orientations in local frame (6D per body)

#pragma once

#include <legged_rl_controllers/ObservationManager.h>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {

/**
 * @brief Base class for motion tracking observations
 * 
 * Provides common functionality for all motion tracking observation terms.
 * Stores a reference to the motion command term to access reference motion data.
 */
class MotionObservation : public ObservationTerm {
 public:
  explicit MotionObservation(const MotionCommandTerm::SharedPtr& commandTerm) : commandTerm_(commandTerm) {}

 protected:
  MotionCommandTerm::SharedPtr commandTerm_;  ///< Access to motion references
};

/**
 * @brief Observation of reference anchor position in local frame
 * 
 * Provides the target position for the anchor body (e.g., torso) relative to
 * its current position. This tells the policy where the anchor should move.
 * 
 * Dimension: 3 (x, y, z in anchor's local frame)
 */
class MotionAnchorPosition final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 3; }

 protected:
  vector_t evaluate() override { return commandTerm_->getAnchorPositionLocal(); }
};

/**
 * @brief Observation of reference anchor orientation in local frame
 * 
 * Provides the target orientation for the anchor body relative to its current
 * orientation using 6D rotation representation (first two columns of rotation matrix).
 * 
 * Dimension: 6 (R00, R01, R10, R11, R20, R21)
 */
class MotionAnchorOrientation final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 6; }

 protected:
  vector_t evaluate() override { return commandTerm_->getAnchorOrientationLocal(); }
};

/**
 * @brief Observation of current robot body positions in local frame
 * 
 * Provides the current positions of all tracked bodies (e.g., pelvis, feet, hands)
 * relative to the anchor body. This gives the policy spatial awareness of the
 * robot's configuration.
 * 
 * Dimension: 3 * num_bodies (x, y, z for each body)
 */
class RobotBodyPosition final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 3 * commandTerm_->getCfg().bodyNames.size(); }

 protected:
  vector_t evaluate() override { return commandTerm_->getRobotBodyPositionLocal(); }
};

/**
 * @brief Observation of current robot body orientations in local frame
 * 
 * Provides the current orientations of all tracked bodies relative to the anchor
 * body using 6D rotation representation. This gives the policy orientation awareness
 * of the robot's configuration.
 * 
 * Dimension: 6 * num_bodies (6D rotation for each body)
 */
class RobotBodyOrientation final : public MotionObservation {
 public:
  using MotionObservation::MotionObservation;
  size_t getSize() const override { return 6 * commandTerm_->getCfg().bodyNames.size(); }

 protected:
  vector_t evaluate() override { return commandTerm_->getRobotBodyOrientationLocal(); }
};

}  // namespace legged