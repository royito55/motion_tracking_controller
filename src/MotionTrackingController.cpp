//
// MotionTrackingController.cpp - ROS2 controller for humanoid motion tracking
//
// This controller implements whole-body motion tracking for humanoid robots using
// reinforcement learning policies. It extends the RlController base class and adds
// motion-specific command and observation terms.
//
// Key features:
//  - Loads pre-trained ONNX neural network policies
//  - Tracks reference motions from the policy (joint positions, velocities, body poses)
//  - Provides custom observations for motion tracking (anchor position/orientation)
//  - Integrates with ros2_control for real-time robot control
//
// The controller follows the ROS2 lifecycle: init -> configure -> activate -> deactivate

#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {
/**
 * @brief Initialize the controller (ROS2 lifecycle state: unconfigured -> inactive)
 * 
 * Declares controller-specific parameters that can be set via configuration files.
 * Currently declares:
 *  - motion.start_step: Which frame of the reference motion to start from (default: 0)
 * 
 * @return SUCCESS if initialization succeeds, ERROR otherwise
 */
controller_interface::CallbackReturn MotionTrackingController::on_init() {
  // Initialize parent class (RlController)
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    // Declare motion-specific parameters
    auto_declare("motion.start_step", 0);  // Starting frame in motion sequence
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

/**
 * @brief Configure the controller (ROS2 lifecycle state: inactive -> inactive)
 * 
 * Loads the ONNX neural network policy and extracts metadata about the motion:
 *  - Anchor body name (reference frame for tracking)
 *  - Tracked body names (list of bodies with reference trajectories)
 * 
 * This metadata comes from the policy and ensures the controller configuration
 * matches the training setup.
 * 
 * @param previous_state The previous lifecycle state
 * @return SUCCESS if configuration succeeds, ERROR otherwise
 */
controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  // Get parameters from configuration
  const auto policyPath = get_node()->get_parameter("policy.path").as_string();
  const auto startStep = static_cast<size_t>(get_node()->get_parameter("motion.start_step").as_int());

  // Load and initialize the ONNX policy
  policy_ = std::make_shared<MotionOnnxPolicy>(policyPath, startStep);
  policy_->init();

  // Extract motion tracking configuration from policy metadata
  auto policy = std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_);
  cfg_.anchorBody = policy->getAnchorBodyName();  // e.g., "torso_link"
  cfg_.bodyNames = policy->getBodyNames();  // e.g., ["pelvis", "left_foot", "right_foot", ...]
  
  RCLCPP_INFO_STREAM(rclcpp::get_logger("MotionTrackingController"), 
                     "Load Onnx model from " << policyPath << " successfully !");

  // Continue with parent class configuration (model loading, term registration, etc.)
  return RlController::on_configure(previous_state);
}

/**
 * @brief Activate the controller (ROS2 lifecycle state: inactive -> active)
 * 
 * Starts the control loop. The parent class handles:
 *  - Resetting the policy state
 *  - Initializing command and observation managers
 *  - Starting the real-time control thread
 * 
 * @param previous_state The previous lifecycle state
 * @return SUCCESS if activation succeeds, ERROR otherwise
 */
controller_interface::CallbackReturn MotionTrackingController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

/**
 * @brief Deactivate the controller (ROS2 lifecycle state: active -> inactive)
 * 
 * Stops the control loop safely. The parent class handles:
 *  - Stopping the real-time control thread
 *  - Resetting internal state
 *  - Preparing for potential reactivation
 * 
 * @param previous_state The previous lifecycle state
 * @return SUCCESS if deactivation succeeds, ERROR otherwise
 */
controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

/**
 * @brief Parse and register command terms from configuration
 * 
 * Command terms provide the "setpoint" or "target" that the controller tries to achieve.
 * This controller adds the "motion" command term which extracts reference joint
 * positions and velocities from the ONNX policy.
 * 
 * Standard command terms (from parent class) might include things like base velocity commands.
 * 
 * @param name The name of the command term to parse
 * @return true if the term was recognized and registered, false otherwise
 */
bool MotionTrackingController::parserCommand(const std::string& name) {
  // Try parsing standard command terms first
  if (RlController::parserCommand(name)) {
    return true;
  }
  
  // Handle motion-specific command term
  if (name == "motion") {
    // Create the motion command term which provides joint position/velocity references
    commandTerm_ = std::make_shared<MotionCommandTerm>(cfg_, std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_));
    commandManager_->addTerm(commandTerm_);
    return true;
  }
  
  return false;  // Unknown command term
}

/**
 * @brief Parse and register observation terms from configuration
 * 
 * Observation terms compute state information that's fed to the neural network policy.
 * This controller adds motion-specific observations:
 *  - motion_anchor_pos_b: Reference anchor position in local frame
 *  - motion_anchor_ori_b: Reference anchor orientation in local frame  
 *  - robot_body_pos: Current body positions in local frame (for imitation)
 *  - robot_body_ori: Current body orientations in local frame (for imitation)
 * 
 * Standard observations (from parent class) include:
 *  - base_lin_vel, base_ang_vel: Base velocities
 *  - joint_pos, joint_vel: Current joint state
 *  - actions: Previous control actions
 * 
 * @param name The name of the observation term to parse
 * @return true if the term was recognized and registered, false otherwise
 */
bool MotionTrackingController::parserObservation(const std::string& name) {
  // Try parsing standard observation terms first
  if (RlController::parserObservation(name)) {
    return true;
  }
  
  // Handle motion-specific observation terms
  if (name == "motion_ref_pos_b" || name == "motion_anchor_pos_b") {
    // Reference position of anchor body in local frame
    observationManager_->addTerm(std::make_shared<MotionAnchorPosition>(commandTerm_));
  } else if (name == "motion_ref_ori_b" || name == "motion_anchor_ori_b") {
    // Reference orientation of anchor body in local frame (6D representation)
    observationManager_->addTerm(std::make_shared<MotionAnchorOrientation>(commandTerm_));
  } else if (name == "robot_body_pos") {
    // Current positions of all tracked bodies in local frame
    observationManager_->addTerm(std::make_shared<RobotBodyPosition>(commandTerm_));
  } else if (name == "robot_body_ori") {
    // Current orientations of all tracked bodies in local frame (6D representation)
    observationManager_->addTerm(std::make_shared<RobotBodyOrientation>(commandTerm_));
  } else {
    return false;  // Unknown observation term
  }
  return true;
}

}  // namespace legged

// Export the controller as a pluginlib plugin
// This allows ros2_control to dynamically load the controller
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)