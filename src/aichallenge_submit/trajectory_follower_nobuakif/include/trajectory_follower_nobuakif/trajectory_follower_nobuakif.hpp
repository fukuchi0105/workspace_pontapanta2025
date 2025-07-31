#ifndef TRAJECTORY_FOLLOWER_NOBUAKIF_HPP_
#define TRAJECTORY_FOLLOWER_NOBUAKIF_HPP_

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>

namespace trajectory_follower_nobuakif {

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PointStamped;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;

class TrajectoryFollower : public rclcpp::Node {
 public:
  explicit TrajectoryFollower();

  // subscribers
  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  
  // publishers
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_raw_cmd_;
  rclcpp::Publisher<PointStamped>::SharedPtr pub_lookahead_point_;
  rclcpp::Publisher<PointStamped>::SharedPtr pub_debug_pt_; // publisher for debug point

  // timer
  rclcpp::TimerBase::SharedPtr timer_;

  // updated by subscribers
  Trajectory::SharedPtr trajectory_;
  Odometry::SharedPtr odometry_;

  // stanley control parameters
  const int vehicle_model_; // 0 = Kinematic, 1 = Dynamic
  const double position_gain_forward_;
  const double position_gain_reverse_;
  const double yaw_rate_feedback_gain_;
  const double steering_feedback_gain_;
  const double wheel_base_;
  const double lf_; // distance from center to front axle
  const double lr_; // distance from center to rear axle
  const double vehicle_mass_; // vehicle mass
  const double front_corner_stiffness_; // front tire corner stiffness
  const double max_steering_angle_deg_; // maximum steering angle in degrees
  const double speed_proportional_gain_;
  const bool use_external_target_vel_;
  const double external_target_vel_;

  double prev_delta_; // previous steering angle
  // // pure pursuit parameters
  // const double lookahead_distance_;
  // const double lookahead_gain_;
  // const double lookahead_min_distance_;
  // const double steering_tire_angle_gain_;


 private:
  void onTimer();
  bool subscribeMessageAvailable();
};

}  // namespace trajectory_follower_nobuakif

#endif  // TRAJECTORY_FOLLOWER_NOBUAKIF_HPP_
