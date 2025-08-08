#ifndef SIMPLE_PURE_PURSUIT_HPP_
#define SIMPLE_PURE_PURSUIT_HPP_

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64.hpp"

namespace simple_pure_pursuit {

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PointStamped;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;

class SimplePurePursuit : public rclcpp::Node {
 public:
  explicit SimplePurePursuit();
  
  // subscribers
  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  
  // publishers
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_raw_cmd_;
  rclcpp::Publisher<PointStamped>::SharedPtr pub_lookahead_point_;  

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_controller_;  
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_out_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_pure_pursuit_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_stanley_e_cte_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_stanley_e_heading_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_stanley_steer_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_pid_output_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_pid_kp_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_pid_ki_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_steer_pid_kd_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_data1_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_data2_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_data3_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_data4_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_debug_msg_cmd_data5_;

  // timer
  rclcpp::TimerBase::SharedPtr timer_;

  // updated by subscribers
  Trajectory::SharedPtr trajectory_;
  Odometry::SharedPtr odometry_;



  // pure pursuit parameters
  const double wheel_base_;
  const double lookahead_gain_;
  const double lookahead_min_distance_;
  const double speed_proportional_gain_;
  const bool use_external_target_vel_;
  const double external_target_vel_;
  const double steering_tire_angle_gain_;


 private:
  void onTimer();
  bool subscribeMessageAvailable();
};

}  // namespace simple_pure_pursuit

#endif  // SIMPLE_PURE_PURSUIT_HPP_
