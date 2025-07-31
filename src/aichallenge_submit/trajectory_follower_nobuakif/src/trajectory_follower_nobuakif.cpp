#include "trajectory_follower_nobuakif/trajectory_follower_nobuakif.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <tf2/utils.h>
#include <algorithm>

namespace trajectory_follower_nobuakif
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

TrajectoryFollower::TrajectoryFollower()
: Node("trajectory_follower_nobuakif"),
  // --- Parameter declarations ---
  // Vehicle model type: 0 = Kinematic, 1 = Dynamic
  vehicle_model_(declare_parameter<int>("vehicle_model", 0)),
  // Position error gains for Stanley term
  position_gain_forward_(declare_parameter<double>("position_gain_forward", 2.5)),
  position_gain_reverse_(declare_parameter<double>("position_gain_reverse", 2.5)),
  // Dynamic model feedback gains
  yaw_rate_feedback_gain_(declare_parameter<double>("yaw_rate_feedback_gain", 2.5)),
  steering_feedback_gain_(declare_parameter<double>("steering_feedback_gain", 2.5)),
  // Vehicle geometry parameters
  wheel_base_(declare_parameter<double>("wheelbase", 1.087)),                 // [m] wheelbase length
  lf_(declare_parameter<double>("dist_cm_to_front_axle", 0.54)),           // [m] center to front axle
  lr_(declare_parameter<double>("dist_cm_to_rear_axle", 0.54)),            // [m] center to rear axle
  // Dynamic bicycle parameters
  vehicle_mass_(declare_parameter<double>("vehicle_mass", 160.0)),       // [kg] vehicle mass
  front_corner_stiffness_(declare_parameter<double>("front_tire_corner_stiffness", 19000.0)), // [N/rad]
  // Control limits
  max_steering_angle_deg_(declare_parameter<double>("max_steering_angle_deg", 80.0)), // [deg]
  // Speed controller gain
  speed_proportional_gain_(declare_parameter<double>("speed_proportional_gain", 1.0)),
  // External speed option
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<double>("external_target_vel", 0.0)),
  prev_delta_(0.0) // Initialize previous steering angle
{
  // Publishers
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  // pub_debug_pt_ = create_publisher<PointStamped>("/control/debug/stanley_point", 1);
  pub_debug_pt_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);

  // Subscriptions
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  sub_kinematics_ = create_subscription<Odometry>("input/kinematics", qos,
    [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>("input/trajectory", qos,
    [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

  // Timer for control loop (100 Hz)
  timer_ = rclcpp::create_timer(
    this, get_clock(), std::chrono::milliseconds(10),
    std::bind(&TrajectoryFollower::onTimer, this));
}

// Helper: produce zeroed Ackermann command
AckermannControlCommand zeroAckermannControlCommand(rclcpp::Time stamp)
{
  AckermannControlCommand cmd;
  cmd.stamp = stamp;
  cmd.longitudinal.stamp = stamp;
  cmd.longitudinal.speed = 0.0;
  cmd.longitudinal.acceleration = 0.0;
  cmd.lateral.stamp = stamp;
  cmd.lateral.steering_tire_angle = 0.0;
  return cmd;
}

void TrajectoryFollower::onTimer()
{
  // Check input availability
  if (!subscribeMessageAvailable()) return;

  // 1) Find nearest trajectory point index
  size_t idx = findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

  // 2) Initialize zero command
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // 3) If at end of path, stop vehicle
  if (idx >= trajectory_->points.size() - 1 || trajectory_->points.size() <= 2) {
    cmd.longitudinal.speed = 0.0;
    cmd.longitudinal.acceleration = -10.0;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Reached goal. Stopping.");
    pub_cmd_->publish(cmd);
    pub_raw_cmd_->publish(cmd);
    return;
  }

  // 4) Retrieve reference point and current speed
  const auto &ref_pt = trajectory_->points[idx];
  double v_nominal = use_external_target_vel_ ? external_target_vel_ : ref_pt.longitudinal_velocity_mps;
  double v_current = odometry_->twist.twist.linear.x;

  // --- Longitudinal control (speed) ---
  // English: simple P control on speed
  // 日本語: 速度 P 制御
  cmd.longitudinal.speed = v_nominal;
  cmd.longitudinal.acceleration = speed_proportional_gain_ * (v_nominal - v_current);

  // --- Compute common errors for Stanley ---
  // English: lateral deviation (cross-track error)
  // 日本語: 横ずれ誤差 (クロストラック誤差)
  double cte = calcLateralDeviation(odometry_->pose.pose, ref_pt.pose.position);
  // English: heading error between vehicle and path tangent
  // 日本語: 車体向きと経路接線の誤差 (ヘディング誤差)
  double heading_err = calcYawDeviation(odometry_->pose.pose, ref_pt.pose);

  // Prevent division by zero
  double v_stanley = std::max(v_current, 0.0) + 1e-3;

  // --- Kinematic Stanley term ---
  // English: choose forward/reverse gain based on direction
  // 日本語: 進行方向に応じたゲイン選択
  double k_pos = (v_current >= 0 ? position_gain_forward_ : position_gain_reverse_);
  // English: stanley term = atan(k_pos * cte / v)
  // 日本語: Stanley 項 = atan(k_pos × 横ずれ / 速度)
  double stanley_term = std::atan2(k_pos * cte, v_stanley);

  // Base steering command (kinematic)
  // English: δ = heading error + stanley term
  // 日本語: δ = ヘディング誤差 + Stanley 項
  double delta = heading_err;// + stanley_term / 10;
  // delta = -delta;

  // --- Dynamic bicycle enhancements ---
  if (vehicle_model_ == 1) {
    // English: add yaw-rate feedback
    // 日本語: ヨーレートフィードバックを追加
    double psi_dot = odometry_->twist.twist.angular.z;
    delta += yaw_rate_feedback_gain_ * psi_dot;

    // English: add steering-angle feedback (prev - curr)
    // 日本語: ステア角フィードバック (前回コマンド - 現在角)
    double curr_steer = cmd.lateral.steering_tire_angle; // assume available sensor or last cmd
    delta += steering_feedback_gain_ * (prev_delta_ - curr_steer);

    // // English: optional feedforward from path curvature
    // // 日本語: 経路曲率によるフィードフォワード
    // if (ref_pt.curvature != 0.0) {
    //   double ff = std::atan(wheel_base_ * ref_pt.curvature);
    //   delta += ff;
    // }
  }
  RCLCPP_INFO(get_logger(), "CTE: %.3f, Heading Error: %.3f, Speed: %.2f\n",
        cte, heading_err, v_current);

  RCLCPP_INFO(get_logger(), "Stanley Term: %.3f, heading err: %.3f, tire angle: %.3f\n",
        stanley_term, heading_err, delta);

  // --- Saturate steering to actuator limits ---
  // English: clamp between ±max angle
  // 日本語: ±最大舵角で制限
  double max_rad = max_steering_angle_deg_ * M_PI / 180.0;
  delta = std::clamp(delta, -max_rad, max_rad);
  prev_delta_ = delta;

  // 5) Publish lateral command
  cmd.lateral.steering_tire_angle = delta;
  pub_cmd_->publish(cmd);

  // 6) Publish raw command (undo any gain scaling)
  pub_raw_cmd_->publish(cmd);

  // 7) Debug: publish the reference point being tracked
  PointStamped dbg;
  dbg.header.stamp = get_clock()->now();
  dbg.header.frame_id = "map";
  dbg.point = ref_pt.pose.position;
  pub_debug_pt_->publish(dbg);
}

bool TrajectoryFollower::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Odometry not available.");
    return false;
  }
  if (!trajectory_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Trajectory not available.");
    return false;
  }
  if (trajectory_->points.empty()) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Trajectory is empty.");
    return false;
  }
  return true;
}

} // namespace trajectory_follower_nobuakif

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<trajectory_follower_nobuakif::TrajectoryFollower>());
  rclcpp::shutdown();
  return 0;
}
