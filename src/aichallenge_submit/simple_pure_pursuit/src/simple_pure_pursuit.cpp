#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>

/*
[概要]
・幾何学ベースでの制御ロジック(前輪の操舵角と加減速コマンドを決定)
・速度が高いほど遠い目標点を設定


*/

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

// コンストラクタ
// ROSノードとして初期化
SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
    // initialize parameters
    wheel_base_(declare_parameter<float>("wheel_base", 2.14)),    // 車両ホイールベース
    lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)), // 速度依存の先行距離ゲイン
    lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)), // 最小先行距離
    speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)), // 速度制御の比例ゲイン
    use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),    // 外部速度を使用するか
    external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)), // 外部速度設定
    steering_tire_angle_gain_(declare_parameter<float>("steering_tire_angle_gain", 1.0))    // 操舵角ゲイン
{
    // Publisherの設定
    pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);    // 制御用Ackermannコマンド
    pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);    // 生のAckermannコマンド
    pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1); // デバッグ用の先行点可視化

    // Subscriberの設定
    const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    sub_kinematics_ = create_subscription<Odometry>(
        "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });    // Odometry購読
    sub_trajectory_ = create_subscription<Trajectory>(
        "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });    // Trajectory購読

    // 10msタイマー周期動作関数
    using namespace std::literals::chrono_literals;
    timer_ =
        rclcpp::create_timer(this, get_clock(), 5ms, std::bind(&SimplePurePursuit::onTimer, this));
}

// 完全停止コマンドを生成する関数 (初期化、ゴール到着時に動作)
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

// メイン制御ループ (タイマー動作関数)
void SimplePurePursuit::onTimer()
{
    double lpf_a0 = 0.3858695451;
    double lpf_a1 = 0.3858695451;
    double lpf_a2 = 0.2282609098;
    static double y_prev_lpf = 0;
    static double x_prev_lpf = 0;

    // check data
    if (!subscribeMessageAvailable()) {
        return;
    }

    // 現在位置に最も近いTrajectoryのインデックスを取得
    size_t closet_traj_point_idx =
        findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

    // publish zero command
    AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // 目標速度・加速度の計算
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // ゴール判定 (最後のポイントに到達 or ポイントのサイズが2以下)
    if (
        (closet_traj_point_idx == trajectory_->points.size() - 1) ||
        (trajectory_->points.size() <= 2)) {
        cmd.longitudinal.speed = 0.0;
        cmd.longitudinal.acceleration = -10.0;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "reached to the goal");

    // ゴールでなければ、制御動作
    } else {
        // 最近点の目標値を取得
        TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

        // 目標速度値の計算 (外部設定値 or Trajectoryの設定値)
        double target_longitudinal_vel;
        if (use_external_target_vel_) {
            target_longitudinal_vel = external_target_vel_;
        } else {
            target_longitudinal_vel = closet_traj_point.longitudinal_velocity_mps;
        }
        double current_longitudinal_vel = odometry_->twist.twist.linear.x;

        // 目標加速度の計算 (目標速度 - 現在速度) × ゲイン
        cmd.longitudinal.speed = target_longitudinal_vel;
        cmd.longitudinal.acceleration =
            speed_proportional_gain_ * (target_longitudinal_vel - current_longitudinal_vel);

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        // 操舵量の計算
        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // 先行距離の計算
        double lookahead_distance = lookahead_gain_ * target_longitudinal_vel + lookahead_min_distance_;

        // リアの車軸中心座標を計算
        double rear_x = odometry_->pose.pose.position.x -
                                        wheel_base_ / 2.0 * std::cos(odometry_->pose.pose.orientation.z);
        double rear_y = odometry_->pose.pose.position.y -
                                        wheel_base_ / 2.0 * std::sin(odometry_->pose.pose.orientation.z);

        // 先行距離以上のTrajectoryポイントを探索 (見つからなければ、最終点を使用)
        auto lookahead_point_itr = std::find_if(
            trajectory_->points.begin() + closet_traj_point_idx, trajectory_->points.end(),
            [&](const TrajectoryPoint & point) {
                return std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >=
                             lookahead_distance;
            });
        if (lookahead_point_itr == trajectory_->points.end()) {
            lookahead_point_itr = trajectory_->points.end() - 1;
        }
        double lookahead_point_x = lookahead_point_itr->pose.position.x;
        double lookahead_point_y = lookahead_point_itr->pose.position.y;

        // デバッグ用出力
        geometry_msgs::msg::PointStamped lookahead_point_msg;
        lookahead_point_msg.header.stamp = get_clock()->now();
        lookahead_point_msg.header.frame_id = "map";
        lookahead_point_msg.point.x = lookahead_point_x;
        lookahead_point_msg.point.y = lookahead_point_y;
        lookahead_point_msg.point.z = closet_traj_point.pose.position.z;
        pub_lookahead_point_->publish(lookahead_point_msg);

        // 操舵角の計算
        double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) -
                                     tf2::getYaw(odometry_->pose.pose.orientation);
        double calc_result_steer =
            steering_tire_angle_gain_ * std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);

        double calc_result_steer_lpf = calc_result_steer * lpf_a0 + x_prev_lpf * lpf_a1 + y_prev_lpf * lpf_a2;
        x_prev_lpf = calc_result_steer;
        y_prev_lpf = calc_result_steer_lpf;

        cmd.lateral.steering_tire_angle = calc_result_steer_lpf;
    }

    // 結果の出力
    pub_cmd_->publish(cmd);
    // cmd.lateral.steering_tire_angle /=    steering_tire_angle_gain_;
    pub_raw_cmd_->publish(cmd);
}

// Odometry、Trajectoryのデータ有無を確認
bool SimplePurePursuit::subscribeMessageAvailable()
{
    if (!odometry_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
        return false;
    }
    if (!trajectory_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "trajectory is not available");
        return false;
    }
    if (trajectory_->points.empty()) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/,    "trajectory points is empty");
            return false;
        }
    return true;
}
}    // namespace simple_pure_pursuit

// ROS2 エンドポイント。 ノードを起動して待機
int main(int argc, char const * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<simple_pure_pursuit::SimplePurePursuit>());
    rclcpp::shutdown();
    return 0;
}
