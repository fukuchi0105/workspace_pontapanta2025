#include "simple_pure_pursuit/simple_pure_pursuit.hpp"
#include "std_msgs/msg/float64.hpp"

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
    // stanley_gain_(declare_parameter<float>("stanley_gain", 1.0))
{
    // Publisherの設定
    pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);    // 制御用Ackermannコマンド
    pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);    // 生のAckermannコマンド
    pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1); // デバッグ用の先行点可視化

    pub_debug_controller_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/controller", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_out_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_out", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_pure_pursuit_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_pure_pursuit", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_stanley_e_cte_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_stanley_e_cte", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_stanley_e_heading_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_stanley_e_heading", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_stanley_steer_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_stanley_steer", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_pid_output_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_pid_output", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_pid_kp_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_pid_kp", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_pid_ki_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_pid_ki", rclcpp::QoS(1));
    pub_debug_msg_cmd_steer_pid_kd_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/steer_pid_kd", rclcpp::QoS(1));
    pub_debug_msg_cmd_data1_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/data1", rclcpp::QoS(1));
    pub_debug_msg_cmd_data2_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/data2", rclcpp::QoS(1));
    pub_debug_msg_cmd_data3_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/data3", rclcpp::QoS(1));
    pub_debug_msg_cmd_data4_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/data4", rclcpp::QoS(1));
    pub_debug_msg_cmd_data5_ = create_publisher<std_msgs::msg::Float64>("/debug/cmd/data5", rclcpp::QoS(1));

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

// 角度差を [-π, π] の範囲に正規化する関数
double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double calcCurvature(geometry_msgs::msg::Point &p0, geometry_msgs::msg::Point &p1, geometry_msgs::msg::Point &p2)
{
    double a = std::hypot(p1.x - p0.x, p1.y - p0.y);
    double b = std::hypot(p2.x - p1.x, p2.y - p1.y);
    double c = std::hypot(p2.x - p0.x, p2.y - p0.y);
    
    // 退避（点が重なる/極端に近い場合）
    constexpr double kEps = 1e-6;
    if (a < kEps || b < kEps || c < kEps) {
        return 0.0;
    }

    // 符号付き 2*三角形面積（z方向クロス積）
    double z = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    
    // 曲率 = 4A / (abc) = 2*z / (a*b*c)  （z は 2A の符号付き）
    double denom = a * b * c;
    
    if (std::fabs(denom) < kEps) {
        return 0.0;
    }

    return (2.0 * z) / denom;
}

// メイン制御ループ (タイマー動作関数)
void SimplePurePursuit::onTimer()
{
    // LPFパラメータ
    double lpf_a0 = 0.3858695451;
    double lpf_a1 = 0.3858695451;
    double lpf_a2 = 0.2282609098;
    static double y_prev_lpf = 0;
    static double x_prev_lpf = 0;

    static int lap_count = 0;
    static int goal_reached_prev = 0;

    int ignore_points_garage = 100;

    // Debugパラメータ 初期化
    std_msgs::msg::Float64 debug_msg_steer_controller;
    std_msgs::msg::Float64 debug_msg_cmd_steer_out;
    std_msgs::msg::Float64 debug_msg_cmd_steer_pure_pursuit;
    std_msgs::msg::Float64 debug_msg_cmd_steer_stanley_e_cte;
    std_msgs::msg::Float64 debug_msg_cmd_steer_stanley_e_heading;
    std_msgs::msg::Float64 debug_msg_cmd_steer_stanley_steer;
    std_msgs::msg::Float64 debug_msg_cmd_steer_pid_output;
    std_msgs::msg::Float64 debug_msg_cmd_steer_pid_kp;
    std_msgs::msg::Float64 debug_msg_cmd_steer_pid_ki;
    std_msgs::msg::Float64 debug_msg_cmd_steer_pid_kd;
    std_msgs::msg::Float64 debug_msg_cmd_data1;
    std_msgs::msg::Float64 debug_msg_cmd_data2;
    std_msgs::msg::Float64 debug_msg_cmd_data3;
    std_msgs::msg::Float64 debug_msg_cmd_data4;
    std_msgs::msg::Float64 debug_msg_cmd_data5;

    // Loopパラメータ
    static long int loop_counter = 0;

    // PID制御パラメータ
    // double pid_kp = 0.25;
    // double pid_ki = 0.0001;
    // double pid_kd = 0.025;
    double pid_kp = 0.2;
    double pid_ki = 0.001;
    double pid_kd = 0.0;
    // double pid_kd = 0.05;
    static double pid_integ = 0.0;
    static double pid_prev_error = 0.0;
    
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

        // --- 周回カウント ---
        if (!goal_reached_prev) {   // 前回未到達ならカウント
          lap_count++;
          // RCLCPP_INFO(get_logger(), "Lap count: %d", lap_count_);
          goal_reached_prev = 1;  // ゴール済みにセット
        }

    // ゴールでなければ、制御動作
    } else {
        goal_reached_prev = false;

        // 現在車速の取得
        double current_longitudinal_vel = odometry_->twist.twist.linear.x;

        // 最近点の目標値を取得
        TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

        // 最近点の曲率を取得
        int rad_cal_diff = 1; 
        int lookahead_curvature = static_cast<int>(current_longitudinal_vel * 0.5); //遅れ時間×車速(mps)で、ターゲットの距離を見る
        int last_idx = static_cast<int>(trajectory_->points.size()) - 1;

        int i0 = closet_traj_point_idx + lookahead_curvature + 4;
        // int i0 = std::min(closet_traj_point_idx + lookahead_curvature, last_idx);
        int i1 = std::min(i0 + rad_cal_diff, last_idx);
        int i2 = std::min(i1 + rad_cal_diff, last_idx);

        auto &p0 = trajectory_->points[i0].pose.position;
        auto &p1 = trajectory_->points[i1].pose.position;
        auto &p2 = trajectory_->points[i2].pose.position;

        double curvature = calcCurvature(p0, p1, p2);

        // 目標速度値の計算 (外部設定値 or Trajectoryの設定値)
        double target_longitudinal_vel;
        if (use_external_target_vel_) {
            target_longitudinal_vel = external_target_vel_;
        } else {
            target_longitudinal_vel = closet_traj_point.longitudinal_velocity_mps;
        }

        // 目標加速度の計算 (目標速度 - 現在速度) × ゲイン
        cmd.longitudinal.speed = target_longitudinal_vel;
        cmd.longitudinal.acceleration =
            speed_proportional_gain_ * (target_longitudinal_vel - current_longitudinal_vel);

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        // 操舵量の計算
        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // 先行距離の計算
        double lookahead_distance = lookahead_gain_ * target_longitudinal_vel + lookahead_min_distance_;

        // リアの車軸中心座標とヨー角を計算
        double rear_x = odometry_->pose.pose.position.x -
                                        wheel_base_ / 2.0 * std::cos(odometry_->pose.pose.orientation.z);
        double rear_y = odometry_->pose.pose.position.y -
                                        wheel_base_ / 2.0 * std::sin(odometry_->pose.pose.orientation.z);
        double yaw = tf2::getYaw(odometry_->pose.pose.orientation);

        // Trajectoryポイントを探索 
        auto lookahead_point_itr = std::find_if(
            trajectory_->points.begin() + closet_traj_point_idx, trajectory_->points.end(),
            [&](const TrajectoryPoint & point) {
                return std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >= lookahead_distance;
            });

        // Trajectoryポイントが最終点似到達している場合は、スタートから再計算し直す。
        if (lookahead_point_itr == trajectory_->points.end()) {
            // lookahead_point_itr = trajectory_->points.end() - 1;
            // lookahead_point_itr = trajectory_->points.begin() + 20;
            lookahead_point_itr = std::find_if(
            trajectory_->points.begin() + ignore_points_garage, trajectory_->points.end(), // ガレージ走行ラインの軌跡は無視する
            [&](const TrajectoryPoint & point) {
                return std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >= lookahead_distance;
            });
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

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        // 操舵角の計算
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        
        // ① Pure Pursuit計算
        double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) - yaw;
        
        double pure_pursuit_steer =
            steering_tire_angle_gain_ * std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);

        // PID制御
        double error = normalizeAngle(alpha);  
        double const dt = 0.01;

        pid_integ += error * dt;
        double derivative = (error - pid_prev_error) / dt;
        double pid_output = pid_kp * error + pid_ki * pid_integ + pid_kd * derivative; 
        pid_prev_error = error;


        // ② Stanley制御計算
        double traj_yaw = tf2::getYaw(closet_traj_point.pose.orientation);

        // 横ずれ誤差 e_cte (車両後軸基準)
        // double e_cte = calcLateralDeviation(closet_traj_point.pose, odometry_->pose.pose.position);
        double e_cte = std::pow(closet_traj_point.pose.position.x - odometry_->pose.pose.position.x, 2);
        e_cte += std::pow(closet_traj_point.pose.position.y - odometry_->pose.pose.position.y, 2);
        e_cte = std::sqrt(e_cte);

        // ヘディング誤差 e_heading
        double e_heading = normalizeAngle(traj_yaw - yaw);

        // Stanleyの操舵角 (横ずれ誤差 + ヘディング誤差)
        double stanley_gain_ = 0.3;
        double velocity_soft = 0.1;
        double stanley_steer = normalizeAngle(e_heading + std::atan2(stanley_gain_ * e_cte, current_longitudinal_vel + velocity_soft));
      
        // 制御量の決定
        // double combined_steer = 1.0 * pure_pursuit_steer + (0.0) * stanley_steer + (0.0) * pid_output + (1.0) * curvature;  // 重み付け可
        // double combined_pure_pursuit_gain = 0.50;
        // double combined_curvature_gain = 1.6;
        // double combined_eheading_gain = 0.3;
        // double combined_e_cte_gain = 0.05;

        // Best (Not Complete)
        // double combined_pure_pursuit_gain = 0.65;
        // double combined_curvature_gain = 1.5;
        // double combined_eheading_gain = 0.25;
        // double combined_e_cte_gain = 0.05;
        
        // Working
        double combined_pure_pursuit_gain;
        double combined_curvature_gain;
        double combined_eheading_gain;
        double combined_e_cte_gain;
        int lookahead_index = std::distance(trajectory_->points.begin(), lookahead_point_itr);
        if (lookahead_index < ignore_points_garage) {
            combined_pure_pursuit_gain = 0.6;
            combined_curvature_gain = 1.5;
            combined_eheading_gain = 0.25;
            combined_e_cte_gain = 0.05;
        }
        else {
            combined_pure_pursuit_gain = 0.20;
            combined_curvature_gain = 1.85;
            combined_eheading_gain = 0.2;
            combined_e_cte_gain = 0.0;
        }
        // combined_pure_pursuit_gain = 0.25;
        // combined_curvature_gain = 1.85;
        // combined_eheading_gain = 0.0;
        // combined_e_cte_gain = 0.0;

        double combined_curvature_result = combined_curvature_gain * curvature;
        if (combined_curvature_result > 0.45 || combined_curvature_result < -0.45) {
            combined_curvature_result *= 2.0;
        }


        double combined_steer = combined_pure_pursuit_gain * pure_pursuit_steer
                                 + combined_curvature_result
                                 + combined_eheading_gain * e_heading
                                 + combined_e_cte_gain * e_cte;

        // LPF適用
        double calc_result_steer_lpf =
            combined_steer * lpf_a0 + x_prev_lpf * lpf_a1 + y_prev_lpf * lpf_a2;
        x_prev_lpf = combined_steer;
        y_prev_lpf = calc_result_steer_lpf;

        cmd.lateral.steering_tire_angle = calc_result_steer_lpf;
        // cmd.lateral.steering_tire_angle = combined_steer;

        // Original Pure Pursuit
        // double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) -
        //                              tf2::getYaw(odometry_->pose.pose.orientation);
        // double calc_result_steer =
        //     steering_tire_angle_gain_ * std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);

        // double calc_result_steer_lpf = calc_result_steer * lpf_a0 + x_prev_lpf * lpf_a1 + y_prev_lpf * lpf_a2;
        // x_prev_lpf = calc_result_steer;
        // y_prev_lpf = calc_result_steer_lpf;

        // cmd.lateral.steering_tire_angle = calc_result_steer_lpf;

        // 検証用 Debug出力
        debug_msg_cmd_steer_out.data = calc_result_steer_lpf;
        debug_msg_cmd_steer_pure_pursuit.data = pure_pursuit_steer;
        debug_msg_cmd_steer_stanley_e_cte.data = e_cte;
        debug_msg_cmd_steer_stanley_e_heading.data = e_heading;
        debug_msg_cmd_steer_stanley_steer.data = stanley_steer;
        debug_msg_cmd_steer_pid_output.data = pid_output;
        debug_msg_cmd_steer_pid_kp.data = pid_kp * error;
        debug_msg_cmd_steer_pid_ki.data = pid_ki * pid_integ;
        debug_msg_cmd_steer_pid_kd.data = pid_kd * derivative;
        debug_msg_steer_controller.data = pure_pursuit_steer;
        debug_msg_cmd_data1.data = current_longitudinal_vel;
        debug_msg_cmd_data2.data = combined_pure_pursuit_gain * pure_pursuit_steer;
        debug_msg_cmd_data3.data = combined_curvature_result;
        debug_msg_cmd_data4.data = combined_eheading_gain * e_heading;
        debug_msg_cmd_data5.data = combined_e_cte_gain * e_cte;


        pub_debug_msg_cmd_steer_out_->publish(debug_msg_cmd_steer_out);
        pub_debug_msg_cmd_steer_pure_pursuit_->publish(debug_msg_cmd_steer_pure_pursuit);
        pub_debug_msg_cmd_steer_stanley_e_cte_->publish(debug_msg_cmd_steer_stanley_e_cte);
        pub_debug_msg_cmd_steer_stanley_e_heading_->publish(debug_msg_cmd_steer_stanley_e_heading);
        pub_debug_msg_cmd_steer_stanley_steer_->publish(debug_msg_cmd_steer_stanley_steer);
        pub_debug_msg_cmd_steer_pid_output_->publish(debug_msg_cmd_steer_pid_output);
        pub_debug_msg_cmd_steer_pid_kp_->publish(debug_msg_cmd_steer_pid_kp);
        pub_debug_msg_cmd_steer_pid_ki_->publish(debug_msg_cmd_steer_pid_ki);
        pub_debug_msg_cmd_steer_pid_kd_->publish(debug_msg_cmd_steer_pid_kd);
        pub_debug_msg_cmd_data1_->publish(debug_msg_cmd_data1);
        pub_debug_msg_cmd_data2_->publish(debug_msg_cmd_data2);
        pub_debug_msg_cmd_data3_->publish(debug_msg_cmd_data3);
        pub_debug_msg_cmd_data4_->publish(debug_msg_cmd_data4);
        pub_debug_msg_cmd_data5_->publish(debug_msg_cmd_data5);
 
        pub_debug_controller_->publish(debug_msg_steer_controller);


    }

    // 結果の出力
    pub_cmd_->publish(cmd);
    // cmd.lateral.steering_tire_angle /= steering_tire_angle_gain_;
    pub_raw_cmd_->publish(cmd);
    

    // Debug データの出力
    // if (loop_counter % 10 == 0) {
    //     // RCLCPP_INFO(get_logger(), "SteerAng: %.2f, PurePur: %.2f, PID: %.2f, Kp: %.3f, Ki: %.2f, Kd: %.2f\n",
    //     //        calc_result_steer_lpf, pure_pursuit_steer, pid_output, error, pid_integ, derivative);

    //     RCLCPP_INFO(get_logger(), "SteerAng:\n", calc_result_steer_lpf);
    // }

    loop_counter++;
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
