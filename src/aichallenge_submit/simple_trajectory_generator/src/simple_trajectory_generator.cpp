// Copyright 2023 Tier IV, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
using TrajectoryPoint = autoware_auto_planning_msgs::msg::TrajectoryPoint;

class CSVToTrajectory : public rclcpp::Node
{
public:
  CSVToTrajectory() : Node("csv_to_trajectory_node")
  {
    resample_interval_m_ = declare_parameter<double>("resample_interval_m", 1.0);

    const auto rb_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    pub_ = this->create_publisher<Trajectory>("trajectory", rb_qos);
    set_parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CSVToTrajectory::on_parameter_event, this, std::placeholders::_1));


    declare_parameter("csv_path", "");
    z_= declare_parameter<float>("z");
    std::string csv_path = get_parameter("csv_path").as_string();
    
    if (csv_path.empty()) {
      RCLCPP_ERROR(get_logger(), "CSV path is not specified");
      return;
    }
    
    if (!loadCSVTrajectory(csv_path)) {
      RCLCPP_ERROR(get_logger(), "Failed to load CSV file: %s", csv_path.c_str());
      return;
    }

    // write csv trajectory as csv
    if (true) {
      std::ofstream out_file("/aichallenge/workspace/src/aichallenge_submit/simple_trajectory_generator/data/trajectory_output.csv");
      // std::ofstream out_file("../data/trajectory_output.csv");
      if (out_file.is_open()) {
        out_file << "x,y,z,orientation_x,orientation_y,orientation_z,orientation_w,curvature\n";
        for (const auto & point : csv_trajectory_.points) {
          out_file << point.pose.position.x << ","
                   << point.pose.position.y << ","
                   << point.pose.position.z << ","
                   << point.pose.orientation.x << ","
                   << point.pose.orientation.y << ","
                   << point.pose.orientation.z << ","
                   << point.pose.orientation.w << ","
                   << point.heading_rate_rps << "\n";
        }
        out_file.close();
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to open output file for writing");
      }
    }
    
    RCLCPP_INFO(get_logger(), "Loaded trajectory from CSV with %zu points", csv_trajectory_.points.size());

    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&CSVToTrajectory::publish_trajectory, this));

  }

private:
  bool loadCSVTrajectory(const std::string & csv_path)
  {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      return false;
    }
    
    std::string line;
    std::getline(file, line);
    
    csv_trajectory_.header.stamp = this->now();
    csv_trajectory_.header.frame_id = "map";

    csv_trajectory_.points.clear();
    
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string token;
      std::vector<double> values;
      
      while (std::getline(ss, token, ',')) {
        values.push_back(std::stod(token));
      }
      
      if (values.size() != 8) {
        RCLCPP_WARN(get_logger(), "Invalid CSV line format, expected 8 values");
        continue;
      }
      
      TrajectoryPoint point;
      point.pose.position.x = values[0];
      point.pose.position.y = values[1];
      point.pose.position.z = z_;

      point.pose.orientation.x = values[3];
      point.pose.orientation.y = values[4];
      point.pose.orientation.z = values[5];
      point.pose.orientation.w = values[6];
      
      point.longitudinal_velocity_mps = values[7];
      
      point.lateral_velocity_mps = 0.0;
      point.acceleration_mps2 = 0.0;
      point.heading_rate_rps = 0.0;
      
      csv_trajectory_.points.push_back(point);
    }

    //resample trajectory
    auto orig = csv_trajectory_.points;
    // csv_trajectory_.points = resampleTrajectory(orig);
    auto resampled = resampleTrajectory(csv_trajectory_);

    csv_trajectory_.points.assign(resampled.begin(), resampled.end());
    
    for (size_t i = 1; i < csv_trajectory_.points.size() - 1; ++i) {
      const auto & prev_pt = csv_trajectory_.points[i - 1];
      auto & curr_pt = csv_trajectory_.points[i];
      const auto & next_pt = csv_trajectory_.points[i + 1];

      // ３点 prev_pt, curr_pt, next_pt を使って曲率 κ を近似
      double dx1 = curr_pt.pose.position.x - prev_pt.pose.position.x;
      double dy1 = curr_pt.pose.position.y - prev_pt.pose.position.y;
      double dx2 = next_pt.pose.position.x - curr_pt.pose.position.x;
      double dy2 = next_pt.pose.position.y - curr_pt.pose.position.y;
      
      double angle1 = std::atan2(dy1, dx1);
      double angle2 = std::atan2(dy2, dx2);
      double dtheta = angle2 - angle1;
      // ── 差を [-π,π] に正規化 ──
      while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
      while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

      double dist1 = std::hypot(dx1, dy1);
      double dist2 = std::hypot(dx2, dy2);
      double ds = 0.5 * (dist1 + dist2); // 平均距離

      double curvature = (ds > 1e-6 ? dtheta / ds : 0.0);

      curr_pt.heading_rate_rps = curvature; // Set to zero for now, can be computed later if needed
      curr_pt.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), 
        angle1)); // Update orientation based on curvature
    }
    csv_trajectory_.points[0].heading_rate_rps = 0.0; // First point has no curvature
    csv_trajectory_.points.back().heading_rate_rps = 0.0; // Last point has no curvature
    csv_trajectory_.points[0].pose.orientation = csv_trajectory_.points[1].pose.orientation; // First point orientation
    csv_trajectory_.points.back().pose.orientation = csv_trajectory_.points[csv_trajectory_.points.size() - 2].pose.orientation; // Last point orientation

    //simple low pas filter for heading_rate_rps
    std::vector<double> heading_rates_filtered(csv_trajectory_.points.size(), 0.0);
    for (size_t i = 1; i < csv_trajectory_.points.size() - 1; ++i) {
      const auto & next_pt = csv_trajectory_.points[i + 1];
      const auto & curr_pt = csv_trajectory_.points[i];
      const auto & prev_pt = csv_trajectory_.points[i - 1];
      // 平均をとる
      heading_rates_filtered[i] = (prev_pt.heading_rate_rps + curr_pt.heading_rate_rps + next_pt.heading_rate_rps) / 3.0;
    }
    for (size_t i = 0; i < csv_trajectory_.points.size(); ++i) {
      csv_trajectory_.points[i].heading_rate_rps = heading_rates_filtered[i];
    }

    return !csv_trajectory_.points.empty();
  }
  
  void publish_trajectory()
  {
    if (csv_trajectory_.points.empty()) {
      RCLCPP_WARN(get_logger(), "No trajectory points to publish");
      return;
    }
    
    csv_trajectory_.header.stamp = this->now();
    pub_->publish(csv_trajectory_);
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(), 60000 /*ms*/, "Published trajectory with %zu points", csv_trajectory_.points.size());
  }

  rcl_interfaces::msg::SetParametersResult on_parameter_event(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "";

    for (const auto & param : parameters) {
      if (param.get_name() == "csv_path") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
          std::string new_csv_path = param.as_string();
          // new_csv_pathがFileSystemのパスであることを確認
          if (!std::filesystem::exists(new_csv_path)) {
            RCLCPP_ERROR(get_logger(), "File does not exist: '%s'", new_csv_path.c_str());
            result.successful = false;
            result.reason = "File does not exist.";
            continue;
          }

          if (new_csv_path != current_csv_path_) {
            RCLCPP_INFO(get_logger(), "csv_path parameter changed from '%s' to '%s'", 
                        current_csv_path_.c_str(), new_csv_path.c_str());
            
            // 新しいCSVファイルの読み込みを試みる
            if (loadCSVTrajectory(new_csv_path)) {
              current_csv_path_ = new_csv_path;
              RCLCPP_INFO(get_logger(), "Successfully loaded new trajectory from CSV: %s with %zu points", 
                          current_csv_path_.c_str(), csv_trajectory_.points.size());
            } else {
              RCLCPP_ERROR(get_logger(), "Failed to load new CSV file: %s. Keeping old trajectory.", new_csv_path.c_str());
              result.successful = false;
              result.reason = "Failed to load new CSV file.";
            }
          }
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'csv_path' received with wrong type. Expected string.");
          result.successful = false;
          result.reason = "Invalid type for csv_path parameter.";
        }
      } else if (param.get_name() == "z") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE || param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          z_ = static_cast<float>(param.as_double());
          RCLCPP_INFO(get_logger(), "z parameter changed to %f", z_);
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'z' received with wrong type. Expected float/double.");
          result.successful = false;
          result.reason = "Invalid type for z parameter.";
        }
      }
    }
    return result;
  }
  // std::vector<TrajectoryPoint> resampleTrajectory(
  //   const std::vector<TrajectoryPoint> & orig)
  std::vector<TrajectoryPoint> resampleTrajectory(
    const Trajectory & origTraj)
  {
    const auto & orig = origTraj.points;

    std::vector<TrajectoryPoint> out;
    if (orig.empty()) return out;

    out.push_back(orig.front());
    double accum_dist = 0.0;
    size_t idx = 0;

    auto current_pts = orig[idx];
    // セグメント毎に進みつつ、resample_interval_m_ 毎に点を追加
    while (idx + 1 < orig.size()) {
      const auto & p0 = current_pts;
      const auto & p1 = orig[idx+1];
      // セグメント長
      double dx = p1.pose.position.x - p0.pose.position.x;
      double dy = p1.pose.position.y - p0.pose.position.y;
      double seg_len = std::hypot(dx, dy);

      if (accum_dist + seg_len < resample_interval_m_) {
        // まだ次のサンプリング点はこのセグメント内に到達しない
        accum_dist += seg_len;
        idx++;
        current_pts = orig[idx];
      } else {
        // サンプリング点はこのセグメント内にある
        double remain = resample_interval_m_ - accum_dist;
        double t = remain / seg_len;  // 補間係数 [0,1]
        TrajectoryPoint np;
        // 位置線形補間
        np.pose.position.x = p0.pose.position.x + dx * t;
        np.pose.position.y = p0.pose.position.y + dy * t;
        np.pose.position.z = this->z_;
        // ヘディング線形補間（補正して範囲内に）
        double yaw0 = tf2::getYaw(p0.pose.orientation);
        double yaw1 = tf2::getYaw(p1.pose.orientation);
        double dyaw = yaw1 - yaw0;
        while (dyaw >  M_PI) dyaw -= 2.0*M_PI;
        while (dyaw < -M_PI) dyaw += 2.0*M_PI;
        double yaw = yaw0 + dyaw * t;
        np.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1), yaw));
        // 速度線形補間
        np.longitudinal_velocity_mps =
          p0.longitudinal_velocity_mps + (p1.longitudinal_velocity_mps - p0.longitudinal_velocity_mps) * t;
        // heading_rate_rps は後でまとめて計算するので一旦 0
        np.heading_rate_rps = 0.0;

        out.push_back(np);
        // 新しい基準点をこの内挿点に設定し、残り距離を 0 にリセット
        accum_dist = 0.0;

        current_pts = np; // 内挿点を次の基準点に設定
      }
    }

    // 最後の点を確実に含める
    out.push_back(orig.back());
    return out;
  }
  
  rclcpp::Publisher<Trajectory>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  Trajectory csv_trajectory_;
  float z_;
  std::string current_csv_path_;
  OnSetParametersCallbackHandle::SharedPtr set_parameter_callback_handle_;
  double resample_interval_m_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CSVToTrajectory>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
