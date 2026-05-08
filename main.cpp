// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FAR Planner native module for dimos NativeModule framework.
//
// Replaces the ROS2 FARMaster node with CLI arg parsing + LCM pub/sub.
// The algorithm modules (ContourDetector, DynamicGraph, GraphPlanner,
// ContourGraph, MapHandler, ScanHandler) are called directly — FARMaster
// is bypassed because it is too tightly coupled to ROS.
//
// Usage:
//   ./far_planner_native \
//       --terrain_map_ext '/terrain_map_ext#sensor_msgs.PointCloud2' \
//       --terrain_map '/terrain_map#sensor_msgs.PointCloud2' \
//       --registered_scan '/registered_scan#sensor_msgs.PointCloud2' \
//       --odometry '/odometry#nav_msgs.Odometry' \
//       --goal '/goal#geometry_msgs.PointStamped' \
//       --way_point '/way_point#geometry_msgs.PointStamped' \
//       --goal_path '/goal_path#nav_msgs.Path' \
//       --update_rate 5.0 --sensor_range 15.0

#include <lcm/lcm-cpp.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// dimos-lcm message types
#include "sensor_msgs/PointCloud2.hpp"
#include "sensor_msgs/PointField.hpp"
#include "nav_msgs/Odometry.hpp"
#include "geometry_msgs/PointStamped.hpp"
#include "std_msgs/Bool.hpp"
#include "nav_msgs/Path.hpp"
#include "std_msgs/Header.hpp"

// SmartNav common helpers
#include "dimos_native_module.hpp"
#include "point_cloud_utils.hpp"

// FAR Planner core (via compat layer — no ROS deps)
#include "far_planner/compat.h"
#include "far_planner/utility.h"
#include "far_planner/contour_detector.h"
#include "far_planner/dynamic_graph.h"
#include "far_planner/graph_planner.h"
#include "far_planner/contour_graph.h"
#include "far_planner/map_handler.h"
#include "far_planner/scan_handler.h"

// ---------------------------------------------------------------------------
// FARUtil static definitions (normally in far_planner.cpp — we bypass that file)
// ---------------------------------------------------------------------------

PointCloudPtr  FARUtil::surround_obs_cloud_  = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::surround_free_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::stack_new_cloud_     = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::cur_new_cloud_       = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::cur_dyobs_cloud_     = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::stack_dyobs_cloud_   = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::cur_scan_cloud_      = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::local_terrain_obs_   = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointCloudPtr  FARUtil::local_terrain_free_  = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
PointKdTreePtr FARUtil::kdtree_new_cloud_    = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
PointKdTreePtr FARUtil::kdtree_filter_cloud_ = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());

const float FARUtil::kEpsilon = 1e-7f;
const float FARUtil::kINF     = std::numeric_limits<float>::max();
std::string FARUtil::worldFrameId;
float   FARUtil::kAngleNoise;
Point3D FARUtil::robot_pos;
Point3D FARUtil::odom_pos;
Point3D FARUtil::map_origin;
Point3D FARUtil::free_odom_p;
float   FARUtil::robot_dim;
float   FARUtil::vehicle_height;
float   FARUtil::kLeafSize;
float   FARUtil::kHeightVoxel;
float   FARUtil::kNavClearDist;
float   FARUtil::kCellLength;
float   FARUtil::kCellHeight;
float   FARUtil::kNewPIThred;
float   FARUtil::kSensorRange;
float   FARUtil::kMarginDist;
float   FARUtil::kMarginHeight;
float   FARUtil::kTerrainRange;
float   FARUtil::kLocalPlanRange;
float   FARUtil::kFreeZ;
float   FARUtil::kVizRatio;
double  FARUtil::systemStartTime;
float   FARUtil::kObsDecayTime;
float   FARUtil::kNewDecayTime;
float   FARUtil::kNearDist;
float   FARUtil::kMatchDist;
float   FARUtil::kProjectDist;
int     FARUtil::kDyObsThred;
int     FARUtil::KNewPointC;
int     FARUtil::kObsInflate;
float   FARUtil::kTolerZ;
float   FARUtil::kAcceptAlign;
bool    FARUtil::IsStaticEnv;
bool    FARUtil::IsDebug;
bool    FARUtil::IsMultiLayer;
TimeMeasure FARUtil::Timer;

// DynamicGraph statics
DynamicGraphParams DynamicGraph::dg_params_;
NodePtrStack DynamicGraph::globalGraphNodes_;
std::size_t  DynamicGraph::id_tracker_;
std::unordered_map<std::size_t, NavNodePtr> DynamicGraph::idx_node_map_;
std::unordered_map<NavNodePtr, std::pair<int, std::unordered_set<NavNodePtr>>> DynamicGraph::out_contour_nodes_map_;

// ContourGraph statics
CTNodeStack ContourGraph::polys_ctnodes_;
CTNodeStack ContourGraph::contour_graph_;
PolygonStack ContourGraph::contour_polygons_;
std::vector<PointPair> ContourGraph::global_contour_;
std::vector<PointPair> ContourGraph::unmatched_contour_;
std::vector<PointPair> ContourGraph::inactive_contour_;
std::vector<PointPair> ContourGraph::boundary_contour_;
std::vector<PointPair> ContourGraph::local_boundary_;
std::unordered_set<NavEdge, navedge_hash> ContourGraph::global_contour_set_;
std::unordered_set<NavEdge, navedge_hash> ContourGraph::boundary_contour_set_;

// MapHandler statics
PointKdTreePtr MapHandler::kdtree_terrain_clould_;
std::vector<int> MapHandler::terrain_grid_occupy_list_;
std::vector<int> MapHandler::terrain_grid_traverse_list_;
std::unordered_set<int> MapHandler::neighbor_obs_indices_;
std::unordered_set<int> MapHandler::extend_obs_indices_;
std::unique_ptr<grid_ns::Grid<PointCloudPtr>> MapHandler::world_free_cloud_grid_;
std::unique_ptr<grid_ns::Grid<PointCloudPtr>> MapHandler::world_obs_cloud_grid_;
std::unique_ptr<grid_ns::Grid<std::vector<float>>> MapHandler::terrain_height_grid_;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};
static lcm::LCM* g_lcm = nullptr;

// Topic strings (set from CLI args)
static std::string g_terrain_map_ext_topic;
static std::string g_terrain_map_topic;
static std::string g_registered_scan_topic;
static std::string g_odometry_topic;
static std::string g_goal_topic;
static std::string g_stop_movement_topic;
static std::string g_way_point_topic;
static std::string g_goal_path_topic;

// World frame for published messages
static std::string g_world_frame = "map";
static float g_wp_churn_dist = 0.0f;  // 0 = publish every cycle (no churn reduction)

// Thread-safe input buffers: odometry
static std::mutex g_odom_mutex;
static nav_msgs::Odometry g_latest_odom;
static bool g_has_odom = false;

// Thread-safe input buffers: terrain_map_ext
static std::mutex g_terrain_ext_mutex;
static sensor_msgs::PointCloud2 g_latest_terrain_ext;
static bool g_has_terrain_ext = false;

// Thread-safe input buffers: terrain_map
static std::mutex g_terrain_mutex;
static sensor_msgs::PointCloud2 g_latest_terrain;
static bool g_has_terrain = false;

// Thread-safe input buffers: registered_scan
static std::mutex g_scan_mutex;
static sensor_msgs::PointCloud2 g_latest_scan;
static bool g_has_scan = false;

// Thread-safe input buffers: goal
static std::mutex g_goal_mutex;
static geometry_msgs::PointStamped g_latest_goal;
static bool g_has_goal = false;
static bool g_goal_consumed = true;  // tracks whether current goal was processed
static bool g_goal_cancelled = false; // NaN goal = cancel navigation

// ---------------------------------------------------------------------------
// LCM callback handlers
// ---------------------------------------------------------------------------

class Handlers {
public:
    void on_odometry(const lcm::ReceiveBuffer* /*rbuf*/,
                     const std::string& /*channel*/,
                     const nav_msgs::Odometry* msg) {
        std::lock_guard<std::mutex> lock(g_odom_mutex);
        g_latest_odom = *msg;
        g_has_odom = true;
    }

    void on_terrain_ext(const lcm::ReceiveBuffer* /*rbuf*/,
                        const std::string& /*channel*/,
                        const sensor_msgs::PointCloud2* msg) {
        std::lock_guard<std::mutex> lock(g_terrain_ext_mutex);
        g_latest_terrain_ext = *msg;
        g_has_terrain_ext = true;
    }

    void on_terrain(const lcm::ReceiveBuffer* /*rbuf*/,
                    const std::string& /*channel*/,
                    const sensor_msgs::PointCloud2* msg) {
        std::lock_guard<std::mutex> lock(g_terrain_mutex);
        g_latest_terrain = *msg;
        g_has_terrain = true;
    }

    void on_registered_scan(const lcm::ReceiveBuffer* /*rbuf*/,
                            const std::string& /*channel*/,
                            const sensor_msgs::PointCloud2* msg) {
        std::lock_guard<std::mutex> lock(g_scan_mutex);
        g_latest_scan = *msg;
        g_has_scan = true;
    }

    void on_goal(const lcm::ReceiveBuffer* /*rbuf*/,
                 const std::string& /*channel*/,
                 const geometry_msgs::PointStamped* msg) {
        std::lock_guard<std::mutex> lock(g_goal_mutex);
        // NaN sentinel = cancel navigation (e.g. teleop took over).
        if (std::isnan(msg->point.x) || std::isnan(msg->point.y) || std::isnan(msg->point.z)) {
            g_goal_cancelled = true;
            g_goal_consumed = true;
            return;
        }
        g_latest_goal = *msg;
        g_has_goal = true;
        g_goal_consumed = false;
        g_goal_cancelled = false;
    }

    void on_stop_movement(const lcm::ReceiveBuffer* /*rbuf*/,
                          const std::string& /*channel*/,
                          const std_msgs::Bool* msg) {
        if (!msg->data) return;
        std::lock_guard<std::mutex> lock(g_goal_mutex);
        g_goal_cancelled = true;
        g_goal_consumed = true;
    }
};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert dimos-lcm PointCloud2 to pcl::PointCloud<PCLPoint> via point_cloud_utils
static PointCloudPtr lcm_pc2_to_pcl(const sensor_msgs::PointCloud2& lcm_msg) {
    PointCloudPtr cloud(new pcl::PointCloud<PCLPoint>());
    auto points = smart_nav::parse_pointcloud2(lcm_msg);
    cloud->reserve(points.size());
    for (const auto& p : points) {
        PCLPoint pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
        pt.intensity = p.intensity;
        cloud->push_back(pt);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

/// Convert Point3D to dimos-lcm geometry_msgs::PointStamped
static geometry_msgs::PointStamped point3d_to_lcm_point_stamped(
    const Point3D& pt, double timestamp) {
    geometry_msgs::PointStamped msg;
    msg.header = dimos::make_header(g_world_frame, timestamp);
    msg.point.x = pt.x;
    msg.point.y = pt.y;
    msg.point.z = pt.z;
    return msg;
}

/// Convert a path (vector of NavNodePtr) to dimos-lcm nav_msgs::Path
static nav_msgs::Path path_to_lcm_path(
    const NodePtrStack& path_nodes, double timestamp) {
    nav_msgs::Path msg;
    msg.header = dimos::make_header(g_world_frame, timestamp);
    msg.poses_length = static_cast<int32_t>(path_nodes.size());
    msg.poses.resize(path_nodes.size());
    for (size_t i = 0; i < path_nodes.size(); ++i) {
        auto& pose = msg.poses[i];
        pose.header = msg.header;
        pose.pose.position.x = path_nodes[i]->position.x;
        pose.pose.position.y = path_nodes[i]->position.y;
        pose.pose.position.z = path_nodes[i]->position.z;
        pose.pose.orientation.w = 1.0;
    }
    return msg;
}

/// Extract robot position from odometry message
static Point3D odom_to_point3d(const nav_msgs::Odometry& odom) {
    return Point3D(
        static_cast<float>(odom.pose.pose.position.x),
        static_cast<float>(odom.pose.pose.position.y),
        static_cast<float>(odom.pose.pose.position.z));
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static void signal_handler(int /*sig*/) {
    g_running.store(false);
}

// ---------------------------------------------------------------------------
// Waypoint projection: push waypoints away from obstacle surfaces
// Based on Algorithm 6 (Figure 14) from the FPS paper.
// ---------------------------------------------------------------------------

/// Project a nav waypoint away from obstacle surfaces into free space.
/// Uses the node's surf_dirs to determine the "away from wall" direction,
/// then ray-traces along that direction using the obstacle cloud to find
/// how far the waypoint can be extended.
/// Project waypoint away from obstacle surfaces (Algorithm 6 / Figure 14).
/// Updates free_dist in-place with actual achieved distance (for heading momentum).
static Point3D ProjectWaypointFromObstacles(
    const NavNodePtr& nav_node,
    const Point3D& robot_position,
    float& free_dist)
{
    if (nav_node == nullptr) return Point3D(0,0,0);

    Point3D waypoint = nav_node->position;

    // Only project CONVEX nodes (nodes on obstacle corners that face outward)
    // For non-CONVEX nodes, set free_dist to 0 so heading extension is minimal
    if (nav_node->free_direct != NodeFreeDirect::CONVEX) { free_dist = 0; return waypoint; }
    if (FARUtil::surround_obs_cloud_ == nullptr || FARUtil::surround_obs_cloud_->empty()) return waypoint;

    // Compute surface normal direction (away from obstacle)
    bool is_wall = false;
    const Point3D surf_dir = -FARUtil::SurfTopoDirect(nav_node->surf_dirs, is_wall);
    if (is_wall || surf_dir.norm() < FARUtil::kEpsilon) { free_dist = 0; return waypoint; }

    // Crop obstacle cloud around the waypoint (matches original ExtendViewpointOnObsCloud)
    PointCloudPtr local_obs(new pcl::PointCloud<PCLPoint>());
    FARUtil::CropPCLCloud(FARUtil::surround_obs_cloud_, local_obs,
                          nav_node->position, free_dist + FARUtil::kNearDist);

    float maxR = std::min((nav_node->position - robot_position).norm(), free_dist) - FARUtil::kNearDist;
    maxR = std::max(maxR, 0.0f);

    if (local_obs->empty()) {
        waypoint = waypoint + surf_dir * maxR;
        free_dist = maxR;
        return waypoint;
    }

    // Build KD-tree from cropped surround obs (matches original kdtree_viewpoint_obs_cloud_)
    PointKdTreePtr local_kdtree(new pcl::KdTreeFLANN<PCLPoint>());
    local_kdtree->setInputCloud(local_obs);

    const float step = FARUtil::kNearDist;
    const float collision_radius = FARUtil::kNearDist / 2.0f + FARUtil::kLeafSize;
    const int collision_threshold = static_cast<int>(std::floor(FARUtil::kNearDist / FARUtil::kLeafSize));

    // Ray-trace outward from the node along surf_dir (matches original ray tracing)
    Point3D start_p = waypoint + surf_dir * step;
    float ray_dist = step;
    bool is_occupied = static_cast<int>(FARUtil::PointInXCounter(start_p, collision_radius, local_kdtree)) > collision_threshold;
    waypoint = start_p;

    while (!is_occupied && ray_dist < free_dist) {
        start_p = start_p + surf_dir * step;
        ray_dist += step;
        is_occupied = static_cast<int>(FARUtil::PointInXCounter(start_p, collision_radius, local_kdtree)) > collision_threshold;
        if (ray_dist < maxR) {
            waypoint = start_p;
        }
    }

    if (is_occupied) {
        waypoint = (nav_node->position + waypoint - surf_dir * step) / 2.0f;
        waypoint.z = nav_node->position.z;
        free_dist = ray_dist - step;
    } else {
        // Full extension without hitting obstacle — free_dist = actual distance extended
        free_dist = (waypoint - nav_node->position).norm();
    }

    return waypoint;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    dimos::NativeModule mod(argc, argv);

    // ── Get topic strings ──────────────────────────────────────────────────
    g_terrain_map_ext_topic = mod.has("terrain_map_ext") ? mod.topic("terrain_map_ext") : "";
    g_terrain_map_topic     = mod.has("terrain_map") ? mod.topic("terrain_map") : "";
    g_registered_scan_topic = mod.has("registered_scan") ? mod.topic("registered_scan") : "";
    g_odometry_topic        = mod.has("odometry") ? mod.topic("odometry") : "";
    g_goal_topic            = mod.has("goal") ? mod.topic("goal") : "";
    g_stop_movement_topic   = mod.has("stop_movement") ? mod.topic("stop_movement") : "";
    g_way_point_topic       = mod.has("way_point") ? mod.topic("way_point") : "";
    g_goal_path_topic       = mod.has("goal_path") ? mod.topic("goal_path") : "";

    // Debug visualization topics
    std::string g_graph_nodes_topic = mod.has("graph_nodes") ? mod.topic("graph_nodes") : "";
    std::string g_graph_edges_topic = mod.has("graph_edges") ? mod.topic("graph_edges") : "";
    std::string g_contour_polygons_topic = mod.has("contour_polygons") ? mod.topic("contour_polygons") : "";
    std::string g_nav_boundary_topic = mod.has("nav_boundary") ? mod.topic("nav_boundary") : "";

    // ── Get config params (mirrors LoadROSParams in original FARMaster) ────
    float update_rate         = mod.arg_float("update_rate", 5.0f);
    float robot_dim           = mod.arg_float("robot_dim", 0.5f);
    float voxel_dim           = mod.arg_float("voxel_dim", 0.1f);
    float sensor_range        = mod.arg_float("sensor_range", 15.0f);
    float terrain_range       = mod.arg_float("terrain_range", 7.5f);
    float local_planner_range = mod.arg_float("local_planner_range", 2.5f);
    float vehicle_height      = mod.arg_float("vehicle_height", 0.75f);
    float visibility_range    = mod.arg_float("visibility_range", 15.0f);
    bool  is_static_env       = mod.arg_bool("is_static_env", false);
    bool  is_viewpoint_extend = mod.arg_bool("is_viewpoint_extend", true);
    float converge_dist       = mod.arg_float("converge_dist", 0.8f);
    g_world_frame             = mod.arg("world_frame", "map");
    g_wp_churn_dist           = mod.arg_float("wp_churn_dist", 0.0f);

    // Remaining params are read inline via mod.arg_*() where they're used.

    printf("[far_planner] Starting FAR Planner native module\n");
    printf("[far_planner] terrain_map_ext: %s\n",
           g_terrain_map_ext_topic.empty() ? "(disabled)" : g_terrain_map_ext_topic.c_str());
    printf("[far_planner] terrain_map: %s\n",
           g_terrain_map_topic.empty() ? "(disabled)" : g_terrain_map_topic.c_str());
    printf("[far_planner] registered_scan: %s\n",
           g_registered_scan_topic.empty() ? "(disabled)" : g_registered_scan_topic.c_str());
    printf("[far_planner] odometry: %s\n",
           g_odometry_topic.empty() ? "(disabled)" : g_odometry_topic.c_str());
    printf("[far_planner] goal: %s\n",
           g_goal_topic.empty() ? "(disabled)" : g_goal_topic.c_str());
    printf("[far_planner] way_point: %s\n",
           g_way_point_topic.empty() ? "(disabled)" : g_way_point_topic.c_str());
    printf("[far_planner] goal_path: %s\n",
           g_goal_path_topic.empty() ? "(disabled)" : g_goal_path_topic.c_str());
    printf("[far_planner] update_rate=%.1f Hz  sensor_range=%.1f  vehicle_height=%.2f\n",
           update_rate, sensor_range, vehicle_height);

    // ── Init LCM ───────────────────────────────────────────────────────────
    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "[far_planner] Error: LCM init failed\n");
        return 1;
    }
    g_lcm = &lcm;

    // ── Subscribe to input topics ──────────────────────────────────────────
    Handlers handlers;
    if (!g_odometry_topic.empty())
        lcm.subscribe(g_odometry_topic, &Handlers::on_odometry, &handlers);
    if (!g_terrain_map_ext_topic.empty())
        lcm.subscribe(g_terrain_map_ext_topic, &Handlers::on_terrain_ext, &handlers);
    if (!g_terrain_map_topic.empty())
        lcm.subscribe(g_terrain_map_topic, &Handlers::on_terrain, &handlers);
    if (!g_registered_scan_topic.empty())
        lcm.subscribe(g_registered_scan_topic, &Handlers::on_registered_scan, &handlers);
    if (!g_goal_topic.empty())
        lcm.subscribe(g_goal_topic, &Handlers::on_goal, &handlers);
    if (!g_stop_movement_topic.empty())
        lcm.subscribe(g_stop_movement_topic, &Handlers::on_stop_movement, &handlers);

    // ── Set FARUtil static parameters ──────────────────────────────────────
    // These correspond to LoadROSParams() in the original FARMaster.
    // Mirror LoadROSParams() from original FARMaster — set FARUtil statics.
    // Note: kEpsilon and kINF are const statics, already initialized.
    FARUtil::vehicle_height  = vehicle_height;
    FARUtil::kSensorRange    = sensor_range;
    FARUtil::kTerrainRange   = std::min(terrain_range, sensor_range);
    FARUtil::kLocalPlanRange = local_planner_range;
    FARUtil::kLeafSize       = voxel_dim;
    FARUtil::kNearDist       = robot_dim;
    FARUtil::kHeightVoxel    = voxel_dim * 2.0f;
    FARUtil::kMatchDist      = robot_dim * 2.0f + voxel_dim;
    FARUtil::kNavClearDist   = robot_dim / 2.0f + voxel_dim;
    FARUtil::kProjectDist    = voxel_dim;
    FARUtil::kMarginDist     = sensor_range - FARUtil::kMatchDist;
    FARUtil::kFreeZ          = 0.1f;
    float floor_height_val   = mod.arg_float("floor_height", 2.0f);
    FARUtil::kTolerZ         = floor_height_val - FARUtil::kHeightVoxel;
    FARUtil::kCellLength     = mod.arg_float("cell_length", 5.0f);
    FARUtil::kCellHeight     = floor_height_val / 2.5f;
    FARUtil::kMarginHeight   = FARUtil::kTolerZ - FARUtil::kCellHeight / 2.0f;
    FARUtil::kVizRatio       = mod.arg_float("visualize_ratio", 0.4f);
    FARUtil::worldFrameId    = g_world_frame;
    FARUtil::kObsInflate     = mod.arg_int("obs_inflate_size", 2);
    FARUtil::kAngleNoise     = mod.arg_float("angle_noise", 15.0f) / 180.0f * M_PI;
    FARUtil::kAcceptAlign    = mod.arg_float("accept_max_align_angle", 15.0f) / 180.0f * M_PI;
    FARUtil::kNewPIThred     = mod.arg_float("new_intensity_thred", 2.0f);
    FARUtil::kObsDecayTime   = mod.arg_float("dynamic_obs_decay_time", 10.0f);
    FARUtil::kNewDecayTime   = mod.arg_float("new_points_decay_time", 2.0f);
    FARUtil::kDyObsThred     = mod.arg_int("dyobs_update_thred", 4);
    FARUtil::KNewPointC      = mod.arg_int("new_point_counter", 10);
    FARUtil::IsStaticEnv     = is_static_env;
    FARUtil::IsDebug         = mod.arg_bool("is_debug_output", false);
    FARUtil::IsMultiLayer    = mod.arg_bool("is_multi_layer", false);
    FARUtil::robot_dim       = robot_dim;
    FARUtil::systemStartTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Init FARUtil kdtrees
    FARUtil::kdtree_new_cloud_ = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
    PointCloudPtr init_cloud(new pcl::PointCloud<PCLPoint>());
    PCLPoint zero_p;
    zero_p.x = zero_p.y = zero_p.z = 0.0f;
    zero_p.intensity = 0.0f;
    init_cloud->push_back(zero_p);
    FARUtil::kdtree_new_cloud_->setInputCloud(init_cloud);
    FARUtil::robot_pos = Point3D(0, 0, 0);
    FARUtil::free_odom_p = Point3D(0, 0, 0);
    FARUtil::odom_pos = Point3D(0, 0, 0);

    // ── Create stub CompatNode for algorithm modules that need rclcpp::Node ─
    auto nh = std::make_shared<CompatNode>("far_planner");

    // ── Initialize algorithm modules ───────────────────────────────────────

    // ContourDetector
    ContourDetector contour_detector;
    {
        ContourDetectParams cd_params;
        cd_params.sensor_range = sensor_range;
        cd_params.voxel_dim    = voxel_dim;
        cd_params.kRatio       = mod.arg_float("resize_ratio", 5.0f);
        cd_params.kThredValue  = mod.arg_int("filter_count_value", 5);
        cd_params.kBlurSize    = static_cast<int>(std::round(FARUtil::kNavClearDist / voxel_dim));
        cd_params.is_save_img  = false;
        cd_params.img_path     = "";
        contour_detector.Init(cd_params);
    }

    // ContourGraph
    ContourGraph contour_graph;
    {
        ContourGraphParams cg_params;
        cg_params.kPillarPerimeter = robot_dim * 4.0f;
        contour_graph.Init(nh, cg_params);
    }

    // DynamicGraph
    DynamicGraph dynamic_graph;
    {
        DynamicGraphParams dg_params;
        dg_params.dumper_thred             = mod.arg_int("clear_dumper_thred", 3);
        dg_params.finalize_thred           = mod.arg_int("node_finalize_thred", 3);
        dg_params.pool_size                = mod.arg_int("filter_pool_size", 12);
        dg_params.votes_size               = mod.arg_int("connect_votes_size", 10);
        dg_params.kConnectAngleThred       = FARUtil::kAcceptAlign;
        dg_params.filter_pos_margin        = FARUtil::kNavClearDist;
        dg_params.filter_dirs_margin       = FARUtil::kAngleNoise;
        dg_params.frontier_perimeter_thred = FARUtil::kMatchDist * 4.0f;
        dynamic_graph.Init(nh, dg_params);
    }

    // GraphPlanner
    GraphPlanner graph_planner;
    {
        GraphPlannerParams gp_params;
        gp_params.converge_dist  = converge_dist;
        gp_params.adjust_radius  = mod.arg_float("goal_adjust_radius", 10.0f);
        gp_params.momentum_dist  = robot_dim / 2.0f;
        gp_params.is_autoswitch  = mod.arg_bool("is_attempt_autoswitch", true);
        gp_params.free_thred     = mod.arg_int("free_counter_thred", 5);
        gp_params.votes_size     = mod.arg_int("reach_goal_vote_size", 5);
        gp_params.momentum_thred = mod.arg_int("path_momentum_thred", 5);
        gp_params.path_switch_cost_ratio = mod.arg_float("path_switch_cost_ratio", 0.85f);
        gp_params.min_path_hold_frames = mod.arg_int("min_path_hold_frames", 10);
        graph_planner.Init(nh, gp_params);
    }

    // MapHandler
    MapHandler map_handler;
    {
        MapHandlerParams mh_params;
        mh_params.sensor_range     = sensor_range;
        mh_params.floor_height     = floor_height_val;
        mh_params.cell_length      = FARUtil::kCellLength;
        mh_params.cell_height      = FARUtil::kCellHeight;
        mh_params.grid_max_length  = mod.arg_float("map_grid_max_length", 1000.0f);
        mh_params.grid_max_height  = mod.arg_float("map_grad_max_height", 100.0f);
        mh_params.height_voxel_dim = FARUtil::kHeightVoxel;
        map_handler.Init(mh_params);
    }

    // ScanHandler
    ScanHandler scan_handler;
    {
        ScanHandlerParams sh_params;
        sh_params.terrain_range = terrain_range;
        sh_params.voxel_size    = voxel_dim;
        sh_params.ceil_height   = floor_height_val;
        scan_handler.Init(sh_params);
    }

    // ── Tracking state for main loop ───────────────────────────────────────
    bool is_map_init = false;
    bool has_received_odom = false;
    Point3D robot_pos(0, 0, 0);
    Point3D last_robot_pos(0, 0, 0);
    int frame_count = 0;
    // Persist the last goal so we can re-set it after planning failures.
    // In the original ROS code, the goal subscriber continuously re-delivers
    // the goal. In our single-shot LCM design, we must handle retries ourselves.
    bool has_pending_goal = false;
    Point3D pending_goal_pos(0, 0, 0);
    double last_goal_retry_time = 0.0;
    const double goal_retry_interval = 2.0;  // seconds between retry attempts
    // Waypoint smoothing: track last published waypoint to reduce churn
    Point3D last_published_waypoint(0, 0, 0);
    bool has_published_waypoint = false;
    NavNodePtr last_nav_node = nullptr;
    // Heading momentum for smooth waypoint projection (matches original nav_heading_)
    Point3D nav_heading(0, 0, 0);

    // ── Main loop ──────────────────────────────────────────────────────────
    float loop_period_ms = 1000.0f / update_rate;
    printf("[far_planner] Starting main loop at %.1f Hz\n", update_rate);
    fflush(stdout);

    while (g_running.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();

        // Handle all pending LCM messages (non-blocking)
        while (lcm.handleTimeout(0) > 0) {}

        double now_sec = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // ── Snapshot input buffers ─────────────────────────────────────────
        nav_msgs::Odometry cur_odom;
        bool have_odom = false;
        {
            std::lock_guard<std::mutex> lock(g_odom_mutex);
            if (g_has_odom) {
                cur_odom = g_latest_odom;
                have_odom = true;
            }
        }

        sensor_msgs::PointCloud2 cur_terrain_ext;
        bool have_terrain_ext = false;
        {
            std::lock_guard<std::mutex> lock(g_terrain_ext_mutex);
            if (g_has_terrain_ext) {
                cur_terrain_ext = g_latest_terrain_ext;
                have_terrain_ext = true;
                g_has_terrain_ext = false;  // consume
            }
        }

        sensor_msgs::PointCloud2 cur_terrain;
        bool have_terrain = false;
        {
            std::lock_guard<std::mutex> lock(g_terrain_mutex);
            if (g_has_terrain) {
                cur_terrain = g_latest_terrain;
                have_terrain = true;
                g_has_terrain = false;
            }
        }

        sensor_msgs::PointCloud2 cur_scan;
        bool have_scan = false;
        {
            std::lock_guard<std::mutex> lock(g_scan_mutex);
            if (g_has_scan) {
                cur_scan = g_latest_scan;
                have_scan = true;
                g_has_scan = false;
            }
        }

        geometry_msgs::PointStamped cur_goal_msg;
        bool have_new_goal = false;
        bool is_cancelled = false;
        {
            std::lock_guard<std::mutex> lock(g_goal_mutex);
            if (g_goal_cancelled) {
                is_cancelled = true;
                g_goal_cancelled = false;
            }
            if (g_has_goal && !g_goal_consumed) {
                cur_goal_msg = g_latest_goal;
                have_new_goal = true;
                g_goal_consumed = true;
            }
        }

        // Need odometry to do anything
        if (!have_odom) {
            auto elapsed = std::chrono::high_resolution_clock::now() - loop_start;
            auto elapsed_ms = std::chrono::duration<float, std::milli>(elapsed).count();
            if (elapsed_ms < loop_period_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>(loop_period_ms - elapsed_ms)));
            }
            continue;
        }

        // ── Update robot position ──────────────────────────────────────────
        robot_pos = odom_to_point3d(cur_odom);
        FARUtil::robot_pos = robot_pos;
        FARUtil::odom_pos = robot_pos;

        if (!has_received_odom) {
            has_received_odom = true;
            last_robot_pos = robot_pos;
            printf("[far_planner] First odometry received: (%.2f, %.2f, %.2f)\n",
                   robot_pos.x, robot_pos.y, robot_pos.z);
            fflush(stdout);
        }

        // Init map origin on first odometry
        if (!is_map_init) {
            map_handler.SetMapOrigin(robot_pos);
            scan_handler.ReInitGrids();
            is_map_init = true;
            printf("[far_planner] Map origin set\n");
            fflush(stdout);
        }

        // Update positions in algorithm modules
        map_handler.UpdateRobotPosition(robot_pos);
        scan_handler.UpdateRobotPosition(robot_pos);
        dynamic_graph.UpdateRobotPosition(robot_pos);

        // ── Process terrain_map_ext (obstacle + free separation) ───────────
        PointCloudPtr obs_cloud(new pcl::PointCloud<PCLPoint>());
        PointCloudPtr free_cloud(new pcl::PointCloud<PCLPoint>());
        PointCloudPtr terrain_height_cloud(new pcl::PointCloud<PCLPoint>());
        bool have_new_clouds = false;

        if (have_terrain_ext) {
            PointCloudPtr terrain_ext_pcl = lcm_pc2_to_pcl(cur_terrain_ext);
            if (!terrain_ext_pcl->empty()) {
                // Upstream uses CropBoxCloud with terrain_range, not sensor_range
                FARUtil::CropPCLCloud(terrain_ext_pcl, robot_pos, FARUtil::kTerrainRange);
                FARUtil::RemoveNanInfPoints(terrain_ext_pcl);
                FARUtil::ExtractFreeAndObsCloud(terrain_ext_pcl, free_cloud, obs_cloud);
                have_new_clouds = true;
                if (!FARUtil::IsStaticEnv) {
                    FARUtil::RemoveOverlapCloud(obs_cloud, FARUtil::stack_dyobs_cloud_, true);
                }
                map_handler.UpdateObsCloudGrid(obs_cloud);
                map_handler.UpdateFreeCloudGrid(free_cloud);
                // NOTE: UpdateTerrainHeightGrid is done later using surround_free_cloud
                // (matching upstream TerrainCallBack lines 746-747)
            }
        }

        // ── Process terrain_map (local terrain for TerrainPlanner) ─────────
        // Upstream: TerrainLocalCallBack populates local_terrain_obs_ from
        // /terrain_local_cloud. In DimOS, terrain_map serves this role.
        // When terrain_map_ext is absent, terrain_map also feeds the grids.
        if (have_terrain) {
            PointCloudPtr terrain_pcl = lcm_pc2_to_pcl(cur_terrain);
            if (!terrain_pcl->empty()) {
                FARUtil::CropPCLCloud(terrain_pcl, robot_pos, FARUtil::kTerrainRange);
                FARUtil::RemoveNanInfPoints(terrain_pcl);

                // Populate local_terrain_obs_ (upstream TerrainLocalCallBack)
                if (!FARUtil::IsStaticEnv) {
                    FARUtil::ExtractFreeAndObsCloud(
                        terrain_pcl, FARUtil::local_terrain_free_, FARUtil::local_terrain_obs_);
                }

                // When terrain_map_ext is absent, also use terrain_map for grids
                if (!have_terrain_ext) {
                    FARUtil::ExtractFreeAndObsCloud(terrain_pcl, free_cloud, obs_cloud);
                    have_new_clouds = true;
                    if (!FARUtil::IsStaticEnv) {
                        FARUtil::RemoveOverlapCloud(obs_cloud, FARUtil::stack_dyobs_cloud_, true);
                    }
                    map_handler.UpdateObsCloudGrid(obs_cloud);
                    map_handler.UpdateFreeCloudGrid(free_cloud);
                }
            }
        }

        // ── Process registered_scan (for dynamic obstacle detection) ───────
        if (have_scan) {
            PointCloudPtr scan_pcl = lcm_pc2_to_pcl(cur_scan);
            if (!scan_pcl->empty()) {
                FARUtil::CropPCLCloud(scan_pcl, robot_pos, sensor_range);
                FARUtil::RemoveNanInfPoints(scan_pcl);
                PointCloudPtr scan_free_cloud(new pcl::PointCloud<PCLPoint>());
                scan_handler.SetCurrentScanCloud(scan_pcl, scan_free_cloud);
            }
        }

        // ── Handle goal cancellation (NaN sentinel from teleop) ───────────
        if (is_cancelled) {
            has_pending_goal = false;
            has_published_waypoint = false;
            last_nav_node = nullptr;
            graph_planner.GoalReset();
            // Publish robot position as stop waypoint
            if (!g_way_point_topic.empty()) {
                auto wp_msg = point3d_to_lcm_point_stamped(robot_pos, now_sec);
                g_lcm->publish(g_way_point_topic, &wp_msg);
            }
            printf("[far_planner] Goal cancelled — idle until new goal\n");
            fflush(stdout);
        }

        // ── Handle new goal ────────────────────────────────────────────────
        if (have_new_goal) {
            Point3D goal_pos(
                static_cast<float>(cur_goal_msg.point.x),
                static_cast<float>(cur_goal_msg.point.y),
                static_cast<float>(cur_goal_msg.point.z));
            graph_planner.UpdateGoal(goal_pos);
            pending_goal_pos = goal_pos;
            has_pending_goal = true;
            last_goal_retry_time = 0.0;  // allow immediate first retry if needed
            printf("[far_planner] New goal received: (%.2f, %.2f, %.2f)\n",
                   goal_pos.x, goal_pos.y, goal_pos.z);
            fflush(stdout);
        }
        // Re-set the goal if it was cleared by a planning failure (rate-limited)
        if (has_pending_goal && graph_planner.GetGoalNodePtr() == nullptr) {
            if (now_sec - last_goal_retry_time >= goal_retry_interval) {
                graph_planner.UpdateGoal(pending_goal_pos);
                last_goal_retry_time = now_sec;
                printf("[far_planner] frame=%d goal re-set after failure (retry)\n", frame_count); fflush(stdout);
            }
        }

        // ── Main graph update loop (mirrors FARMaster::MainLoopCallBack) ───
        // Order matches original FARMaster exactly (lines 169-236).
        if (have_new_clouds && is_map_init) {
            // Extract new observation points (before GetSurroundObsCloud updates the grid)
            // Original: ExtractNewObsPointCloud(temp_obs_ptr_, surround_obs_cloud_, cur_new_cloud_)
            //   cloudIn = current frame obs (intensity 0), cloudRefer = surround (intensity 255)
            //   Result: points in current frame NOT in surround = genuinely new observations
            PointCloudPtr new_obs_cloud(new pcl::PointCloud<PCLPoint>());
            if (!FARUtil::surround_obs_cloud_->empty()) {
                FARUtil::ExtractNewObsPointCloud(obs_cloud, FARUtil::surround_obs_cloud_, new_obs_cloud);
            }

            // Update surround clouds from grid (original lines 746-749)
            PointCloudPtr surround_obs(new pcl::PointCloud<PCLPoint>());
            PointCloudPtr surround_free(new pcl::PointCloud<PCLPoint>());
            map_handler.GetSurroundFreeCloud(surround_free);
            // Terrain height uses surround free cloud, not frame free cloud
            // (upstream TerrainCallBack lines 746-747)
            map_handler.UpdateTerrainHeightGrid(surround_free, terrain_height_cloud);
            map_handler.GetSurroundObsCloud(surround_obs);
            FARUtil::surround_obs_cloud_ = surround_obs;

            // Dynamic obstacle handling (original lines 751-769)
            // Original calls FARMaster::ExtractDynamicObsFromScan which does:
            //   scan_handler.ReInitGrids()
            //   scan_handler.SetCurrentScanCloud(scanCloud, freeCloud)
            //   scan_handler.ExtractDyObsCloud(obsCloud, dyObsOut)
            // SetCurrentScanCloud was already called in the scan processing section above.
            FARUtil::cur_dyobs_cloud_->clear();
            if (!FARUtil::IsStaticEnv) {
                scan_handler.SetSurroundObsCloud(surround_obs, true);
                PointCloudPtr dyobs_cloud(new pcl::PointCloud<PCLPoint>());
                scan_handler.ExtractDyObsCloud(surround_obs, dyobs_cloud);
                if (static_cast<int>(dyobs_cloud->size()) > FARUtil::kDyObsThred) {
                    FARUtil::InflateCloud(dyobs_cloud, voxel_dim, 1, true);
                    map_handler.RemoveObsCloudFromGrid(dyobs_cloud);
                    FARUtil::RemoveOverlapCloud(surround_obs, dyobs_cloud);
                    FARUtil::FilterCloud(dyobs_cloud, voxel_dim);
                    *new_obs_cloud += *dyobs_cloud;
                    FARUtil::FilterCloud(new_obs_cloud, voxel_dim);
                }
                // Accumulate dynamic obstacles with time decay (original line 768)
                FARUtil::StackCloudByTime(dyobs_cloud, FARUtil::stack_dyobs_cloud_, FARUtil::kObsDecayTime, nh);
                FARUtil::cur_dyobs_cloud_ = dyobs_cloud;
            } else {
                scan_handler.SetSurroundObsCloud(surround_obs, false);
            }

            // Accumulate new observations with time decay, then update KD-trees
            // (original lines 772-773: StackCloudByTime + UpdateKdTrees(stack_new_cloud_))
            FARUtil::StackCloudByTime(new_obs_cloud, FARUtil::stack_new_cloud_, FARUtil::kNewDecayTime, nh);
            FARUtil::UpdateKdTrees(FARUtil::stack_new_cloud_);

            const NodePtrStack& nav_graph = dynamic_graph.GetNavGraph();

            const NavNodePtr graph_odom_node = dynamic_graph.GetOdomNode();
            if (graph_odom_node != nullptr) {
                // Contour extraction (original lines 193-194)
                std::vector<PointStack> realworld_contour;
                contour_detector.BuildTerrainImgAndExtractContour(
                    graph_odom_node, surround_obs, realworld_contour);
                contour_graph.UpdateContourGraph(graph_odom_node, realworld_contour);

                // Adjust heights BEFORE matching (original lines 200-201)
                map_handler.AdjustCTNodeHeight(ContourGraph::contour_graph_);
                map_handler.AdjustNodesHeight(nav_graph);

                // Update near nodes BEFORE matching (original lines 203-204)
                dynamic_graph.UpdateGlobalNearNodes();
                const NodePtrStack& near_nodes = dynamic_graph.GetExtendLocalNode();

                // Match contour with nav graph (original line 206)
                CTNodeStack new_convex_vertices;
                contour_graph.MatchContourWithNavGraph(nav_graph, near_nodes, new_convex_vertices);

                // Extract and update graph (original lines 214-222)
                dynamic_graph.ExtractGraphNodes(new_convex_vertices);

                const bool is_freeze_vgraph = false;
                NodePtrStack clear_nodes;
                dynamic_graph.UpdateNavGraph(
                    dynamic_graph.GetNewNodes(), is_freeze_vgraph, clear_nodes);

                if (!clear_nodes.empty()) {
                    PointCloudPtr clear_cloud(new pcl::PointCloud<PCLPoint>());
                    for (const auto& node : clear_nodes) {
                        clear_cloud->push_back(FARUtil::Point3DToPCLPoint(node->position));
                    }
                    map_handler.RemoveObsCloudFromGrid(clear_cloud);
                }

                contour_graph.ExtractGlobalContours();

                // Periodic status
                if (frame_count % 50 == 0) {
                    const auto& polys = ContourGraph::GetContourPolygons();
                    int total_edges = 0;
                    for (const auto& n : dynamic_graph.GetNavGraph()) total_edges += n->connect_nodes.size();
                    printf("[far_planner] frame=%d nodes=%zu edges=%d polys=%zu global_contours=%zu boundary=%zu robot=(%.1f,%.1f)\n",
                           frame_count, dynamic_graph.GetNavGraph().size(), total_edges / 2,
                           polys.size(), ContourGraph::global_contour_.size(),
                           ContourGraph::boundary_contour_.size(),
                           robot_pos.x, robot_pos.y);
                    fflush(stdout);
                }
                // NOTE: UpdaetVGraph is called in the planning block below,
                // AFTER UpdateGoalNavNodeConnects, matching the original order.

                // Publish navigation boundary (local obstacle edges near robot)
                // Matches original FARMaster::LocalBoundaryHandler
                if (!g_nav_boundary_topic.empty()) {
                    const auto& local_bnd = ContourGraph::local_boundary_;
                    // Filter to edges within local_planner_range
                    std::vector<std::pair<Point3D, Point3D>> bnd_edges;
                    for (const auto& edge : local_bnd) {
                        if (FARUtil::DistanceToLineSeg2D(robot_pos, edge) <= local_planner_range) {
                            bnd_edges.push_back({edge.first, edge.second});
                        }
                    }
                    if (!bnd_edges.empty()) {
                        nav_msgs::Path bnd_msg;
                        bnd_msg.header = dimos::make_header(g_world_frame, now_sec);
                        bnd_msg.poses_length = static_cast<int32_t>(bnd_edges.size() * 2);
                        bnd_msg.poses.resize(bnd_edges.size() * 2);
                        for (size_t i = 0; i < bnd_edges.size(); ++i) {
                            auto& p1 = bnd_msg.poses[i * 2];
                            auto& p2 = bnd_msg.poses[i * 2 + 1];
                            p1.header = bnd_msg.header;
                            p2.header = bnd_msg.header;
                            p1.pose.position.x = bnd_edges[i].first.x;
                            p1.pose.position.y = bnd_edges[i].first.y;
                            p1.pose.position.z = bnd_edges[i].first.z;
                            p1.pose.orientation.w = 1.0;
                            p2.pose.position.x = bnd_edges[i].second.x;
                            p2.pose.position.y = bnd_edges[i].second.y;
                            p2.pose.position.z = bnd_edges[i].second.z;
                            p2.pose.orientation.w = 1.0;
                        }
                        g_lcm->publish(g_nav_boundary_topic, &bnd_msg);
                    }
                }

                // Publish debug visualization: graph nodes (as nav_msgs/Path, decoded as GraphNodes3D)
                // orientation.w encodes node type: 0=normal, 1=odom, 2=goal, 3=frontier, 4=navpoint
                if (!g_graph_nodes_topic.empty() && frame_count % 5 == 0) {
                    const NodePtrStack& viz_graph = dynamic_graph.GetNavGraph();
                    nav_msgs::Path nodes_msg;
                    nodes_msg.header = dimos::make_header(g_world_frame, now_sec);
                    nodes_msg.poses_length = static_cast<int32_t>(viz_graph.size());
                    nodes_msg.poses.resize(viz_graph.size());
                    for (size_t i = 0; i < viz_graph.size(); ++i) {
                        const auto& n = viz_graph[i];
                        auto& pose = nodes_msg.poses[i];
                        pose.header = nodes_msg.header;
                        pose.pose.position.x = n->position.x;
                        pose.pose.position.y = n->position.y;
                        pose.pose.position.z = n->position.z;
                        float node_type = 0.0f;
                        if (n->is_odom) node_type = 1.0f;
                        else if (n->is_goal) node_type = 2.0f;
                        else if (n->is_frontier) node_type = 3.0f;
                        else if (n->is_navpoint) node_type = 4.0f;
                        pose.pose.orientation.w = node_type;
                    }
                    g_lcm->publish(g_graph_nodes_topic, &nodes_msg);
                }

                // Publish debug visualization: graph edges (as nav_msgs/Path, decoded as LineSegments3D)
                // Consecutive pose pairs form edge segments.
                // orientation.w encodes traversability:
                //   1.0 = both endpoints traversable (reachable from robot)
                //   0.5 = one endpoint traversable
                //   0.0 = neither endpoint traversable (unreachable)
                if (!g_graph_edges_topic.empty() && frame_count % 5 == 0) {
                    const NodePtrStack& viz_graph = dynamic_graph.GetNavGraph();
                    // Collect unique edges with traversability label
                    struct EdgeInfo { Point3D a, b; float trav; };
                    std::vector<EdgeInfo> edges;
                    for (const auto& n : viz_graph) {
                        for (const auto& neighbor : n->connect_nodes) {
                            if (n->id < neighbor->id) {
                                float trav = 0.0f;
                                if (n->is_traversable && neighbor->is_traversable) trav = 1.0f;
                                else if (n->is_traversable || neighbor->is_traversable) trav = 0.5f;
                                edges.push_back({n->position, neighbor->position, trav});
                            }
                        }
                    }
                    nav_msgs::Path edges_msg;
                    edges_msg.header = dimos::make_header(g_world_frame, now_sec);
                    edges_msg.poses_length = static_cast<int32_t>(edges.size() * 2);
                    edges_msg.poses.resize(edges.size() * 2);
                    for (size_t i = 0; i < edges.size(); ++i) {
                        auto& p1 = edges_msg.poses[i * 2];
                        auto& p2 = edges_msg.poses[i * 2 + 1];
                        p1.header = edges_msg.header;
                        p2.header = edges_msg.header;
                        p1.pose.position.x = edges[i].a.x;
                        p1.pose.position.y = edges[i].a.y;
                        p1.pose.position.z = edges[i].a.z;
                        p1.pose.orientation.w = edges[i].trav;
                        p2.pose.position.x = edges[i].b.x;
                        p2.pose.position.y = edges[i].b.y;
                        p2.pose.position.z = edges[i].b.z;
                        p2.pose.orientation.w = edges[i].trav;
                    }
                    g_lcm->publish(g_graph_edges_topic, &edges_msg);
                }

                // Publish debug visualization: contour polygons (as PointCloud2, decoded as ContourPolygons3D)
                // Each polygon's vertices are points with intensity = polygon_id
                if (!g_contour_polygons_topic.empty() && frame_count % 5 == 0) {
                    const auto& polys = ContourGraph::GetContourPolygons();
                    std::vector<smart_nav::PointXYZI> poly_pts;
                    int non_pillar_id = 0;
                    for (size_t pid = 0; pid < polys.size(); ++pid) {
                        const auto& poly = polys[pid];
                        if (poly->is_pillar) continue;
                        float poly_id = static_cast<float>(non_pillar_id++);
                        for (const auto& v : poly->vertices) {
                            poly_pts.push_back({v.x, v.y, v.z, poly_id});
                        }
                    }
                    if (frame_count % 50 == 0) {
                        printf("[far_planner] contour_polygons: total=%zu non_pillar=%d pts=%zu\n",
                               polys.size(), non_pillar_id, poly_pts.size());
                        fflush(stdout);
                    }
                    auto poly_msg = smart_nav::build_pointcloud2(poly_pts, g_world_frame, now_sec);
                    g_lcm->publish(g_contour_polygons_topic, &poly_msg);
                }
            }
        }

        // ── Planning loop (mirrors FARMaster::PlanningCallBack) ────────────
        // Matches the original FARMaster::PlanningCallBack order exactly.
        // In the original, odom_node_ptr_ is set from previous timer iterations.
        // In our single-loop design, we run a no-goal UpdateGraphTraverability
        // every frame to keep odom_node_ptr_ warm for ReEvaluateGoalPosition.
        const NavNodePtr goal_ptr = graph_planner.GetGoalNodePtr();
        const NavNodePtr odom_node = dynamic_graph.GetOdomNode();
        if (goal_ptr == nullptr && is_map_init && odom_node != nullptr) {
            // No goal — still update graph + traversability (matches original line 281)
            graph_planner.UpdaetVGraph(dynamic_graph.GetNavGraph());
            graph_planner.UpdateGraphTraverability(odom_node, nullptr);
        } else if (goal_ptr != nullptr && is_map_init && odom_node != nullptr) {
            // Matches original FARMaster::PlanningCallBack order (lines 286-303):
            // 1. UpdateFreeTerrainGrid
            const Point3D ori_p = graph_planner.GetOriginNodePos(true);
            PointCloudPtr goal_obs(new pcl::PointCloud<PCLPoint>());
            PointCloudPtr goal_free(new pcl::PointCloud<PCLPoint>());
            map_handler.GetCloudOfPoint(ori_p, goal_obs, CloudType::OBS_CLOUD, true);
            map_handler.GetCloudOfPoint(ori_p, goal_free, CloudType::FREE_CLOUD, true);
            graph_planner.UpdateFreeTerrainGrid(ori_p, goal_obs, goal_free);

            // 2. ReEvaluateGoalPosition (uses odom_node_ptr_ from prev frame)
            graph_planner.ReEvaluateGoalPosition(goal_ptr, !FARUtil::IsMultiLayer);

            // 3. UpdateGoalNavNodeConnects — creates edges TO the goal node
            graph_planner.UpdateGoalNavNodeConnects(goal_ptr);

            // 4. UpdaetVGraph — MUST be after step 3 so goal edges are included
            graph_planner.UpdaetVGraph(dynamic_graph.GetNavGraph());

            // 5. UpdateGraphTraverability (Dijkstra — uses edges from steps 3+4)
            graph_planner.UpdateGraphTraverability(odom_node, goal_ptr);

            // Plan path to goal
            NodePtrStack global_path;
            NavNodePtr nav_waypoint = nullptr;
            Point3D goal_p;
            bool is_fails = false;
            bool is_succeed = false;
            bool is_free_nav = false;

            bool plan_success = graph_planner.PathToGoal(
                goal_ptr, global_path, nav_waypoint, goal_p,
                is_fails, is_succeed, is_free_nav);

            if (plan_success && nav_waypoint != nullptr) {
                Point3D waypoint;
                if (nav_waypoint != goal_ptr) {
                    // Project waypoint away from obstacle surfaces (Algorithm 6 / Figure 14)
                    // free_dist is updated by reference — shortened if wall hit
                    float free_dist = local_planner_range;
                    if (is_viewpoint_extend) {
                        waypoint = ProjectWaypointFromObstacles(
                            nav_waypoint, robot_pos, free_dist);
                    } else {
                        waypoint = nav_waypoint->position;
                    }

                    // Heading momentum (matches original ProjectNavWaypoint lines 392-411)
                    bool is_momentum = (last_nav_node == nav_waypoint) ||
                        (last_nav_node != nullptr &&
                         (last_nav_node->position - nav_waypoint->position).norm() < FARUtil::kNearDist);
                    const Point3D diff_p = waypoint - robot_pos;
                    Point3D new_heading;
                    if (is_momentum && nav_heading.norm() > FARUtil::kEpsilon) {
                        const float hdist = free_dist / 2.0f;
                        const float ratio = std::min(hdist, diff_p.norm()) / hdist;
                        new_heading = diff_p.normalize() * ratio + nav_heading * (1.0f - ratio);
                    } else {
                        new_heading = diff_p.normalize();
                    }
                    if (nav_heading.norm() > FARUtil::kEpsilon && new_heading.norm_dot(nav_heading) < 0.0f) {
                        Point3D temp_heading(nav_heading.y, -nav_heading.x, nav_heading.z);
                        if (temp_heading.norm_dot(new_heading) < 0.0f) {
                            temp_heading.x = -temp_heading.x;
                            temp_heading.y = -temp_heading.y;
                        }
                        new_heading = temp_heading;
                    }
                    nav_heading = new_heading.normalize();
                    if (diff_p.norm() < free_dist) {
                        waypoint = waypoint + nav_heading * (free_dist - diff_p.norm());
                    }
                } else {
                    // Waypoint IS the goal — use goal position directly.
                    // No projection, no heading momentum extension.
                    // (Matches original lines 315-318: only project non-goal waypoints)
                    waypoint = goal_ptr->position;
                    nav_heading = Point3D(0, 0, 0);
                }

                // Churn reduction: only publish if waypoint moved significantly
                // or the nav node changed. Set wp_churn_dist=0 to disable.
                const float wp_change_dist = (waypoint - last_published_waypoint).norm();
                const bool nav_node_changed = (nav_waypoint != last_nav_node);
                const bool should_publish = !has_published_waypoint ||
                    nav_node_changed ||
                    (g_wp_churn_dist > 0 ? wp_change_dist > g_wp_churn_dist : true);

                if (should_publish) {
                    if (!g_way_point_topic.empty()) {
                        auto wp_msg = point3d_to_lcm_point_stamped(waypoint, now_sec);
                        g_lcm->publish(g_way_point_topic, &wp_msg);
                    }
                    if (!g_goal_path_topic.empty() && !global_path.empty()) {
                        auto path_msg = path_to_lcm_path(global_path, now_sec);
                        g_lcm->publish(g_goal_path_topic, &path_msg);
                    }
                    last_published_waypoint = waypoint;
                    has_published_waypoint = true;
                    last_nav_node = nav_waypoint;
                }
            } else if (is_fails) {
                // Planning failed — publish robot position as waypoint to stop movement
                if (!g_way_point_topic.empty()) {
                    auto wp_msg = point3d_to_lcm_point_stamped(robot_pos, now_sec);
                    g_lcm->publish(g_way_point_topic, &wp_msg);
                }
                has_published_waypoint = false;
                last_nav_node = nullptr;
            }

            if (is_succeed) {
                has_pending_goal = false;
                has_published_waypoint = false;
                last_nav_node = nullptr;
                printf("[far_planner] Goal reached!\n"); fflush(stdout);
            }
        }

        frame_count++;
        last_robot_pos = robot_pos;

        // ── Rate control ───────────────────────────────────────────────────
        auto elapsed = std::chrono::high_resolution_clock::now() - loop_start;
        auto elapsed_ms = std::chrono::duration<float, std::milli>(elapsed).count();
        if (elapsed_ms < loop_period_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(loop_period_ms - elapsed_ms)));
        }
    }

    printf("[far_planner] Shutting down\n");
    g_lcm = nullptr;
    return 0;
}
