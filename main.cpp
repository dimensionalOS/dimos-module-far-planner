// FAR Planner — dimos NativeModule port
// Ported from ROS2 packages:
//   src/route_planner/far_planner/
//   src/route_planner/boundary_handler/
//   src/route_planner/graph_decoder/
//   src/route_planner/visibility_graph_msg/
//
// Builds and maintains a visibility graph from obstacle boundaries detected in
// registered point clouds.  Uses contour detection (OpenCV) to extract obstacle
// polygons, constructs a dynamic navigation graph with shortest-path planning
// to the navigation goal, and publishes intermediate waypoints for the local
// planner.
//
// LCM inputs:  registered_scan (PointCloud2), odometry (Odometry), goal (PointStamped)
// LCM outputs: way_point (PointStamped)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <lcm/lcm-cpp.hpp>

#include "dimos_native_module.hpp"
#include "point_cloud_utils.hpp"

// ---------------------------------------------------------------------------
// Debug logging (enabled at runtime via DIMOS_DEBUG=1)
// ---------------------------------------------------------------------------
static bool dimosDebug = false;
#define DBG(...) do { if (dimosDebug) { printf("[FAR][DEBUG] " __VA_ARGS__); fflush(stdout); } } while (0)
#define DBG_EVERY(N, ...) do { if (dimosDebug) { static int _c=0; if ((++_c % (N)) == 0) { printf("[FAR][DEBUG] " __VA_ARGS__); fflush(stdout); } } } while (0)

#include "sensor_msgs/PointCloud2.hpp"
#include "nav_msgs/Odometry.hpp"
#include "nav_msgs/Path.hpp"
#include "geometry_msgs/PointStamped.hpp"
#include "geometry_msgs/PoseStamped.hpp"

#ifdef USE_PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <boost/functional/hash.hpp>
#endif

#ifdef HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif

using namespace std;

// ---------------------------------------------------------------------------
//  Signal handling
// ---------------------------------------------------------------------------
static std::atomic<bool> g_shutdown{false};
static void signal_handler(int) { g_shutdown.store(true); }

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------
#define EPSILON_VAL 1e-7f

// ---------------------------------------------------------------------------
//  Point3D — lightweight 3D point with arithmetic operators
//  (Port of far_planner/point_struct.h)
// ---------------------------------------------------------------------------
struct Point3D {
    float x, y, z;
    float intensity;
    Point3D() : x(0), y(0), z(0), intensity(0) {}
    Point3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z), intensity(0) {}
    Point3D(float _x, float _y, float _z, float _i) : x(_x), y(_y), z(_z), intensity(_i) {}
    Point3D(Eigen::Vector3f v) : x(v(0)), y(v(1)), z(v(2)), intensity(0) {}
    Point3D(Eigen::Vector3d v) : x(v(0)), y(v(1)), z(v(2)), intensity(0) {}

    bool operator==(const Point3D& p) const {
        return fabs(x-p.x)<EPSILON_VAL && fabs(y-p.y)<EPSILON_VAL && fabs(z-p.z)<EPSILON_VAL;
    }
    bool operator!=(const Point3D& p) const { return !(*this == p); }
    float operator*(const Point3D& p) const { return x*p.x + y*p.y + z*p.z; }
    Point3D operator*(float f) const { return {x*f, y*f, z*f}; }
    Point3D operator/(float f) const { return {x/f, y/f, z/f}; }
    Point3D operator+(const Point3D& p) const { return {x+p.x, y+p.y, z+p.z}; }
    Point3D operator-(const Point3D& p) const { return {x-p.x, y-p.y, z-p.z}; }
    Point3D operator-() const { return {-x, -y, -z}; }

    float norm() const { return std::hypot(x, std::hypot(y, z)); }
    float norm_flat() const { return std::hypot(x, y); }

    Point3D normalize() const {
        float n = norm();
        return (n > EPSILON_VAL) ? Point3D(x/n, y/n, z/n) : Point3D(0,0,0);
    }
    Point3D normalize_flat() const {
        float n = norm_flat();
        return (n > EPSILON_VAL) ? Point3D(x/n, y/n, 0.0f) : Point3D(0,0,0);
    }
    float norm_dot(Point3D p) const {
        float n1 = norm(), n2 = p.norm();
        if (n1 < EPSILON_VAL || n2 < EPSILON_VAL) return 0.f;
        float d = (x*p.x + y*p.y + z*p.z) / (n1*n2);
        return std::min(std::max(-1.0f, d), 1.0f);
    }
    float norm_flat_dot(Point3D p) const {
        float n1 = norm_flat(), n2 = p.norm_flat();
        if (n1 < EPSILON_VAL || n2 < EPSILON_VAL) return 0.f;
        float d = (x*p.x + y*p.y) / (n1*n2);
        return std::min(std::max(-1.0f, d), 1.0f);
    }
};

typedef std::pair<Point3D, Point3D> PointPair;
typedef std::vector<Point3D> PointStack;

// ---------------------------------------------------------------------------
//  Node enums and structures
//  (Port of far_planner/node_struct.h)
// ---------------------------------------------------------------------------
enum NodeFreeDirect { UNKNOW=0, CONVEX=1, CONCAVE=2, PILLAR=3 };

struct NavNode;
typedef std::shared_ptr<NavNode> NavNodePtr;
typedef std::pair<NavNodePtr, NavNodePtr> NavEdge;

struct Polygon {
    std::size_t N;
    std::vector<Point3D> vertices;
    bool is_robot_inside;
    bool is_pillar;
    float perimeter;
};
typedef std::shared_ptr<Polygon> PolygonPtr;
typedef std::vector<PolygonPtr> PolygonStack;

struct CTNode {
    Point3D position;
    bool is_global_match;
    bool is_contour_necessary;
    bool is_ground_associate;
    std::size_t nav_node_id;
    NodeFreeDirect free_direct;
    PointPair surf_dirs;
    PolygonPtr poly_ptr;
    std::shared_ptr<CTNode> front;
    std::shared_ptr<CTNode> back;
    std::vector<std::shared_ptr<CTNode>> connect_nodes;
};
typedef std::shared_ptr<CTNode> CTNodePtr;
typedef std::vector<CTNodePtr> CTNodeStack;

struct NavNode {
    std::size_t id;
    Point3D position;
    PointPair surf_dirs;
    std::deque<Point3D> pos_filter_vec;
    std::deque<PointPair> surf_dirs_vec;
    CTNodePtr ctnode;
    bool is_active, is_block_frontier, is_contour_match;
    bool is_odom, is_goal, is_near_nodes, is_wide_near, is_merged;
    bool is_covered, is_frontier, is_finalized, is_navpoint, is_boundary;
    int clear_dumper_count;
    std::deque<int> frontier_votes;
    std::unordered_set<std::size_t> invalid_boundary;
    std::vector<NavNodePtr> connect_nodes;
    std::vector<NavNodePtr> poly_connects;
    std::vector<NavNodePtr> contour_connects;
    std::unordered_map<std::size_t, std::deque<int>> contour_votes;
    std::unordered_map<std::size_t, std::deque<int>> edge_votes;
    std::vector<NavNodePtr> potential_contours;
    std::vector<NavNodePtr> potential_edges;
    std::vector<NavNodePtr> trajectory_connects;
    std::unordered_map<std::size_t, std::size_t> trajectory_votes;
    std::unordered_map<std::size_t, std::size_t> terrain_votes;
    NodeFreeDirect free_direct;
    // planner members
    bool is_block_to_goal, is_traversable, is_free_traversable;
    float gscore, fgscore;
    NavNodePtr parent, free_parent;
};

typedef std::vector<NavNodePtr> NodePtrStack;
typedef std::vector<std::size_t> IdxStack;
typedef std::unordered_set<std::size_t> IdxSet;

#ifdef USE_PCL
typedef pcl::PointXYZI PCLPoint;
typedef pcl::PointCloud<PCLPoint> PointCloud;
typedef pcl::PointCloud<PCLPoint>::Ptr PointCloudPtr;
typedef pcl::KdTreeFLANN<PCLPoint>::Ptr PointKdTreePtr;
#endif

// ---------------------------------------------------------------------------
//  Hash/comparison functors for nodes and edges
// ---------------------------------------------------------------------------
struct nodeptr_hash {
    std::size_t operator()(const NavNodePtr& n) const { return std::hash<std::size_t>()(n->id); }
};
struct nodeptr_equal {
    bool operator()(const NavNodePtr& a, const NavNodePtr& b) const { return a->id == b->id; }
};
struct navedge_hash {
    std::size_t operator()(const NavEdge& e) const {
        std::size_t seed = 0;
        seed ^= std::hash<std::size_t>()(e.first->id) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<std::size_t>()(e.second->id) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};
struct nodeptr_gcomp {
    bool operator()(const NavNodePtr& a, const NavNodePtr& b) const { return a->gscore > b->gscore; }
};
struct nodeptr_fgcomp {
    bool operator()(const NavNodePtr& a, const NavNodePtr& b) const { return a->fgscore > b->fgscore; }
};
struct nodeptr_icomp {
    bool operator()(const NavNodePtr& a, const NavNodePtr& b) const { return a->position.intensity < b->position.intensity; }
};

// ---------------------------------------------------------------------------
//  Line-segment intersection (port of far_planner/intersection.h)
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV
namespace POLYOPS {
static bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r) {
    return q.x<=max(p.x,r.x) && q.x>=min(p.x,r.x) && q.y<=max(p.y,r.y) && q.y>=min(p.y,r.y);
}
static int orientation(cv::Point2f p, cv::Point2f q, cv::Point2f r) {
    double val = (q.y-p.y)*(r.x-q.x) - (q.x-p.x)*(r.y-q.y);
    if (abs(val)<1e-7) return 0;
    return (val>0)?1:2;
}
static bool doIntersect(cv::Point2f p1, cv::Point2f q1, cv::Point2f p2, cv::Point2f q2) {
    int o1=orientation(p1,q1,p2), o2=orientation(p1,q1,q2);
    int o3=orientation(p2,q2,p1), o4=orientation(p2,q2,q1);
    if (o1!=o2 && o3!=o4) return true;
    if (o1==0 && onSegment(p1,p2,q1)) return true;
    if (o2==0 && onSegment(p1,q2,q1)) return true;
    if (o3==0 && onSegment(p2,p1,q2)) return true;
    if (o4==0 && onSegment(p2,q1,q2)) return true;
    return false;
}
}
#endif

// ---------------------------------------------------------------------------
//  ConnectPair, HeightPair — edge helper structures
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV
struct ConnectPair {
    cv::Point2f start_p, end_p;
    ConnectPair() = default;
    ConnectPair(const cv::Point2f& p1, const cv::Point2f& p2) : start_p(p1), end_p(p2) {}
    ConnectPair(const Point3D& p1, const Point3D& p2) {
        start_p.x = p1.x; start_p.y = p1.y;
        end_p.x = p2.x; end_p.y = p2.y;
    }
};
#endif

struct HeightPair {
    float minH, maxH;
    HeightPair() = default;
    HeightPair(float mn, float mx) : minH(mn), maxH(mx) {}
    HeightPair(const Point3D& p1, const Point3D& p2) {
        minH = std::min(p1.z, p2.z);
        maxH = std::max(p1.z, p2.z);
    }
};

// ---------------------------------------------------------------------------
//  3D Grid template (port of far_planner/grid.h)
// ---------------------------------------------------------------------------
namespace grid_ns {
template <typename _T>
class Grid {
public:
    explicit Grid(const Eigen::Vector3i& sz, _T init, const Eigen::Vector3d& orig = Eigen::Vector3d(0,0,0),
                  const Eigen::Vector3d& res = Eigen::Vector3d(1,1,1), int dim = 3)
        : origin_(orig), size_(sz), resolution_(res), dimension_(dim) {
        for (int i=0; i<dimension_; i++) resolution_inv_(i) = 1.0/resolution_(i);
        cell_number_ = size_.x()*size_.y()*size_.z();
        cells_.resize(cell_number_);
        for (int i=0; i<cell_number_; i++) cells_[i] = init;
    }
    int GetCellNumber() const { return cell_number_; }
    Eigen::Vector3i GetSize() const { return size_; }
    Eigen::Vector3d GetOrigin() const { return origin_; }
    void SetOrigin(const Eigen::Vector3d& o) { origin_ = o; }
    Eigen::Vector3d GetResolution() const { return resolution_; }
    void ReInitGrid(const _T& v) { std::fill(cells_.begin(), cells_.end(), v); }
    bool InRange(const Eigen::Vector3i& s) const {
        bool r=true;
        for (int i=0; i<dimension_; i++) r &= s(i)>=0 && s(i)<size_(i);
        return r;
    }
    bool InRange(int ind) const { return ind>=0 && ind<cell_number_; }
    int Sub2Ind(int x, int y, int z) const { return x + y*size_.x() + z*size_.x()*size_.y(); }
    int Sub2Ind(const Eigen::Vector3i& s) const { return Sub2Ind(s.x(),s.y(),s.z()); }
    Eigen::Vector3i Ind2Sub(int ind) const {
        Eigen::Vector3i s;
        s.z() = ind/(size_.x()*size_.y());
        ind -= s.z()*size_.x()*size_.y();
        s.y() = ind/size_.x();
        s.x() = ind%size_.x();
        return s;
    }
    Eigen::Vector3d Sub2Pos(const Eigen::Vector3i& s) const {
        Eigen::Vector3d p(0,0,0);
        for (int i=0; i<dimension_; i++) p(i) = origin_(i) + s(i)*resolution_(i) + resolution_(i)/2.0;
        return p;
    }
    Eigen::Vector3d Ind2Pos(int ind) const { return Sub2Pos(Ind2Sub(ind)); }
    Eigen::Vector3i Pos2Sub(double px, double py, double pz) const { return Pos2Sub(Eigen::Vector3d(px,py,pz)); }
    Eigen::Vector3i Pos2Sub(const Eigen::Vector3d& p) const {
        Eigen::Vector3i s(0,0,0);
        for (int i=0; i<dimension_; i++) s(i) = p(i)-origin_(i)>-1e-7 ? (int)((p(i)-origin_(i))*resolution_inv_(i)) : -1;
        return s;
    }
    int Pos2Ind(const Eigen::Vector3d& p) const { return Sub2Ind(Pos2Sub(p)); }
    _T& GetCell(int ind) { return cells_[ind]; }
    _T& GetCell(const Eigen::Vector3i& s) { return cells_[Sub2Ind(s)]; }
    _T GetCellValue(int ind) const { return cells_[ind]; }
private:
    Eigen::Vector3d origin_, resolution_, resolution_inv_;
    Eigen::Vector3i size_;
    std::vector<_T> cells_;
    int cell_number_, dimension_;
};
} // namespace grid_ns

// ---------------------------------------------------------------------------
//  TimeMeasure utility (port of far_planner/time_measure.h)
// ---------------------------------------------------------------------------
class TimeMeasure {
    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::string, std::chrono::time_point<Clock>> timers_;
public:
    void start_time(const std::string& n, bool reset=false) {
        auto it = timers_.find(n);
        auto now = Clock::now();
        if (it == timers_.end()) timers_.insert({n, now});
        else if (reset) it->second = now;
    }
    double end_time(const std::string& n, bool print=true) {
        auto it = timers_.find(n);
        if (it != timers_.end()) {
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-it->second);
            double ms = dur.count()/1000.0;
            if (print) printf("    %s Time: %.2fms\n", n.c_str(), ms);
            timers_.erase(it);
            return ms;
        }
        return -1.0;
    }
    double record_time(const std::string& n) {
        auto it = timers_.find(n);
        if (it != timers_.end()) {
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-it->second);
            return dur.count()/1000.0;
        }
        return -1.0;
    }
};

// ---------------------------------------------------------------------------
//  Global utility class (port of FARUtil statics)
// ---------------------------------------------------------------------------
struct FARGlobals {
    // constants
    static constexpr float kEpsilon = 1e-7f;
    static constexpr float kINF = std::numeric_limits<float>::max();

    // configurable parameters
    bool is_static_env = true;
    bool is_debug = false;
    bool is_multi_layer = false;
    Point3D robot_pos, odom_pos, map_origin, free_odom_p;
    float robot_dim = 0.8f;
    float vehicle_height = 0.75f;
    float kLeafSize = 0.2f;
    float kHeightVoxel = 0.4f;
    float kNavClearDist = 1.0f;  // node padding from contour (was 0.5 — equal to robot_dim with zero margin)
    float kNearDist = 0.8f;
    float kMatchDist = 1.8f;
    float kProjectDist = 0.2f;
    float kSensorRange = 30.0f;
    float kMarginDist = 28.0f;
    float kMarginHeight = 1.2f;
    float kTerrainRange = 15.0f;
    float kLocalPlanRange = 5.0f;
    float kAngleNoise = 0.2618f;  // 15 degrees in rad
    float kAcceptAlign = 0.2618f;
    float kCellLength = 5.0f;
    float kCellHeight = 0.8f;
    float kNewPIThred = 2.0f;
    float kFreeZ = 0.1f;
    float kVizRatio = 1.0f;
    float kTolerZ = 1.6f;
    float kObsDecayTime = 10.0f;
    float kNewDecayTime = 2.0f;
    int kDyObsThred = 4;
    int KNewPointC = 10;
    int kObsInflate = 2;
    double systemStartTime = 0.0;
    std::string worldFrameId = "map";
    TimeMeasure Timer;

#ifdef USE_PCL
    PointCloudPtr surround_obs_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr surround_free_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr stack_new_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr cur_new_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr cur_dyobs_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr stack_dyobs_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr cur_scan_cloud = PointCloudPtr(new PointCloud());
    PointCloudPtr local_terrain_obs = PointCloudPtr(new PointCloud());
    PointCloudPtr local_terrain_free = PointCloudPtr(new PointCloud());
    PointKdTreePtr kdtree_new_cloud = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
    PointKdTreePtr kdtree_filter_cloud = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());

    // --- PCL utility methods ---
    void FilterCloud(const PointCloudPtr& cloud, float leaf) {
        pcl::VoxelGrid<PCLPoint> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf, leaf, leaf);
        pcl::PointCloud<PCLPoint> filtered;
        vg.filter(filtered);
        *cloud = filtered;
    }
    void CropPCLCloud(const PointCloudPtr& cloudIn, const PointCloudPtr& out,
                      const Point3D& c, float range) {
        out->clear();
        out->resize(cloudIn->size());
        std::size_t idx = 0;
        for (const auto& p : cloudIn->points) {
            if ((Point3D(p.x,p.y,p.z) - c).norm() < range) { out->points[idx++] = p; }
        }
        out->resize(idx);
    }
    PCLPoint Point3DToPCL(const Point3D& p) {
        PCLPoint pp; pp.x=p.x; pp.y=p.y; pp.z=p.z; pp.intensity=p.intensity; return pp;
    }
    void ExtractNewObsPointCloud(const PointCloudPtr& cloudIn, const PointCloudPtr& refer, const PointCloudPtr& out) {
        PointCloudPtr temp(new PointCloud());
        for (auto& p : cloudIn->points) p.intensity = 0.0f;
        for (auto& p : refer->points) p.intensity = 255.0f;
        out->clear(); temp->clear();
        *temp = *cloudIn + *refer;
        FilterCloud(temp, kLeafSize*2.0f);
        for (const auto& p : temp->points) {
            if (p.intensity < kNewPIThred) out->points.push_back(p);
        }
    }
    // Ref: utility.cpp:205-226  FARUtil::ExtractFreeAndObsCloud
    void ExtractFreeAndObsCloud(const PointCloudPtr& in, const PointCloudPtr& free_out, const PointCloudPtr& obs_out) {
        free_out->clear(); obs_out->clear();
        for (const auto& p : in->points) {
            if (p.intensity < kFreeZ) free_out->points.push_back(p);
            else obs_out->points.push_back(p);
        }
    }
    void UpdateKdTrees(const PointCloudPtr& newObs) {
        if (!newObs->empty()) kdtree_new_cloud->setInputCloud(newObs);
        else {
            PCLPoint tmp; tmp.x=tmp.y=tmp.z=0.f;
            newObs->resize(1); newObs->points[0]=tmp;
            kdtree_new_cloud->setInputCloud(newObs);
        }
    }
    std::size_t PointInXCounter(const Point3D& p, float radius, const PointKdTreePtr& tree) {
        std::vector<int> idx; std::vector<float> dist;
        PCLPoint pp; pp.x=p.x; pp.y=p.y; pp.z=p.z;
        if (!std::isfinite(pp.x) || !std::isfinite(pp.y) || !std::isfinite(pp.z)) return 0;
        tree->radiusSearch(pp, radius, idx, dist);
        return idx.size();
    }
    bool IsPointNearNewPoints(const Point3D& p, bool is_creation=false) {
        int near_c = (int)PointInXCounter(p, kMatchDist, kdtree_new_cloud);
        int limit = is_creation ? (int)std::round(KNewPointC/2.0f) : KNewPointC;
        return near_c > limit;
    }
#endif

    // --- Point-in-polygon (Randolph Franklin) ---
    template <typename Point>
    bool PointInsideAPoly(const std::vector<Point>& poly, const Point& p) const {
        int i,j,c=0, npol=(int)poly.size();
        if (npol<3) return false;
        for (i=0,j=npol-1; i<npol; j=i++) {
            if ((((poly[i].y<=p.y)&&(p.y<poly[j].y))||((poly[j].y<=p.y)&&(p.y<poly[i].y)))&&
                (p.x<(poly[j].x-poly[i].x)*(p.y-poly[i].y)/(poly[j].y-poly[i].y)+poly[i].x)) c=!c;
        }
        return c;
    }

    bool IsPointInToleratedHeight(const Point3D& p, float h) const {
        return fabs(p.z - robot_pos.z) < h;
    }
    bool IsPointInLocalRange(const Point3D& p, bool large_h=false) const {
        float H = large_h ? kTolerZ+kHeightVoxel : kTolerZ;
        return IsPointInToleratedHeight(p, H) && (p-odom_pos).norm() < kSensorRange;
    }
    bool IsPointInMarginRange(const Point3D& p) const {
        return IsPointInToleratedHeight(p, kMarginHeight) && (p-odom_pos).norm() < kMarginDist;
    }
    bool IsFreeNavNode(const NavNodePtr& n) const { return n->is_odom || n->is_navpoint; }
    bool IsStaticNode(const NavNodePtr& n) const { return n->is_odom || n->is_goal; }
    bool IsOutsideGoal(const NavNodePtr& n) const { return n->is_goal && !n->is_navpoint; }
    int Mod(int a, int b) const { return (b+(a%b))%b; }
    bool IsSamePoint3D(const Point3D& p1, const Point3D& p2) const { return (p2-p1).norm()<kEpsilon; }

    void EraseNodeFromStack(const NavNodePtr& n, NodePtrStack& stack) {
        for (auto it=stack.begin(); it!=stack.end();) {
            if (*it==n) it=stack.erase(it); else ++it;
        }
    }
    template <typename T>
    bool IsTypeInStack(const T& e, const std::vector<T>& s) const {
        return std::find(s.begin(), s.end(), e) != s.end();
    }
    float NoiseCosValue(float dot_val, bool is_large, float noise) const {
        float theta = std::acos(std::max(-1.0f, std::min(1.0f, dot_val)));
        int sign = is_large ? 1 : -1;
        double m = theta + sign*noise;
        m = std::min(std::max(m, 0.0), (double)M_PI);
        return (float)cos(m);
    }
    float MarginAngleNoise(float dist, float max_shift, float angle_noise) const {
        float m = angle_noise;
        if (dist*sin(m) < max_shift) m = std::asin(max_shift/std::max(dist, max_shift));
        return m;
    }
    bool IsOutReducedDirs(const Point3D& diff, const PointPair& dirs) const {
        Point3D nd = diff.normalize_flat();
        float man = MarginAngleNoise(diff.norm_flat(), kNearDist, kAngleNoise);
        Point3D opp = -dirs.second;
        float thrd = NoiseCosValue(dirs.first*opp, true, man);
        if (nd*dirs.first>thrd && nd*opp>thrd) return true;
        opp = -dirs.first;
        thrd = NoiseCosValue(dirs.second*opp, true, man);
        if (nd*dirs.second>thrd && nd*opp>thrd) return true;
        return false;
    }
    bool IsOutReducedDirs(const Point3D& diff, const NavNodePtr& n) const {
        if (n->free_direct != PILLAR) { if (!IsOutReducedDirs(diff, n->surf_dirs)) return false; }
        return true;
    }
    Point3D SurfTopoDirect(const PointPair& dirs) const {
        Point3D td = dirs.first + dirs.second;
        return (td.norm_flat() > kEpsilon) ? td.normalize_flat() : Point3D(0,0,0);
    }
    bool IsVoteTrue(const std::deque<int>& votes, bool balanced=true) const {
        int N=(int)votes.size();
        float s = std::accumulate(votes.begin(), votes.end(), 0.0f);
        float f = balanced ? 2.0f : 3.0f;
        return s > std::floor(N/f);
    }
    bool IsConvexPoint(const PolygonPtr& poly, const Point3D& ev_p) const {
        return PointInsideAPoly(poly->vertices, ev_p) != poly->is_robot_inside;
    }
    template <typename N1, typename N2>
    bool IsAtSameLayer(const N1& n1, const N2& n2) const {
        if (is_multi_layer && fabs(n1->position.z - n2->position.z) > kTolerZ) return false;
        return true;
    }
    bool IsNodeInLocalRange(const NavNodePtr& n, bool lh=false) const { return IsPointInLocalRange(n->position, lh); }
    bool IsNodeInExtendMatchRange(const NavNodePtr& n) const {
        return IsPointInToleratedHeight(n->position, kTolerZ*1.5f) && (n->position-odom_pos).norm()<kSensorRange;
    }
    float ClampAbsRange(float v, float range) const { range=fabs(range); return std::min(std::max(-range,v),range); }
    float ContourSurfDirs(const Point3D& end_p, const Point3D& start_p, const Point3D& center_p, float radius) const {
        // Returns direction angle; simplified for the port
        float D = (center_p - end_p).norm_flat();
        float phi = std::acos((center_p-end_p).norm_flat_dot(start_p-end_p));
        float H = D*sin(phi);
        if (H < kEpsilon) return 0;
        return std::asin(ClampAbsRange(H/radius, 1.0f));
    }
    Point3D ContourSurfDirsVec(const Point3D& end_p, const Point3D& start_p, const Point3D& center_p, float radius) const {
        float D = (center_p - end_p).norm_flat();
        float phi = std::acos((center_p-end_p).norm_flat_dot(start_p-end_p));
        float H = D*sin(phi);
        if (H < kEpsilon) return (end_p - center_p).normalize_flat();
        float theta = asin(ClampAbsRange(H/radius, 1.0f));
        Point3D dir = (start_p - end_p).normalize_flat();
        Point3D V_p = end_p + dir * D * cos(phi);
        Point3D K_p = V_p - dir * radius * cos(theta);
        return (K_p - center_p).normalize_flat();
    }
    bool IsInCoverageDirPairs(const Point3D& diff, const NavNodePtr& n) const {
        if (n->free_direct == PILLAR) return false;
        Point3D nd = diff.normalize_flat();
        float man = MarginAngleNoise(diff.norm_flat(), kNearDist, kAngleNoise*2.0f);
        float dv = NoiseCosValue(n->surf_dirs.first * n->surf_dirs.second, true, man);
        if (n->free_direct == CONCAVE) {
            if (nd*n->surf_dirs.first>dv && nd*n->surf_dirs.second>dv) return true;
        } else if (n->free_direct == CONVEX) {
            if (nd*(-n->surf_dirs.second)>dv && nd*(-n->surf_dirs.first)>dv) return true;
        }
        return false;
    }
    bool IsInContourDirPairs(const Point3D& diff, const PointPair& dirs) const {
        float man = MarginAngleNoise(diff.norm_flat(), kNearDist, kAngleNoise);
        float mc = cos(man);
        if (dirs.first.norm_dot(diff) > mc) return true;
        if (dirs.second.norm_dot(diff) > mc) return true;
        return false;
    }
    float VerticalDistToLine2D(const Point3D& sp, const Point3D& ep, const Point3D& cp) const {
        Point3D ld = ep - sp;
        Point3D dp = cp - sp;
        float dv = ld.norm_flat_dot(dp);
        return sin(acos(dv)) * dp.norm_flat();
    }
    bool IsInCylinder(const Point3D& from, const Point3D& to, const Point3D& cur, float radius, bool is2d=false) const {
        Point3D ua = is2d ? (to-from).normalize_flat() : (to-from).normalize();
        Point3D v = cur - from;
        float ps = v * ua;
        float tl = is2d ? (to-from).norm_flat() : (to-from).norm();
        if (ps < -radius || ps > tl+radius) return false;
        Point3D va = ua * ps;
        float dl = is2d ? (v-va).norm_flat() : (v-va).norm();
        return dl <= radius;
    }
    float DistanceToLineSeg2D(const Point3D& p, const PointPair& line) const {
        float A=(p-line.first).x, B=(p-line.first).y;
        float C=(line.second-line.first).x, D=(line.second-line.first).y;
        float dot=A*C+B*D, len_sq=C*C+D*D;
        float param = (len_sq!=0.0f) ? dot/len_sq : -1.0f;
        float xx,yy;
        if (param<0) { xx=line.first.x; yy=line.first.y; }
        else if (param>1) { xx=line.second.x; yy=line.second.y; }
        else { xx=line.first.x+param*C; yy=line.first.y+param*D; }
        return sqrt((p.x-xx)*(p.x-xx)+(p.y-yy)*(p.y-yy));
    }
    float LineMatchPercentage(const PointPair& l1, const PointPair& l2) const {
        float ds = (l1.first-l2.first).norm_flat();
        float theta = acos((l1.second-l1.first).norm_flat_dot(l2.second-l2.first));
        if (theta > kAcceptAlign || ds > kNavClearDist) return 0.0f;
        float cds = (l2.second-l2.first).norm_flat();
        float mds = cds;
        if (theta > kEpsilon) mds = std::min(mds, kNavClearDist/tan(theta));
        return mds/cds;
    }
    int VoteRankInVotes(int c, const std::vector<int>& ov) const {
        int idx=0;
        while (idx<(int)ov.size() && c<ov[idx]) idx++;
        return idx;
    }
    float DirsDistance(const PointPair& r, const PointPair& c) const {
        return std::acos(r.first.norm_dot(c.first)) + std::acos(r.second.norm_dot(c.second));
    }
    Point3D RANSACPosition(const std::deque<Point3D>& pf, float margin, std::size_t& inlier_sz) const {
        inlier_sz = 0;
        PointStack best;
        for (const auto& p : pf) {
            PointStack tmp;
            for (const auto& cp : pf) { if ((p-cp).norm_flat()<margin) tmp.push_back(cp); }
            if (tmp.size()>inlier_sz) { best=tmp; inlier_sz=tmp.size(); }
        }
        return AveragePoints(best);
    }
    Point3D AveragePoints(const PointStack& ps) const {
        Point3D m(0,0,0);
        if (ps.empty()) return m;
        for (const auto& p : ps) m = m + p;
        return m / (float)ps.size();
    }
    PointPair RANSACSurfDirs(const std::deque<PointPair>& sd, float margin, std::size_t& isz) const {
        isz = 0;
        std::vector<PointPair> best;
        PointPair pillar_dir(Point3D(0,0,-1), Point3D(0,0,-1));
        std::size_t pc = 0;
        for (const auto& d : sd) if (d.first==Point3D(0,0,-1)&&d.second==Point3D(0,0,-1)) pc++;
        for (const auto& d : sd) {
            if (d.first==Point3D(0,0,-1)&&d.second==Point3D(0,0,-1)) continue;
            std::vector<PointPair> tmp;
            for (const auto& cd : sd) {
                if (cd.first==Point3D(0,0,-1)&&cd.second==Point3D(0,0,-1)) continue;
                if (DirsDistance(d,cd)<margin) tmp.push_back(cd);
            }
            if (tmp.size()>isz) { best=tmp; isz=tmp.size(); }
        }
        if (pc>isz) { isz=pc; return pillar_dir; }
        // average dirs
        Point3D m1(0,0,0), m2(0,0,0);
        for (const auto& d : best) { m1=m1+d.first; m2=m2+d.second; }
        return {m1.normalize(), m2.normalize()};
    }
    void CorrectDirectOrder(const PointPair& ref, PointPair& d) const {
        if (ref.first*d.first + ref.second*d.second < ref.first*d.second + ref.second*d.first)
            std::swap(d.first, d.second);
    }
};

// Global instance
static FARGlobals G;

// ---------------------------------------------------------------------------
//  Graph ID tracker and global graph storage
// ---------------------------------------------------------------------------
static std::size_t g_id_tracker = 1;
static NodePtrStack g_global_graph_nodes;
static std::unordered_map<std::size_t, NavNodePtr> g_idx_node_map;

// Contour graph global statics
static CTNodeStack g_contour_graph;
static PolygonStack g_contour_polygons;
static CTNodeStack g_polys_ctnodes;
static std::vector<PointPair> g_global_contour;
static std::vector<PointPair> g_boundary_contour;
static std::vector<PointPair> g_local_boundary;
static std::vector<PointPair> g_inactive_contour;
static std::vector<PointPair> g_unmatched_contour;
static std::unordered_set<NavEdge, navedge_hash> g_global_contour_set;

// Ray-cast obstacle check callback — set in main() after MapHandler is created.
// Returns true if the line from p1 to p2 is blocked by accumulated obstacles.
static std::function<bool(const Point3D&, const Point3D&)> g_obstacle_raycast = nullptr;
static std::unordered_set<NavEdge, navedge_hash> g_boundary_contour_set;

// ---------------------------------------------------------------------------
//  CreateNavNodeFromPoint — factory for navigation nodes
// ---------------------------------------------------------------------------
static void AssignGlobalNodeID(const NavNodePtr& n) {
    n->id = g_id_tracker;
    g_idx_node_map.insert({n->id, n});
    g_id_tracker++;
}

static void CreateNavNodeFromPoint(const Point3D& p, NavNodePtr& n, bool is_odom,
                                   bool is_navpoint=false, bool is_goal=false, bool is_boundary=false) {
    n = std::make_shared<NavNode>();
    n->pos_filter_vec.clear();
    n->surf_dirs_vec.clear();
    n->ctnode = nullptr;
    n->is_active = true;
    n->is_block_frontier = false;
    n->is_contour_match = false;
    n->is_odom = is_odom;
    n->is_near_nodes = true;
    n->is_wide_near = true;
    n->is_merged = false;
    n->is_covered = (is_odom||is_navpoint||is_goal);
    n->is_frontier = false;
    n->is_finalized = is_navpoint;
    n->is_traversable = is_odom;
    n->is_navpoint = is_navpoint;
    n->is_boundary = is_boundary;
    n->is_goal = is_goal;
    n->clear_dumper_count = 0;
    n->frontier_votes.clear();
    n->invalid_boundary.clear();
    n->connect_nodes.clear();
    n->poly_connects.clear();
    n->contour_connects.clear();
    n->contour_votes.clear();
    n->potential_contours.clear();
    n->trajectory_connects.clear();
    n->trajectory_votes.clear();
    n->terrain_votes.clear();
    n->free_direct = (is_odom||is_navpoint) ? PILLAR : UNKNOW;
    n->is_block_to_goal = false;
    n->gscore = G.kINF;
    n->fgscore = G.kINF;
    n->is_traversable = true;
    n->is_free_traversable = true;
    n->parent = nullptr;
    n->free_parent = nullptr;
    n->position = p;
    n->pos_filter_vec.push_back(p);
    AssignGlobalNodeID(n);
}

// ---------------------------------------------------------------------------
//  Graph edge helpers
// ---------------------------------------------------------------------------
static void AddEdge(const NavNodePtr& n1, const NavNodePtr& n2) {
    if (n1==n2) return;
    if (!G.IsTypeInStack(n2, n1->connect_nodes) && !G.IsTypeInStack(n1, n2->connect_nodes)) {
        n1->connect_nodes.push_back(n2);
        n2->connect_nodes.push_back(n1);
    }
}
static void EraseEdge(const NavNodePtr& n1, const NavNodePtr& n2) {
    G.EraseNodeFromStack(n2, n1->connect_nodes);
    G.EraseNodeFromStack(n1, n2->connect_nodes);
}
static void AddPolyEdge(const NavNodePtr& n1, const NavNodePtr& n2) {
    if (n1==n2) return;
    if (!G.IsTypeInStack(n2, n1->poly_connects) && !G.IsTypeInStack(n1, n2->poly_connects)) {
        n1->poly_connects.push_back(n2);
        n2->poly_connects.push_back(n1);
    }
}
static void ErasePolyEdge(const NavNodePtr& n1, const NavNodePtr& n2) {
    G.EraseNodeFromStack(n2, n1->poly_connects);
    G.EraseNodeFromStack(n1, n2->poly_connects);
}
static void AddNodeToGraph(const NavNodePtr& n) {
    if (n) g_global_graph_nodes.push_back(n);
}

// ---------------------------------------------------------------------------
//  Contour graph helpers — add/delete contour to sets
// ---------------------------------------------------------------------------
static void AddContourToSets(const NavNodePtr& n1, const NavNodePtr& n2) {
    NavEdge e = (n1->id < n2->id) ? NavEdge(n1,n2) : NavEdge(n2,n1);
    g_global_contour_set.insert(e);
    if (n1->is_boundary && n2->is_boundary) g_boundary_contour_set.insert(e);
}
static void DeleteContourFromSets(const NavNodePtr& n1, const NavNodePtr& n2) {
    NavEdge e = (n1->id < n2->id) ? NavEdge(n1,n2) : NavEdge(n2,n1);
    g_global_contour_set.erase(e);
    if (n1->is_boundary && n2->is_boundary) g_boundary_contour_set.erase(e);
}
static void AddContourConnect(const NavNodePtr& n1, const NavNodePtr& n2) {
    if (!G.IsTypeInStack(n1, n2->contour_connects) && !G.IsTypeInStack(n2, n1->contour_connects)) {
        n1->contour_connects.push_back(n2);
        n2->contour_connects.push_back(n1);
        AddContourToSets(n1, n2);
    }
}

// ---------------------------------------------------------------------------
//  Collision checking with boundary segments
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV
static bool IsEdgeCollideSegment(const PointPair& line, const ConnectPair& edge) {
    cv::Point2f sp(line.first.x, line.first.y), ep(line.second.x, line.second.y);
    return POLYOPS::doIntersect(sp, ep, edge.start_p, edge.end_p);
}
static bool IsEdgeCollidePoly(const PointStack& poly, const ConnectPair& edge) {
    int N=(int)poly.size();
    for (int i=0; i<N; i++) {
        PointPair l(poly[i], poly[G.Mod(i+1,N)]);
        if (IsEdgeCollideSegment(l, edge)) return true;
    }
    return false;
}
// ---------------------------------------------------------------------------
//  Goal reprojection — move goal from inside polygon to free space nearby
//  Ref: contour_graph.cpp:546-586  ReprojectPointOutsidePolygons
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV
// Returns true if point was moved (was inside a polygon). Moves it to just
// outside the nearest polygon vertex by free_radius distance.
static bool ReprojectPointOutsidePolygons(Point3D& point, float free_radius) {
    for (const auto& poly : g_contour_polygons) {
        if (poly->is_pillar) continue;
        if (!G.PointInsideAPoly(poly->vertices, point)) continue;
        // Point is inside this polygon, but odom should be outside
        if (G.PointInsideAPoly(poly->vertices, G.odom_pos)) continue;
        // Find nearest vertex and project outward
        float near_dist = G.kINF;
        Point3D reproject_p = point;
        Point3D free_dir(0, 0, -1);
        int N = (int)poly->vertices.size();
        for (int idx = 0; idx < N; idx++) {
            Point3D vertex(poly->vertices[idx].x, poly->vertices[idx].y, point.z);
            float temp_dist = (vertex - point).norm_flat();
            if (temp_dist < near_dist) {
                // Compute outward direction at this vertex
                int prev_i = G.Mod(idx-1, N), next_i = G.Mod(idx+1, N);
                Point3D prev(poly->vertices[prev_i].x, poly->vertices[prev_i].y, point.z);
                Point3D next(poly->vertices[next_i].x, poly->vertices[next_i].y, point.z);
                float len1 = (prev - vertex).norm_flat();
                float len2 = (next - vertex).norm_flat();
                if (len1 < 0.001f || len2 < 0.001f) continue;
                Point3D d1 = (prev - vertex) * (1.0f / len1);
                Point3D d2 = (next - vertex) * (1.0f / len2);
                Point3D d = d1 + d2;
                float dlen = d.norm_flat();
                if (dlen < 0.001f) continue;
                d = d * (1.0f / dlen);
                // Check if this direction points outward (test point + small step)
                Point3D test = vertex;
                test.x += d.x * G.kLeafSize; test.y += d.y * G.kLeafSize;
                if (G.PointInsideAPoly(poly->vertices, test)) {
                    // Convex vertex — direction points inside, use it
                    reproject_p = vertex;
                    near_dist = temp_dist;
                    free_dir = d;
                }
            }
        }
        if (near_dist < G.kINF) {
            float orig_z = point.z;
            point.x = reproject_p.x - free_dir.x * free_radius;
            point.y = reproject_p.y - free_dir.y * free_radius;
            point.z = orig_z;
            printf("[FAR] Goal reprojected from inside polygon to (%.2f, %.2f)\n", point.x, point.y);
            return true;
        }
    }
    return false;
}
#else
static bool ReprojectPointOutsidePolygons(Point3D&, float) { return false; }
#endif

// Ref: contour_graph.cpp:94-112  IsNavNodesConnectFreePolygon
// Ref: contour_graph.cpp:156-208  IsPointsConnectFreePolygon
static bool IsNavNodesConnectFreePolygon(const NavNodePtr& n1, const NavNodePtr& n2) {
    ConnectPair cedge(n1->position, n2->position);
    // Check boundary contour segments (from boundary handler — empty without it)
    for (const auto& c : g_boundary_contour) {
        if (IsEdgeCollideSegment(c, cedge)) return false;
    }
    // Check global contour segments for edges where at least one endpoint
    // is outside local range (same as reference: global_contour_ is only
    // checked when is_global_check=true).
    // Ref: contour_graph.cpp:193-206  global check branch
    bool is_global = !G.IsPointInLocalRange(n1->position) ||
                     !G.IsPointInLocalRange(n2->position);
    if (is_global) {
        for (const auto& c : g_global_contour) {
            if (IsEdgeCollideSegment(c, cedge)) return false;
        }
    }
    // Polygon collision check with center-point same-side test.
    // Ref: contour_graph.cpp:172-181 IsPointsConnectFreePolygon (local check branch)
    // A polygon blocks an edge only if the edge center is on the OPPOSITE side
    // from the robot (i.e., inside the polygon when robot is outside, or vice versa).
    // This allows edges within the same "free space" region (including room interiors)
    // even if the polygon encloses both endpoints.
    // Additionally, skip the source polygon of either endpoint (pointer-match within
    // the current frame — works because CTNodes hold refs to current-frame polygons).
    auto skip1 = n1->ctnode ? n1->ctnode->poly_ptr : nullptr;
    auto skip2 = n2->ctnode ? n2->ctnode->poly_ptr : nullptr;
    Point3D center((n1->position.x + n2->position.x) * 0.5f,
                   (n1->position.y + n2->position.y) * 0.5f,
                   (n1->position.z + n2->position.z) * 0.5f);
    for (const auto& poly : g_contour_polygons) {
        if (poly->is_pillar) continue;
        if (poly == skip1 || poly == skip2) continue;
        // Center-point same-side test: block only if center is on opposite side from robot
        // Ref: contour_graph.cpp:175 is_robot_inside != PointInsideAPoly(vertices, center_p)
        bool center_inside = G.PointInsideAPoly(poly->vertices, center);
        if (poly->is_robot_inside != center_inside) {
            // Center is on the opposite side from the robot — also check edge collision
            if (IsEdgeCollidePoly(poly->vertices, cedge)) return false;
        }
    }
    return true;
}
#else
// Without OpenCV, provide stub that always returns true
static bool IsNavNodesConnectFreePolygon(const NavNodePtr&, const NavNodePtr&) { return true; }
#endif

// ---------------------------------------------------------------------------
//  Dijkstra-based traversability + A* path planning
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/src/graph_planner.cpp
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/graph_planner.h
// ---------------------------------------------------------------------------
struct GraphPlanner {
    NavNodePtr odom_node = nullptr;
    NavNodePtr goal_node = nullptr;
    Point3D origin_goal_pos;
    bool is_goal_init = false;
    bool is_use_internav_goal = false;
    bool is_global_path_init = false;
    float converge_dist = 1.0f;
    NodePtrStack current_graph;
    NodePtrStack recorded_path;
    Point3D next_waypoint;
    int path_momentum_counter = 0;
    int momentum_thred = 5;

    void UpdateGraphTraverability(const NavNodePtr& odom, const NavNodePtr& goal_ptr) {
        if (!odom || current_graph.empty()) return;
        odom_node = odom;
        // Debug: verify goal is in graph
        // Init all node states
        for (auto& n : current_graph) {
            n->gscore = G.kINF; n->fgscore = G.kINF;
            n->is_traversable = false; n->is_free_traversable = false;
            n->parent = nullptr; n->free_parent = nullptr;
        }
        // Dijkstra from odom
        odom_node->gscore = 0.0f;
        IdxSet open_set, close_set;
        std::priority_queue<NavNodePtr, NodePtrStack, nodeptr_gcomp> oq;
        oq.push(odom_node); open_set.insert(odom_node->id);
        while (!open_set.empty()) {
            auto cur = oq.top(); oq.pop();
            open_set.erase(cur->id); close_set.insert(cur->id);
            cur->is_traversable = true;
            for (const auto& nb : cur->connect_nodes) {
                if (close_set.count(nb->id)) continue;
                float ed = (cur->position - nb->position).norm();
                float tg = cur->gscore + ed;
                if (tg < nb->gscore) {
                    nb->parent = cur; nb->gscore = tg;
                    if (!open_set.count(nb->id)) { oq.push(nb); open_set.insert(nb->id); }
                }
            }
        }
        // Free-space expansion
        odom_node->fgscore = 0.0f;
        IdxSet fopen, fclose;
        std::priority_queue<NavNodePtr, NodePtrStack, nodeptr_fgcomp> fq;
        fq.push(odom_node); fopen.insert(odom_node->id);
        while (!fopen.empty()) {
            auto cur = fq.top(); fq.pop();
            fopen.erase(cur->id); fclose.insert(cur->id);
            cur->is_free_traversable = true;
            for (const auto& nb : cur->connect_nodes) {
                if (!nb->is_covered || fclose.count(nb->id)) continue;
                float ed = (cur->position - nb->position).norm();
                float tfg = cur->fgscore + ed;
                if (tfg < nb->fgscore) {
                    nb->free_parent = cur; nb->fgscore = tfg;
                    if (!fopen.count(nb->id)) { fq.push(nb); fopen.insert(nb->id); }
                }
            }
        }
    }

    void UpdateGoalConnects(const NavNodePtr& goal_ptr) {
        if (!goal_ptr || is_use_internav_goal) return;
        int poly_blocked = 0, ray_blocked = 0, connected = 0;
        for (const auto& n : current_graph) {
            if (n == goal_ptr) continue;
            // Connect goal to any visible node (not just traversable ones).
            // Traversability is determined by Dijkstra from odom — limiting
            // goal edges to traversable nodes creates a circular dependency
            // when the odom-to-graph connectivity is sparse.
            bool poly_free = IsNavNodesConnectFreePolygon(n, goal_ptr);
            bool ray_free = true;
            // Ray-cast sanity check against accumulated obstacle cloud.
            if (poly_free && g_obstacle_raycast) {
                if (g_obstacle_raycast(n->position, goal_ptr->position)) {
                    poly_free = false;
                    ray_free = false;
                }
            }
            if (poly_free) {
                AddPolyEdge(n, goal_ptr); AddEdge(n, goal_ptr);
                n->is_block_to_goal = false;
                connected++;
            } else {
                ErasePolyEdge(n, goal_ptr); EraseEdge(n, goal_ptr);
                n->is_block_to_goal = true;
                if (!ray_free) ray_blocked++;
                else poly_blocked++;
            }
        }
        static int goal_dbg_ctr = 0;
        if (++goal_dbg_ctr % 5 == 0) {
            printf("[FAR] GoalConnect: goal=(%.2f,%.2f) connected=%d poly_blocked=%d ray_blocked=%d total=%zu\n",
                   goal_ptr->position.x, goal_ptr->position.y,
                   connected, poly_blocked, ray_blocked, current_graph.size());
            fflush(stdout);
        }
    }

    bool ReconstructPath(const NavNodePtr& goal_ptr, NodePtrStack& path) {
        if (!goal_ptr || !goal_ptr->parent) return false;
        path.clear();
        NavNodePtr c = goal_ptr;
        path.push_back(c);
        while (c->parent) { path.push_back(c->parent); c = c->parent; }
        std::reverse(path.begin(), path.end());
        return true;
    }

    NavNodePtr NextWaypoint(const NodePtrStack& path, const NavNodePtr& goal_ptr) {
        if (path.size()<2) return goal_ptr;
        std::size_t idx = 1;
        NavNodePtr wp = path[idx];
        float dist = (wp->position - odom_node->position).norm();
        while (dist < converge_dist && idx+1 < path.size()) {
            idx++; wp = path[idx];
            dist = (wp->position - odom_node->position).norm();
        }
        return wp;
    }

    void UpdateGoal(const Point3D& goal) {
        GoalReset();
        is_use_internav_goal = false;
        // Reproject goal if inside obstacle polygon — ref: graph_planner.cpp:355 ReEvaluateGoalPosition
        // Ref: contour_graph.cpp:546 ReprojectPointOutsidePolygons
        Point3D adj_goal = goal;
        ReprojectPointOutsidePolygons(adj_goal, G.kNearDist);
        // Check if near an existing internav node
        float min_dist = G.kNearDist;
        for (const auto& n : current_graph) {
            if (n->is_navpoint) {
                float d = (n->position - adj_goal).norm();
                if (d < min_dist) {
                    is_use_internav_goal = true;
                    goal_node = n;
                    min_dist = d;
                    goal_node->is_goal = true;
                }
            }
        }
        if (!is_use_internav_goal) {
            CreateNavNodeFromPoint(adj_goal, goal_node, false, false, true);
            AddNodeToGraph(goal_node);
        }
        is_goal_init = true;
        is_global_path_init = false;
        origin_goal_pos = goal_node->position;
        path_momentum_counter = 0;
        recorded_path.clear();
        printf("[FAR] New goal set at (%.2f, %.2f, %.2f)  input=(%.2f, %.2f)  %s\n",
               goal_node->position.x, goal_node->position.y, goal_node->position.z,
               goal.x, goal.y,
               is_use_internav_goal ? "(snapped to internav)" : "(new node)");
    }

    bool PathToGoal(const NavNodePtr& goal_ptr, NodePtrStack& global_path,
                    NavNodePtr& nav_wp, Point3D& goal_p,
                    bool& is_fail, bool& is_succeed) {
        if (!is_goal_init || !odom_node || !goal_ptr || current_graph.empty()) return false;
        is_fail = false; is_succeed = false;
        global_path.clear();
        goal_p = goal_ptr->position;

        // Use 2D (flat) distance for goal convergence — goal z may differ from robot z
        // (e.g. clicked_point has z=0 but robot is at z=1.24)
        if ((odom_node->position - goal_p).norm_flat() < converge_dist ||
            (odom_node->position - origin_goal_pos).norm_flat() < converge_dist) {
            is_succeed = true;
            global_path.push_back(odom_node);
            global_path.push_back(goal_ptr);
            nav_wp = goal_ptr;
            GoalReset();
            is_goal_init = false;
            printf("[FAR] *** Goal Reached! ***\n");
            return true;
        }

        if (goal_ptr->parent) {
            NodePtrStack path;
            if (ReconstructPath(goal_ptr, path)) {
                NavNodePtr new_wp = NextWaypoint(path, goal_ptr);
                // Momentum: if new waypoint direction is inconsistent with previous
                // (heading dot product < 0), use recorded path for up to momentum_thred cycles.
                // Ref: graph_planner.cpp:198-223 momentum navigation
                if (is_global_path_init && path_momentum_counter < momentum_thred && !recorded_path.empty()) {
                    // Compare travel direction: old = (prev_wp - odom), new = (new_wp - odom).
                    // dot < 0 ⟹ new path points opposite the previous first hop (heading reversal).
                    NavNodePtr prev_wp = (recorded_path.size() >= 2) ? recorded_path[1] : nullptr;
                    if (prev_wp && new_wp != prev_wp) {
                        Point3D old_dir = prev_wp->position - odom_node->position;
                        Point3D new_dir = new_wp->position - odom_node->position;
                        float dot = old_dir.x*new_dir.x + old_dir.y*new_dir.y;
                        if (dot < 0.0f) {
                            // Heading reversal — use momentum path
                            global_path = recorded_path;
                            nav_wp = NextWaypoint(recorded_path, goal_ptr);
                            path_momentum_counter++;
                            return true;
                        }
                    }
                }
                // Accept new path
                nav_wp = new_wp;
                global_path = path;
                recorded_path = path;
                is_global_path_init = true;
                path_momentum_counter = 0;
                return true;
            }
        }
        // No path found
        if (is_global_path_init && path_momentum_counter < momentum_thred) {
            global_path = recorded_path;
            nav_wp = NextWaypoint(global_path, goal_ptr);
            path_momentum_counter++;
            return true;
        }
        // Don't reset the goal — keep it alive so we can retry once the
        // visibility graph grows (robot needs to move first).
        is_fail = true;
        return false;
    }

    void GoalReset() {
        if (goal_node && !is_use_internav_goal) {
            // Remove goal from graph
            for (auto& cn : goal_node->connect_nodes) G.EraseNodeFromStack(goal_node, cn->connect_nodes);
            for (auto& pn : goal_node->poly_connects) G.EraseNodeFromStack(goal_node, pn->poly_connects);
            goal_node->connect_nodes.clear();
            goal_node->poly_connects.clear();
            G.EraseNodeFromStack(goal_node, g_global_graph_nodes);
        } else if (goal_node) {
            goal_node->is_goal = false;
        }
        goal_node = nullptr;
    }
};

// ---------------------------------------------------------------------------
//  Dynamic graph manager — simplified
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/src/dynamic_graph.cpp
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/dynamic_graph.h
// ---------------------------------------------------------------------------
struct DynamicGraphManager {
    NavNodePtr odom_node = nullptr;
    NavNodePtr cur_internav = nullptr;
    NavNodePtr last_internav = nullptr;
    NodePtrStack near_nav_nodes, wide_near_nodes, extend_match_nodes;
    NodePtrStack new_nodes;
    Point3D last_connect_pos;
    int finalize_thred = 3;
    int votes_size = 10;
    int dumper_thred = 3;

    void UpdateRobotPosition(const Point3D& rp) {
        if (!odom_node) {
            CreateNavNodeFromPoint(rp, odom_node, true);
            AddNodeToGraph(odom_node);
        } else {
            odom_node->position = rp;
            odom_node->pos_filter_vec.clear();
            odom_node->pos_filter_vec.push_back(rp);
        }
        G.odom_pos = odom_node->position;
    }

    void UpdateGlobalNearNodes() {
        near_nav_nodes.clear(); wide_near_nodes.clear(); extend_match_nodes.clear();
        for (auto& n : g_global_graph_nodes) {
            n->is_near_nodes = false; n->is_wide_near = false;
            if (G.IsNodeInExtendMatchRange(n)) {
                if (G.IsOutsideGoal(n)) continue;
                extend_match_nodes.push_back(n);
                if (G.IsNodeInLocalRange(n)) {
                    wide_near_nodes.push_back(n); n->is_wide_near = true;
                    if (n->is_active || n->is_boundary) {
                        near_nav_nodes.push_back(n); n->is_near_nodes = true;
                    }
                }
            }
        }
    }

    // Remove stale contour-match nodes that haven't been re-observed.
    // Called after ExtractGraphNodes so that matched nodes have had
    // their clear_dumper_count reset.
    void CleanupStaleNodes() {
        NodePtrStack keep;
        keep.reserve(g_global_graph_nodes.size());
        for (auto& n : g_global_graph_nodes) {
            if (n->is_odom || n->is_navpoint || n->is_goal || n->is_finalized) {
                keep.push_back(n);
                continue;
            }
            // Contour-match nodes in local range that weren't re-matched
            // get their dumper count incremented.
            if (n->is_contour_match && n->is_wide_near) {
                n->clear_dumper_count++;
            }
            if (n->clear_dumper_count > dumper_thred) {
                // Remove all edges to/from this node
                for (auto& other : n->connect_nodes) {
                    other->connect_nodes.erase(
                        std::remove(other->connect_nodes.begin(), other->connect_nodes.end(), n),
                        other->connect_nodes.end());
                    other->poly_connects.erase(
                        std::remove(other->poly_connects.begin(), other->poly_connects.end(), n),
                        other->poly_connects.end());
                }
                n->connect_nodes.clear();
                n->poly_connects.clear();
                // Don't add to keep — node is removed
            } else {
                keep.push_back(n);
            }
        }
        if (keep.size() != g_global_graph_nodes.size()) {
            g_global_graph_nodes = std::move(keep);
            // Rebuild index map
            g_idx_node_map.clear();
            for (auto& n : g_global_graph_nodes) {
                g_idx_node_map[n->id] = n;
            }
        }
    }

    bool ExtractGraphNodes() {
        new_nodes.clear();
        // Check if we need a trajectory waypoint
        if (!cur_internav || (G.free_odom_p - last_connect_pos).norm() > G.kNearDist) {
            NavNodePtr np;
            CreateNavNodeFromPoint(G.free_odom_p, np, false, true);
            new_nodes.push_back(np);
            last_connect_pos = G.free_odom_p;
            if (!cur_internav) cur_internav = np;
            last_internav = cur_internav;
            cur_internav = np;
            DBG("internav-node created at (%.2f,%.2f)\n",
                G.free_odom_p.x, G.free_odom_p.y);
        }

        // Per-call stats for contour node generation
        int ct_total = 0, ct_pillar = 0, ct_far = 0, ct_badz = 0, ct_zero_dir = 0;
        int created = 0, updated = 0, too_close = 0;

        // Promote contour vertices to nav graph nodes, or update existing
        // nodes that match.  The original does this via MatchContourWithNavGraph
        // with RANSAC position filtering.  This simplified version either:
        //   - updates an existing contour-match node's position (running average)
        //   - creates a new nav node if no match exists
        for (const auto& ct : g_contour_graph) {
            ct_total++;
            if (ct->free_direct == PILLAR) { ct_pillar++; continue; }

            // Only process vertices within sensor range
            if ((ct->position - G.robot_pos).norm_flat() > G.kSensorRange) { ct_far++; continue; }

            // Compute offset direction into free space.
            //
            // For CONCAVE vertices, surf_dirs already point into free space
            // (see IsInCoverageDirPairs line ~617). For CONVEX vertices, the
            // same function uses NEGATED surf_dirs, meaning the raw vectors
            // point into the obstacle. Flip the sign accordingly so nav_pos
            // is always pushed out of the wall.
            Point3D offset_dir;
            if (ct->surf_dirs.first.z > -0.5f && ct->surf_dirs.second.z > -0.5f) {
                Point3D raw_sum = ct->surf_dirs.first + ct->surf_dirs.second;
                if (ct->free_direct == CONVEX) raw_sum = -raw_sum;
                offset_dir = raw_sum;
                float len = offset_dir.norm_flat();
                if (len > EPSILON_VAL) {
                    offset_dir = offset_dir * (1.0f / len);
                } else {
                    ct_zero_dir++;
                    DBG("vertex skipped: colinear surf_dirs at (%.2f,%.2f) "
                        "dirs=[(%.2f,%.2f),(%.2f,%.2f)] free_direct=%d\n",
                        ct->position.x, ct->position.y,
                        ct->surf_dirs.first.x, ct->surf_dirs.first.y,
                        ct->surf_dirs.second.x, ct->surf_dirs.second.y,
                        (int)ct->free_direct);
                    continue;
                }
            } else {
                ct_badz++;
                continue;
            }
            Point3D nav_pos = ct->position + offset_dir * G.kNavClearDist;

            // Try to match with an existing contour-derived node
            NavNodePtr matched = nullptr;
            float best_dist = G.kMatchDist;
            for (const auto& existing : g_global_graph_nodes) {
                if (!existing->is_contour_match) continue;
                float d = (existing->position - nav_pos).norm_flat();
                if (d < best_dist) {
                    best_dist = d;
                    matched = existing;
                }
            }

            if (matched) {
                // Update existing node with running average (simple position filter)
                float alpha = 0.3f;
                Point3D pre_pos = matched->position;
                matched->position.x = matched->position.x * (1-alpha) + nav_pos.x * alpha;
                matched->position.y = matched->position.y * (1-alpha) + nav_pos.y * alpha;
                matched->surf_dirs = ct->surf_dirs;
                matched->free_direct = ct->free_direct;
                matched->ctnode = ct;
                ct->is_global_match = true;  // mark CTNode as matched
                matched->is_active = true;
                matched->is_covered = true;  // re-observed → mark covered for Dijkstra
                matched->is_boundary = false;  // boundary flag is for boundary handler module
                matched->clear_dumper_count = 0;  // reset stale counter
                updated++;
                DBG("node-update id=%d was=(%.2f,%.2f) → (%.2f,%.2f)  "
                    "ct=(%.2f,%.2f) off=(%.2f,%.2f)×%.2f free_direct=%d\n",
                    matched->id, pre_pos.x, pre_pos.y,
                    matched->position.x, matched->position.y,
                    ct->position.x, ct->position.y,
                    offset_dir.x, offset_dir.y, G.kNavClearDist,
                    (int)ct->free_direct);
            } else {
                // Also check against new nodes added this cycle
                bool close_to_new = false;
                for (const auto& nn : new_nodes) {
                    if ((nn->position - nav_pos).norm_flat() < G.kMatchDist) {
                        close_to_new = true;
                        break;
                    }
                }
                if (close_to_new) { too_close++; continue; }

                NavNodePtr np;
                CreateNavNodeFromPoint(nav_pos, np, false, false);
                np->surf_dirs = ct->surf_dirs;
                np->free_direct = ct->free_direct;
                np->is_contour_match = true;
                np->is_covered = true;  // visible → eligible for Dijkstra expansion
                np->is_boundary = false;  // boundary flag is for boundary handler module
                np->ctnode = ct;
                ct->is_global_match = true;  // mark CTNode as matched
                new_nodes.push_back(np);
                created++;
                DBG("node-create id=%d at (%.2f,%.2f)  ct=(%.2f,%.2f) "
                    "off=(%.2f,%.2f)×%.2f free_direct=%d surf=[(%.2f,%.2f),(%.2f,%.2f)]\n",
                    np->id, np->position.x, np->position.y,
                    ct->position.x, ct->position.y,
                    offset_dir.x, offset_dir.y, G.kNavClearDist,
                    (int)ct->free_direct,
                    ct->surf_dirs.first.x, ct->surf_dirs.first.y,
                    ct->surf_dirs.second.x, ct->surf_dirs.second.y);
            }
        }
        DBG_EVERY(10, "extract summary: contours=%d pillar=%d far=%d bad_z=%d "
                      "zero_dir=%d too_close=%d  created=%d updated=%d  "
                      "kNavClearDist=%.2f\n",
                  ct_total, ct_pillar, ct_far, ct_badz,
                  ct_zero_dir, too_close, created, updated, G.kNavClearDist);
        // Establish contour connections between adjacent matched nav nodes
        // on the same polygon. This populates g_boundary_contour_set via
        // AddContourConnect → AddContourToSets, giving the collision check
        // line-segment barriers that block paths through wall gaps.
        // Ref: dynamic_graph.cpp:237-238 RecordContourVote
        // Ref: dynamic_graph.cpp:551 AddContourConnect
        // Ref: contour_graph.cpp:218-254 IsCTNodesConnectFromContour
        for (const auto& n1 : g_global_graph_nodes) {
            if (!n1->is_contour_match || !n1->ctnode) continue;
            for (const auto& n2 : g_global_graph_nodes) {
                if (n2 == n1 || !n2->is_contour_match || !n2->ctnode) continue;
                if (n1->ctnode->poly_ptr != n2->ctnode->poly_ptr) continue;
                if (G.IsTypeInStack(n2, n1->contour_connects)) continue;
                // Walk front from n1→ctnode to see if we reach n2→ctnode
                // without crossing another matched ctnode (adjacency check)
                bool adjacent = false;
                CTNodePtr cur = n1->ctnode->front;
                while (cur && cur != n1->ctnode) {
                    if (cur == n2->ctnode) { adjacent = true; break; }
                    if (cur->is_global_match) break;
                    cur = cur->front;
                }
                if (!adjacent) {
                    cur = n1->ctnode->back;
                    while (cur && cur != n1->ctnode) {
                        if (cur == n2->ctnode) { adjacent = true; break; }
                        if (cur->is_global_match) break;
                        cur = cur->back;
                    }
                }
                if (adjacent) {
                    AddContourConnect(n1, n2);
                }
            }
        }

        return !new_nodes.empty();
    }

    void UpdateNavGraph(const NodePtrStack& new_nodes_in, bool is_freeze) {
        if (is_freeze) return;
        // Add new nodes
        for (const auto& nn : new_nodes_in) {
            AddNodeToGraph(nn);
            nn->is_near_nodes = true;
            nn->is_wide_near = true;
            near_nav_nodes.push_back(nn);
            wide_near_nodes.push_back(nn);
        }
        // Build visibility edges between odom and near nodes
        for (const auto& n : wide_near_nodes) {
            if (n->is_odom) continue;
            bool poly_free = IsNavNodesConnectFreePolygon(odom_node, n);
            if (poly_free && g_obstacle_raycast) {
                if (g_obstacle_raycast(odom_node->position, n->position))
                    poly_free = false;
            }
            if (poly_free) {
                AddPolyEdge(odom_node, n); AddEdge(odom_node, n);
            } else {
                ErasePolyEdge(odom_node, n); EraseEdge(odom_node, n);
            }
        }
        // Connect near nodes to each other
        for (std::size_t i=0; i<near_nav_nodes.size(); i++) {
            auto n1 = near_nav_nodes[i];
            if (n1->is_odom) continue;
            for (std::size_t j=i+1; j<near_nav_nodes.size(); j++) {
                auto n2 = near_nav_nodes[j];
                if (n2->is_odom) continue;
                bool n_poly_free = IsNavNodesConnectFreePolygon(n1, n2);
                if (n_poly_free && g_obstacle_raycast) {
                    if (g_obstacle_raycast(n1->position, n2->position))
                        n_poly_free = false;
                }
                if (n_poly_free) {
                    AddPolyEdge(n1, n2); AddEdge(n1, n2);
                } else {
                    ErasePolyEdge(n1, n2); EraseEdge(n1, n2);
                }
            }
        }
    }

    const NodePtrStack& GetNavGraph() const { return g_global_graph_nodes; }
    NavNodePtr GetOdomNode() const { return odom_node; }

    void ResetCurrentGraph() {
        odom_node = nullptr; cur_internav = nullptr; last_internav = nullptr;
        g_id_tracker = 1;
        g_idx_node_map.clear();
        near_nav_nodes.clear(); wide_near_nodes.clear(); extend_match_nodes.clear();
        new_nodes.clear();
        g_global_graph_nodes.clear();
    }
};

// ---------------------------------------------------------------------------
//  Contour detector — simplified OpenCV contour extraction
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/src/contour_detector.cpp
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/contour_detector.h
//  (Only built with HAS_OPENCV)
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV
struct ContourDetector {
    float sensor_range = 30.0f;
    float voxel_dim = 0.2f;
    float kRatio = 5.0f;
    int kThredValue = 5;
    int kBlurSize = 3;
    int MAT_SIZE, CMAT, MAT_RESIZE, CMAT_RESIZE;
    float DIST_LIMIT, ALIGN_ANGLE_COS, VOXEL_DIM_INV;
    Point3D odom_pos;
    cv::Mat img_mat;
    std::vector<std::vector<cv::Point2f>> refined_contours;
    std::vector<cv::Vec4i> refined_hierarchy;

    void Init() {
        MAT_SIZE = (int)std::ceil(sensor_range*2.0f/voxel_dim);
        if (MAT_SIZE%2==0) MAT_SIZE++;
        MAT_RESIZE = MAT_SIZE*(int)kRatio;
        CMAT = MAT_SIZE/2; CMAT_RESIZE = MAT_RESIZE/2;
        img_mat = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
        DIST_LIMIT = kRatio * 1.2f;
        ALIGN_ANGLE_COS = cos(G.kAcceptAlign/2.0f);
        VOXEL_DIM_INV = 1.0f/voxel_dim;
    }

    void PointToImgSub(const Point3D& p, int& row, int& col, bool resized=false) {
        float ratio = resized ? kRatio : 1.0f;
        int ci = resized ? CMAT_RESIZE : CMAT;
        row = ci + (int)std::round((p.x-odom_pos.x)*VOXEL_DIM_INV*ratio);
        col = ci + (int)std::round((p.y-odom_pos.y)*VOXEL_DIM_INV*ratio);
        int ms = resized ? MAT_RESIZE : MAT_SIZE;
        row = std::max(0, std::min(row, ms-1));
        col = std::max(0, std::min(col, ms-1));
    }

    Point3D CVToPoint3D(const cv::Point2f& cv_p) {
        Point3D p;
        p.x = (cv_p.y - CMAT_RESIZE)*voxel_dim/kRatio + odom_pos.x;
        p.y = (cv_p.x - CMAT_RESIZE)*voxel_dim/kRatio + odom_pos.y;
        p.z = odom_pos.z;
        return p;
    }

    // Build 2D occupancy image from obstacle cloud, extract contours
    void BuildAndExtract(const Point3D& odom_p,
                         const std::vector<smartnav::PointXYZI>& obs_points,
                         std::vector<PointStack>& realworld_contours) {
        odom_pos = odom_p;
        img_mat = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
        // Project points into image
        for (const auto& pp : obs_points) {
            Point3D p3(pp.x, pp.y, pp.z);
            int r, c;
            PointToImgSub(p3, r, c, false);
            if (r>=0 && r<MAT_SIZE && c>=0 && c<MAT_SIZE) {
                // Inflate each point by ±1 pixel (matching reference).
                // Ref: contour_detector.cpp:44 inflate_vec{-1, 0, 1}
                for (int dr=-1; dr<=1; dr++) for (int dc=-1; dc<=1; dc++) {
                    int rr=r+dr, cc=c+dc;
                    if (rr>=0&&rr<MAT_SIZE&&cc>=0&&cc<MAT_SIZE) img_mat.at<float>(rr,cc)+=1.0f;
                }
            }
        }
        if (G.is_static_env) {
            // no threshold for static
        } else {
            cv::threshold(img_mat, img_mat, kThredValue, 1.0, cv::ThresholdTypes::THRESH_BINARY);
        }
        // Resize and blur
        cv::Mat rimg;
        img_mat.convertTo(rimg, CV_8UC1, 255);
        cv::resize(rimg, rimg, cv::Size(), kRatio, kRatio, cv::INTER_LINEAR);
        cv::boxFilter(rimg, rimg, -1, cv::Size(kBlurSize, kBlurSize), cv::Point(-1,-1), false);
        // Find contours
        std::vector<std::vector<cv::Point2i>> raw_contours;
        refined_hierarchy.clear();
        cv::findContours(rimg, raw_contours, refined_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_L1);
        refined_contours.resize(raw_contours.size());
        for (std::size_t i=0; i<raw_contours.size(); i++) {
            cv::approxPolyDP(raw_contours[i], refined_contours[i], DIST_LIMIT, true);
        }
        // Convert to real-world coordinates
        realworld_contours.clear();
        realworld_contours.resize(refined_contours.size());
        for (std::size_t i=0; i<refined_contours.size(); i++) {
            for (const auto& cvp : refined_contours[i]) {
                realworld_contours[i].push_back(CVToPoint3D(cvp));
            }
        }
    }
};
#endif

// ---------------------------------------------------------------------------
//  Contour graph manager — simplified
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/src/contour_graph.cpp
//  Ref: ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/contour_graph.h
// ---------------------------------------------------------------------------
struct ContourGraphManager {
    NavNodePtr odom_node = nullptr;
    float kPillarPerimeter = 3.2f;

    void UpdateContourGraph(const NavNodePtr& odom, const std::vector<PointStack>& contours) {
        odom_node = odom;
        g_contour_graph.clear();
        g_contour_polygons.clear();
        g_polys_ctnodes.clear();
        for (const auto& poly_pts : contours) {
            if (poly_pts.size() < 3) continue;
            auto poly = std::make_shared<Polygon>();
            poly->N = poly_pts.size();
            poly->vertices = poly_pts;
            poly->is_robot_inside = G.PointInsideAPoly(poly_pts, odom->position);
            // Check if pillar
            float perim = 0;
            for (std::size_t i=1; i<poly_pts.size(); i++)
                perim += (poly_pts[i]-poly_pts[i-1]).norm_flat();
            poly->perimeter = perim;
            poly->is_pillar = (perim <= kPillarPerimeter);
            g_contour_polygons.push_back(poly);

            if (poly->is_pillar) {
                auto ct = std::make_shared<CTNode>();
                ct->position = G.AveragePoints(poly_pts);
                ct->is_global_match = false;
                ct->is_contour_necessary = false;
                ct->is_ground_associate = false;
                ct->nav_node_id = 0;
                ct->free_direct = PILLAR;
                ct->poly_ptr = poly;
                ct->front = nullptr; ct->back = nullptr;
                g_contour_graph.push_back(ct);
            } else {
                CTNodeStack ctstack;
                int N = (int)poly_pts.size();
                for (int idx=0; idx<N; idx++) {
                    auto ct = std::make_shared<CTNode>();
                    ct->position = poly_pts[idx];
                    ct->is_global_match = false;
                    ct->is_contour_necessary = false;
                    ct->is_ground_associate = false;
                    ct->nav_node_id = 0;
                    ct->free_direct = UNKNOW;
                    ct->poly_ptr = poly;
                    ct->front = nullptr; ct->back = nullptr;
                    ctstack.push_back(ct);
                }
                for (int idx=0; idx<N; idx++) {
                    ctstack[idx]->front = ctstack[G.Mod(idx-1,N)];
                    ctstack[idx]->back  = ctstack[G.Mod(idx+1,N)];
                    g_contour_graph.push_back(ctstack[idx]);
                }
                if (!ctstack.empty()) g_polys_ctnodes.push_back(ctstack.front());
            }
        }
        // Analyse surface angles and convexity
        for (auto& ct : g_contour_graph) {
            if (ct->free_direct == PILLAR || ct->poly_ptr->is_pillar) {
                ct->surf_dirs = {Point3D(0,0,-1), Point3D(0,0,-1)};
                ct->free_direct = PILLAR;
                continue;
            }
            // Front direction
            auto next = ct->front;
            float ed = (next->position - ct->position).norm_flat();
            Point3D sp = ct->position, ep = next->position;
            while (next && next!=ct && ed < G.kNavClearDist) {
                sp = ep; next = next->front; ep = next->position;
                ed = (ep - ct->position).norm_flat();
            }
            if (ed < G.kNavClearDist) {
                ct->surf_dirs = {Point3D(0,0,-1), Point3D(0,0,-1)};
                ct->free_direct = PILLAR; continue;
            }
            ct->surf_dirs.first = G.ContourSurfDirsVec(ep, sp, ct->position, G.kNavClearDist);
            // Back direction
            next = ct->back;
            sp = ct->position; ep = next->position;
            ed = (ep - ct->position).norm_flat();
            while (next && next!=ct && ed < G.kNavClearDist) {
                sp = ep; next = next->back; ep = next->position;
                ed = (ep - ct->position).norm_flat();
            }
            if (ed < G.kNavClearDist) {
                ct->surf_dirs = {Point3D(0,0,-1), Point3D(0,0,-1)};
                ct->free_direct = PILLAR; continue;
            }
            ct->surf_dirs.second = G.ContourSurfDirsVec(ep, sp, ct->position, G.kNavClearDist);
            // Convexity analysis
            Point3D topo = G.SurfTopoDirect(ct->surf_dirs);
            if (topo.norm_flat() < G.kEpsilon) { ct->free_direct = UNKNOW; continue; }
            Point3D ev_p = ct->position + topo * G.kLeafSize;
            ct->free_direct = G.IsConvexPoint(ct->poly_ptr, ev_p) ? CONVEX : CONCAVE;
        }
    }

    void ExtractGlobalContours() {
        g_global_contour.clear();
        g_boundary_contour.clear();
        g_local_boundary.clear();
        g_inactive_contour.clear();
        g_unmatched_contour.clear();
        for (const auto& e : g_global_contour_set) {
            g_global_contour.push_back({e.first->position, e.second->position});
        }
        for (const auto& e : g_boundary_contour_set) {
            g_boundary_contour.push_back({e.first->position, e.second->position});
        }
    }

    void ResetCurrentContour() {
        g_contour_graph.clear();
        g_contour_polygons.clear();
        g_polys_ctnodes.clear();
        g_global_contour_set.clear();
        g_boundary_contour_set.clear();
        odom_node = nullptr;
    }
};

// ---------------------------------------------------------------------------
//  MapHandler — persistent 3D world grid for obstacle/free-space accumulation
//  Port of:
//    ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/map_handler.h
//    ros-navigation-autonomy-stack/src/route_planner/far_planner/src/map_handler.cpp
//  Grid utility:
//    ros-navigation-autonomy-stack/src/route_planner/far_planner/include/far_planner/grid.h
//
//  Accumulates obstacle and free-space points across frames so the contour
//  detector receives dense, properly-closed wall polygons.
// ---------------------------------------------------------------------------
#ifdef USE_PCL
struct MapHandler {
    // Configuration (set before Init)
    float sensor_range_ = 15.0f;
    float floor_height_ = 1.5f;
    float cell_length_ = 1.5f;
    float cell_height_ = 0.6f;
    float grid_max_length_ = 300.0f;
    float grid_max_height_ = 3.0f;
    float height_voxel_dim_ = 0.2f;

    // Grids
    std::unique_ptr<grid_ns::Grid<PointCloudPtr>> world_obs_cloud_grid_;
    std::unique_ptr<grid_ns::Grid<PointCloudPtr>> world_free_cloud_grid_;
    std::unique_ptr<grid_ns::Grid<std::vector<float>>> terrain_height_grid_;

    // Neighbor cell indices (recomputed each frame around robot)
    std::unordered_set<int> neighbor_obs_indices_;
    std::unordered_set<int> neighbor_free_indices_;
    std::unordered_set<int> extend_obs_indices_;

    // State
    Eigen::Vector3i robot_cell_sub_;
    int neighbor_Lnum_ = 0;
    int neighbor_Hnum_ = 5;
    int INFLATE_N = 1;
    bool is_init_ = false;

    // Terrain KdTree (flat cloud with height stored in intensity)
    PointCloudPtr flat_terrain_cloud_;
    PointKdTreePtr kdtree_terrain_cloud_;

    // Per-cell tracking lists
    std::vector<int> global_visited_indices_;
    std::vector<int> util_obs_modified_list_;
    std::vector<int> util_free_modified_list_;
    std::vector<int> terrain_grid_occupy_list_;
    std::vector<int> terrain_grid_traverse_list_;

    // Ref: map_handler.cpp:13-65  MapHandler::Init
    void Init() {
        const int row_num = static_cast<int>(std::ceil(grid_max_length_ / cell_length_));
        const int col_num = row_num;
        int level_num = static_cast<int>(std::ceil(grid_max_height_ / cell_height_));
        neighbor_Lnum_ = static_cast<int>(std::ceil(sensor_range_ * 2.0f / cell_length_)) + 1;
        if (level_num % 2 == 0) level_num++;
        if (neighbor_Lnum_ % 2 == 0) neighbor_Lnum_++;

        Eigen::Vector3i pc_size(row_num, col_num, level_num);
        Eigen::Vector3d pc_origin(0, 0, 0);
        Eigen::Vector3d pc_res(cell_length_, cell_length_, cell_height_);
        PointCloudPtr null_cloud;
        world_obs_cloud_grid_ = std::make_unique<grid_ns::Grid<PointCloudPtr>>(
            pc_size, null_cloud, pc_origin, pc_res, 3);
        world_free_cloud_grid_ = std::make_unique<grid_ns::Grid<PointCloudPtr>>(
            pc_size, null_cloud, pc_origin, pc_res, 3);

        const int n_cell = world_obs_cloud_grid_->GetCellNumber();
        for (int i = 0; i < n_cell; i++) {
            world_obs_cloud_grid_->GetCell(i) = PointCloudPtr(new PointCloud);
            world_free_cloud_grid_->GetCell(i) = PointCloudPtr(new PointCloud);
        }
        global_visited_indices_.assign(n_cell, 0);
        util_obs_modified_list_.assign(n_cell, 0);
        util_free_modified_list_.assign(n_cell, 0);

        // Terrain height grid: 2D grid (z-dim=1) at robot_dim resolution
        int height_dim = static_cast<int>(
            std::ceil((sensor_range_ + cell_length_) * 2.0f / G.robot_dim));
        if (height_dim % 2 == 0) height_dim++;
        Eigen::Vector3i h_size(height_dim, height_dim, 1);
        Eigen::Vector3d h_origin(0, 0, 0);
        Eigen::Vector3d h_res(G.robot_dim, G.robot_dim, G.kLeafSize);
        std::vector<float> empty_vec;
        terrain_height_grid_ = std::make_unique<grid_ns::Grid<std::vector<float>>>(
            h_size, empty_vec, h_origin, h_res, 3);

        const int n_terrain = terrain_height_grid_->GetCellNumber();
        terrain_grid_occupy_list_.assign(n_terrain, 0);
        terrain_grid_traverse_list_.assign(n_terrain, 0);

        flat_terrain_cloud_ = PointCloudPtr(new PointCloud());
        kdtree_terrain_cloud_ = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
        kdtree_terrain_cloud_->setSortedResults(false);

        printf("[FAR] MapHandler: obs_grid=%dx%dx%d (%d cells)  "
               "terrain_grid=%dx%d  cell=%.1fm  sensor=%.1fm\n",
               row_num, col_num, level_num, n_cell,
               height_dim, height_dim, cell_length_, sensor_range_);
    }

    // Ref: map_handler.cpp:129-140  MapHandler::SetMapOrigin
    void SetMapOrigin(const Point3D& robot_pos) {
        const Eigen::Vector3i dim = world_obs_cloud_grid_->GetSize();
        float ox = robot_pos.x - (cell_length_ * dim.x()) / 2.0f;
        float oy = robot_pos.y - (cell_length_ * dim.y()) / 2.0f;
        float oz = robot_pos.z - (cell_height_ * dim.z()) / 2.0f - G.vehicle_height;
        Eigen::Vector3d origin(ox, oy, oz);
        world_obs_cloud_grid_->SetOrigin(origin);
        world_free_cloud_grid_->SetOrigin(origin);
        is_init_ = true;
        printf("[FAR] MapHandler: grid origin set at (%.2f, %.2f, %.2f)\n", ox, oy, oz);
    }

    // Ref: map_handler.cpp:172-181  MapHandler::SetTerrainHeightGridOrigin
    void SetTerrainHeightGridOrigin(const Point3D& robot_pos) {
        const Eigen::Vector3d res = terrain_height_grid_->GetResolution();
        const Eigen::Vector3i dim = terrain_height_grid_->GetSize();
        Eigen::Vector3d origin;
        origin.x() = robot_pos.x - (res.x() * dim.x()) / 2.0f;
        origin.y() = robot_pos.y - (res.y() * dim.y()) / 2.0f;
        origin.z() = 0.0f - (res.z() * dim.z()) / 2.0f;
        terrain_height_grid_->SetOrigin(origin);
    }

    // Ref: map_handler.cpp:142-170  MapHandler::UpdateRobotPosition
    void UpdateRobotPosition(const Point3D& odom_pos) {
        if (!is_init_) SetMapOrigin(odom_pos);
        robot_cell_sub_ = world_obs_cloud_grid_->Pos2Sub(
            Eigen::Vector3d(odom_pos.x, odom_pos.y, odom_pos.z));
        neighbor_free_indices_.clear();
        neighbor_obs_indices_.clear();
        const int N = neighbor_Lnum_ / 2;
        const int H = neighbor_Hnum_ / 2;
        Eigen::Vector3i nsub;
        for (int i = -N; i <= N; i++) {
            nsub.x() = robot_cell_sub_.x() + i;
            for (int j = -N; j <= N; j++) {
                nsub.y() = robot_cell_sub_.y() + j;
                // Extra layer below for terrain free points
                nsub.z() = robot_cell_sub_.z() - H - 1;
                if (world_obs_cloud_grid_->InRange(nsub)) {
                    neighbor_free_indices_.insert(world_obs_cloud_grid_->Sub2Ind(nsub));
                }
                for (int k = -H; k <= H; k++) {
                    nsub.z() = robot_cell_sub_.z() + k;
                    if (world_obs_cloud_grid_->InRange(nsub)) {
                        int ind = world_obs_cloud_grid_->Sub2Ind(nsub);
                        neighbor_obs_indices_.insert(ind);
                        neighbor_free_indices_.insert(ind);
                    }
                }
            }
        }
        SetTerrainHeightGridOrigin(odom_pos);
    }

    // Ref: map_handler.cpp:201-221  MapHandler::UpdateObsCloudGrid
    void UpdateObsCloudGrid(const PointCloudPtr& obsCloud) {
        if (!is_init_ || obsCloud->empty()) return;
        std::fill(util_obs_modified_list_.begin(), util_obs_modified_list_.end(), 0);
        for (const auto& point : obsCloud->points) {
            Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(
                Eigen::Vector3d(point.x, point.y, point.z));
            if (!world_obs_cloud_grid_->InRange(sub)) continue;
            const int ind = world_obs_cloud_grid_->Sub2Ind(sub);
            if (neighbor_obs_indices_.count(ind)) {
                world_obs_cloud_grid_->GetCell(ind)->points.push_back(point);
                util_obs_modified_list_[ind] = 1;
                global_visited_indices_[ind] = 1;
            }
        }
        // Voxel-filter modified cells to prevent unbounded growth
        for (int i = 0; i < world_obs_cloud_grid_->GetCellNumber(); ++i) {
            if (util_obs_modified_list_[i] == 1)
                G.FilterCloud(world_obs_cloud_grid_->GetCell(i), G.kLeafSize);
        }
    }

    // Ref: map_handler.cpp:223-238  MapHandler::UpdateFreeCloudGrid
    void UpdateFreeCloudGrid(const PointCloudPtr& freeCloud) {
        if (!is_init_ || freeCloud->empty()) return;
        std::fill(util_free_modified_list_.begin(), util_free_modified_list_.end(), 0);
        for (const auto& point : freeCloud->points) {
            Eigen::Vector3i sub = world_free_cloud_grid_->Pos2Sub(
                Eigen::Vector3d(point.x, point.y, point.z));
            if (!world_free_cloud_grid_->InRange(sub)) continue;
            const int ind = world_free_cloud_grid_->Sub2Ind(sub);
            world_free_cloud_grid_->GetCell(ind)->points.push_back(point);
            util_free_modified_list_[ind] = 1;
            global_visited_indices_[ind] = 1;
        }
        for (int i = 0; i < world_free_cloud_grid_->GetCellNumber(); ++i) {
            if (util_free_modified_list_[i] == 1)
                G.FilterCloud(world_free_cloud_grid_->GetCell(i), G.kLeafSize);
        }
    }

    // Ref: map_handler.cpp:183-190  MapHandler::GetSurroundObsCloud
    void GetSurroundObsCloud(const PointCloudPtr& out) {
        if (!is_init_) return;
        out->clear();
        for (const auto& ind : neighbor_obs_indices_) {
            if (world_obs_cloud_grid_->GetCell(ind)->empty()) continue;
            *out += *(world_obs_cloud_grid_->GetCell(ind));
        }
    }

    // Ref: map_handler.cpp:192-199  MapHandler::GetSurroundFreeCloud
    void GetSurroundFreeCloud(const PointCloudPtr& out) {
        if (!is_init_) return;
        out->clear();
        for (const auto& ind : neighbor_free_indices_) {
            if (world_free_cloud_grid_->GetCell(ind)->empty()) continue;
            *out += *(world_free_cloud_grid_->GetCell(ind));
        }
    }

    // Ref: map_handler.h:56-77  MapHandler::NearestHeightOfRadius
    template <typename Position>
    float NearestHeightOfRadius(const Position& p, float radius,
                                float& minH, float& maxH, bool& is_matched) {
        std::vector<int> pIdxK;
        std::vector<float> pdDistK;
        PCLPoint pcl_p;
        pcl_p.x = p.x; pcl_p.y = p.y; pcl_p.z = 0.0f; pcl_p.intensity = 0.0f;
        minH = maxH = p.z;
        is_matched = false;
        if (kdtree_terrain_cloud_->getInputCloud() &&
            !kdtree_terrain_cloud_->getInputCloud()->empty() &&
            kdtree_terrain_cloud_->radiusSearch(pcl_p, radius, pIdxK, pdDistK) > 0) {
            const auto& cloud = kdtree_terrain_cloud_->getInputCloud();
            float avgH = cloud->points[pIdxK[0]].intensity;
            minH = maxH = avgH;
            for (std::size_t i = 1; i < pIdxK.size(); i++) {
                float h = cloud->points[pIdxK[i]].intensity;
                if (h < minH) minH = h;
                if (h > maxH) maxH = h;
                avgH += h;
            }
            avgH /= static_cast<float>(pIdxK.size());
            is_matched = true;
            return avgH;
        }
        return p.z;
    }

    // Ref: map_handler.cpp:240-256  MapHandler::TerrainHeightOfPoint
    float TerrainHeightOfPoint(const Point3D& p, bool& is_matched) {
        is_matched = false;
        const Eigen::Vector3i sub = terrain_height_grid_->Pos2Sub(
            Eigen::Vector3d(p.x, p.y, 0.0f));
        if (terrain_height_grid_->InRange(sub)) {
            const int ind = terrain_height_grid_->Sub2Ind(sub);
            if (terrain_grid_traverse_list_[ind] != 0) {
                is_matched = true;
                return terrain_height_grid_->GetCell(ind)[0];
            }
        }
        return p.z;
    }

    void Expansion2D(const Eigen::Vector3i& csub,
                     std::vector<Eigen::Vector3i>& subs, int n) {
        subs.clear();
        for (int ix = -n; ix <= n; ix++) {
            for (int iy = -n; iy <= n; iy++) {
                Eigen::Vector3i sub = csub;
                sub.x() += ix;
                sub.y() += iy;
                subs.push_back(sub);
            }
        }
    }

    void AssignFlatTerrainCloud(const PointCloudPtr& ref, PointCloudPtr& flatOut) {
        const int N = static_cast<int>(ref->size());
        flatOut->resize(N);
        for (int i = 0; i < N; i++) {
            PCLPoint p = ref->points[i];
            p.intensity = p.z;  // store height in intensity
            p.z = 0.0f;        // flatten for 2D KdTree search
            flatOut->points[i] = p;
        }
    }

    // Ref: map_handler.cpp:391-422  MapHandler::UpdateTerrainHeightGrid
    void UpdateTerrainHeightGrid(const PointCloudPtr& freeCloud,
                                 const PointCloudPtr& heightOut) {
        if (freeCloud->empty()) return;
        // Voxel-filter free cloud at terrain grid resolution
        PointCloudPtr filtered(new PointCloud());
        pcl::copyPointCloud(*freeCloud, *filtered);
        {
            pcl::VoxelGrid<PCLPoint> vg;
            vg.setInputCloud(filtered);
            Eigen::Vector3d res = terrain_height_grid_->GetResolution();
            vg.setLeafSize(static_cast<float>(res.x()),
                           static_cast<float>(res.y()),
                           static_cast<float>(res.z()));
            PointCloud tmp;
            vg.filter(tmp);
            *filtered = tmp;
        }
        // Populate terrain height grid cells
        std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
        for (const auto& point : filtered->points) {
            Eigen::Vector3i csub = terrain_height_grid_->Pos2Sub(
                Eigen::Vector3d(point.x, point.y, 0.0f));
            std::vector<Eigen::Vector3i> subs;
            Expansion2D(csub, subs, INFLATE_N);
            for (const auto& sub : subs) {
                if (!terrain_height_grid_->InRange(sub)) continue;
                const int ind = terrain_height_grid_->Sub2Ind(sub);
                if (terrain_grid_occupy_list_[ind] == 0) {
                    terrain_height_grid_->GetCell(ind).resize(1);
                    terrain_height_grid_->GetCell(ind)[0] = point.z;
                } else {
                    terrain_height_grid_->GetCell(ind).push_back(point.z);
                }
                terrain_grid_occupy_list_[ind] = 1;
            }
        }
        // BFS traversable analysis from robot position
        TraversableAnalysis(heightOut);
        // Build flat terrain KdTree for height queries
        if (heightOut->empty()) {
            flat_terrain_cloud_->clear();
        } else {
            AssignFlatTerrainCloud(heightOut, flat_terrain_cloud_);
            kdtree_terrain_cloud_->setInputCloud(flat_terrain_cloud_);
        }
        // Filter obs neighbor indices by terrain height — only when terrain
        // data is dense enough to be reliable.  With sparse terrain (common
        // during early exploration or in sim), skipping this avoids discarding
        // valid obs cells that simply have no terrain height association.
        const int MIN_TERRAIN_PTS = 200;
        if (static_cast<int>(heightOut->size()) >= MIN_TERRAIN_PTS) {
            ObsNeighborCloudWithTerrain();
        }
    }

    // Ref: map_handler.cpp:424-504  MapHandler::TraversableAnalysis
    void TraversableAnalysis(const PointCloudPtr& heightOut) {
        const Eigen::Vector3i robot_sub = terrain_height_grid_->Pos2Sub(
            Eigen::Vector3d(G.robot_pos.x, G.robot_pos.y, 0.0f));
        heightOut->clear();
        if (!terrain_height_grid_->InRange(robot_sub)) return;

        const float H_THRED = height_voxel_dim_;
        std::fill(terrain_grid_traverse_list_.begin(),
                  terrain_grid_traverse_list_.end(), 0);

        auto IsTraversableNeighbor = [&](int cur_id, int ref_id) -> bool {
            if (terrain_grid_occupy_list_[ref_id] == 0) return false;
            float cur_h = terrain_height_grid_->GetCell(cur_id)[0];
            float ref_h = 0.0f;
            int counter = 0;
            for (const auto& e : terrain_height_grid_->GetCell(ref_id)) {
                if (fabs(e - cur_h) > H_THRED) continue;
                ref_h += e;
                counter++;
            }
            if (counter > 0) {
                terrain_height_grid_->GetCell(ref_id).resize(1);
                terrain_height_grid_->GetCell(ref_id)[0] = ref_h / static_cast<float>(counter);
                return true;
            }
            return false;
        };

        auto AddTraversePoint = [&](int idx) {
            Eigen::Vector3d cpos = terrain_height_grid_->Ind2Pos(idx);
            cpos.z() = terrain_height_grid_->GetCell(idx)[0];
            PCLPoint p;
            p.x = static_cast<float>(cpos.x());
            p.y = static_cast<float>(cpos.y());
            p.z = static_cast<float>(cpos.z());
            p.intensity = 0.0f;
            heightOut->points.push_back(p);
            terrain_grid_traverse_list_[idx] = 1;
        };

        const int robot_idx = terrain_height_grid_->Sub2Ind(robot_sub);
        const std::array<int, 4> dx = {-1, 0, 1, 0};
        const std::array<int, 4> dy = { 0, 1, 0,-1};
        std::deque<int> q;
        bool robot_terrain_init = false;
        std::unordered_set<int> visited;
        q.push_back(robot_idx);
        visited.insert(robot_idx);

        while (!q.empty()) {
            int cur_id = q.front();
            q.pop_front();
            if (terrain_grid_occupy_list_[cur_id] != 0) {
                if (!robot_terrain_init) {
                    // Initialize from robot's current height
                    float avg_h = 0.0f;
                    int counter = 0;
                    for (const auto& e : terrain_height_grid_->GetCell(cur_id)) {
                        if (fabs(e - G.robot_pos.z + G.vehicle_height) > H_THRED)
                            continue;
                        avg_h += e;
                        counter++;
                    }
                    if (counter > 0) {
                        avg_h /= static_cast<float>(counter);
                        terrain_height_grid_->GetCell(cur_id).resize(1);
                        terrain_height_grid_->GetCell(cur_id)[0] = avg_h;
                        AddTraversePoint(cur_id);
                        robot_terrain_init = true;
                        q.clear();  // restart BFS from this cell
                    }
                } else {
                    AddTraversePoint(cur_id);
                }
            } else if (robot_terrain_init) {
                continue;  // skip unoccupied cells after init
            }
            const Eigen::Vector3i csub = terrain_height_grid_->Ind2Sub(cur_id);
            for (int i = 0; i < 4; i++) {
                Eigen::Vector3i ref_sub = csub;
                ref_sub.x() += dx[i];
                ref_sub.y() += dy[i];
                if (!terrain_height_grid_->InRange(ref_sub)) continue;
                int ref_id = terrain_height_grid_->Sub2Ind(ref_sub);
                if (!visited.count(ref_id) &&
                    (!robot_terrain_init ||
                     IsTraversableNeighbor(cur_id, ref_id))) {
                    q.push_back(ref_id);
                    visited.insert(ref_id);
                }
            }
        }
    }

    // Ref: map_handler.cpp:362-389  MapHandler::ObsNeighborCloudWithTerrain
    void ObsNeighborCloudWithTerrain() {
        std::unordered_set<int> neighbor_copy = neighbor_obs_indices_;
        neighbor_obs_indices_.clear();
        const float R = cell_length_ * 0.7071f;  // sqrt(2)/2 * cell diagonal
        for (const auto& idx : neighbor_copy) {
            Point3D pos(world_obs_cloud_grid_->Ind2Pos(idx));
            bool inRange = false;
            float minH, maxH;
            NearestHeightOfRadius(pos, R, minH, maxH, inRange);
            if (inRange &&
                pos.z + cell_height_ > minH &&
                pos.z - cell_height_ < maxH + G.kTolerZ) {
                neighbor_obs_indices_.insert(idx);
            }
        }
        // Build extended obs indices (one layer below each obs cell)
        extend_obs_indices_.clear();
        const std::vector<int> inflate_vec{-1, 0};
        for (const int& idx : neighbor_obs_indices_) {
            const Eigen::Vector3i csub = world_obs_cloud_grid_->Ind2Sub(idx);
            for (const int& plus : inflate_vec) {
                Eigen::Vector3i sub = csub;
                sub.z() += plus;
                if (!world_obs_cloud_grid_->InRange(sub)) continue;
                extend_obs_indices_.insert(world_obs_cloud_grid_->Sub2Ind(sub));
            }
        }
    }

    // Ref: map_handler.cpp:344-360  MapHandler::AdjustCTNodeHeight
    void AdjustCTNodeHeight(const CTNodeStack& ctnodes) {
        if (ctnodes.empty()) return;
        const float H_MAX = G.robot_pos.z + G.kTolerZ;
        const float H_MIN = G.robot_pos.z - G.kTolerZ;
        for (auto& ct : ctnodes) {
            float min_th, max_th;
            NearestHeightOfRadius(ct->position, G.kMatchDist,
                                  min_th, max_th, ct->is_ground_associate);
            if (ct->is_ground_associate) {
                ct->position.z = min_th + G.vehicle_height;
            } else {
                ct->position.z = TerrainHeightOfPoint(
                    ct->position, ct->is_ground_associate);
                ct->position.z += G.vehicle_height;
            }
            ct->position.z = std::max(std::min(ct->position.z, H_MAX), H_MIN);
        }
    }

    // Ref: map_handler.cpp:324-342  MapHandler::AdjustNodesHeight
    void AdjustNodesHeight(const NodePtrStack& nodes) {
        if (nodes.empty()) return;
        for (const auto& n : nodes) {
            if (!n->is_active || n->is_boundary || G.IsFreeNavNode(n) ||
                G.IsOutsideGoal(n) ||
                !G.IsPointInLocalRange(n->position, true))
                continue;
            bool is_match = false;
            float terrain_h = TerrainHeightOfPoint(n->position, is_match);
            if (is_match) {
                terrain_h += G.vehicle_height;
                if (n->pos_filter_vec.empty()) {
                    n->position.z = terrain_h;
                } else {
                    n->pos_filter_vec.back().z = terrain_h;
                    n->position.z = G.AveragePoints(PointStack(
                        n->pos_filter_vec.begin(),
                        n->pos_filter_vec.end())).z;
                }
            }
        }
    }

    // Ray-cast sanity check: test if a 2D line segment from p1 to p2 is
    // blocked by accumulated obstacle points in the world grid.
    // Walks along the segment in cell_length_ steps, collecting obstacle
    // points within robot_dim/2 of the line. Returns true if blocked.
    bool IsEdgeBlockedByObstacles(const Point3D& p1, const Point3D& p2,
                                   float robot_dim, int min_blocked_pts = 3) const {
        if (!is_init_ || !world_obs_cloud_grid_) return false;
        float dx = p2.x - p1.x, dy = p2.y - p1.y;
        float dist = std::sqrt(dx*dx + dy*dy);
        if (dist < 0.01f) return false;

        float half_dim = robot_dim * 0.5f;
        // Direction unit vector
        float ux = dx/dist, uy = dy/dist;
        // Step size: half cell to ensure we check each cell
        float step = cell_length_ * 0.5f;
        int n_steps = std::max(2, (int)(dist / step));
        int blocked_count = 0;

        for (int s = 0; s <= n_steps; s++) {
            float t = (float)s / (float)n_steps;
            float px = p1.x + dx * t;
            float py = p1.y + dy * t;
            // Skip endpoints (within 0.2m of start/end) to avoid self-blocking
            float d1 = std::sqrt((px-p1.x)*(px-p1.x)+(py-p1.y)*(py-p1.y));
            float d2 = std::sqrt((px-p2.x)*(px-p2.x)+(py-p2.y)*(py-p2.y));
            if (d1 < 0.2f || d2 < 0.2f) continue;

            // Check the grid cell and neighbors for obstacle points near the line
            Eigen::Vector3d pos(px, py, p1.z);
            Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(pos);
            if (!world_obs_cloud_grid_->InRange(sub)) continue;

            // Check 3x3 neighborhood of cells
            for (int di=-1; di<=1; di++) {
                for (int dj=-1; dj<=1; dj++) {
                    Eigen::Vector3i nsub(sub.x()+di, sub.y()+dj, sub.z());
                    if (!world_obs_cloud_grid_->InRange(nsub)) continue;
                    int ind = world_obs_cloud_grid_->Sub2Ind(nsub);
                    const auto& cell = world_obs_cloud_grid_->GetCell(ind);
                    if (!cell || cell->empty()) continue;

                    for (const auto& pt : cell->points) {
                        // 2D distance from point to line segment
                        float apx = pt.x - p1.x, apy = pt.y - p1.y;
                        float proj = apx*ux + apy*uy;
                        if (proj < 0.2f || proj > dist - 0.2f) continue;
                        float perp = std::abs(apx*(-uy) + apy*ux);
                        if (perp < half_dim) {
                            blocked_count++;
                            if (blocked_count >= min_blocked_pts) return true;
                        }
                    }
                }
            }
        }
        return false;
    }
};
#endif  // USE_PCL

// ---------------------------------------------------------------------------
//  Message state — latest received LCM messages
// ---------------------------------------------------------------------------
static std::mutex g_state_mutex;

static bool g_odom_init = false;
static bool g_cloud_init = false;
static bool g_goal_received = false;
static Point3D g_robot_pos;
static Point3D g_goal_point;

// Cached obstacle points for contour detection (from terrain_map_ext)
static std::vector<smartnav::PointXYZI> g_obs_points;

// Raw terrain cloud with intensity-encoded free/obs classification (from terrain_map)
// This is the equivalent of /terrain_cloud in the reference — used by MapHandler.
static std::vector<smartnav::PointXYZI> g_terrain_cloud;
static bool g_terrain_cloud_init = false;

// ---------------------------------------------------------------------------
//  LCM message handlers
// ---------------------------------------------------------------------------
static void on_odometry(const lcm::ReceiveBuffer*, const std::string&,
                        const nav_msgs::Odometry* msg) {
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_robot_pos.x = (float)msg->pose.pose.position.x;
    g_robot_pos.y = (float)msg->pose.pose.position.y;
    g_robot_pos.z = (float)msg->pose.pose.position.z;
    G.robot_pos = g_robot_pos;
    if (!g_odom_init) {
        G.systemStartTime = msg->header.stamp.sec + msg->header.stamp.nsec/1e9;
        G.map_origin = g_robot_pos;
        g_odom_init = true;
        printf("[FAR] Odometry initialized at (%.2f, %.2f, %.2f)\n",
               g_robot_pos.x, g_robot_pos.y, g_robot_pos.z);
    }
}

// terrain_map_ext: classified obstacle cloud from TerrainMapExt — primary input
// for contour detection and visibility graph construction.
static void on_terrain_map_ext(const lcm::ReceiveBuffer*, const std::string&,
                               const sensor_msgs::PointCloud2* msg) {
    auto pts = smartnav::parse_pointcloud2(*msg);
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_obs_points = std::move(pts);
    g_cloud_init = true;
}

// terrain_map: raw terrain cloud from TerrainAnalysis with intensity-encoded
// free/obs classification — used by MapHandler for persistent grid accumulation.
// Ref: far_planner.cpp:28  terrain_sub_ subscribes to /terrain_cloud
// Ref: terrainAnalysis.cpp:865  intensity = disZ (elevation above ground)
static void on_terrain_map(const lcm::ReceiveBuffer*, const std::string&,
                           const sensor_msgs::PointCloud2* msg) {
    auto pts = smartnav::parse_pointcloud2(*msg);
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_terrain_cloud = std::move(pts);
    g_terrain_cloud_init = true;
}

// registered_scan: raw lidar scan — kept for future dynamic obstacle detection.
static void on_registered_scan(const lcm::ReceiveBuffer*, const std::string&,
                               const sensor_msgs::PointCloud2* msg) {
    (void)msg; // unused in static env mode; reserved for dynamic obs
}

static void on_goal(const lcm::ReceiveBuffer*, const std::string&,
                    const geometry_msgs::PointStamped* msg) {
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_goal_point.x = (float)msg->point.x;
    g_goal_point.y = (float)msg->point.y;
    g_goal_point.z = (float)msg->point.z;
    g_goal_received = true;
    printf("[FAR] Goal received: (%.2f, %.2f, %.2f)\n",
           g_goal_point.x, g_goal_point.y, g_goal_point.z);
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Signal handling for clean shutdown
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT,  signal_handler);

    // Line-buffer stdout so logs are visible in real-time through subprocess pipe
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // Enable verbose debug logging when DIMOS_DEBUG env var is set to a
    // non-empty, non-zero value.
    if (const char* dbg = std::getenv("DIMOS_DEBUG")) {
        dimosDebug = (dbg[0] != '\0' && std::string(dbg) != "0");
    }

    dimos::NativeModule mod(argc, argv);

    // --- Read configurable parameters from CLI args ---
    G.robot_dim        = mod.arg_float("robot_dim", 0.8f);
    G.vehicle_height   = mod.arg_float("vehicle_height", 0.75f);
    G.kLeafSize        = mod.arg_float("voxel_dim", 0.1f);
    G.kSensorRange     = mod.arg_float("sensor_range", 15.0f);
    G.kTerrainRange    = mod.arg_float("terrain_range", 15.0f);
    G.kLocalPlanRange  = mod.arg_float("local_planner_range", 5.0f);
    G.is_static_env    = mod.arg_bool("is_static_env", false);
    G.is_debug         = mod.arg_bool("is_debug", false);
    G.is_multi_layer   = mod.arg_bool("is_multi_layer", false);
    float main_freq    = mod.arg_float("update_rate", 5.0f);
    float converge_d   = mod.arg_float("converge_dist", 0.4f);
    int momentum_thr   = mod.arg_int("momentum_thred", 5);

    // Compute derived parameters (same as LoadROSParams)
    float floor_height = mod.arg_float("floor_height", 1.5f);
    G.kFreeZ           = mod.arg_float("terrain_free_Z", 0.15f);
    float cell_length  = mod.arg_float("cell_length", 1.5f);
    float grid_max_len = mod.arg_float("grid_max_length", 300.0f);
    G.kHeightVoxel     = G.kLeafSize * 2.0f;
    G.kNearDist        = G.robot_dim;
    G.kMatchDist       = G.robot_dim * 2.0f + G.kLeafSize;
    G.kNavClearDist    = G.robot_dim * 2.0f + G.kLeafSize;  // 2×radius + voxel — real safety margin (was 0.35m with the /2 formula, equal to robot radius with zero margin)
    G.kProjectDist     = G.kLeafSize;
    G.kTolerZ          = floor_height - G.kHeightVoxel;
    float cell_height  = floor_height / 2.5f;
    G.kCellHeight      = cell_height;
    G.kMarginDist      = G.kSensorRange - G.kMatchDist;
    G.kMarginHeight    = G.kTolerZ - G.kCellHeight / 2.0f;
    float angle_noise_deg = mod.arg_float("angle_noise", 15.0f);
    float accept_align_deg = mod.arg_float("accept_align", 15.0f);
    G.kAngleNoise      = angle_noise_deg / 180.0f * (float)M_PI;
    G.kAcceptAlign     = accept_align_deg / 180.0f * (float)M_PI;

    // Verbose logging only when DEBUG=1
    const char* debug_env = std::getenv("DEBUG");
    bool verbose = (debug_env && std::string(debug_env) == "1");

    printf("[FAR] Configuration:\n");
    printf("  robot_dim=%.2f  sensor_range=%.1f  voxel=%.2f  freq=%.1f  verbose=%d\n",
           G.robot_dim, G.kSensorRange, G.kLeafSize, main_freq, verbose);
    printf("  static_env=%d  multi_layer=%d  converge_dist=%.2f\n",
           G.is_static_env, G.is_multi_layer, converge_d);

    // --- LCM setup ---
    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "[FAR] ERROR: LCM init failed\n");
        return 1;
    }

    std::string topic_terrain_ext = mod.topic("terrain_map_ext");
    std::string topic_terrain     = mod.topic("terrain_map");
    std::string topic_scan        = mod.topic("registered_scan");
    std::string topic_odom        = mod.topic("odometry");
    std::string topic_goal        = mod.topic("goal");
    std::string topic_wp          = mod.topic("way_point");
    std::string topic_goal_path   = mod.topic("goal_path");

    // LCM subscribe requires member-function + object pointer; wrap free fns
    // in a trivial handler struct.
    struct LcmHandler {
        void odom(const lcm::ReceiveBuffer* b, const std::string& c,
                  const nav_msgs::Odometry* m) { on_odometry(b, c, m); }
        void terrain_ext(const lcm::ReceiveBuffer* b, const std::string& c,
                         const sensor_msgs::PointCloud2* m) { on_terrain_map_ext(b, c, m); }
        void terrain(const lcm::ReceiveBuffer* b, const std::string& c,
                     const sensor_msgs::PointCloud2* m) { on_terrain_map(b, c, m); }
        void scan(const lcm::ReceiveBuffer* b, const std::string& c,
                  const sensor_msgs::PointCloud2* m) { on_registered_scan(b, c, m); }
        void goal(const lcm::ReceiveBuffer* b, const std::string& c,
                  const geometry_msgs::PointStamped* m) { on_goal(b, c, m); }
    } lcm_handler;
    lcm.subscribe(topic_odom,        &LcmHandler::odom,        &lcm_handler);
    lcm.subscribe(topic_terrain_ext, &LcmHandler::terrain_ext, &lcm_handler);
    lcm.subscribe(topic_terrain,     &LcmHandler::terrain,     &lcm_handler);
    lcm.subscribe(topic_scan,        &LcmHandler::scan,        &lcm_handler);
    lcm.subscribe(topic_goal,        &LcmHandler::goal,        &lcm_handler);

    printf("[FAR] Subscribed: terrain_ext=%s  terrain=%s  scan=%s  odom=%s  goal=%s\n",
           topic_terrain_ext.c_str(), topic_terrain.c_str(),
           topic_scan.c_str(), topic_odom.c_str(), topic_goal.c_str());
    printf("[FAR] Publishing: way_point=%s\n", topic_wp.c_str());

    // --- Module objects ---
    DynamicGraphManager graph_mgr;
    GraphPlanner planner;
    ContourGraphManager contour_mgr;
    planner.converge_dist = converge_d;
    planner.momentum_thred = momentum_thr;
    graph_mgr.finalize_thred = mod.arg_int("finalize_thred", 3);
    graph_mgr.votes_size = mod.arg_int("votes_size", 10);
    graph_mgr.dumper_thred = mod.arg_int("dumper_thred", 3);
    contour_mgr.kPillarPerimeter = G.robot_dim * 4.0f;

#ifdef HAS_OPENCV
    ContourDetector contour_det;
    contour_det.sensor_range = G.kSensorRange;
    contour_det.voxel_dim = G.kLeafSize;
    contour_det.kRatio = mod.arg_float("resize_ratio", 3.0f);
    contour_det.kThredValue = mod.arg_int("filter_count_value", 5);
    contour_det.kBlurSize = (int)std::round(G.kNavClearDist / G.kLeafSize);
    contour_det.Init();
#endif

#ifdef USE_PCL
    MapHandler map_handler;
    map_handler.sensor_range_ = G.kSensorRange;
    map_handler.floor_height_ = floor_height;
    map_handler.cell_length_ = cell_length;
    map_handler.cell_height_ = floor_height / 2.5f;
    map_handler.grid_max_length_ = grid_max_len;
    map_handler.grid_max_height_ = floor_height * 2.0f;
    map_handler.height_voxel_dim_ = G.kLeafSize * 2.0f;
    map_handler.Init();
    // Set up the ray-cast obstacle check callback
    g_obstacle_raycast = [&map_handler](const Point3D& p1, const Point3D& p2) -> bool {
        // Use wider corridor (1.5x robot dim) for better wall detection, and
        // lower threshold (3 blocked pts) to catch walls while allowing
        // navigation near furniture.
        return map_handler.IsEdgeBlockedByObstacles(p1, p2, G.robot_dim * 1.5f, 3);
    };
#endif

    bool is_graph_init = false;
    const int loop_ms = (int)(1000.0f / main_freq);

    printf("[FAR] Entering main loop (period=%dms)...\n", loop_ms);

    // Waypoint hysteresis state: skip re-publishing when the new chosen
    // waypoint lies within wp_hysteresis_dist of the last published one for
    // the same goal. Without this the local planner gets spammed with
    // micro-updates as obstacle polygons / ray-casts flicker between cycles
    // and Dijkstra picks slightly different first hops.
    Point3D last_published_wp;
    Point3D last_published_goal;
    bool has_published_wp = false;
    const float wp_hysteresis_dist = 0.3f;  // meters

    // Sticky first-hop state: keep publishing the same graph node across
    // cycles as long as it's still reachable from the robot and the goal
    // hasn't changed. Resists path thrashing when the graph grows/edits
    // faster than the robot can advance. Only switch when:
    //   - sticky node is no longer reachable via direct robot→node line,
    //   - robot has closed to within converge_dist (time to advance),
    //   - goal coordinates changed, or
    //   - sticky node pointer was pruned from the graph.
    NavNodePtr sticky_wp_node = nullptr;
    Point3D sticky_wp_goal;

    // --- Main loop ---
    // Drain LCM messages continuously, run planning at update_rate.
    // Ref: far_planner.cpp uses ROS spin + timer callbacks at separate rates.
    // Here we use a single thread: drain messages with short timeout, then
    // check if enough time has elapsed for the next planning cycle.
    auto last_plan_time = std::chrono::steady_clock::now();
    const auto plan_period = std::chrono::milliseconds(loop_ms);
    const int drain_timeout_ms = 10;  // short timeout to keep draining messages

    while (!g_shutdown.load()) {
        // Drain all pending LCM messages with short timeout
        lcm.handleTimeout(drain_timeout_ms);

        // Only run planning cycle at the configured update_rate
        auto now = std::chrono::steady_clock::now();
        if (now - last_plan_time < plan_period) continue;
        last_plan_time = now;

        // Check preconditions
        bool odom_ok, cloud_ok, terrain_ok, goal_pending;
        Point3D robot_p, goal_p;
        std::vector<smartnav::PointXYZI> obs_snap;
        std::vector<smartnav::PointXYZI> terrain_snap;
        {
            std::lock_guard<std::mutex> lk(g_state_mutex);
            odom_ok = g_odom_init;
            cloud_ok = g_cloud_init;
            terrain_ok = g_terrain_cloud_init;
            goal_pending = g_goal_received;
            robot_p = g_robot_pos;
            goal_p = g_goal_point;
            if (cloud_ok) {
                obs_snap = g_obs_points; // copy
            }
            if (terrain_ok) {
                terrain_snap = g_terrain_cloud; // copy
            }
            if (goal_pending) g_goal_received = false;
        }

        // Debug: periodic status (every ~2s at 5Hz)
        if (verbose) {
            static int dbg_ctr = 0;
            if (++dbg_ctr % 10 == 0) {
                auto gp_tmp = planner.goal_node;
                float goal_dist = gp_tmp ? (robot_p - Point3D(gp_tmp->position.x, gp_tmp->position.y, gp_tmp->position.z)).norm_flat() : 0.0f;
                printf("[FAR] status: odom=%d cloud=%d graph_init=%d "
                       "graph_nodes=%zu  robot=(%.2f,%.2f)  "
                       "has_goal=%d  goal=(%.2f,%.2f)  goal_dist=%.1fm  "
                       "obs_pts=%zu\n",
                       odom_ok, cloud_ok, is_graph_init,
                       g_global_graph_nodes.size(),
                       robot_p.x, robot_p.y,
                       (gp_tmp != nullptr), goal_p.x, goal_p.y, goal_dist,
                       obs_snap.size());
                fflush(stdout);
            }
            // Dump graph summary every 100 cycles (~20s)
            static int dump_ctr = 0;
            if (++dump_ctr % 100 == 0) {
                int odom_count=0, nav_count=0, contour_count=0, goal_count=0;
                for (const auto& n : g_global_graph_nodes) {
                    if (n->is_odom) odom_count++;
                    if (n->is_navpoint) nav_count++;
                    if (n->is_contour_match) contour_count++;
                    if (n->is_goal) goal_count++;
                }
                printf("[FAR] graph: total=%zu odom=%d nav=%d contour=%d goal=%d polys=%zu\n",
                       g_global_graph_nodes.size(), odom_count, nav_count, contour_count, goal_count,
                       g_contour_polygons.size());
                fflush(stdout);
            }
        }

        if (!odom_ok || !cloud_ok) continue;

        // --- Main graph update cycle (port of MainLoopCallBack) ---
        G.Timer.start_time("V-Graph Update");

        // 1. Update robot position in graph
        graph_mgr.UpdateRobotPosition(robot_p);
        auto odom_node = graph_mgr.GetOdomNode();
        if (!odom_node) continue;

        // free_odom_p: for now, same as odom
        G.free_odom_p = odom_node->position;

        // 1b. MapHandler: accumulate obs/free into persistent grid, get surround obs
        //     Ref: far_planner.cpp:722-781  FARMaster::TerrainCallBack
        //     - Uses terrain_map (raw terrain cloud with intensity-encoded
        //       free/obs classification) same as reference /terrain_cloud
        //     - Crops, splits by intensity, feeds into MapHandler grids
        //     - Extracts accumulated surround obs for contour detection
#ifdef USE_PCL
        if (terrain_ok) {
            // Convert terrain_map (raw terrain cloud) to PCL
            PointCloudPtr terrain_pcl(new PointCloud());
            terrain_pcl->points.reserve(terrain_snap.size());
            for (const auto& pp : terrain_snap) {
                PCLPoint p;
                p.x = pp.x; p.y = pp.y; p.z = pp.z; p.intensity = pp.intensity;
                terrain_pcl->points.push_back(p);
            }
            // Crop box around robot (terrain_range x terrain_range x kTolerZ)
            // Same as reference: FARUtil::CropBoxCloud(temp_cloud_ptr_, robot_pos_, ...)
            PointCloudPtr cropped(new PointCloud());
            for (const auto& p : terrain_pcl->points) {
                if (fabs(p.x - robot_p.x) < G.kTerrainRange &&
                    fabs(p.y - robot_p.y) < G.kTerrainRange &&
                    fabs(p.z - robot_p.z) < G.kTolerZ) {
                    cropped->points.push_back(p);
                }
            }
            // Split by intensity: free (< kFreeZ) vs obs (>= kFreeZ)
            // Same as reference: FARUtil::ExtractFreeAndObsCloud(...)
            PointCloudPtr free_cloud(new PointCloud());
            PointCloudPtr obs_cloud(new PointCloud());
            G.ExtractFreeAndObsCloud(cropped, free_cloud, obs_cloud);
            if (verbose) {
                static int mh_ctr = 0;
                if (++mh_ctr % 10 == 0) {
                    printf("[FAR] maphandler: terrain_raw=%zu cropped=%zu free=%zu obs=%zu kFreeZ=%.3f\n",
                           terrain_snap.size(), cropped->size(),
                           free_cloud->size(), obs_cloud->size(), G.kFreeZ);
                    fflush(stdout);
                }
            }
            // Feed into MapHandler persistent grids
            // Same as reference: map_handler_.UpdateRobotPosition + grids
            map_handler.UpdateRobotPosition(robot_p);
            map_handler.UpdateObsCloudGrid(obs_cloud);
            map_handler.UpdateFreeCloudGrid(free_cloud);
            // Terrain height analysis from accumulated free cloud
            // Same as reference: GetSurroundFreeCloud → UpdateTerrainHeightGrid
            PointCloudPtr surround_free(new PointCloud());
            PointCloudPtr terrain_height(new PointCloud());
            map_handler.GetSurroundFreeCloud(surround_free);
            map_handler.UpdateTerrainHeightGrid(surround_free, terrain_height);
            // Get accumulated surround obs for contour detection
            // Same as reference: map_handler_.GetSurroundObsCloud(surround_obs_cloud_)
            PointCloudPtr surround_obs(new PointCloud());
            map_handler.GetSurroundObsCloud(surround_obs);
            if (verbose) {
                static int mh2_ctr = 0;
                if (++mh2_ctr % 10 == 0) {
                    printf("[FAR] maphandler: surround_obs=%zu surround_free=%zu terrain_h=%zu\n",
                           surround_obs->size(), surround_free->size(), terrain_height->size());
                    fflush(stdout);
                }
            }
            // Replace single-frame obs_snap with accumulated surround obs
            obs_snap.clear();
            obs_snap.reserve(surround_obs->size());
            for (const auto& p : surround_obs->points) {
                smartnav::PointXYZI pp;
                pp.x = p.x; pp.y = p.y; pp.z = p.z; pp.intensity = p.intensity;
                obs_snap.push_back(pp);
            }
        }
#endif

        // 2. Extract contours from obstacle cloud (now using accumulated surround obs)
        std::vector<PointStack> realworld_contours;
#ifdef HAS_OPENCV
        contour_det.BuildAndExtract(odom_node->position, obs_snap, realworld_contours);
        if (verbose) {
            int total_vertices = 0;
            for (const auto& c : realworld_contours) total_vertices += c.size();
            static int contour_dbg_ctr = 0;
            if (++contour_dbg_ctr % 10 == 0) {
                printf("[FAR] contour: obs_pts=%zu  contours=%zu  vertices=%d\n",
                       obs_snap.size(), realworld_contours.size(), total_vertices);
                fflush(stdout);
            }
        }
#endif

        // 3. Update contour graph
        contour_mgr.UpdateContourGraph(odom_node, realworld_contours);

        // 3b. Adjust contour node heights from terrain
#ifdef USE_PCL
        map_handler.AdjustCTNodeHeight(g_contour_graph);
#endif

        // 4. Update global near nodes
        graph_mgr.UpdateGlobalNearNodes();

        // 5. Extract new graph nodes (trajectory nodes)
        NodePtrStack new_nodes;
        if (graph_mgr.ExtractGraphNodes()) {
            new_nodes = graph_mgr.new_nodes;
        }

        // 6. Update navigation graph edges
        graph_mgr.UpdateNavGraph(new_nodes, false);

        // 6b. Remove stale contour-match nodes not re-observed
        graph_mgr.CleanupStaleNodes();

        // 6c. Adjust nav node heights from terrain
#ifdef USE_PCL
        map_handler.AdjustNodesHeight(g_global_graph_nodes);
#endif

        // 7. Extract global contours for polygon collision checking
        contour_mgr.ExtractGlobalContours();

        auto nav_graph = graph_mgr.GetNavGraph();
        planner.current_graph = nav_graph;

        double vg_time = G.Timer.end_time("V-Graph Update", false);

        if (!is_graph_init && !nav_graph.empty()) {
            is_graph_init = true;
            printf("[FAR] V-Graph initialized with %zu nodes\n", nav_graph.size());
        }

        // --- Goal handling ---
        if (goal_pending) {
            planner.UpdateGoal(goal_p);
        }

        // --- Planning cycle (port of PlanningCallBack) ---
        if (!is_graph_init) continue;

        auto gp = planner.goal_node;
        if (!gp) {
            planner.UpdateGraphTraverability(odom_node, nullptr);
        } else {
            // Update goal connectivity
            planner.UpdateGoalConnects(gp);
            planner.current_graph = graph_mgr.GetNavGraph();

            // Dijkstra traversability
            planner.UpdateGraphTraverability(odom_node, gp);

            // Path to goal
            NodePtrStack global_path;
            NavNodePtr nav_wp = nullptr;
            Point3D cur_goal;
            bool is_fail = false, is_succeed = false;

            planner.PathToGoal(gp, global_path, nav_wp, cur_goal, is_fail, is_succeed);

            if (is_succeed) {
                // Goal reached — publish robot's current position as the
                // waypoint so LocalPlanner stops driving. Historically this
                // branch fell through and published `nav_wp->position`
                // (the just-reached goal node), which confused LocalPlanner
                // into chasing a stale point — especially when UpdateGoal
                // snapped the new goal to an existing internav node.
                geometry_msgs::PointStamped stop_wp;
                stop_wp.header = dimos::make_header(G.worldFrameId,
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                stop_wp.point.x = odom_node->position.x;
                stop_wp.point.y = odom_node->position.y;
                stop_wp.point.z = odom_node->position.z;
                lcm.publish(topic_wp, &stop_wp);
                has_published_wp = false;  // clear hysteresis state on reach
                sticky_wp_node = nullptr;  // clear sticky on goal reached
            } else if (!is_fail && nav_wp) {
                // Defensive re-validation: Dijkstra may select a first hop
                // whose edge from the robot crosses known obstacles (because
                // graph edges aren't always re-verified against the current
                // obstacle cloud). Re-run poly + ray checks on the direct
                // robot→nav_wp segment; if blocked, walk forward along the
                // path looking for the first node that IS visible from the
                // robot. If none found, publish a stop.
                auto robot_to_node_free = [&](const NavNodePtr& n) -> bool {
                    if (!n) return false;
                    if (!IsNavNodesConnectFreePolygon(odom_node, n)) return false;
                    if (g_obstacle_raycast &&
                        g_obstacle_raycast(odom_node->position, n->position)) {
                        return false;
                    }
                    return true;
                };
                {
                    bool first_hop_free = robot_to_node_free(nav_wp);
                    if (!first_hop_free) {
                        // Walk further along the global path looking for a
                        // later node that IS directly visible from the robot.
                        // Search from the end toward the front (prefer the
                        // furthest-visible node, closest to the goal).
                        NavNodePtr alt = nullptr;
                        int alt_idx = -1;
                        int skipped = 0;
                        for (int i = (int)global_path.size() - 1; i >= 1; i--) {
                            if (global_path[i] == nav_wp) continue;  // already tried
                            if (robot_to_node_free(global_path[i])) {
                                alt = global_path[i];
                                alt_idx = i;
                                break;
                            }
                            skipped++;
                        }
                        if (alt) {
                            static int skip_dbg = 0;
                            if (++skip_dbg % 5 == 0) {
                                printf("[FAR] WP skipped blocked first hop (%.2f,%.2f) → "
                                       "using path[%d]=(%.2f,%.2f) (searched %d, path_len=%zu)\n",
                                       nav_wp->position.x, nav_wp->position.y,
                                       alt_idx, alt->position.x, alt->position.y,
                                       skipped, global_path.size());
                                fflush(stdout);
                            }
                            nav_wp = alt;
                        } else {
                            geometry_msgs::PointStamped stop_wp;
                            stop_wp.header = dimos::make_header(G.worldFrameId,
                                std::chrono::duration<double>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            stop_wp.point.x = odom_node->position.x;
                            stop_wp.point.y = odom_node->position.y;
                            stop_wp.point.z = odom_node->position.z;
                            lcm.publish(topic_wp, &stop_wp);
                            has_published_wp = false;
                            sticky_wp_node = nullptr;  // everything blocked, drop sticky
                            static int blocked_dbg = 0;
                            if (++blocked_dbg % 5 == 0) {
                                printf("[FAR] WP BLOCKED → every node on path[1..%zu] blocked from "
                                       "robot=(%.2f,%.2f) (publishing stop)\n",
                                       global_path.size(),
                                       odom_node->position.x, odom_node->position.y);
                                fflush(stdout);
                            }
                            goto post_publish;
                        }
                    }
                }

                // Sticky first-hop: if the previously-chosen first hop is
                // still in the graph, still visible from the robot, and
                // the robot hasn't approached it (within converge_dist),
                // keep using it. Prevents Dijkstra from thrashing the path
                // as new nodes/edges are added each cycle.
                {
                    bool sticky_goal_same =
                        sticky_wp_node &&
                        std::abs(cur_goal.x - sticky_wp_goal.x) < 1e-3f &&
                        std::abs(cur_goal.y - sticky_wp_goal.y) < 1e-3f;
                    // Check that sticky node is still present in the current graph
                    // (it may have been pruned by CleanupStaleNodes).
                    bool sticky_alive = false;
                    if (sticky_goal_same) {
                        for (const auto& n : planner.current_graph) {
                            if (n == sticky_wp_node) { sticky_alive = true; break; }
                        }
                    }
                    if (sticky_alive) {
                        float d_robot = (sticky_wp_node->position -
                                         odom_node->position).norm_flat();
                        bool advance = d_robot < planner.converge_dist;
                        bool sticky_reachable = robot_to_node_free(sticky_wp_node);
                        if (sticky_reachable && !advance && sticky_wp_node != nav_wp) {
                            // Stay the course.
                            static int sticky_dbg = 0;
                            if (++sticky_dbg % 10 == 0) {
                                printf("[FAR] WP sticky: keeping (%.2f,%.2f) "
                                       "(Dijkstra wanted (%.2f,%.2f); robot dist=%.2f)\n",
                                       sticky_wp_node->position.x,
                                       sticky_wp_node->position.y,
                                       nav_wp->position.x, nav_wp->position.y,
                                       d_robot);
                                fflush(stdout);
                            }
                            nav_wp = sticky_wp_node;
                        }
                    }
                }

                // Waypoint hysteresis: if the goal hasn't changed and the
                // newly chosen first hop is within wp_hysteresis_dist of the
                // previously published one, skip the republish. Stops the
                // LocalPlanner from being spammed with micro-updates when
                // Dijkstra flips between neighboring first hops due to
                // polygon/ray-cast flicker.
                bool same_goal =
                    has_published_wp &&
                    std::abs(cur_goal.x - last_published_goal.x) < 1e-3f &&
                    std::abs(cur_goal.y - last_published_goal.y) < 1e-3f;
                float wp_drift = has_published_wp
                    ? (nav_wp->position - last_published_wp).norm_flat()
                    : std::numeric_limits<float>::infinity();
                bool skip_publish = same_goal && wp_drift < wp_hysteresis_dist;
                if (skip_publish) {
                    // Too close to last published — suppress spam.
                    // Still fall through to the is_succeed GOAL REACHED print.
                } else {

                // Publish graph-planned waypoint
                geometry_msgs::PointStamped wp_msg;
                wp_msg.header = dimos::make_header(G.worldFrameId,
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                wp_msg.point.x = nav_wp->position.x;
                wp_msg.point.y = nav_wp->position.y;
                // Use robot's z — graph node z can be wrong with sparse terrain
                // data. The local planner only uses x,y for 2D path generation.
                wp_msg.point.z = odom_node->position.z;
                lcm.publish(topic_wp, &wp_msg);
                last_published_wp = nav_wp->position;
                last_published_goal = cur_goal;
                has_published_wp = true;
                // Remember what we stuck with so subsequent cycles can compare.
                sticky_wp_node = nav_wp;
                sticky_wp_goal = cur_goal;

                // Publish full planned path for visualization
                {
                    nav_msgs::Path path_msg;
                    path_msg.header = wp_msg.header;
                    path_msg.poses_length = static_cast<int32_t>(global_path.size());
                    path_msg.poses.resize(global_path.size());
                    for (std::size_t i = 0; i < global_path.size(); i++) {
                        auto& ps = path_msg.poses[i];
                        ps.header = wp_msg.header;
                        ps.pose.position.x = global_path[i]->position.x;
                        ps.pose.position.y = global_path[i]->position.y;
                        ps.pose.position.z = odom_node->position.z;
                        ps.pose.orientation.w = 1.0;
                    }
                    lcm.publish(topic_goal_path, &path_msg);
                }

                float dist_to_goal = (odom_node->position - cur_goal).norm_flat();
                if (verbose) {
                    printf("[FAR] GRAPH PATH → wp=(%.2f,%.2f,%.2f)  "
                           "path_nodes=%zu  graph_nodes=%zu  robot=(%.2f,%.2f)  "
                           "goal=(%.2f,%.2f)  dist_to_goal=%.1fm  vg_time=%.1fms\n",
                           nav_wp->position.x, nav_wp->position.y, nav_wp->position.z,
                           global_path.size(), nav_graph.size(),
                           odom_node->position.x, odom_node->position.y,
                           cur_goal.x, cur_goal.y, dist_to_goal, vg_time);
                    fflush(stdout);
                }
                }  // end: if (!skip_publish)
            } else if (is_fail) {
                // No valid path — publish robot's current position as waypoint
                // to stop the robot from driving into walls. This matches the
                // original's behavior in PlanningCallBack.
                {
                    geometry_msgs::PointStamped stop_wp;
                    stop_wp.header = dimos::make_header(G.worldFrameId,
                        std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch()).count());
                    stop_wp.point.x = odom_node->position.x;
                    stop_wp.point.y = odom_node->position.y;
                    stop_wp.point.z = odom_node->position.z;
                    lcm.publish(topic_wp, &stop_wp);
                }
                has_published_wp = false;  // clear hysteresis on stop
                sticky_wp_node = nullptr;  // NO ROUTE, drop sticky

                // Count how many graph nodes are traversable and connected to goal
                int traversable_count = 0, goal_connected = 0;
                for (const auto& n : nav_graph) {
                    if (n->is_traversable) traversable_count++;
                }
                for (const auto& cn : gp->connect_nodes) {
                    (void)cn; goal_connected++;
                }

                printf("[FAR] NO ROUTE → goal=(%.2f,%.2f,%.2f)  "
                       "robot=(%.2f,%.2f)  graph_nodes=%zu  traversable=%d  "
                       "goal_edges=%d  dist=%.1fm  (published stop)\n",
                       cur_goal.x, cur_goal.y, cur_goal.z,
                       odom_node->position.x, odom_node->position.y,
                       nav_graph.size(), traversable_count, goal_connected,
                       (odom_node->position - cur_goal).norm_flat());
                fflush(stdout);
            }
            post_publish:;

            if (is_succeed) {
                printf("[FAR] *** GOAL REACHED *** at (%.2f,%.2f)  "
                       "goal was (%.2f,%.2f)  graph_nodes=%zu\n",
                       odom_node->position.x, odom_node->position.y,
                       cur_goal.x, cur_goal.y, nav_graph.size());
                fflush(stdout);
            }
        }
    }

    printf("[FAR] Shutdown complete.\n");
    return 0;
}
