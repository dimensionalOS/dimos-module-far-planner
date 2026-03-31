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

#include "sensor_msgs/PointCloud2.hpp"
#include "nav_msgs/Odometry.hpp"
#include "geometry_msgs/PointStamped.hpp"

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
    float kNavClearDist = 0.5f;
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
static bool IsNavNodesConnectFreePolygon(const NavNodePtr& n1, const NavNodePtr& n2) {
    // simplified check against boundary contours
    ConnectPair cedge(n1->position, n2->position);
    HeightPair hp(n1->position, n2->position);
    for (const auto& c : g_boundary_contour) {
        if (IsEdgeCollideSegment(c, cedge)) return false;
    }
    for (const auto& poly : g_contour_polygons) {
        if (poly->is_pillar) continue;
        if (IsEdgeCollidePoly(poly->vertices, cedge)) return false;
    }
    return true;
}
#else
// Without OpenCV, provide stub that always returns true
static bool IsNavNodesConnectFreePolygon(const NavNodePtr&, const NavNodePtr&) { return true; }
#endif

// ---------------------------------------------------------------------------
//  Dijkstra-based traversability + A* path planning
//  (Port of graph_planner.cpp)
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
        for (const auto& n : current_graph) {
            if (n == goal_ptr) continue;
            if (n->is_traversable && IsNavNodesConnectFreePolygon(n, goal_ptr)) {
                AddPolyEdge(n, goal_ptr); AddEdge(n, goal_ptr);
                n->is_block_to_goal = false;
            } else {
                ErasePolyEdge(n, goal_ptr); EraseEdge(n, goal_ptr);
                n->is_block_to_goal = true;
            }
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
        // Check if near an existing internav node
        float min_dist = G.kNearDist;
        for (const auto& n : current_graph) {
            if (n->is_navpoint) {
                float d = (n->position - goal).norm();
                if (d < min_dist) {
                    is_use_internav_goal = true;
                    goal_node = n;
                    min_dist = d;
                    goal_node->is_goal = true;
                }
            }
        }
        if (!is_use_internav_goal) {
            CreateNavNodeFromPoint(goal, goal_node, false, false, true);
            AddNodeToGraph(goal_node);
        }
        is_goal_init = true;
        is_global_path_init = false;
        origin_goal_pos = goal_node->position;
        path_momentum_counter = 0;
        recorded_path.clear();
        printf("[FAR] New goal set at (%.2f, %.2f, %.2f)\n", goal.x, goal.y, goal.z);
    }

    bool PathToGoal(const NavNodePtr& goal_ptr, NodePtrStack& global_path,
                    NavNodePtr& nav_wp, Point3D& goal_p,
                    bool& is_fail, bool& is_succeed) {
        if (!is_goal_init || !odom_node || !goal_ptr || current_graph.empty()) return false;
        is_fail = false; is_succeed = false;
        global_path.clear();
        goal_p = goal_ptr->position;

        if ((odom_node->position - goal_p).norm() < converge_dist ||
            (odom_node->position - origin_goal_pos).norm() < converge_dist) {
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
                nav_wp = NextWaypoint(path, goal_ptr);
                global_path = path;
                recorded_path = path;
                is_global_path_init = true;
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
//  (Core of dynamic_graph.h / dynamic_graph.cpp)
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
        }
        return !new_nodes.empty();
    }

    void UpdateNavGraph(const NodePtrStack& new_nodes_in, bool is_freeze) {
        if (is_freeze) return;
        // Add new nodes
        for (const auto& nn : new_nodes_in) {
            AddNodeToGraph(nn);
            nn->is_near_nodes = true;
            near_nav_nodes.push_back(nn);
        }
        // Build visibility edges between odom and near nodes
        for (const auto& n : wide_near_nodes) {
            if (n->is_odom) continue;
            if (IsNavNodesConnectFreePolygon(odom_node, n)) {
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
                if (IsNavNodesConnectFreePolygon(n1, n2)) {
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
//  (Port of contour_detector.cpp — only built with HAS_OPENCV)
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
//  (Port of contour_graph.cpp)
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
//  Message state — latest received LCM messages
// ---------------------------------------------------------------------------
static std::mutex g_state_mutex;

static bool g_odom_init = false;
static bool g_cloud_init = false;
static bool g_goal_received = false;
static Point3D g_robot_pos;
static Point3D g_goal_point;

// Cached obstacle points for contour detection (from registered_scan)
static std::vector<smartnav::PointXYZI> g_obs_points;

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

static void on_registered_scan(const lcm::ReceiveBuffer*, const std::string&,
                               const sensor_msgs::PointCloud2* msg) {
    auto pts = smartnav::parse_pointcloud2(*msg);
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_obs_points = std::move(pts);
    g_cloud_init = true;
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

    dimos::NativeModule mod(argc, argv);

    // --- Read configurable parameters from CLI args ---
    G.robot_dim        = mod.arg_float("robot_dim", 0.8f);
    G.vehicle_height   = mod.arg_float("vehicle_height", 0.75f);
    G.kLeafSize        = mod.arg_float("voxel_dim", 0.2f);
    G.kSensorRange     = mod.arg_float("sensor_range", 30.0f);
    G.kTerrainRange    = mod.arg_float("terrain_range", 15.0f);
    G.kLocalPlanRange  = mod.arg_float("local_planner_range", 5.0f);
    G.is_static_env    = mod.arg_bool("is_static_env", true);
    G.is_debug         = mod.arg_bool("is_debug", false);
    G.is_multi_layer   = mod.arg_bool("is_multi_layer", false);
    float main_freq    = mod.arg_float("update_rate", 5.0f);
    float converge_d   = mod.arg_float("converge_dist", 1.0f);
    int momentum_thr   = mod.arg_int("momentum_thred", 5);

    // Compute derived parameters (same as LoadROSParams)
    float floor_height = mod.arg_float("floor_height", 2.0f);
    G.kHeightVoxel     = G.kLeafSize * 2.0f;
    G.kNearDist        = G.robot_dim;
    G.kMatchDist       = G.robot_dim * 2.0f + G.kLeafSize;
    G.kNavClearDist    = G.robot_dim / 2.0f + G.kLeafSize;
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

    std::string topic_scan  = mod.topic("registered_scan");
    std::string topic_odom  = mod.topic("odometry");
    std::string topic_goal  = mod.topic("goal");
    std::string topic_wp    = mod.topic("way_point");

    // LCM subscribe requires member-function + object pointer; wrap free fns
    // in a trivial handler struct.
    struct LcmHandler {
        static void odom_cb(const lcm::ReceiveBuffer* b, const std::string& c,
                            const nav_msgs::Odometry* m) { on_odometry(b, c, m); }
        static void scan_cb(const lcm::ReceiveBuffer* b, const std::string& c,
                            const sensor_msgs::PointCloud2* m) { on_registered_scan(b, c, m); }
        static void goal_cb(const lcm::ReceiveBuffer* b, const std::string& c,
                            const geometry_msgs::PointStamped* m) { on_goal(b, c, m); }
        void odom(const lcm::ReceiveBuffer* b, const std::string& c,
                  const nav_msgs::Odometry* m) { on_odometry(b, c, m); }
        void scan(const lcm::ReceiveBuffer* b, const std::string& c,
                  const sensor_msgs::PointCloud2* m) { on_registered_scan(b, c, m); }
        void goal(const lcm::ReceiveBuffer* b, const std::string& c,
                  const geometry_msgs::PointStamped* m) { on_goal(b, c, m); }
    } lcm_handler;
    lcm.subscribe(topic_odom, &LcmHandler::odom, &lcm_handler);
    lcm.subscribe(topic_scan, &LcmHandler::scan, &lcm_handler);
    lcm.subscribe(topic_goal, &LcmHandler::goal, &lcm_handler);

    printf("[FAR] Subscribed: scan=%s  odom=%s  goal=%s\n",
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
    contour_det.kRatio = mod.arg_float("resize_ratio", 5.0f);
    contour_det.kThredValue = mod.arg_int("filter_count_value", 5);
    contour_det.kBlurSize = (int)std::round(G.kNavClearDist / G.kLeafSize);
    contour_det.Init();
#endif

    bool is_graph_init = false;
    const int loop_ms = (int)(1000.0f / main_freq);

    printf("[FAR] Entering main loop (period=%dms)...\n", loop_ms);

    // --- Main loop ---
    while (!g_shutdown.load()) {
        // Handle pending LCM messages (non-blocking with timeout)
        lcm.handleTimeout(loop_ms);

        // Check preconditions
        bool odom_ok, cloud_ok, goal_pending;
        Point3D robot_p, goal_p;
        std::vector<smartnav::PointXYZI> obs_snap;
        {
            std::lock_guard<std::mutex> lk(g_state_mutex);
            odom_ok = g_odom_init;
            cloud_ok = g_cloud_init;
            goal_pending = g_goal_received;
            robot_p = g_robot_pos;
            goal_p = g_goal_point;
            if (cloud_ok) {
                obs_snap = g_obs_points; // copy
            }
            if (goal_pending) g_goal_received = false;
        }

        // Debug: periodic status (every ~2s at 5Hz)
        if (verbose) {
            static int dbg_ctr = 0;
            if (++dbg_ctr % 10 == 0) {
                auto gp_tmp = planner.goal_node;
                float goal_dist = gp_tmp ? (robot_p - Point3D(gp_tmp->position.x, gp_tmp->position.y, gp_tmp->position.z)).norm() : 0.0f;
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

        // 2. Extract contours from obstacle cloud
        std::vector<PointStack> realworld_contours;
#ifdef HAS_OPENCV
        contour_det.BuildAndExtract(odom_node->position, obs_snap, realworld_contours);
#endif

        // 3. Update contour graph
        contour_mgr.UpdateContourGraph(odom_node, realworld_contours);

        // 4. Update global near nodes
        graph_mgr.UpdateGlobalNearNodes();

        // 5. Extract new graph nodes (trajectory nodes)
        NodePtrStack new_nodes;
        if (graph_mgr.ExtractGraphNodes()) {
            new_nodes = graph_mgr.new_nodes;
        }

        // 6. Update navigation graph edges
        graph_mgr.UpdateNavGraph(new_nodes, false);

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

            if (planner.PathToGoal(gp, global_path, nav_wp, cur_goal, is_fail, is_succeed) && nav_wp) {
                // Publish graph-planned waypoint
                geometry_msgs::PointStamped wp_msg;
                wp_msg.header = dimos::make_header(G.worldFrameId,
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                wp_msg.point.x = nav_wp->position.x;
                wp_msg.point.y = nav_wp->position.y;
                wp_msg.point.z = nav_wp->position.z;
                lcm.publish(topic_wp, &wp_msg);

                float dist_to_goal = (odom_node->position - cur_goal).norm();
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
            } else if (is_fail) {
                // Graph too sparse to plan — do NOT publish the goal
                // directly as waypoint (that drives the robot into walls).
                // Wait for the graph to grow via exploration or manual driving.

                // Count how many graph nodes are traversable and connected to goal
                int traversable_count = 0, goal_connected = 0;
                for (const auto& n : nav_graph) {
                    if (n->is_traversable) traversable_count++;
                }
                for (const auto& cn : gp->connect_nodes) {
                    (void)cn; goal_connected++;
                }

                if (verbose) {
                    printf("[FAR] NO ROUTE → goal=(%.2f,%.2f,%.2f)  "
                           "robot=(%.2f,%.2f)  graph_nodes=%zu  traversable=%d  "
                           "goal_edges=%d  dist=%.1fm\n",
                           cur_goal.x, cur_goal.y, cur_goal.z,
                           odom_node->position.x, odom_node->position.y,
                           nav_graph.size(), traversable_count, goal_connected,
                           (odom_node->position - cur_goal).norm());
                    fflush(stdout);
                }
            }

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
