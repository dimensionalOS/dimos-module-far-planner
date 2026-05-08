#pragma once

// ROS-to-standalone compatibility header for FAR Planner.
// Replaces ROS2 constructs with minimal C++ stubs so algorithm code compiles
// without any ROS dependency.  Real I/O goes through LCM in main.cpp.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Logging macros (replace RCLCPP_* — the `logger` argument is ignored)
// ---------------------------------------------------------------------------

#define RCLCPP_INFO(logger, ...) \
    fprintf(stdout, "[INFO] " __VA_ARGS__), fprintf(stdout, "\n")
#define RCLCPP_WARN(logger, ...) \
    fprintf(stderr, "[WARN] " __VA_ARGS__), fprintf(stderr, "\n")
#define RCLCPP_ERROR(logger, ...) \
    fprintf(stderr, "[ERROR] " __VA_ARGS__), fprintf(stderr, "\n")

#define RCLCPP_WARN_ONCE(logger, ...)                       \
    do {                                                     \
        static bool _once = false;                           \
        if (!_once) {                                        \
            _once = true;                                    \
            RCLCPP_WARN(logger, __VA_ARGS__);                \
        }                                                    \
    } while (0)

#define RCLCPP_INFO_STREAM(logger, expr)                                \
    do {                                                                 \
        std::ostringstream _ss;                                          \
        _ss << expr;                                                     \
        fprintf(stdout, "[INFO] %s\n", _ss.str().c_str());              \
    } while (0)

#define RCLCPP_WARN_STREAM(logger, expr)                                \
    do {                                                                 \
        std::ostringstream _ss;                                          \
        _ss << expr;                                                     \
        fprintf(stderr, "[WARN] %s\n", _ss.str().c_str());              \
    } while (0)

#define RCLCPP_ERROR_STREAM(logger, expr)                               \
    do {                                                                 \
        std::ostringstream _ss;                                          \
        _ss << expr;                                                     \
        fprintf(stderr, "[ERROR] %s\n", _ss.str().c_str());             \
    } while (0)

#define RCLCPP_DEBUG(logger, ...) (void)0
#define RCLCPP_DEBUG_STREAM(logger, expr) (void)0

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

struct CompatTime {
    double sec_ = 0.0;
    double seconds() const { return sec_; }
    double nanoseconds() const { return sec_ * 1e9; }
    static CompatTime now() {
        auto tp = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(tp.time_since_epoch()).count();
        CompatTime t;
        t.sec_ = s;
        return t;
    }
};

struct TimeStamp {
    int32_t sec = 0;
    uint32_t nanosec = 0;
    double seconds() const { return sec + nanosec * 1e-9; }
    TimeStamp() = default;
    TimeStamp(const CompatTime& t) {
        sec = static_cast<int32_t>(t.sec_);
        nanosec = static_cast<uint32_t>((t.sec_ - sec) * 1e9);
    }
};

// ---------------------------------------------------------------------------
// Header (std_msgs::msg::Header)
// ---------------------------------------------------------------------------

namespace std_msgs { namespace msg {

struct Header {
    std::string frame_id;
    TimeStamp stamp;
};

struct Bool   { bool data = false; using SharedPtr = std::shared_ptr<Bool>; };
struct Empty  { using SharedPtr = std::shared_ptr<Empty>; };
struct String { std::string data; using SharedPtr = std::shared_ptr<String>; };
struct Float32 { float data = 0.0f; using SharedPtr = std::shared_ptr<Float32>; };
struct ColorRGBA { float r = 0, g = 0, b = 0, a = 1; };

}} // namespace std_msgs::msg

// ---------------------------------------------------------------------------
// geometry_msgs
// ---------------------------------------------------------------------------

namespace geometry_msgs { namespace msg {

struct Point   { double x = 0, y = 0, z = 0; };
struct Point32 { float  x = 0, y = 0, z = 0; };
struct Vector3 { double x = 0, y = 0, z = 0; };

struct Quaternion { double x = 0, y = 0, z = 0, w = 1; };

struct Pose {
    Point position;
    Quaternion orientation;
};

struct PoseWithCovariance {
    Pose pose;
};

struct Twist {
    Vector3 linear;
    Vector3 angular;
};

struct TwistWithCovariance {
    Twist twist;
};

struct PoseStamped {
    std_msgs::msg::Header header;
    Pose pose;
};

struct Polygon {
    std::vector<Point32> points;
};

struct PolygonStamped {
    std_msgs::msg::Header header;
    Polygon polygon;
};

struct PointStamped {
    std_msgs::msg::Header header;
    Point point;
};

struct Transform {
    Vector3 translation;
    Quaternion rotation;
};

struct TransformStamped {
    std_msgs::msg::Header header;
    std::string child_frame_id;
    Transform transform;
};

}} // namespace geometry_msgs::msg

// ---------------------------------------------------------------------------
// nav_msgs
// ---------------------------------------------------------------------------

namespace nav_msgs { namespace msg {

struct Odometry {
    std_msgs::msg::Header header;
    std::string child_frame_id;
    geometry_msgs::msg::PoseWithCovariance pose;
    geometry_msgs::msg::TwistWithCovariance twist;
};

struct Path {
    std_msgs::msg::Header header;
    std::vector<geometry_msgs::msg::PoseStamped> poses;
};

}} // namespace nav_msgs::msg

// ---------------------------------------------------------------------------
// sensor_msgs
// ---------------------------------------------------------------------------

namespace sensor_msgs { namespace msg {

struct PointField {
    std::string name;
    uint32_t offset = 0;
    uint8_t datatype = 0;
    uint32_t count = 0;
    static constexpr uint8_t FLOAT32 = 7;
};

struct PointCloud2 {
    std_msgs::msg::Header header;
    uint32_t height = 0;
    uint32_t width = 0;
    std::vector<PointField> fields;
    bool is_bigendian = false;
    uint32_t point_step = 0;
    uint32_t row_step = 0;
    std::vector<uint8_t> data;
    bool is_dense = false;
};

struct Joy {
    std_msgs::msg::Header header;
    std::vector<float> axes;
    std::vector<int> buttons;
};

}} // namespace sensor_msgs::msg

// ---------------------------------------------------------------------------
// visualization_msgs
// ---------------------------------------------------------------------------

namespace visualization_msgs { namespace msg {

struct Marker {
    std_msgs::msg::Header header;
    std::string ns;
    int32_t id = 0;
    int32_t type = 0;
    int32_t action = 0;
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Vector3 scale;
    std_msgs::msg::ColorRGBA color;
    CompatTime lifetime;
    bool frame_locked = false;
    std::vector<geometry_msgs::msg::Point> points;
    std::vector<std_msgs::msg::ColorRGBA> colors;
    std::string text;

    static constexpr int32_t ARROW = 0;
    static constexpr int32_t CUBE = 1;
    static constexpr int32_t SPHERE = 2;
    static constexpr int32_t CYLINDER = 3;
    static constexpr int32_t LINE_STRIP = 4;
    static constexpr int32_t LINE_LIST = 5;
    static constexpr int32_t CUBE_LIST = 6;
    static constexpr int32_t SPHERE_LIST = 7;
    static constexpr int32_t POINTS = 8;
    static constexpr int32_t TEXT_VIEW_FACING = 9;
    static constexpr int32_t MESH_RESOURCE = 10;
    static constexpr int32_t TRIANGLE_LIST = 11;

    static constexpr int32_t ADD = 0;
    static constexpr int32_t MODIFY = 0;
    static constexpr int32_t DELETE = 2;
    static constexpr int32_t DELETEALL = 3;

    bool operator==(const Marker& other) const {
        return type == other.type && ns == other.ns && id == other.id;
    }
};

struct MarkerArray {
    std::vector<Marker> markers;
};

}} // namespace visualization_msgs::msg

// ---------------------------------------------------------------------------
// visibility_graph_msg
// ---------------------------------------------------------------------------

namespace visibility_graph_msg { namespace msg {

struct Node {
    std_msgs::msg::Header header;
    int64_t id = 0;
    geometry_msgs::msg::Point position;
    int freetype = 0;
    int is_covered = 0;
    int is_frontier = 0;
    int is_navpoint = 0;
    int is_boundary = 0;
    std::vector<geometry_msgs::msg::Point> surface_dirs;
    std::vector<int64_t> connect_nodes;
    std::vector<int64_t> poly_connects;
    std::vector<int64_t> contour_connects;
    std::vector<int64_t> traj_connects;
    std::vector<int64_t> trajectory_connects;
};

struct Graph {
    std_msgs::msg::Header header;
    std::vector<Node> nodes;
    int size = 0;
    int robot_id = 0;
    using SharedPtr = std::shared_ptr<Graph>;
};

}} // namespace visibility_graph_msg::msg

// ---------------------------------------------------------------------------
// Stub publisher / subscription
// ---------------------------------------------------------------------------

template <typename MsgT>
struct StubPublisher {
    using SharedPtr = std::shared_ptr<StubPublisher<MsgT>>;
    void publish(const MsgT& /*msg*/) {}
};

template <typename MsgT>
struct StubSubscription {
    using SharedPtr = std::shared_ptr<StubSubscription<MsgT>>;
};

// ---------------------------------------------------------------------------
// rclcpp shim
// ---------------------------------------------------------------------------

namespace rclcpp {

struct Logger {};

struct Duration {
    double sec_;
    explicit Duration(double s) : sec_(s) {}
    Duration(int32_t s, uint32_t ns) : sec_(s + ns * 1e-9) {}
    double seconds() const { return sec_; }
};

struct Time {
    double sec_ = 0.0;
    double seconds() const { return sec_; }
    double nanoseconds() const { return sec_ * 1e9; }
};

struct QoS {
    int depth_;
    explicit QoS(int d) : depth_(d) {}
    QoS& reliable() { return *this; }
    QoS& best_effort() { return *this; }
    QoS& durability_volatile() { return *this; }
    QoS& keep_last(int) { return *this; }
    QoS& transient_local() { return *this; }
};

inline QoS SystemDefaultsQoS() { return QoS(10); }
inline QoS SensorDataQoS() { return QoS(5); }

inline void init(int /*argc*/, char** /*argv*/) {}
inline void shutdown() {}
inline void spin(std::shared_ptr<void>) {}
inline bool ok() { return true; }

} // namespace rclcpp

// ---------------------------------------------------------------------------
// CompatNode — minimal rclcpp::Node replacement
// ---------------------------------------------------------------------------

class CompatNode {
public:
    using SharedPtr = std::shared_ptr<CompatNode>;

    explicit CompatNode(const std::string& name) : name_(name) {}

    rclcpp::Logger get_logger() const { return {}; }

    CompatTime now() const { return CompatTime::now(); }

    const std::string& get_name() const { return name_; }

    void* get_clock() const { return nullptr; }

    template <typename MsgT, typename Callback>
    std::shared_ptr<StubSubscription<MsgT>> create_subscription(
        const std::string& /*topic*/, const rclcpp::QoS& /*qos*/,
        Callback&& /*cb*/) {
        return nullptr;
    }

    template <typename MsgT, typename Callback>
    std::shared_ptr<StubSubscription<MsgT>> create_subscription(
        const std::string& /*topic*/, int /*qos*/, Callback&& /*cb*/) {
        return nullptr;
    }

    template <typename MsgT>
    std::shared_ptr<StubPublisher<MsgT>> create_publisher(
        const std::string& /*topic*/, const rclcpp::QoS& /*qos*/) {
        return std::make_shared<StubPublisher<MsgT>>();
    }

    template <typename MsgT>
    std::shared_ptr<StubPublisher<MsgT>> create_publisher(
        const std::string& /*topic*/, int /*qos*/) {
        return std::make_shared<StubPublisher<MsgT>>();
    }

    struct TimerBase {};

    template <typename Callback>
    std::shared_ptr<TimerBase> create_wall_timer(
        std::chrono::milliseconds /*period*/, Callback&& /*cb*/) {
        return std::make_shared<TimerBase>();
    }

    template <typename T>
    void declare_parameter(const std::string& /*name*/, const T& /*default_val*/) {}

    template <typename T>
    T get_parameter_or(const std::string& /*name*/, const T& default_val) {
        return default_val;
    }

private:
    std::string name_;
};

namespace rclcpp {
using Node = CompatNode;

template <typename MsgT>
using Publisher = StubPublisher<MsgT>;

template <typename MsgT>
using Subscription = StubSubscription<MsgT>;

using TimerBase = CompatNode::TimerBase;
} // namespace rclcpp

// ---------------------------------------------------------------------------
// tf2 stubs
// ---------------------------------------------------------------------------

namespace tf2 {

struct Matrix3x3 {
    void getRPY(double& roll, double& pitch, double& yaw) const {
        roll = 0; pitch = 0; yaw = 0;
    }
};

struct Vector3 {
    double x_ = 0, y_ = 0, z_ = 0;
    double x() const { return x_; }
    double y() const { return y_; }
    double z() const { return z_; }
};

struct Quaternion {
    double x_ = 0, y_ = 0, z_ = 0, w_ = 1;
};

struct Transform {
    Vector3 origin_;
    Matrix3x3 basis_;

    const Vector3& getOrigin() const { return origin_; }
    const Matrix3x3& getBasis() const { return basis_; }
    void setOrigin(const Vector3& v) { origin_ = v; }
};

inline void fromMsg(const geometry_msgs::msg::Quaternion& /*msg*/, Quaternion& /*q*/) {}
inline void fromMsg(const geometry_msgs::msg::Transform& /*in*/, Transform& /*out*/) {}
inline void fromMsg(const geometry_msgs::msg::Pose& /*msg*/, Transform& /*t*/) {}

struct TransformException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename MsgT>
inline void doTransform(const MsgT& in, MsgT& out,
                        const geometry_msgs::msg::TransformStamped& /*tf*/) {
    out = in; // No-op in standalone mode
}

inline CompatTime TimePointZero{};

} // namespace tf2

namespace tf2_ros {

class Buffer {
public:
    Buffer() = default;
    explicit Buffer(void* /*clock*/) {}

    geometry_msgs::msg::TransformStamped lookupTransform(
        const std::string& /*target*/, const std::string& /*source*/,
        const CompatTime& /*time*/) const {
        return {};
    }

    geometry_msgs::msg::TransformStamped lookupTransform(
        const std::string& /*target*/, const std::string& /*source*/,
        const CompatTime& /*time*/,
        const rclcpp::Duration& /*timeout*/) const {
        return {};
    }

    bool canTransform(const std::string& /*target*/, const std::string& /*source*/,
                      const CompatTime& /*time*/) const {
        return false;
    }

    bool canTransform(const std::string& /*target*/, const std::string& /*source*/,
                      const CompatTime& /*time*/, const std::chrono::seconds& /*timeout*/) const {
        return false;
    }

    template <typename MsgT>
    void transform(const MsgT& in, MsgT& out, const std::string& /*target*/) const {
        out = in;
    }
};

class TransformListener {
public:
    explicit TransformListener(Buffer& /*buf*/) {}
    TransformListener(Buffer& /*buf*/, bool /*spin_thread*/) {}
};

} // namespace tf2_ros

// pcl_conversions stubs are defined after PCL headers in utility.h

// Provide empty header paths that algorithm code may include.
// The actual includes are satisfied by this single compat header.
