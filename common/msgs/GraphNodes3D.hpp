// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Typed C++ helper mirroring the Python `dimos.msgs.nav_msgs.GraphNodes3D`
// wrapper. Wire format is `nav_msgs::Path`: each `PoseStamped`'s
// `position` is a node, and `orientation.w` encodes its `node_type`.
// The Python `GraphNodes3D.lcm_decode` reads exactly this layout — keep
// the two in sync.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <lcm/lcm-cpp.hpp>

#include "geometry_msgs/PoseStamped.hpp"
#include "nav_msgs/Path.hpp"

#include "dimos_native_module.hpp"

namespace dimos {

class GraphNodes3D {
public:
    // Must match GraphNodes3D.py's TYPE_COLORS keys.
    enum NodeType : int32_t {
        NORMAL = 0,
        ODOM = 1,
        GOAL = 2,
        FRONTIER = 3,
        NAVPOINT = 4,
    };

    GraphNodes3D(std::string frame_id, double ts)
        : frame_id_(std::move(frame_id)), ts_(ts) {}

    void reserve(size_t capacity) { nodes_.reserve(capacity); }

    void add(float x, float y, float z, int32_t node_type = NORMAL) {
        nodes_.push_back({x, y, z, node_type});
    }

    size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }

    nav_msgs::Path to_lcm_path() const {
        nav_msgs::Path msg;
        msg.header = make_header(frame_id_, ts_);
        msg.poses_length = static_cast<int32_t>(nodes_.size());
        msg.poses.resize(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto& pose = msg.poses[i];
            pose.header = msg.header;
            pose.pose.position.x = nodes_[i].x;
            pose.pose.position.y = nodes_[i].y;
            pose.pose.position.z = nodes_[i].z;
            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = 0.0;
            // orientation.w carries node_type (see GraphNodes3D.py).
            pose.pose.orientation.w = static_cast<double>(nodes_[i].type);
        }
        return msg;
    }

    int publish(lcm::LCM& lcm, const std::string& channel) const {
        nav_msgs::Path msg = to_lcm_path();
        return lcm.publish(channel, &msg);
    }

private:
    struct Node {
        float x;
        float y;
        float z;
        int32_t type;
    };

    std::string frame_id_;
    double ts_;
    std::vector<Node> nodes_;
};

}  // namespace dimos
