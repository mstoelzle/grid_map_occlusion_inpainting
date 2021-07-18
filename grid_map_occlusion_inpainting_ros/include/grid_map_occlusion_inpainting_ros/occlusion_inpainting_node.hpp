/*
 * PointCloud2_to_GridMap_msg_node.hpp
 *
 *  Created on: July 03, 2021
 *      Author: Maximilian Stölzle
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>

#include <grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp>

namespace grid_map_occlusion_inpainting {

class OcclusionInpaintingNode
{
    public:
        /*!
        * Constructor.
        * @param nodeHandle the ROS node handle.
        */
        OcclusionInpaintingNode(ros::NodeHandle& nodeHandle);
        ~OcclusionInpaintingNode();

        /*!
        * Callback function for the point cloud 2.
        * @param message the point cloud2 message to be converted to a grid map msg
        */
        void sub_callback(const grid_map_msgs::GridMap & occGridMapMsg);

    private:
        std::string occGridMapTopic_;
        std::string recGridMapTopic_;
        std::string compGridMapTopic_;

        //! ROS nodehandle.
        ros::NodeHandle nodeHandle_;

        // Publishers and subscribers
        ros::Subscriber sub_;
        ros::Publisher recPub_;
        ros::Publisher compPub_;

        grid_map_occlusion_inpainting::OcclusionInpainter occInpainter_;
};

} /* namespace */