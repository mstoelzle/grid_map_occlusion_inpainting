/*
 * occlusion_inpainting_node.cpp
 *
 *  Created on: July 18, 2021
 *      Author: Maximilian Stölzle
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include "grid_map_occlusion_inpainting_ros/occlusion_inpainting_node.hpp"
#include <grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp>

namespace grid_map_occlusion_inpainting {

OcclusionInpaintingNode::OcclusionInpaintingNode(ros::NodeHandle& nodeHandle)
{
  ROS_INFO("OcclusionInpaintingNode started");
  nodeHandle_ = nodeHandle;

  grid_map_occlusion_inpainting::OcclusionInpainter occInpainter_();

  // Subscriber
  nodeHandle_.param<std::string>("occ_grid_map_topic", occGridMapTopic_, "occ_grid_map");
  sub_ = nodeHandle_.subscribe(occGridMapTopic_, 1, &OcclusionInpaintingNode::sub_callback, this);

  // Publisher
  nodeHandle_.param<std::string>("rec_grid_map_topic", recGridMapTopic_, "rec_grid_map");
  pub_ = nodeHandle_.advertise<grid_map_msgs::GridMap>(recGridMapTopic_, 1, true);
}

OcclusionInpaintingNode::~OcclusionInpaintingNode()
{
  ROS_INFO("OcclusionInpaintingNode deconstructed");
}

void OcclusionInpaintingNode::sub_callback(const grid_map_msgs::GridMap & occGridMapMsg) 
{
  ROS_INFO("Received occluded GridMap message");

  // load occluded grip map
  grid_map::GridMap occGridMap;
  grid_map::GridMapRosConverter::fromMessage(occGridMapMsg, occGridMap);

  occInpainter_.setOccGridMap(occGridMap);
  if (!occInpainter_.inpaintGridMap()){
    ROS_WARN("Could not inpaint grid map");
  }
  
  grid_map::GridMap recGridMap = occInpainter_.getGridMap();

  // publish reconstructed DEM
  grid_map_msgs::GridMap recGridMapMsg;
  grid_map::GridMapRosConverter::toMessage(recGridMap, recGridMapMsg);
  pub_.publish(recGridMapMsg);
}

} /* namespace */

int main(int argc, char** argv) {
  ROS_INFO("Launched occlusion_inpainting_node");

  ros::init(argc, argv, "occlusion_inpainting_node");
  ros::NodeHandle nodeHandle("~");

  grid_map_occlusion_inpainting::OcclusionInpaintingNode node(nodeHandle);

  // run
  ros::spin();
  return EXIT_SUCCESS;
}