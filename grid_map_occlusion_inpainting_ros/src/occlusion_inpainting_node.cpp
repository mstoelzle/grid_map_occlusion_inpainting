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

OcclusionInpaintingNode::OcclusionInpaintingNode(ros::NodeHandle& nodeHandle)
{
  ROS_INFO("OcclusionInpaintingNode started");
  nodeHandle_ = nodeHandle;

  // Subscriber
  nodeHandle_.param<std::string>("occ_grid_map_topic", occGridMapTopic_, "occ_grid_map");
  sub_ = nodeHandle_.subscribe(occGridMapTopic_, 1, &OcclusionInpaintingNode::sub_callback, this);

  // reconstructed DEMs publisher
  nodeHandle_.param<std::string>("rec_grid_map_topic", recGridMapTopic_, "rec_grid_map");
  recPub_ = nodeHandle_.advertise<grid_map_msgs::GridMap>(recGridMapTopic_, 1, true);

  // composed DEMs publisher
  nodeHandle_.param<std::string>("comp_grid_map_topic", compGridMapTopic_, "comp_grid_map");
  compPub_ = nodeHandle_.advertise<grid_map_msgs::GridMap>(compGridMapTopic_, 1, true);
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

  occInpainter_.loadOccGridMap(occGridMap);
  occInpainter_.inpaintGridMap();
  grid_map::GridMap recGridMap = occInpainter_.getRecGridMap();
  grid_map::GridMap compGridMap = occInpainter_.getCompGridMap();

  // publish reconstructed DEM
  grid_map_msgs::GridMap recGridMapMsg;
  grid_map::GridMapRosConverter::toMessage(recGridMap, recGridMapMsg);
  rec_pub_.publish(recGridMapMsg);

  // publish composed DEM
  grid_map_msgs::GridMap compGridMapMsg;
  grid_map::GridMapRosConverter::toMessage(compGridMap, compGridMapMsg);
  rec_pub_.publish(compGridMapMsg);
}

int main(int argc, char** argv) {
  ROS_INFO("Launched occlusion_inpainting_node");

  ros::init(argc, argv, "occlusion_inpainting_node");
  ros::NodeHandle nodeHandle("~");

  OcclusionInpaintingNode node = OcclusionInpaintingNode(nodeHandle);

  // run
  ros::spin();
  return EXIT_SUCCESS;
}