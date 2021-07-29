/*
 * occlusion_inpainting_node.cpp
 *
 *  Created on: July 18, 2021
 *      Author: Maximilian Stölzle
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>
#include <ros/package.h>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include "grid_map_occlusion_inpainting_ros/occlusion_inpainting_node.hpp"
#include <grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp>

namespace gmoi = grid_map_occlusion_inpainting;

namespace grid_map_occlusion_inpainting {

OcclusionInpaintingNode::OcclusionInpaintingNode(ros::NodeHandle& nodeHandle)
{
  ROS_INFO("OcclusionInpaintingNode started");
  nodeHandle_ = nodeHandle;

  grid_map_occlusion_inpainting::OcclusionInpainter occInpainter_();

  // Subscriber
  nodeHandle_.param<std::string>("input_grid_map_topic", inputGridMapTopic_, "input_grid_map");
  sub_ = nodeHandle_.subscribe(inputGridMapTopic_, 1, &OcclusionInpaintingNode::sub_callback, this);

  // Publisher
  nodeHandle_.param<std::string>("rec_grid_map_topic", recGridMapTopic_, "rec_grid_map");
  pub_ = nodeHandle_.advertise<grid_map_msgs::GridMap>(recGridMapTopic_, 1, true);
}

OcclusionInpaintingNode::~OcclusionInpaintingNode()
{
  ROS_INFO("OcclusionInpaintingNode deconstructed");
}

void OcclusionInpaintingNode::configure() {
  // Configure OcclusionInpainter class
  nodeHandle_.param<int>("inpaint_method", occInpainter_.inpaintMethod_, 0);
  nodeHandle_.param<std::string>("input_layer", occInpainter_.inputLayer_, "occ_grid_map");
  nodeHandle_.param<bool>("resizing/resize", occInpainter_.resize_, false);
  nodeHandle_.param<double>("resizing/target_resolution", occInpainter_.targetResolution_, 0.1);
  nodeHandle_.param<double>("inpaint_radius", occInpainter_.inpaintRadius_, 0.3);  
  nodeHandle_.param<bool>("use_gpu", occInpainter_.useGpu_, false);
  nodeHandle_.param<double>("NaN_replacement", occInpainter_.NaN_replacement_, 0.);
  nodeHandle_.param<bool>("subgrids/divide_into_subgrids", occInpainter_.divideIntoSubgrids_, false);
  nodeHandle_.param<int>("subgrids/subgrid_rows", occInpainter_.subgridRows_, 64);
  nodeHandle_.param<int>("subgrids/subgrid_cols", occInpainter_.subgridCols_, 64);
  nodeHandle_.param<double>("subgrids/subgrid_max_occ_ratio_thresh", occInpainter_.subgridMaxOccRatioThresh_, 1.);
  nodeHandle_.param<bool>("visualize_with_open_cv", occInpainter_.visualizeWithOpenCV_, false);

  std::string relNeuralNetworkPath;
  nodeHandle_.param<std::string>("neural_network_path", relNeuralNetworkPath, "models/default.pt");
  occInpainter_.neuralNetworkPath_ = ros::package::getPath("grid_map_occlusion_inpainting_core") + "/" + relNeuralNetworkPath;
  ROS_INFO_STREAM(occInpainter_.neuralNetworkPath_);

  if (occInpainter_.inpaintMethod_ == gmoi::INPAINT_NN) {
    #if USE_TORCH
      occInpainter_.loadNeuralNetworkModel();
    #else
      throw std::invalid_argument("The occlusion inpainting node was compiled without libtorch / PyTorch support." );
    #endif
  }
}

void OcclusionInpaintingNode::sub_callback(const grid_map_msgs::GridMap & inputGridMapMsg) 
{
  ROS_INFO("Received occluded GridMap message");

  // load occluded grip map
  grid_map::GridMap inputGridMap;
  grid_map::GridMapRosConverter::fromMessage(inputGridMapMsg, inputGridMap);

  occInpainter_.setInputGridMap(inputGridMap);
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
  node.configure();

  // run
  ros::spin();
  return EXIT_SUCCESS;
}