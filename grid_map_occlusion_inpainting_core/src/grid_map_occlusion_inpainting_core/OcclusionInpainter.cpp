
/*
 * OcclusionInpainter.cpp
 *
 *  Created on: Jul 18, 2021
 *      Author: Maximilian Stoelzle
 *	 Institute: ETH Zurich
 */

#include "grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp"

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_msgs/GridMap.h>

namespace grid_map_occlusion_inpainting {

OcclusionInpainter::OcclusionInpainter()
{

}

OcclusionInpainter::~OcclusionInpainter()
{
    
}

void OcclusionInpainter::setOccGridMap(const grid_map::GridMap occGridMap)
{
    occGridMap_ = occGridMap;
}

grid_map::GridMap OcclusionInpainter::getRecGridMap()
{
    return recGridMap_;
}

grid_map::GridMap OcclusionInpainter::getCompGridMap()
{
    return compGridMap_;
}

bool OcclusionInpainter::inpaintGridMap()
{
    recGridMap_ = occGridMap_;
    compGridMap_ = occGridMap_;
    return true;
}

} /* namespace */