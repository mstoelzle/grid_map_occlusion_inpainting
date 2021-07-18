
/*
 * OcclusionInpainter.cpp
 *
 *  Created on: Jul 18, 2021
 *      Author: Maximilian Stoelzle
 *	 Institute: ETH Zurich
 */

#include "grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp"

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_cv/GridMapCvConverter.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

namespace gmoi = grid_map_occlusion_inpainting;

namespace grid_map_occlusion_inpainting {

OcclusionInpainter::OcclusionInpainter()
{

}

OcclusionInpainter::~OcclusionInpainter()
{
    
}

void OcclusionInpainter::setOccGridMap(const grid_map::GridMap occGridMap)
{
    gridMap_ = occGridMap;

    gridMap_["occ_grid_map"] = gridMap_[inputLayer_];
    addOccMask();
}

grid_map::GridMap OcclusionInpainter::getGridMap()
{
    return gridMap_;
}

void OcclusionInpainter::addOccMask() {
    gridMap_.add("occ_mask", 0.0);
    // mapOut.setBasicLayers(std::vector<std::string>());
    for (grid_map::GridMapIterator iterator(gridMap_); !iterator.isPastEnd(); ++iterator) {
        if (!gridMap_.isValid(*iterator, inputLayer_)) {
            gridMap_.at("occ_mask", *iterator) = 1.0;
        }
    }
}

/* Add composed grid map */
void OcclusionInpainter::addCompLayer() {
    gridMap_.add("comp_grid_map", 0.0);
    // mapOut.setBasicLayers(std::vector<std::string>());
    for (grid_map::GridMapIterator iterator(gridMap_); !iterator.isPastEnd(); ++iterator) {
        if (gridMap_.at("occ_mask", *iterator) == 1.0) {
            gridMap_.at("comp_grid_map", *iterator) = gridMap_.at("rec_grid_map", *iterator);
        } else {
            gridMap_.at("comp_grid_map", *iterator) = gridMap_.at("occ_grid_map", *iterator);
        }
    }
}

bool OcclusionInpainter::inpaintGridMap()
{
    if (inpaint_method_ == gmoi::INPAINT_NS || inpaint_method_ == gmoi::INPAINT_TELEA)
    {
        // occ_img_cv = grid_map::GridMapCvConverter::toImage(gridMap_, "occ_grid_map", );

        // cv::inpaint(occ_img_cv, occ_mask_cv, rec_img_cv, inpaint_radius_, inpaint_method_);
    } else if (inpaint_method_ == gmoi::INPAINT_NN) {

    } else {
        throw std::invalid_argument("The chosen inpaint method is not implemented." );
    }

    return true;
}

} /* namespace */