/*
 * OcclusionInpainter.hpp
 *
 *  Created on: Jul 18, 2021
 *      Author: Maximilian Stoelzle
 *	 Institute: ETH Zurich
 */

#pragma once

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_msgs/GridMap.h>

namespace grid_map_occlusion_inpainting {

class OcclusionInpainter
{
    public:
        /*!
        * Default constructor.
        */
        OcclusionInpainter();

        /*!
        * Destructor.
        */
        virtual ~OcclusionInpainter();

        // getters and setters
        void setOccGridMap(const grid_map::GridMap occGridMap);
        grid_map::GridMap getRecGridMap();
        grid_map::GridMap getCompGridMap();

        // logic functions
        bool inpaintGridMap();

    private:
        // Grid maps
        grid_map::GridMap occGridMap_;
        grid_map::GridMap recGridMap_;
        grid_map::GridMap compGridMap_;

        std::string method_; // Telea, Navier-Stokes or NN
};

} /* namespace */
