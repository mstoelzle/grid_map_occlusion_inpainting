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

enum {
    INPAINT_NS = 0, //!< Use Navier-Stokes based method
    INPAINT_TELEA = 1, //!< Use the algorithm proposed by Alexandru Telea @cite Telea04
    INPAINT_NN = 2 // inpainting using a pretrained neural network   
};

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

        // Parameters
        int inpaint_method_ = 0; // Telea, Navier-Stokes or NN
        double inpaint_radius_ = 3.; // inpaint radius for Telea, Navier-Stokes

        std::string inputLayer_ = "occ_grid_map";

        // getters and setters
        void setOccGridMap(const grid_map::GridMap occGridMap);
        grid_map::GridMap getGridMap();

        // logic functions
        bool inpaintGridMap();
        void addOccMask();
        void addCompLayer();

    protected:
        // Grid maps
        grid_map::GridMap gridMap_;

        bool inpaintOpenCV();
        #if USE_TORCH
            bool inpaintNeuralNetwork();
        #endif
};

} /* namespace */
