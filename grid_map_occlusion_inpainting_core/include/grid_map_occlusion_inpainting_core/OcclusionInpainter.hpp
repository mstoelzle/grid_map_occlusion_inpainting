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

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#if USE_TORCH
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#endif

namespace grid_map_occlusion_inpainting {

enum {
    INPAINT_NS = 0, //!< Use Navier-Stokes based method
    INPAINT_TELEA = 1, //!< Use the algorithm proposed by Alexandru Telea @cite Telea04
    INPAINT_NN = 2 // inpainting using a pretrained neural network   
};

template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

class OcclusionInpainter
{
    public:
        /*!
        * Default constructor.
        */
        OcclusionInpainter(int inpaintMethod = 0, const std::string& inputLayer = "occ_grid_map");

        /*!
        * Destructor.
        */
        virtual ~OcclusionInpainter();

        /*** Parameters ***/
        int inpaintMethod_; // Navier-Stokes, Telea or NN
        std::string inputLayer_;

        // resizing of grid map prior to inference
        bool resize_ = false;
        double targetResolution_ = 0.1; // [m/px] resize in example from 300x300px with resolution of 0.1m/px to 256x256px 

        // OpenCV parameters
        double inpaintRadius_ = 0.3; // inpaint radius for Telea, Navier-Stokes [m]

        // neural network parameters
        std::string neuralNetworkPath_ = "/home/mstolzle/catkin_ws/src/grid_map_occlusion_inpainting/grid_map_occlusion_inpainting_core/models/gonzen.pt";
        bool useGpu_ = false;
        double NaN_replacement_ = 0.; // replacement values for NaNs in occluded grid map before inputting into neural network

        // division into subgrids for NN inference
        bool divideIntoSubgrids_ = false;
        int subgridRows_ = -1;
        int subgridCols_ = -1;
        // We only run inference on subgrids with less than x% occlusion, otherwise we use the occluded input subgrid
        double subgridMaxOccRatioThresh_ = 1.;

        // visualization
        bool visualizeWithOpenCV_ = false;

        // getters and setters
        void setInputGridMap(const grid_map::GridMap& occGridMap);
        grid_map::GridMap getGridMap();

        // logic functions
        bool inpaintGridMap();

        static void addOccMask(grid_map::GridMap& gridMap, const std::string& layer) {
            gridMap.add("occ_mask", 0.0);
            // mapOut.setBasicLayers(std::vector<std::string>());
            for (grid_map::GridMapIterator iterator(gridMap); !iterator.isPastEnd(); ++iterator) {
                if (!gridMap.isValid(*iterator, layer)) {
                    gridMap.at("occ_mask", *iterator) = 1.0;
                }
            }
        }

        /* Add composed grid map */
        static void addCompLayer(grid_map::GridMap& gridMap) {
            gridMap.add("comp_grid_map", 0.0);
            for (grid_map::GridMapIterator iterator(gridMap); !iterator.isPastEnd(); ++iterator) {
                if (gridMap.at("occ_mask", *iterator) == 1.0) {
                    gridMap.at("comp_grid_map", *iterator) = gridMap.at("rec_grid_map", *iterator);
                } else {
                    gridMap.at("comp_grid_map", *iterator) = gridMap.at("occ_grid_map", *iterator);
                }
            }
        }

        // static helper methods
        /* get ratio of occluded cells */
        static double getOccRatio(const grid_map::GridMap& gridMap){
            int nocc_cells = gridMap["occ_mask"].numberOfFinites();
            int total_cells = gridMap.getSize()[0] *  gridMap.getSize()[1];
            double occ_ratio = 1 - nocc_cells / ((double) total_cells);
            return occ_ratio;
        }

        void visualizeGridMapWithOpenCV(grid_map::GridMap& gridMap);

        // libtorch
        #if USE_TORCH
        bool loadNeuralNetworkModel();
        static void tensorToGridMapLayer(const torch::Tensor& tensor, const std::string& layer, grid_map::GridMap& gridMap) {
            assert(tensor.ndimension() == 2);

            float* data = tensor.data_ptr<float>();
            Eigen::Map<Eigen::MatrixXf> ef(data, tensor.size(0), tensor.size(1));

            gridMap.add(layer, ef);
        }
        static void gridMapLayerToTensor(grid_map::GridMap& gridMap, const std::string& layer, torch::Tensor& tensor) {
            tensor = torch::from_blob(gridMap[layer].data(), {1, 1,  gridMap.getSize()[0], gridMap.getSize()[1]}, at::kFloat);
        }
        #endif

    protected:
        // Grid maps
        grid_map::GridMap inputGridMap_;
        grid_map::GridMap gridMap_;

        // inpainting methods
        bool inpaintOpenCV(grid_map::GridMap& gridMap);

        // helper methods

        void replaceNaNs(grid_map::GridMap& gridMap, const std::string& inputLayer, const std::string& outputLayer) {
            gridMap.add(outputLayer, 0.0);
            gridMap[outputLayer] = gridMap[inputLayer];
            for (grid_map::GridMapIterator iterator(gridMap); !iterator.isPastEnd(); ++iterator) {
                if (!gridMap.isValid(*iterator, inputLayer)) {
                    gridMap.at(outputLayer, *iterator) = NaN_replacement_;
                }
            }
        }

        // libtorch
        #if USE_TORCH
            // libtorch attributes
            torch::jit::script::Module module_;

            static torch::Device getDevice(bool useGpu){
                torch::Device device((torch::cuda::is_available() && useGpu) ? torch::kCUDA : torch::kCPU);
                return device;
            }

            bool inpaintNeuralNetwork(grid_map::GridMap& gridMap);
        
        #endif
};

} /* namespace */
