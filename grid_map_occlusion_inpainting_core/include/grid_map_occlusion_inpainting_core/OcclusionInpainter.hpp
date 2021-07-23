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
        OcclusionInpainter();

        /*!
        * Destructor.
        */
        virtual ~OcclusionInpainter();

        // Parameters
        int inpaint_method_ = 0; // Telea, Navier-Stokes or NN
        double inpaint_radius_ = 0.3; // inpaint radius for Telea, Navier-Stokes [m]
        double NaN_replacement_ = 0.; // replacement values for NaNs in occluded grid map before inputting into neural network

        std::string inputLayer_ = "occ_grid_map";

        std::string neuralNetworkPath_ = "models/gonzen.pt";

        // getters and setters
        void setOccGridMap(const grid_map::GridMap occGridMap);
        grid_map::GridMap getGridMap();

        // logic functions
        bool inpaintGridMap();
        void addOccMask();
        void addCompLayer();

        // static helper methods

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
        grid_map::GridMap gridMap_;

        // inpainting methods
        bool inpaintOpenCV(grid_map::GridMap gridMap);

        // helper methods
        void replaceNaNs(grid_map::GridMap gridMap, const std::string& inputLayer, const std::string& outputLayer);

        // libtorch
        #if USE_TORCH
            // libtorch attributes
            torch::jit::script::Module module_;

            bool inpaintNeuralNetwork(grid_map::GridMap gridMap);
        
            /* static torch::Tensor eigenVectorToTorchTensor(Eigen::VectorXd e){
                auto t = torch::rand({e.rows()});
                float* data = t.data_ptr<float>();

                Eigen::Map<Eigen::VectorXf> ef(data,t.size(0),1);
                ef = e.cast<float>();

                t.requires_grad_(true);
                return t;
            }

            static torch::Tensor eigenMatrixToTorchTensor(Eigen::MatrixXd e){
                auto t = torch::rand({e.cols(),e.rows()});
                float* data = t.data_ptr<float>();

                Eigen::Map<Eigen::MatrixXf> ef(data,t.size(1),t.size(0));
                ef = e.cast<float>();
                t.requires_grad_(true);
                return t.transpose(0,1);
            }

            static Eigen::MatrixXd torchTensorToEigenMatrix(torch::Tensor t){
                auto t = torch::rand({e.cols(),e.rows()});


                float* data = t.data_ptr<float>();

                Eigen::Map<Eigen::MatrixXf> ef(data,t.size(1),t.size(0));
                ef = e.cast<float>();
                t.requires_grad_(true);
                return e.transpose(0,1);
            } */
        #endif
};

} /* namespace */
