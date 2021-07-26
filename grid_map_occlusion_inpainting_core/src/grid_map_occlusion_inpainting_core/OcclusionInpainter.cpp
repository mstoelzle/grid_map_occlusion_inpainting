
/*
 * OcclusionInpainter.cpp
 *
 *  Created on: Jul 18, 2021
 *      Author: Maximilian Stoelzle
 *	 Institute: ETH Zurich
 */
#include <cmath>

#include "grid_map_occlusion_inpainting_core/OcclusionInpainter.hpp"

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <grid_map_cv/GridMapCvConverter.hpp>
#include <grid_map_cv/GridMapCvProcessing.hpp>

#include <Eigen/Dense>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

#if USE_TORCH
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#endif

namespace gmoi = grid_map_occlusion_inpainting;

namespace grid_map_occlusion_inpainting {

OcclusionInpainter::OcclusionInpainter(int inpaintMethod, const std::string& inputLayer)
{
    inpaintMethod_ = inpaintMethod;
    inputLayer_ = inputLayer;
}

OcclusionInpainter::~OcclusionInpainter()
{
    
}

void OcclusionInpainter::setOccGridMap(const grid_map::GridMap occGridMap)
{
    inputGridMap_ = occGridMap;

    if (resize_) {
        grid_map::GridMapCvProcessing::changeResolution(occGridMap, gridMap_, targetResolution_);
    } else {
        gridMap_ = occGridMap;
    }

    gridMap_.add("occ_grid_map" ,gridMap_[inputLayer_]);
    OcclusionInpainter::addOccMask(gridMap_, inputLayer_);
}

grid_map::GridMap OcclusionInpainter::getGridMap()
{
    return gridMap_;
}


bool OcclusionInpainter::inpaintGridMap()
{
    if (inpaintMethod_ == gmoi::INPAINT_NS || inpaintMethod_ == gmoi::INPAINT_TELEA)
    {
        if (!inpaintOpenCV(gridMap_)){
            return false;
        }
    } else if (inpaintMethod_ == gmoi::INPAINT_NN) {
        #if USE_TORCH
            if (!inpaintNeuralNetwork(gridMap_)){
                return false;
            }
        #else
            throw std::invalid_argument("The library was compiled without libtorch / PyTorch support." );
        #endif
    } else {
        throw std::invalid_argument("The chosen inpaint method is not implemented." );
    }

    OcclusionInpainter::addCompLayer(gridMap_);

    /*
    std::vector<std::string> basicLayers = {"rec_grid_map", "comp_grid_map"};
    gridMap_.setBasicLayers(basicLayers);
    */

    return true;
}


bool OcclusionInpainter::inpaintOpenCV(grid_map::GridMap gridMap) {
    // TODO: some validation of inputs and parameters

    const float minValue = gridMap.get("occ_grid_map").minCoeffOfFinites();
    const float maxValue = gridMap.get("occ_grid_map").maxCoeffOfFinites();

    cv::Mat occImage;
    cv::Mat maskImage;
    cv::Mat recImage;
    grid_map::GridMapCvConverter::toImage<unsigned char, 3>(gridMap, "occ_grid_map", CV_8UC3, minValue, maxValue, occImage);
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(gridMap, "occ_mask", CV_8UC1, maskImage);

    const double radiusInPixels = inpaintRadius_ / gridMap.getResolution();
    cv::inpaint(occImage, maskImage, recImage, radiusInPixels, inpaintMethod_);

    gridMap.add("rec_grid_map", 0.0);
    grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 3>(recImage, "rec_grid_map", gridMap, minValue, maxValue);

    return true;
}

#if USE_TORCH
bool OcclusionInpainter::loadNeuralNetworkModel() {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(neuralNetworkPath_);
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Could not load the neural network model");
        return false;
    }
   
    return true;
}

bool OcclusionInpainter::inpaintNeuralNetwork(grid_map::GridMap gridMap) {
    // number of rows and cols
    int rows = gridMap.getSize()[0];
    int cols = gridMap.getSize()[1];

    // replace NaNs
    replaceNaNs(gridMap, "occ_grid_map", "nn_input_grid_map");

    // normalization of input
    /* double grid_map_mean = gridMap["occ_grid_map"].meanOfFinites();
    gridMap["norm_occ_grid_map"] = gridMap["occ_grid_map"].array() - grid_map_mean; */

    // torch device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // init torch tensors
    auto occGridMapTensor = torch::zeros({1, 1, rows, cols});
    OcclusionInpainter::gridMapLayerToTensor(gridMap, "nn_input_grid_map", occGridMapTensor);
    auto occMaskTensor = torch::zeros({1, 1, rows, cols});
    OcclusionInpainter::gridMapLayerToTensor(gridMap, "occ_mask", occMaskTensor);

    // assemble channels
    torch::Tensor inputTensorUnsplit = torch::cat({occGridMapTensor, occMaskTensor}, 1);

    // division into subgrids
    int subgridRows = subgridRows_;
    int subgridCols = subgridCols_;
    if (!divideIntoSubgrids_) {
        subgridRows = rows;
        subgridCols = cols;
    }
    // the implementation currently only works if our reconstructed grid map has the same width and length than the occluded
    assert(rows % subgridRows == 0);
    assert(cols % subgridCols == 0);
    int numSubgrids = (rows / subgridRows) * (cols / subgridCols);

    auto inputTensorBatch = torch::rand({numSubgrids, inputTensorUnsplit.sizes()[1], subgridRows, subgridCols});
    std::vector<float> subgridMeans = {};
    int subgrid_idx = 0;
    int start_row_idx = 0;
    int start_col_idx = 0;
    int stop_row_idx = subgridRows;
    int stop_col_idx = subgridCols;
    
    // row-major assembly of subgrids into batch
    while (stop_row_idx < rows) {
        while (stop_col_idx < cols) {
            auto rowSlice = torch::indexing::Slice(start_row_idx, stop_row_idx, 1);
            auto colSlice = torch::indexing::Slice(start_col_idx, stop_col_idx, 1);
            torch::Tensor inputTensorSubgrid = inputTensorUnsplit.index({"...", rowSlice, colSlice});
            auto subgridNoccSelector = torch::eq(inputTensorSubgrid.index({torch::indexing::Slice(), torch::indexing::Slice(1,2,1), "..."}), 0);

            // computation of occusion ratio for subgrid
            int noccSubgridCells = torch::sum(subgridNoccSelector).item<int>();
            int totalSubgridCells = subgridRows * subgridCols;
            float occRatio = 1 - noccSubgridCells / ((double) totalSubgridCells);
            
            if (occRatio <= subgridMaxOccRatioThresh_) {
                // normalization of subgrid
                auto subgridMean = torch::mean(inputTensorSubgrid.index({subgridNoccSelector})).item<float>();
                subgridMeans.push_back(subgridMean);
                inputTensorSubgrid.index_put_({subgridNoccSelector}, inputTensorSubgrid.index({subgridNoccSelector})-subgridMean);

                // insert subgrid into batch
                inputTensorBatch.index_put_({subgrid_idx, torch::indexing::Slice(), rowSlice, colSlice}, inputTensorSubgrid.index({"...", rowSlice, colSlice})); 

                subgrid_idx += 1;
            } else {
                subgridMeans.push_back((float) std::nan(""));
            }

            start_col_idx += subgridCols;
            stop_col_idx += subgridCols;
        }
        start_row_idx += subgridRows;
        stop_row_idx += subgridRows;
    }

    // send torch tensors to device
    torch::Tensor inputTensor = inputTensorBatch.to(device);

    // assemble forward function inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    // Execute the model and turn its output into a tensor.
    torch::Tensor outputs = module_.forward(inputs).toTensor();
    // std::cout << outputs.slice(1,  0, 5) << '\n';

    // row-major disassembly of subgrids from batch into complete grid map
    auto recGridMapTensor = torch::zeros({rows, cols});
    subgrid_idx = 0;
    start_row_idx = 0;
    start_col_idx = 0;
    stop_row_idx = subgridRows;
    stop_col_idx = subgridCols;
    while (stop_row_idx < rows) {
        while (stop_col_idx < cols) {
            auto rowSlice = torch::indexing::Slice(start_row_idx, stop_row_idx, 1);
            auto colSlice = torch::indexing::Slice(start_col_idx, stop_col_idx, 1);
            if (!std::isnan(subgridMeans[subgrid_idx])) {
                // we ran inference for this subgrid
                recGridMapTensor.index_put_({rowSlice, colSlice}, outputs.index({subgrid_idx, 0, rowSlice, colSlice}) + subgridMeans[subgrid_idx]);
                subgrid_idx += 1;
            } else {
                // we didn't run inference for this subgrid
                recGridMapTensor.index_put_({rowSlice, colSlice}, occGridMapTensor.index({subgrid_idx, 0, rowSlice, colSlice}));
            }

            start_col_idx += subgridCols;
            stop_col_idx += subgridCols;
        }
        start_row_idx += subgridRows;
        stop_row_idx += subgridRows;
    }

    OcclusionInpainter::tensorToGridMapLayer(recGridMapTensor, "rec_grid_map", gridMap);

    // denormalization of output
    // gridMap["rec_grid_map"] = gridMap["norm_rec_grid_map"].array() + grid_map_mean;

    return true;
}
#endif

} /* namespace */