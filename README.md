# Grid Map Occlusion Inpainting

## Usage

### Installation
Create catkin workspace directory at `~/catkin_ws` and clone repository into `~/catkin_ws/src`.

#### Install Dependencies
```
rosdep install -y --from-paths src --rosdistro noetic
```

#### Install PyTorch for C++ #####

PyTorch is used to load neural network models to encoder a wind grid and predict the cost/validity of Dubins paths in wind, given a start and goal pose.

- Download libtorch (you may also download a version with CUDA if your hardware supports it): https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu102.zip (Stable (1.9) --> Linux --> libtorch --> C++/Java --> select GPU support --> cxx11 ABI)
- Unzip the download
- Add to bash profile: `export Torch_DIR=/absolute/path/to/libtorch/share/cmake/Torch`

#### Build Package
```
catkin_make
```
