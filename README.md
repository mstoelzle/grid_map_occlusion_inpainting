# Grid Map Occlusion Inpainting


## Installation
Create catkin workspace directory at `~/catkin_ws` and clone repository into `~/catkin_ws/src`.

### Install Dependencies
```bash
rosdep install -y --from-paths src --rosdistro noetic --skip-keys grid_map_occlusion_inpainting_core
```

### Install PyTorch for C++ #####

PyTorch is used inpaint the occlusions in grid maps using a pretrained neural network.

If we want to use a GPU for PyTorch neural network inference, we first need to install [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux).

Enable NVIDIA repo:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin 

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
```
Install cuDNN library repo:
```bash
sudo apt-get install libcudnn8=8.2.2.26-1+cuda11.4
sudo apt-get install libcudnn8-dev=8.2.2.26-1+cuda11.4
```

Then we can install PyTorch:

- Download libtorch (you may also download a version with CUDA if your hardware supports it): https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu102.zip (Stable (1.9) --> Linux --> libtorch --> C++/Java --> select GPU support --> cxx11 ABI)
- Unzip the download
- Add to bash profile: `export Torch_DIR=/absolute/path/to/libtorch/share/cmake/Torch`

### Build Package
```bash
catkin_make
```

## Configuration
ROS parameters are used for configuration. A sample .YAML configuration files can be found in the `grid_map_occlusion_inpainting_ros/config` folder. In particular, the inpainting method and the input topic and input grid map layer can be chosen

## Usage
Run node:
```bash
rosrun grid_map_occlusion_inpainting_ros node
```
with rviz visualization of grid map:
```bash
roslaunch grid_map_occlusion_inpainting_ros occlusion_inpainting.launch config_file:=solving_occlusion.yaml
```

## Published Topics
The ROS node publishes several topics:
1. The GridMapMsg published at `/grid_map_occlusion_inpainting/all_grid_map` contains all layers of the original input GridMapMsg and additionally the layers of the occluded grid map `occ_grid_map`, the occlusion mask at `occ_mask`, the reconstructed grid map at the layer `rec_grid_map` and finally the composed grid map at layer `comp_grid_map`.
2. The GridMapMsg published at `/grid_map_occlusion_inpainting/occ_grid_map` contains the occluded grid map at layer `occ_grid_map`.
3. The GridMapMsg published at `/grid_map_occlusion_inpainting/rec_grid_map` contains the reconstructed grid map at layer `rec_grid_map`.
4. The GridMapMsg published at `/grid_map_occlusion_inpainting/comp_grid_map` contains the composed grid map at layer `comp_grid_map`.

## Helpful commands
Source workspace
```
source ~/catkin_ws/devel/setup.bash
```
Replay rosbag:
```bash
rosbag play mission_1.bag -r 0.2
```
Record published gonzen mine rosbag:
```bash
rosbag record /grid_map_occlusion_inpainting/occ_grid_map /grid_map_occlusion_inpainting/rec_grid_map /grid_map_occlusion_inpainting/comp_grid_map /grid_map_occlusion_inpainting/all_grid_map /elevation_mapping/elevation_map_recordable /state_estimator/anymal_state
```