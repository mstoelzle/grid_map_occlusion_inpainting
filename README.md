# Grid Map Occlusion Inpainting


## Installation
Create catkin workspace directory at `~/catkin_ws` and clone repository into `~/catkin_ws/src`.

### Install Dependencies
```bash
rosdep install -y --from-paths src --rosdistro noetic
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

## Usage
Run node:
```
rosrun grid_map_occlusion_inpainting_ros node
```
