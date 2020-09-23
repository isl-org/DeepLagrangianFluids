# Lagrangian Fluid Simulation with Continuous Convolutions

This repository contains code for our ICLR 2020 paper. 
We show how to train particle-based fluid simulation networks as CNNs using 
continuous convolutions. The code allows you to generate data, train your own 
model or just run a pretrained model.

<p align="center"> <img src="images/canyon.gif" alt="canyon video"> </p>

Please cite our paper [(pdf)](https://openreview.net/pdf?id=B1lDoJSYDH) if you find this code useful:
```
@inproceedings{Ummenhofer2020Lagrangian,
        title     = {Lagrangian Fluid Simulation with Continuous Convolutions},
        author    = {Benjamin Ummenhofer and Lukas Prantl and Nils Thuerey and Vladlen Koltun},
        booktitle = {International Conference on Learning Representations},
        year      = {2020},
}
```

To stay informed about updates we recommend to watch this repository.

## Dependencies

- Tensorflow 2.0 or PyTorch 1.6
- Open3D master with ML module (https://github.com/intel-isl/Open3D/)
- SPlisHSPlasH 2.4.0 (for generating training data and fluid particle sampling, https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- Tensorpack DataFlow (for reading data, ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```)
- python-prctl (needed by Tensorpack DataFlow; depends on libcap-dev, install with ```apt install libcap-dev``` )
- msgpack (```pip install msgpack``` )
- msgpack-numpy (```pip install msgpack-numpy```)
- python-zstandard (```pip install zstandard``` https://github.com/indygreg/python-zstandard)
- partio (https://github.com/wdas/partio)
- SciPy
- OpenVDB with python binding (optional for creating surface meshes, https://github.com/AcademySoftwareFoundation/openvdb)
- plyfile (optional for creating surface meshes, ```pip install plyfile```)

The versions match the configuration that we have tested on a system with Ubuntu 18.04.
SPlisHSPlasH 2.4.0 is required for generating training data (ensure that it is compiled in *Release* mode).
We recommend to use the latest versions for all other packages.


<!--### Open3D 0.11 and later-->
<!--The ML module is included in Open3D 0.11 and later and can simply be installed with ```pip install open3d```.-->
<!--Make sure that the version of your ML framework matches the version for which the ML ops in Open3D have been built.-->
<!--For Open3D 0.11 this is CUDA 10.1, TensorFlow 2.3 and PyTorch 1.6.-->
<!--If you cannot match this configuration it is recommended to build Open3D from source.-->


### Building Open3D with ML module from source.
At the moment Open3D needs to be build from source to make the code in this 
repo work. To build Open3D with the ML ops for Tensorflow and PyTorch do the 
following
```bash
git clone --recursive https://github.com/intel-isl/Open3D.git
# check the file Open3D/util/scripts/install-deps-ubuntu.sh
# for dependencies and install them. For more instructions see the Open3D documentation

mkdir Open3D/build
cd Open3D/build

# This builds the ml ops for both TensorFlow and PyTorch.
# If you don't need both frameworks you can disable the one you don't need with OFF.
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TENSORFLOW_OPS=ON -DBUILD_PYTORCH_OPS=ON -DBUILD_CUDA_MODULE=ON
make install-pip-package
```

## Running the pretrained model

The pretrained network weights are in ```scripts/pretrained_model_weights.h5``` for TensorFlow and in ```scripts/pretrained_model_weights.pt``` for PyTorch.
The following code runs the network on the example scene
```bash
cd scripts
# with TensorFlow
./run_network.py --weights pretrained_model_weights.h5 \
                 --scene example_scene.json \
                 --output example_out \
                 --write-ply \
                 train_network_tf.py
# or with PyTorch
./run_network.py --weights pretrained_model_weights.pt \
                 --scene example_scene.json \
                 --output example_out \
                 --write-ply \
                 train_network_torch.py
```
The script writes point clouds with the particle positions as .ply files, which can be visualized with Open3D.
Note that SPlisHSPlasH is required for sampling the initial fluid volumes from ```.obj``` files.


## Training the network

### Data generation
The data generation scripts are in the ```datasets``` subfolder.
To generate the training and validation data 
 1. Set the path to the ```DynamicBoundarySimulator``` of SPlisHSPlasH in the ```datasets/splishsplash_config.py``` script.
 2. Run the script from within the datasets folder 
    ```bash
    cd datasets
    ./create_data.sh
    ```

### Data download
If you want to skip the data generation step you can download training and validation data from this [link](https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07).
The training data has been generated with the scripts in this repository. 
The validation data corresponds to the data used in the paper.

### Training scripts
To train the model with the generated data simply run one of the ```train_network_x.py``` scripts from within the ```scripts``` folder. 
```bash
cd scripts
# TensorFlow version
./train_network_tf.py
# PyTorch version
./train_network_torch.py
```
The scripts will create a folder ```train_network_tf``` or ```train_network_torch``` respectively with snapshots and log files.
The log files can be viewed with Tensorboard.

### Evaluating the network
To evaluate the network run the ```scripts/evaluate_network.py``` script like this
```bash
./evaluate_network.py --trainscript train_network_tf.py
# or
./evaluate_network.py --trainscript train_network_torch.py
```

This will create the file ```train_network_{tf,torch}.py_eval_50000.json```, which contains the 
individual errors between frame pairs.

The script will also print the overall errors. The output should look like 
this if you use the generated the data:
```{'err_n1': 0.000859004137852537, 'err_n2': 0.0024183266885233934, 'whole_seq_err': 0.030323669719872864}```

Note that the numbers differ from the numbers in the paper due to changes in 
the data generation:
 - We use Open3D to sample surface points to avoid shipping a modified 
   SPlisHSPlasH
 - The sequence of pseudorandom numbers used in the data generation is 
   different, which results in different scenes for training and testing.

If you have downloaded the validation data then the output should be similar to the numbers in the paper.
```{'err_n1': 0.000665973493194656, 'err_n2': 0.0018649007299291042, 'whole_seq_err': 0.03081335372162257}```

## Rendering

See the [scenes](scenes/README.md) directory for instructions on how to create and render the example scenes like the canyon.

## Licenses

Code and scripts are under the MIT license.

Data files in ```datasets/models``` and ```scripts/pretrained_model_weights.{h5,pt}``` are under the CDLA-Permissive-1.0 license.
