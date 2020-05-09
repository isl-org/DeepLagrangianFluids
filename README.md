# Lagrangian Fluid Simulation with Continuous Convolutions

This repository contains code for our ICLR 2020 paper. 
We show how to train particle-based fluid simulation networks as CNNs using 
continuous convolutions. The code allows you to generate data, train your own 
model or just run a pretrained model.

<p align="center"> <img src="images/canyon.gif" alt="canyon video"> </p>

Please cite our paper if you find this code useful:
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

- Tensorflow 2.0
- Open3D with ML module (https://github.com/intel-isl/Open3D/tree/ml-module)
- SPlisHSPlasH 2.4.0 (for generating training data, https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- Tensorpack (for reading data, https://github.com/tensorpack/tensorpack)
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


### Building Open3D with ML module
At the moment Open3D needs to be build from source to make the code in this 
repo work. To build Open3D with the ML module and Tensorflow ops do the 
following
```bash
git clone --branch ml-module https://github.com/intel-isl/Open3D.git

mkdir Open3D/build
cd Open3D/build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TENSORFLOW_OPS=ON
make install-pip-package
```



## Running the pretrained model

The pretrained network weights are in ```scripts/pretrained_model_weights.h5```.
The following code runs the network on the example scene
```bash
cd scripts
./run_network.py --weights pretrained_model_weights.h5 \
                 --scene example_scene.json \
                 --output example_out \
                 --write-ply \
                 train_network.py
```
The script writes point clouds with the particle positions as .ply files, which can be visualized with Open3D.


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

### Training scripts
To train the model with the generated data simply run ```scripts/train_network.py``` from within the ```scripts``` folder.
The script will create a folder ```train_network``` with snapshots and log files.

### Evaluating the network
To evaluate the network run the ```scripts/evaluate_network.py``` script like this
```bash
./evaluate_network.py --trainscript train_network.py
```

This will create the file ```train_network.py_eval_50000.json```, which contains the 
individual errors between frame pairs.

The script will also print the overall errors. The output should look like 
this:
```{'err_n1': 0.000859004137852537, 'err_n2': 0.0024183266885233934, 'whole_seq_err': 0.030323669719872864}```

Note that the numbers differ from the numbers in the paper due to changes in 
the data generation:
 - We use Open3D to sample surface points to avoid shipping a modified 
   SPlisHSPlasH
 - The sequence of pseudorandom numbers used in the data generation is 
   different, which results in different scenes for training and testing.

## Rendering

See the [scenes](scenes/README.md) directory for instructions on how to create and render the example scenes like the canyon.

## Licenses

Code and scripts are under the MIT license.

Data files in ```datasets/models``` and ```scripts/pretrained_model_weights.h5``` are under the CDLA-Permissive-1.0 license.
