# Single Image-Guided Deformation Network for Point Cloud Reconstruction
![image](https://github.com/VIMLab-hfut/Single-Image-Guided-Deformation-Network-for-Point-Cloud-Reconstruction/blob/main/network.png)  
## Introduction
In this work, we propose a deformation-based multi-resolution point cloud reconstruction network. First, the input image features are extracted through the image encoding stage. Then the image features are merged with the initial random point cloud in the point cloud encoding stage to extract the point cloud features. Finally, the initial random point cloud shape is deformed into the final reconstructed point cloud using the point cloud features in the point cloud decoding stage. Due to the randomness introduced by the initial point cloud, the point coordinates of each reconstructed point cloud shift randomly. Overlapping multiple reconstruction results get high-resolution point clouds.
## Installation
Install [Tensorflow](https://www.tensorflow.org/install/). The code is tested under TF1.4 GPU version and Python 2.7 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like cv2 etc. It's highly recommended that you have access to GPUs.
## Usage
The data sets we use are ShapeNet and Pix3D. The script commands used for training and testing are as follows:
* Run the training script:  
``` python train.py xxxxxxx ```
* Run the evaluation script after training finished:  
``` python test.py xxxxxxx ```

