# Comprehensive Self-Supervised method for Reconstruction High Field from Low field MRI Scans

Self-Supervised method is unique method in learning community, one of the powerful Self-Supervised usage is the ability to use very small dataset for getting good predictive results. The paper represents comprehensive approach for transforming Low Field MRI images to High Field MRI with the help of sophisticated AI algorithm and without MRI tool hardware changes. The algorithm combined two major components: The first one is physics-based MRI simulator and the second is U-Net network used as a prior for image reconstruction. Since the algorithm is physics-based - big dataset is no needed, only few samples are enough to achieve good results for versatile MRI use cases. For our knowledge such MRI approach not investigate yet and it should be a breakthrough.  

Overview
--------------
## Data

Provided data for fat-water is located at Data folder and can be [Here](https://www.ismrm.org/workshops/FatWater12/data.htm)

## Pre-Processing 

No need for Pre-Processing since it's integrated part of Unet_MRI_7T3.py.

## Model

The provided model is basically concatenate of two models, but with a twist. First model is U-Net model with modification to provided raw data type. 
The second model is MRI physics-based simulator, the fact that is physics-based simulator allow to use Self-Supervised model.
See picture below for understanding the overall Model architecture.

![ModelHighLevelArchitecture.png](img/ModelHighLevelArchitecture.png)



Code Structure
--------------
### Main Function:

**U-Net network**

    # U-net network implement as integrated part of overall algorithm with some modification for raw-data properties
    # Refernce for U-Net can be seen at: https://github.com/zhixuhao/unet.git
    
**lowfieldsim.py**

    low_field_image, high_field_image, _ ,_ = lowfieldsim(K_space)
    # Low Field Simulatore takes High Field (3T) K-space raw data 
    # and transform it to Low Field (0.5T)

**Unet_MRI_7T3.py**

    # The main enginge for this work.
    # Takes K-Space High Field raw data and transfom it to Low Field return it again to inverse model for getting High Field data.
    

How to use:
--------------
### Dependencies
This code depends on the following libraries:
* Scikit-Image 
* PyTorch
* TensorboardX
* [fastMRI](https://github.com/Apahima/fastMRI.git) 

### Define and Model execution

* The Model is parser based execution. All needed parameters for optimization are exposed as parser.
* Parameters:
    * --num_coil - Define the number of coils used to get the K-space data
    * --num-chnas - Define the first U-Net layer span of layers.
    * --batch-size - Since using as Self-Supervised model only one sample is learned.
    * --lr - Define Learning rate
    * --exp-dir - Define where to save Model results to be able to visualize that with TensorBoard.


**Example:**

    srun -c 2 --gres=gpu:1 --pty python -m Unet_MRI_7T3  --num_coil 8 --num-chans 24 --batch-size 1 --checkpoint checkpoint/best_model.pt  --challenge multicoil --lr 0.2 --num-epochs 10000 --data-path \temp --report-interval 1000 --exp-dir checkpoints/Eval  

    