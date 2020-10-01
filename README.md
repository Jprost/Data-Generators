# Data-Generator sand video deep learning model

This repo is a will present a reduced version of the work done at the Harvard Medical School in the biomedical informatics department between September 2019 and March 2020. The complet version with more details will be publicly available when the project will be published.

In this project, the use of data-generator and an example of deep learning architecture on video will be presented. Please visit the `DataGenerator_video_model.ipynb` note+bookfile.


## Data Generator

They have two main advantage :

- The loading of the data is done by batches to avoid memory overflowing. In a CPU, a data generator retrieves a batch_size number of files, processes them, and then loads its output into the GPU used for the model optimization. The data-generator class inherits from the Sequence object from Keras to guarantee a proper use of multiprocessing and integration to the pipeline. Dedicated CPUs will prepare a batch of data while another CPU will be "busy" providing it the GPU. It ensures that the data processing is not the bottleneck factor that slows down the computation.

- Transformation of the data is done live increasing the possibilities of data augmentation. The class will operate at the sample scale (classic data augmentation - shifting, rotating, normalizing) and at the distribution scale (uniformization of the labels for regression task or over-sampling for imbalanced classes in binary classification)
The scheme bellow depicts the features that have been implemented :

![DataGenerator](./Images/DataGenerator.png?raw=true "VideoDataGenerator")

## Deep Learning Architecture

The video data format is complexe and bulky. Many strategies were employed to decode their information. The time and spatial (image) component of a video enable to tackle the modelling in two steps :
- First, extract the features contained in each frame with a CNN. A CNN per frame could be optimised but it would be computationally expensive. Instead, in this example, a architecture shares its weights across all frames. The images are encoded into a more compact format of extracted features.

- Second, the resulting array goes through a RNN :  the temporal information is decoded.

Another strategy would be to use 3D CNN were all the video array is provided to a convolutional layer. The temporal and spatial information is not dissociated (not presented in this repo).

![Model](./Images/model.png?raw=true )
