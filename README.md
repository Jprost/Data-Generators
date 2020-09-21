# Data-Generatorsand video deep learning model
This repo is a will present a reduced version of the work done at the Harvard Medical School in the biomedical informatics department between September 2019 and March 2020. The complet version with more details will be publicly available when the project will be published.

In this project, the use of data generator will be exposed. They have two main advantage :

The loading of the data is done by batches to avoid memory overflowing. In a CPU, a data generator retrieves a batch_size number of files, processes them, and then loads its output into the GPU used for the model optimization. The data-generator class inherits from the Sequence object from Keras to guarantee a proper use of multiprocessing and integration to the pipeline. Dedicated CPUs will prepare a batch of data while another CPU will be "busy" providing it the GPU. It ensures that the data processing is not the bottleneck factor that slows down the computation.
Transformation of the data is done live increasing the possibilities of data augmentation. The class will operate at the sample scale (classic data augmentation - shifting, rotating, normalizing) and at the distribution scale (uniformization of the labels for regression task or over-sampling for imbalanced classes in binary classification)
The scheme bellow depicts the features that have been implemented :

![Alt text](./Images/DataGenerator.png?raw=true "VideoDataGenerator")

The regression on the age is used as an example.
