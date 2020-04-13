# Classifying Radio Galaxies using Convolutional Neural Networks with Transfer Learning

Please see below for a description of the resources included in this repository: 

1. **data-preparation.txt** <br>

2. **VLASS webscraping.py** <br> This program performs webscraping from a web directory of VLASS quicklook images. 

1. **data-preparation.txt** <br> Program to prepare training and validation sets from data saved as raw 150x150 pixel images. The images 
are sigma clipped, split into training and validation sets and augmented via rotation and flipping. Training and validation labels to match
the datasets are also created. The datasets and labels are saved as numpy arrays. 

2. **Alhassan.py** <br> Contains a model of the Alhassan FIRST classifier. Pre-prepared data should be imported (prepared using 
data-preparation.txt) as numpy arrays. Once the model has been trained, outputs such as the training/validation loss and training/validation
accuracy are saved as numpy arrays to be processed in an interactive notebook. The weights from the model can also be saved. 


3. 
