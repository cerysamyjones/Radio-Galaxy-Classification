# Classifying Radio Galaxies using Convolutional Neural Networks with Transfer Learning

Please see below for a description of the resources included in this repository: 

1. **simpleneuralnetwork.ipynb** <br> Simple neural network model designed to distinguish between extended and boring sources.

2. **VLASS webscraping.py** <br> This program performs webscraping from a web directory of VLASS quicklook images. 

1. **data-preparation.txt** <br> Program to prepare training and validation sets from data saved as raw 150x150 pixel images. The images 
are sigma clipped, split into training and validation sets and augmented via rotation and flipping. Training and validation labels to match
the datasets are also created. The datasets and labels are saved as numpy arrays. 

2. **Alhassan.py** <br> Contains a model of the Alhassan FIRST classifier. Pre-prepared data should be imported (prepared using 
data-preparation.txt) as numpy arrays. Once the model has been trained, outputs such as the training/validation loss and training/validation
accuracy are saved as numpy arrays to be processed in an interactive notebook. The weights from the model can also be saved. 

3. **Aniyan.py** I did build and run the Aniyan model during the project, with a view to comparing the results of transfer learning 
between the Aniyan and Alhassan models. It achieved an accuracy of 95% for FIRST but only a 50% accuracy for VLASS when trained independently, so it was clear that it wasn't generalising to VLASS well at all, so I focussed on Alhassan instead. (plus I didn't have any space in the report for the Aniyan model anyway). 

3. **Final Data Prep.ipynb** <br> Notebook used to plot final graphs for the report and calculate recall, precision and F1 score. 
It isn't very easy to plot graphs on BlueBear because I had to access it through hydra and I didn't know how to enable X11 forwarding through both remote directories, so instead the output files were transferred onto my laptop where I could mess around with the graphs much more easily.
