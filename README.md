# Classifying Radio Galaxies using Convolutional Neural Networks with Transfer Learning

Please see below for a description of the resources included in this repository: 

1. **Source Densities.ipynb** <br> This notebook contains graphs of a poisson distribution of the number of sources that you would expect to find in a 2.5 x 2.5 arcminute tile in FIRST,VLASS and SKA1-MID, based on the density of sources in each survey. It shows that while more than one source in this tile is very rare in FIRST, it is very likely that there will be several sources per tile found in the SKA1-MID, and consequently it will be difficult to isolate sources before they are passed through a neural network.

2. **simpleneuralnetwork.ipynb** <br> Simple neural network model designed to distinguish between extended and boring sources.

3. **VLASS webscraping.py** <br> This program performs webscraping from a web directory of VLASS quicklook images. It also includes a 
cutout generator, that takes a set of source coordinates, downloads the 1x1 degree image that contains them, and then generates a cutout of the source.

4. **Finding more extended sources.py** <br> Code used to find more extended sources and boring sources in FIRST in order to increase the size of the training set for the simple neural network. Also contains webscraping for FIRST.

5. **Finding good FRIs and FRIIs VLASS.py** <br> Displays VLASS images of FRIs and FRIIS from the FRICAT,FRIICAT and CoNFiG catalogs, 
so the bad images could be removed. 

6. **data-preparation.txt** <br> Program to prepare training and validation sets from data saved as raw 150x150 pixel images. The images 
are sigma clipped, split into training and validation sets and augmented via rotation and flipping. Training and validation labels to match
the datasets are also created. The datasets and labels are saved as numpy arrays. 

7. **Alhassan.py** <br> Contains a model of the Alhassan FIRST classifier. Pre-prepared data should be imported (prepared using 
data-preparation.txt) as numpy arrays. Once the model has been trained, outputs such as the training/validation loss and training/validation
accuracy are saved as numpy arrays to be processed in an interactive notebook. The weights from the model can also be saved. 

8. **Aniyan.py** <br> I did build and run the Aniyan model during the project, with a view to comparing the results of transfer learning 
between the Aniyan and Alhassan models. It achieved an accuracy of 95% for FIRST but only a 50% accuracy for VLASS when trained independently, so it was clear that it wasn't generalising to VLASS well at all, so I focussed on Alhassan instead. (plus I didn't have any space in the report for the Aniyan model anyway). 

9. **Final Data Prep.ipynb** <br> Notebook used to plot final graphs for the report and calculate recall, precision and F1 score. 
It isn't very easy to plot graphs on BlueBear because I had to access it through hydra and I didn't know how to enable X11 forwarding through both remote directories, so instead the output files were transferred onto my laptop where I could mess around with the graphs much more easily.

10. **Transfer.py** <br> Neural network used for transfer learning with the Alhassan network from VLASS to FIRST and vice versa. It is essentially the same as Alhassan.py with a few changes to import the old weights and specify how much of the network to retrain. 

11. **TGSS web scraping.py** <br> Webscraping to download sources from the TGSS website.

12. **Resize.py** <br> Program used to resize VLASS/FIRST/TGSS images where necessary. The function in the program could easily be included as part of the data preparation, but because it is difficult to debug things when running them on bluebear, it was easier to work in stages and keep each part separate. 
