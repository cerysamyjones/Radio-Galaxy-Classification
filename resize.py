# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:30:08 2020

@author: Cerys
"""
import numpy as np
import os
from astropy.io import fits

def resize(images,x_size,y_size):
    """
    Resize a set of images to a required size.
    Images are either cropped or padded with zeros.
    Independent of the original image size.
    Resized around the centre of the image.
    """
    # This allows an input of 2D images or 3D arrays of images
    shape = np.shape(images)
    if len(shape) == 2:
        dummy = np.empty((1,shape[0],shape[1]))
        dummy[0,:,:] = images[:,:]
        images = dummy
        length = 1
    if len(shape) == 3:
        length = len(images)
    
    # Create an array to return new images
    resize = np.empty(shape=(length,y_size,x_size))
    
    # Decide whether to crop or pad based on relative image size
    for i in range(length):
        image = images[i,:,:]
        # FITS images are stored as [z,y,x]
        new_image = np.zeros(shape=(y_size,x_size))
        x_pix = image.shape[1]
        y_pix = image.shape[0]

        # Pixel numbers have to be integers so code is slightly
        # different depending on odd/even input
        x_diff = x_size - x_pix
        if x_diff % 2 != 0:
            x_odd = 0.5
        else:
            x_odd = 0
        y_diff = y_size - y_pix
        
        if y_diff % 2 != 0:
            y_odd = 0.5
        else: 
            y_odd = 0
        
        x = np.abs(x_diff/2)
        y = np.abs(y_diff/2)
        
        if x_diff > 0: # Requires zero padding
            x_min = int(x - x_odd)
            x_max = int(x + x_pix - x_odd)
            if y_diff == 0:
                new_image[:,x_min:x_max] = image[:,:]
            if y_diff > 0: # Requires zero padding
                y_min = int(y - y_odd)
                y_max = int(y + y_pix - y_odd)
                new_image[y_min:y_max,x_min:x_max] = image[:,:]
            if y_diff < 0: # Requires cropping
                y_min = int(y - y_odd)
                y_max = int(y_pix - y - y_odd)
                new_image[:,x_min:x_max] = image[y_min:y_max,:]    
                
        else: # Requires cropping
            x_min = int(x - x_odd)
            x_max = int(x_pix - x - x_odd)
            if y_diff == 0:
                new_image[:,:] = image[:,x_min:x_max]
            if y_diff > 0: # Requires zero padding
                y_min = int(y - y_odd)
                y_max = int(y + y_pix - y_odd)
                new_image[y_min:y_max,:] = image[:,x_min:x_max]
            if y_diff < 0: # Requires cropping
                y_min = int(y - y_odd)
                y_max = int(y_pix - y - y_odd)
                new_image[:,:] = image[y_min:y_max,x_min:x_max]
                
        resize[i,:,:] = new_image[:,:]
    return resize
        
directory_path = r'C:\Users\Cerys\Documents\Physics\Y4 Project\Semester 2\new TGSS images'
directory = os.fsencode(directory_path)

files = [f for f in os.listdir(directory_path) if f.endswith('.fits')]
x_size = 83
y_size = 83
dataset = np.empty(shape=(len(files),y_size,x_size))

for i in range(len(files)):
    filename = os.fsdecode(files[i]) # could use if filename.endswith(".fits")
    filepath = os.path.join(directory_path, filename)# or #filepath = directory_path + '\\' + filename
    data = fits.getdata(filepath, ext=0)
    crop = resize(data[0,0,:,:],y_size,x_size)
    dataset[i,:,:] = crop[0,:,:]