# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:04:49 2020

@author: Cerys
"""
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
import time
import pandas as pd
import os

data = pd.read_csv('Radio Galaxy Zoo AGN with Jets.txt',sep='  ',
                   engine='python',dtype=None,header=None)
ra = data.iloc[:,1] #convert to hours
dec = data.iloc[:,2] #given in degrees 
object_list = []
for i in range(len(ra)):
    object_list.append('{} {}'.format(ra[i],dec[i]))
    
file_name = []
for i in range(len(object_list)):
    object_name = object_list[i]
    file = object_name.replace(" ","")
    file_name.append('{}{}'.format(file[:6],file[11:18]))

driver = webdriver.Chrome()
driver.maximize_window()
driver.get('https://vo.astron.nl/tgssadr/q_fits/cutout/form')

directory = r'C:\Users\Cerys\Downloads'
home = r'C:\Users\Cerys\Documents\Physics\Y4 Project\Semester 2\new TGSS images'

for i in range(len(object_list)):
    try:
        object_name = object_list[i]
        position = driver.find_element_by_name('hPOS')
        driver.execute_script("arguments[0].value=arguments[1]",position,'{}'.format(object_name))
        # Object size in degrees
        object_size = 0.1
        size = driver.find_element_by_name('hSIZE')
        driver.execute_script("arguments[0].value=arguments[1]",size,'{}'.format(object_size))
        intersect = driver.find_element_by_name('hINTERSECT')
        intersect.click()
        submit = driver.find_element_by_name('submit')
        submit.click()
        element = driver.find_element_by_class_name('productlink')
        element.click()
        time.sleep(2)
        file = [f for f in os.listdir(directory) if f.endswith('.FITS')]
        os.rename('{}\\{}'.format(directory,file[0]), '{}\\J{}.fits'.format(home,file_name[i]))
        time.sleep(1)
    except NoSuchElementException:
        #Exception incase the input RA and dec are out of range of the database
        #Resets the webpage and continues loop through remaining values
        print ('Could not find element {}'.format(i))
        driver.get('https://vo.astron.nl/tgssadr/q_fits/cutout/form')
        continue