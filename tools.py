import os
import shutil
import cv2
import numpy as np

# 1. GLOBAL FUNCTIONS
def create_n_erase_dir(logger, dir_name):
	try:
		logger.info('Creation of the directory %s ...', dir_name)
		if os.path.isdir(dir_name): # If it already exists we only have to delete everything stored in it
			shutil.rmtree(dir_name)
		os.mkdir(dir_name)          # before recreate it!
		logger.info('\tDone.')
	except:
		logger.critical("Something went wrong with the creation of the directory %s", dir_name)

def construct_name_for_jpg(dir, file_name, ext, process, dpi='', nb=''):
	n = dir + file_name + '_' + process
	if dpi != '':
		n += '_' + str(dpi) + 'dpi'
	if nb != '':
		n += '_' + str(nb)
	n += ext
	return n

# 2. FUNCTIONS FOR TESSERACT
# Found on here: https://nanonets.com/blog/ocr-with-tesseract/
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

#dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
    
#erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

