import sys
import os
import io
import tools # My own tools (see tools.py)
import yaml
import logging
from logging.handlers import RotatingFileHandler
import cv2 
import pytesseract
from termcolor import colored
from pdf2image import convert_from_path


def create_jpg_images(logger, config):
	logger.info('Creation of the images (they will be stored into %s)...', config['directories']['img'])
	for file_name in os.listdir(config['directories']['source']): # For every file found in the corpus dir 
		if '.pdf' in file_name:
			pages = convert_from_path(config['directories']['source'] + file_name, dpi=config['params']['dpi'])
			for idx, page in enumerate(pages):
				# In case the pdf have to be split. On Gallica, the first to pages are generated and they are not 
				# an historical document ! So we skip them.
				if config['params']['split'] == True and idx < config['params']['nb_split']: 
					continue
				name = tools.construct_name_for_jpg(dir=config['directories']['img'], file_name=file_name.replace('.pdf', ''), ext='.jpg', process='img', dpi=config['params']['dpi'], nb=str(idx))
				page.save(name, 'JPEG')
		if '.jpg' in file_name or '.png' in file_name:
			cmd = 'cp ' + config['directories']['source'] + file_name + ' ' + config['directories']['img'] + file_name
			os.system(cmd)
	logger.info('\tDone.')

def preprocess_images(logger, config):
	tools.create_n_erase_dir(logger, config['directories']['img_for_ocr'])
	logger.info('Images preprocessings...')
	try:
		for file_name in os.listdir(config['directories']['img']):
			if config['params']['ext'] not in file_name:
				continue
			img_path = config['directories']['img'] + file_name
			img = cv2.imread(img_path)
			processes = ''
			if config['preprocessings']['grayscale'] == True: 
				img = tools.get_grayscale(img)
				processes += '_grayscale'
			if config['preprocessings']['thresholding'] == True: 
				img = tools.thresholding(img)
				processes += '_thresholding'
			if config['preprocessings']['remove_noise'] == True: 
				img = tools.remove_noise(img)
				processes += '_removeNoise'
			if config['preprocessings']['dilation'] == True: 
				img = tools.dilate(img)
				processes += '_dilation'
			if config['preprocessings']['erosion'] == True: 
				img = tools.erode(img)
				processes += 'erosion'
			if config['preprocessings']['opening'] == True: 
				img = tools.opening(img)
				processes += 'opening'
			if config['preprocessings']['canny_edge_detection'] == True: 
				img = tools.canny(img)
				processes += 'canny_edge_detection'
			image_name = tools.construct_name_for_jpg(dir=config['directories']['img_for_ocr'], file_name=file_name.replace(config['params']['ext'], ''), ext=config['params']['ext'], process=processes)
			cv2.imwrite(image_name, img)
		logger.info('\tDone.')
	except:
		logger.critical('Impossible to process the preprocessing of the images!')

def process_ocr_tesseract(logger, config):
	"""
	# Afficher les bounding boxes - chaque caractère est entourné d'un rectangle
	img = cv2.imread('image.jpg')
	h, w, c = img.shape
	boxes = pytesseract.image_to_boxes(img) 
	for b in boxes.splitlines():
	    b = b.split(' ')
	    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
	cv2.imshow('img', img)
	# Récupérer les rectangles pour les mots
	d = pytesseract.image_to_data(img, output_type=Output.DICT)
	n_boxes = len(d['text'])
	for i in range(n_boxes):
	    if int(d['conf'][i]) > 60:
	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imshow('img', img)
	"""
	logger.info('Processing the ocr with tesseract...')
	try:
		for image_name in os.listdir(config['directories']['img_for_ocr']):
			img_path = config['directories']['img_for_ocr'] + image_name
			img = cv2.imread(img_path)
			#custom_config = r'--oem 3 --psm 6' # Adding custom options
			out_name = tools.construct_name_for_jpg(dir=config['directories']['ocr'], file_name=image_name.replace('.jpg', ''), ext='.txt', process='ocr_tesseract_default')
			outFile = io.open(out_name, 'w')
			#txt = pytesseract.image_to_string(img, config=custom_config)
			txt = pytesseract.image_to_string(img, lang='fra') # ------------> essayer avec ita_old
			outFile.write(txt)
			outFile.close()
		logger.info('\tDone.')
	except:
		logger.critical('Impossible to process the OCR with tesseract!')

def process_ocr_kraken(logger, config):
	#kraken -I bpt6k57083288_img_300dpi_6.jpg -o _binarized.jpg binarize
	#kraken -I bpt6k57083288_img_300dpi_6_binarized.jpg -o .txt segment ocr -m kraken_CORPUS17.mlmodel
	ext = config['params']['ext']
	logger.info('Processing the ocr with kraken...')
	if config['preprocessings']['binarization_kraken'] == False:
		try:
			# a. Binarization
			logger.warning('Kraken needs a binarized image. So the preprocessed images will get another preprocessing...')
			logger.info('Binarization of the images...')
			cmd1 = 'kraken -I \"' + config['directories']['img_for_ocr'] + '*' + ext + '\" -o _binarized' + ext + ' binarize'
			os.system(cmd1)
			logger.info('\tDone.')
		except:
			logger.critical('Something went wrong with the images binarization!')
	try:
		# b. OCR
		logger.info('Segmentation and OCR (default model)...')
		cmd2a = 'kraken -I \"' + config['directories']['img_for_ocr'] + '*_binarized' + ext + '\" -o _ocr_kraken_default.txt segment ocr -m ' + config['kraken']['model_default']
		os.system(cmd2a)
		logger.info('\tDone.')

		if config['kraken']['model_17'] != '': # That means we want to use a specific model
			logger.info('Segmentation and OCR (model ' + config['kraken']['model_17'] + ')...')
			model = config['kraken']['model_17'].split('/')[-1]
			cmd2b = 'kraken -I \"' + config['directories']['img_for_ocr'] + '*_binarized' + ext + '\" -o _ocr_kraken_' + model + '.txt segment ocr -m ' + config['kraken']['model_17']
			os.system(cmd2b)
			logger.info('\tDone.')
	except:
		logger.critical('Something went wrong with the segmentation and the OCR!')
	try:
		logger.info('Moving the OCR results into the directory ' + config['directories']['ocr'] + '...')
		# c. Moving the ocr results into the correct directory
		for name in os.listdir(config['directories']['img_for_ocr']):
			if 'ocr_kraken' in name:
				cmd3 = 'mv ' + config['directories']['img_for_ocr'] + name + ' ' + config['directories']['ocr'] + name
				os.system(cmd3)
		logger.info('\tDone.')
	except:
		logger.warning('Impossible to move the ocr resulting files into the following directory ' + config['directories']['ocr'])

def binarization_kraken(logger, config):
	try:
		logger.info('Binarization of the images...')
		ext = config['params']['ext']
		cmd1 = 'kraken -I \"' + config['directories']['img'] + '*' + ext + '\" -o _binarized' + ext + ' binarize'
		os.system(cmd1)
		tools.create_n_erase_dir(logger, config['directories']['img_for_ocr'])
		for name in os.listdir(config['directories']['img']):
			if '_binarized' in name:
				cmd2 = 'mv ' + config['directories']['img'] + name + ' ' + config['directories']['img_for_ocr'] + name
				os.system(cmd2)
		logger.info('\tDone.')
	except:
		logger.critical('Something went wrong with the images binarization!')

if __name__ == "__main__":

	# 1. LOGGER
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG) # First, we want it to write everything
	formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
	# File
	file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	# Console
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.DEBUG)
	logger.addHandler(stream_handler)

	# 2. CONFIG: Read the configs and store them into a dictionary 
	try:
		with open('config.yml', 'r') as ymlfile:
			config = yaml.safe_load(ymlfile)
			logger.info('Config file:')
			logger.info(config)
	except (IOError, OSError):
		logger.critical('Impossible to read the configutations file. This program needs a config file named \'config.yml\'.')


	# 3. MAIN PROGRAM
	
	# a. Images creation
	tools.create_n_erase_dir(logger, config['directories']['img'])
	create_jpg_images(logger, config)
	
	# b. Preprocessings
	if config['preprocessings']['binarization_kraken'] == True:
		binarization_kraken(logger, config)
	else:
		preprocess_images(logger, config)

	# c. OCR
	tools.create_n_erase_dir(logger, config['directories']['ocr']) 
	process_ocr_tesseract(logger, config)
	process_ocr_kraken(logger, config)
