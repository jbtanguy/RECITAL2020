import os
import io
import yaml
import logging
from math import *
from pyexcel_ods import save_data
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from pickle import dump
from pickle import load
from JBTools.simple_proba_language_model_char_based import CharBasedSimpleProbaLanguageModel
from JBTools.neural_language_model_char_based import CharBasedNeuralLanguageModel

def get_logger():
	# Initialisation
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
	return logger

def get_txt_from_file(logger, file_name):
	txt = ''
	try:
		file = io.open(file_name, mode='r', encoding='utf-8')
		txt += file.read()
		file.close()
		return txt
	except:
		logger.critical('Something went wrong with the reading of %s', file_name)

def get_file_names_from_dir(logger, dir_name):
	try:
		file_names = [n for n in os.listdir(dir_name) if '.txt' in n]
		return file_names
	except:
		logger.critical('Something went wrong with this directory when trying to get the files in it %s', dir_name)

def get_txts_from_dir(logger, dir_name):
	file_names = get_file_names_from_dir(logger, dir_name)
	res = {f:get_txt_from_file(logger, dir_name+f) for f in file_names}
	return res

def get_length_from_LM_name(model_name):
	for i in range(1, 11):
		if str(i)+'gram' in model_name:
			return i
			break

def get_model_type_from_LM_name(model_name):
	for t in ['_simple_', '_lstm_', '_bilstm_']: # Je mets les _ pour distinguer lstm de bilstm
		if t in model_name:
			return t.replace('_', '')
			break

def get_sum_proba(model_tuple, txt):
	tot = 0
	nb = 0
	length = model_tuple[2]
	model = model_tuple[1]
	for i in range(0, len(txt)-length-1):
		seq = txt[i:i+length]
		char_next = txt[length+1]
		proba = model.get_proba(seq, char_next)
		tot += proba
		nb += 1
	return (nb, tot)

def get_product_proba(model_tuple, txt):
	pro = 0
	nb = 0
	length = model_tuple[2]
	model = model_tuple[1]
	for i in range(0, len(txt)-length-1):
		seq = txt[i:i+length]
		char_next = txt[length+1]
		proba = model.get_proba(seq, char_next)
		if pro == 0: # Initialisation
			if proba == 0: 
				pro = 1/121
			else:
				pro = proba
		else:
			if proba == 0:
				pro *= 1/121
			else:
				pro *= proba
		nb += 1
	if nb == 0: nb = 1
	if pro == 0: pro = 1/121
	return (nb, pro)

def get_estimate_char_accuracy(nb_sum, sum_proba):
	if nb_sum != 0:
		return 100*sum_proba/nb_sum
	else:
		return 0

def get_entropy(nb, pro):
	return -(1/nb)*log2(pro)

def get_perplexity(nb_pro, pro_proba):
	return 2**get_entropy(nb_pro, pro_proba)

def get_log_perplexity(nb_pro, pro_proba):
	return log(get_perplexity(nb_pro, pro_proba))

def mean(liste):
	nb = 0
	tot = 0
	for val in liste:
		tot += val
		nb += 1
	if nb != 0:
		return tot / nb
	else: return 0

def get_metrics(model_tuple, ocr_result):
	tokens = ocr_result.split('\n') # Ligne
	nb_sums, sums, nb_pros, pros, CAs, pers, logpers = [],[],[],[],[],[],[]
	for token in tokens:
		if len(token) < model_tuple[2]:
			continue
		nb_sum, sum_proba = get_sum_proba(model_tuple, token)
		nb_pro, pro_proba = get_product_proba(model_tuple, token) # product of all the proba
		estimate_CA = get_estimate_char_accuracy(nb_sum, sum_proba)
		perplexity = get_perplexity(nb_pro, pro_proba)
		log_perplexity = get_log_perplexity(nb_pro, pro_proba)
		nb_sums.append(nb_sum)
		sums.append(sum_proba)
		nb_pros.append(nb_pro)
		pros.append(pro_proba)
		CAs.append(estimate_CA)
		pers.append(perplexity)
		logpers.append(log_perplexity)
	return [round(mean(nb_sums), 0), mean(sums), round(mean(nb_pros), 0), mean(pros), mean(CAs), mean(pers), mean(logpers)]

""" NIVEAU DE LA PAGE
def get_metrics(model_tuple, ocr_result):
	nb_sum, sum_proba = get_sum_proba(model_tuple, ocr_result)
	nb_pro, pro_proba = get_product_proba(model_tuple, ocr_result) # product of all the proba
	estimate_CA = get_estimate_char_accuracy(nb_sum, sum_proba)
	perplexity = get_perplexity(nb_pro, pro_proba)
	log_perplexity = get_log_perplexity(nb_pro, pro_proba)
	return [float(nb_sum), float(sum_proba), float(nb_pro), float(pro_proba), float(estimate_CA), float(perplexity), float(log_perplexity)]
"""
if __name__ == "__main__":
	# 1. Logger
	logger = get_logger()
	# 2. Config file
	try:
		with open('config.yml', 'r') as ymlfile:
			config = yaml.safe_load(ymlfile)
			logger.info('Config file:')
			logger.info(config)
	except (IOError, OSError):
		logger.critical('Impossible to read the configutations file. This program needs a config file named \'config.yml\'.')
	# 3. Get the OCR results, the ground truth and their names
	try:
		logger.info('Getting files and their content beforehand.')
		gt_txt = get_txts_from_dir(logger, config['directories']['gt'])
		ocr_txt = get_txts_from_dir(logger, config['directories']['ocr'])
		logger.info('\tDone.')
	except:
		logger.critical('Impossible to get the files and their content.')
	root_names = [n.replace('.txt', '') for n in list(gt_txt.keys())]
	names = {config['directories']['gt'] + n + '.txt':
		[config['directories']['ocr'] + n + ocr_engine + '.txt' for ocr_engine in config['ocr_engines']] for n in root_names}
	# 4. Get the language models' name and create the objects, stored into a list
	models = []
	logger.info('Creation of the language models...')
	try:
		for model_name in config['lm_models']:
			model_type = get_model_type_from_LM_name(model_name) # Different type, different object
			length = get_length_from_LM_name(model_name)
			if model_type == 'simple':
				model = CharBasedSimpleProbaLanguageModel(length=length, model_path=model_name)
				models.append(('simple', model, length))
			elif model_type == 'lstm':
				mapping_name = model_name.replace('.model', '.mapping')
				model = CharBasedNeuralLanguageModel(l=length, model_p=model_name, mapping_p=mapping_name)
				models.append(('lstm', model, length))
			else: # bilstm
				mapping_name = model_name.replace('.model', '.mapping')
				model = CharBasedNeuralLanguageModel(l=length, bilstm=True, model_p=model_name, mapping_p=mapping_name)
				models.append(('bilstm', model, length))
		logger.info('\tDone')
	except:
		logger.critical('Impossible to create language models. Please verify paths in the config file.')
	# 5. Evaluation
	
	res_evaluation = {'simple': {}, 'lstm': {}, 'bilstm': {}}
	for model_type in ['simple', 'lstm', 'bilstm']:
		dic_model_type = {}
		for ocr_engine in config['ocr_engines']:
			dic_ocr_engine = {}
			for gt, ocr_results in names.items():
				dic_ocr_engine[gt] = {}
				for ocr_result in ocr_results:
					if ocr_engine in ocr_result:
						key_ocr_txt = ocr_result.split('/')[-1]
						for model_tuple in models:
							if model_tuple[0] != model_type:
								continue
							ngram = model_tuple[0]+'-'+str(model_tuple[2])
							dic_ocr_engine[gt][ngram] = get_metrics(model_tuple, ocr_txt[key_ocr_txt])
			dic_model_type[ocr_engine] = dic_ocr_engine
		res_evaluation[model_type] = dic_model_type


	dump(res_evaluation, open('res_evaluation_ligne.pickle', 'wb'))


	# 6. CSVs creation
	logger.info('CSV files creation...')
	for model_type, res_eval in res_evaluation.items():
		ngrams = [] # Un premier tour de boucle pour récupérer les noms des modèles et savoir à quel modèle correspondent les métriques
					# NECESSAIRE !!!
		for ocr_engine, res_eval_ocr_engine in res_eval.items():
			for gt, model_results in res_eval_ocr_engine.items():
				for name, res in model_results.items():
					ngrams.append(name)
				break
			break

		for ocr_engine, res_eval_ocr_engine in res_eval.items():
			file_name = './temp/' + model_type + '_language_model_res_evaluation__' + ocr_engine + '.ods'
			data = OrderedDict()
			d = []
			metrics_names = ['nb_sum', 'sum_proba', 'nb_pro', 'pro_proba', 'estimate_CA', 'perplexity', 'log_perplexity']
			first_row = ['']
			second_row = ['test page names']
			for n in ngrams:
				first_row += [n for i in range(len(metrics_names))]
				second_row += metrics_names
			d.append(first_row)
			d.append(second_row)
			for gt, model_results in res_eval_ocr_engine.items():
				row = [gt.split('/')[-1]]
				for name, res in model_results.items():
					row += res
				d.append(row)
			data.update({'Feuille 1': d})
			save_data(file_name, data)