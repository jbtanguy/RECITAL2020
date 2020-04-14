import os
import io
import yaml
from pickle import dump
from pickle import load
from JBTools.simple_proba_language_model_char_based import CharBasedSimpleProbaLanguageModel
from JBTools.neural_language_model_char_based import CharBasedNeuralLanguageModel

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

config = {}
config['lm_models'] = [
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_2gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_3gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_4gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_5gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_6gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_7gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_8gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_9gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_10gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_2gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_3gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_4gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_5gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_6gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_7gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_8gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_9gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_10gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_2gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_3gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_4gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_5gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_6gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_7gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_8gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_9gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_10gram.model'
]

# 4. Get the language models' name and create the objects, stored into a list
models = []
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

for model_tuple in models:
	print(model_tuple[1].generate_seq('Il faut remarquer', 50), end='\t')
	print(model_tuple[1].generate_seq('tant qu’ils reçoiuent', 50), end='\t')
	print(model_tuple[1].generate_seq('Que la deuotion eſt', 50), end='\t')
	print(model_tuple[1].generate_seq('Anßi bien faudra', 50))