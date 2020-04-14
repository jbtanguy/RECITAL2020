import sys
import io
from JBTools.neural_language_model_char_based import CharBasedNeuralLanguageModel


train_txt = io.open('SimonGabay_train.txt', mode='r', encoding='utf-8').read()

for n in range(2, 11):
	# Simple character based models
	print('Learning ' + str(n) + '-gram simple character model...')
	name = './models/SimonGabay_train_100%_simple_proba_LM_' + str(n) + 'gram.model'
	model_simple = CharBasedSimpleProbaLanguageModel(txt=train_txt, length=n, model_path=name)

	# LSTM
	print('Learning ' + str(n) + '-gram lstm model...')
	name = './models/SimonGabay_train_100%_lstm_' + str(n) + 'gram_100epochs'
	model = CharBasedNeuralLanguageModel(t=train_txt,  bilstm=False, l=n, model_p=name+'.model', mapping_p=name+'.mapping')

	# biLSTM
	print('Learning ' + str(n) + '-gram bilstm model...')
	name = './models/SimonGabay_train_100%_bilstm_' + str(n) + 'gram_100epochs'
	model = CharBasedNeuralLanguageModel(t=train_txt,  bilstm=True, l=n, model_p=name+'.model', mapping_p=name+'.mapping')
