import os
import io
import yaml
from math import *
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
        return (nb, pro)

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

config = {}
config['lm_models'] = [
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_2gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_3gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_4gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_5gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_6gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_7gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_8gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_9gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_simple_proba_LM_10gram.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_2gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_3gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_4gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_5gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_6gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_7gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_8gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_9gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_lstm_10gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_2gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_3gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_4gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_5gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_6gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_7gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_8gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_9gram_100epochs.model',
'/home/jbtanguy/git/These/RECITAL2020/evaluation/Language_models/models/SimonGabay_train_100%_bilstm_10gram_100epochs.model'


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

txt_test = io.open('SimonGabay_test.txt', mode='r', encoding='utf-8').read()
lines = txt_test.split('\n')
perplexities = []

for model_tuple in models:
        for line in lines:
                if len(line) < model_tuple[2]:
                        continue
                nb, pro = get_product_proba(model_tuple, line)
                perplexities.append(get_perplexity(nb, pro))
        print(model_tuple[0]+'-'+str(model_tuple[2])+'\t'+str(mean(perplexities)))