#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess Articles for JSTOR Classifier Training with Deep Learning
@author: Jaren Haber, PhD, Dartmouth College
@coauthors: Prof. Heather Haveman, UC Berkeley; Thomas Lu, UC Berkeley; Nancy Xu, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: jhaber@berkeley.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: December 7, 2020
@description: Preprocesses ALL JSTOR article data (filtered) for deep learning, specifically. Retains stopwords as transformers use these; does not create vectorizers as transformers don't use these. Saves the preprocessed full data to disk. 
'''


###############################################
#                  Initialize                 #
###############################################

# import packages
import imp, importlib # For working with modules
import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import pickle # For working with .pkl files
import re # for regex magic
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import csv
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
from datetime import date # For working with dates & times
from nltk import sent_tokenize
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.phrases import Phrases # for gathering multi-word expressions
import tables
import random
import os; from os import listdir; from os.path import isfile, join

# Custom scripts for working with texts in Python
from clean_text_utils import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache, get_maxlen, preprocess_text # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files

# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])


###############################################
#              Define file paths              #
###############################################

cwd = os.getcwd()
root = str.replace(cwd, 'tlu_test/preprocess', '') + '/'
thisday = date.today().strftime("%m%d%y")

# directory for prepared data: save files here
data_fp = root + 'tlu_test/data/'
model_fp = root + 'tlu_storage/'
prepped_fp = model_fp #root + 'models_storage/preprocessed_texts/'

# Current article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths

# Labeled data
training_cult_raw_fp = data_fp + 'training_cultural_raw_022621.pkl'
training_relt_raw_fp = data_fp + 'training_relational_raw_022621.pkl'
training_demog_raw_fp = data_fp + 'training_demographic_raw_022621.pkl'
training_orgs_raw_fp = data_fp + 'training_orgs_raw_022621.pkl'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = model_fp + f'vectorizer_cult_{str(thisday)}.joblib'
relt_vec_fp = model_fp + f'vectorizer_relt_{str(thisday)}.joblib'
demog_vec_fp = model_fp + f'vectorizer_demog_{str(thisday)}.joblib'
orgs_vec_fp = model_fp + f'vectorizer_orgs_{str(thisday)}.joblib'

# Vocab of vectorizers (for verification purposes)
cult_vec_feat_fp = model_fp + f'vectorizer_features_cult_{str(thisday)}.csv'
relt_vec_feat_fp = model_fp + f'vectorizer_features_relt_{str(thisday)}.csv'
demog_vec_feat_fp = model_fp + f'vectorizer_features_demog_{str(thisday)}.csv'
orgs_vec_feat_fp = model_fp + f'vectorizer_features_orgs_{str(thisday)}.csv'

# Output
training_cult_prepped_fp = model_fp + f'training_cultural_preprocessed_{str(thisday)}.pkl'
training_relt_prepped_fp = model_fp + f'training_relational_preprocessed_{str(thisday)}.pkl'
training_demog_prepped_fp = model_fp + f'training_demographic_preprocessed_{str(thisday)}.pkl'
training_orgs_prepped_fp = model_fp + f'training_orgs_preprocessed_{str(thisday)}.pkl'


###############################################
#                  Load data                  #
###############################################

coded_cult = quickpickle_load(training_cult_raw_fp)
coded_relt = quickpickle_load(training_relt_raw_fp)
coded_demog = quickpickle_load(training_demog_raw_fp)
coded_orgs = quickpickle_load(training_orgs_raw_fp)

# Read full list of articles for new sample selection
tqdm.pandas(desc='Correcting file paths...')
articles = (pd.read_csv(article_paths_fp, low_memory=False, header=None, names=['file_name']))
articles['file_name'] = articles['file_name'].progress_apply(lambda fp: re.sub('/home/jovyan/work/', root, fp))

# Read text data from files
tqdm.pandas(desc='Loading ALL text files...')
articles['text'] = articles['file_name'].progress_apply(lambda fp: read_text(fp, shell = True))

# Use articles data to define file name for ALL JSTOR preprocessed text
all_prepped_fp = prepped_fp + f'filtered_preprocessed_texts_{str(len(articles))}_{str(thisday)}.pkl'


###############################################
#           Preprocess text files             #
###############################################

tqdm.pandas(desc='Cleaning ALL text files...')
articles['text'] = articles['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))
                
                
#########################################################################
# Detect & fix multi-word expressions (MWEs) from original dictionaries #
#########################################################################

# Load original dictionaries
cult_orig = pd.read_csv(dict_fp + 'cultural_original.csv', header=None)[0].apply(lambda x: x.replace(',', ' '))
dem_orig = pd.read_csv(dict_fp + 'demographic_original.csv', header=None)[0].apply(lambda x: x.replace(',', ' '))
relt_orig = pd.read_csv(dict_fp + 'relational_original.csv', header=None)[0].apply(lambda x: x.replace(',', ' '))

# Filter dicts to MWEs/bigrams & trigrams
orig_dicts = (pd.concat((cult_orig, dem_orig, relt_orig))).tolist() # full list of dictionaries
orig_bigrams = set([term for term in orig_dicts if len(term) > 1]) # filter to MWEs

# Detect & fix MWEs
tqdm.pandas(desc='Fixing dict MWEs...')
articles['text'] = articles['text'].progress_apply(
    lambda text: fix_ngrams(text))

                
###############################################
#        Save preprocessed text files         #
###############################################

# Save full, preprocessed text data
quickpickle_dump(articles, all_prepped_fp)

print("Saved preprocessed text to file.")

sys.exit() # Close script to be safe
