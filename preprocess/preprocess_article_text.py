#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess Articles for JSTOR Classifier Training
@author: Jaren Haber, PhD, Georgetown University
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: November 2020
@description: Preprocesses article data for classifier training purposes. 
'''


###############################################
# Initialize
###############################################

# import packages
import imp, importlib # For working with modules
import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import pickle # For working with .pkl files
import re # for regex magic
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
import datetime # For working with dates & times
from nltk import sent_tokenize
import tables
import random
import os; from os import listdir; from os.path import isfile, join

# Custom scripts for working with texts in Python
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence_apache # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # for quick saving & loading to pickle format


###############################################
# Define file paths
###############################################

cwd = os.getcwd()
root = str.replace(cwd, 'classification/preprocess', '')

# directory for prepared data: save files here
data_fp = root + 'classification/data/'

# Training data
training_cult_raw_fp = data_fp + 'training_cultural_raw_112420.pkl'
training_relt_raw_fp = data_fp + 'training_relational_raw_112420.pkl'
training_demog_raw_fp = data_fp + 'training_demographic_raw_112420.pkl'

# Output
training_cult_prepped_fp = data_fp + 'training_cultural_preprocessed_112420.pkl'
training_relt_prepped_fp = data_fp + 'training_relational_preprocessed_112420.pkl'
training_demog_prepped_fp = data_fp + 'training_demographic_preprocessed_112420.pkl'


###############################################
# Load data
###############################################

coded_cult = quickpickle_load(training_cult_raw_fp)
coded_relt = quickpickle_load(training_relt_raw_fp)
coded_demog = quickpickle_load(training_demog_raw_fp)


###############################################
# Preprocess text files
###############################################

def preprocess_text(article):
    '''Cleans up articles by ...
    
    Args:
        article: in str format, often long
        
    Returns:
        str: cleaned string'''
    
    # Remove page marker junk
    article = article.replace('<plain_text><page sequence="1">', '')
    article = re.sub(r'</page>(\<.*?\>)', ' \n ', article)
    
    doc = [] # list to hold tokenized sentences making up article
    for sent in sent_tokenize(article):
        sent = clean_sentence_apache(sent, unhyphenate=True, remove_propernouns=False, remove_acronyms=False)
        sent = [word for word in sent if word != '']
        if len(sent) > 0:
            doc.append(sent)
    
    return doc

tqdm.pandas(desc='Cleaning text files...')
coded_cult['text'] = coded_cult['text'].progress_apply(lambda text: preprocess_text(text))
coded_relt['text'] = coded_relt['text'].progress_apply(lambda text: preprocess_text(text))
coded_demog['text'] = coded_demog['text'].progress_apply(lambda text: preprocess_text(text))


###############################################
# Save preprocessed text files
###############################################

# Save training data for classifiers: true positives + negatives for each perspective
quickpickle_dump(coded_cult, training_cult_prepped_fp)
quickpickle_dump(coded_relt, training_relt_prepped_fp)
quickpickle_dump(coded_demog, training_demog_prepped_fp)