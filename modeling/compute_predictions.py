#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess Articles for JSTOR Classifier Training
@author: Jaren Haber, PhD, Georgetown University
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: February 2021
@description: 
'''

###############################################
# Initialize
###############################################

# import packages
import imp, importlib # For working with modules
import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import re # for regex magic
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import timeit # For counting time taken for a process
from datetime import date # For working with dates & times
import logging
import tables
import random
import pickle # For working with .pkl files
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import os; from os import listdir; from os.path import isfile, join
from nltk import word_tokenize

# For working with models in sklearn & keras
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model

# Custom scripts for working with texts in Python
import sys; sys.path.insert(0, "../preprocess/") # For loading functions from files in other directory
from clean_text import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files


###############################################
# Define file paths
###############################################

cwd = os.getcwd()
root = str.replace(cwd, 'classification/modeling', '')
thisday = date.today().strftime("%m%d%y")

# Root paths for prepared data and models
data_fp = root + 'classification/data/'
model_fp = root + 'classification/models/'
logs_fp = model_fp + 'logs/'
prepped_fp = root + 'models_storage/preprocessed_texts/'

logging.basicConfig(
    format='%(asctime)s - %(message)s', 
    filename=logs_fp+f'compute_predictions_{str(thisday)}.log', 
    filemode='w', 
    level=logging.INFO)

# Path to ALL JSTOR preprocessed text
all_prepped_fp = prepped_fp + 'filtered_preprocessed_texts_65365_022421.pkl'

# Model filepaths
cult_model_fp = model_fp + 'cult_mlp_keras_022521_notk' #'classifier_cult_MLP_012521.joblib'
relt_model_fp = model_fp + 'relt_mlp_keras_022521_notk' #'classifier_relt_MLP_012521.joblib'
demog_model_fp = model_fp + 'demog_mlp_keras_022521_notk' #'classifier_demog_MLP_012521.joblib'
orgs_model_fp = model_fp + 'orgs_mlp_keras_022521_notk' #'classifier_demog_MLP_012521.joblib'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = model_fp + 'vectorizer_cult_022421.joblib'
relt_vec_fp = model_fp + 'vectorizer_relt_022421.joblib'
demog_vec_fp = model_fp + 'vectorizer_demog_022421.joblib'
orgs_vec_fp = model_fp + 'vectorizer_orgs_022421.joblib'

logging.info("Initialized environment.")


###############################################
# Load data
###############################################

# Read preprocessed text data from files
logging.info("Reading preprocessed article text data...")
articles = quickpickle_load(all_prepped_fp)

# Use preprocessed data on ALL JSTOR articles to define file path for ML predictions on texts
predicted_fp = model_fp + f'predictions_MLP_{str(len(articles))}_{str(thisday)}.pkl'


######################################################
# Load ML models for classifying org. perspectives
######################################################

logging.info("Loading models...")
cult_model = load_model(cult_model_fp)
relt_model = load_model(relt_model_fp)
demog_model = load_model(demog_model_fp)
orgs_model = load_model(orgs_model_fp)


######################################################
# Load vectorizers to keep vocab consistent with training data
######################################################

logging.info("Loading vectorizers...")
cult_vec = joblib.load(cult_vec_fp) 
relt_vec = joblib.load(relt_vec_fp) 
demog_vec = joblib.load(demog_vec_fp)
orgs_vec = joblib.load(orgs_vec_fp)


###############################################
# Compute predictions using models
###############################################

def compute_predictions(text, vectorizer_model, class_model):
    '''
    Predicts the label for an input article using a given model trained to classify organizational perspectives in articles. 
    Uses vectorizer_model to restrict the vocab of the input article so it's consistent with vocab in class_model (avoids errors).
    
    Args:
        text: preprocessed article text in format of list of sentences, each a str or list of tokens
        vectorizer_model: fitted text vectorizer
        class_model: trained classification model
    Returns:
        label: label for text predicted by model, false for tie
        prob: probability for label
    '''
    
    # For prediction, text should be a list containing a single string
    # If the text has more than one text chunk, merge them
    if len(text)>0: #word_tokenize(text[0])
        text = [' '.join([chunk for chunk in text])] # join chunks
        #text = [word for word in word_tokenize(text)] # vectorizer-friendly iterable
        
    X = vectorizer_model.transform(text) # create TF-IDF-weighted DTM from text
    X.sort_indices()
    probabilities = class_model.predict(X) # keras
    #probabilities = class_model.predict_proba(X) # sklearn
    
    #logging.info(f'Length and content of input text: {str(len(text))}, {text}')
    #logging.info(f'Length and content of input text vector: {str(X.shape[0])}, {X}')
    #logging.info(f'Length and content of probabilities output: {str(len(probabilities))}, {probabilities}')
    
    label = 'no'
    prob_yes = probabilities[0][0]
    prob_no = 1-prob_yes
    
    # predicted label is one with greater probability
    if prob_yes > prob_no:
        label = 'yes'
        
    return label, prob_yes, prob_no


# Make predictions and preview results
try:
    tqdm.pandas(desc = "Predicting: cultural persp.")
    articles[['prediction_cult','prediction_cult_prob_yes','prediction_cult_prob_no']] = articles['text'].progress_apply(
        lambda sentlist: pd.Series(compute_predictions(
            [' '.join(sent) for sent in sentlist], cult_vec, cult_model)))
    logging.info('Predictions for cultural perspective:\n{}'.format(articles['prediction_cult'].value_counts()))

    tqdm.pandas(desc = "Predicting: relational persp.")
    articles[['prediction_relt','prediction_relt_prob_yes','prediction_relt_prob_no']] = articles['text'].progress_apply(
        lambda sentlist: pd.Series(compute_predictions(
            [' '.join(sent) for sent in sentlist], relt_vec, relt_model)))
    logging.info('Predictions for relational perspective:\n{}'.format(articles['prediction_relt'].value_counts()))

    tqdm.pandas(desc = "Predicting: demographic persp.")
    articles[['prediction_demog','prediction_demog_prob_yes','prediction_demog_prob_no']] = articles['text'].progress_apply(
        lambda sentlist: pd.Series(compute_predictions(
            [' '.join(sent) for sent in sentlist], demog_vec, demog_model)))
    logging.info('Predictions for demographic perspective:\n{}'.format(articles['prediction_demog'].value_counts()))
    
    tqdm.pandas(desc = "Predicting: org-soc persp.")
    articles[['prediction_orgs','prediction_orgs_prob_yes','prediction_orgs_prob_no']] = articles['text'].progress_apply(
        lambda sentlist: pd.Series(compute_predictions(
            [' '.join(sent) for sent in sentlist], orgs_vec, orgs_model)))
    logging.info('Predictions for org-soc perspective:\n{}'.format(articles['prediction_orgs'].value_counts()))

except Exception as e:
    logging.info(f'Error encountered during prediction:{e}')
    sys.exit()

###############################################
# Save DF with predictions to file
###############################################

articles = articles.drop(columns = 'text') # save memory by dropping cleaned text from predictions DF--can merge using col 'file_name'

quickpickle_dump(articles, predicted_fp) # efficiently save to pickle format

logging.info("Saved predictions to file.")

sys.exit() # Close script to be safe