#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess Articles for JSTOR Classifier Training
@author: Jaren Haber, PhD, Georgetown University
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: December 7, 2020
@description: General script for preprocessing JSTOR article data. As of now does three things: Preprocesses article data for classifier training purposes; preprocesses ALL filtered article data for future sample selection; and creates vectorizers for training each classifier. Saves the preprocessed data (labeled and full) and vectorizers to disk. 
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
import tables
import random
import pickle # For working with .pkl files
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import os; from os import listdir; from os.path import isfile, join

# For working with models in sklearn
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

# Path to ALL JSTOR preprocessed text
all_prepped_fp = data_fp + 'filtered_preprocessed_texts_65365_120720.pkl'

# Model filepaths
cult_model_fp = model_fp + 'classifier_cult_121420.joblib'
relt_model_fp = model_fp + 'classifier_relt_121420.joblib'
demog_model_fp = model_fp + 'classifier_demog_121420.joblib'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = model_fp + 'vectorizer_cult_121120.joblib'
relt_vec_fp = model_fp + 'vectorizer_relt_121120.joblib'
demog_vec_fp = model_fp + 'vectorizer_demog_121120.joblib'


###############################################
# Load data
###############################################

# Read preprocessed text data from files
print("Reading preprocessed article text data...")
articles = quickpickle_load(all_prepped_fp)

# Use preprocessed data on ALL JSTOR articles to define file path for ML predictions on texts
predicted_fp = model_fp + f'predictions_RF_{str(len(articles))}_{str(thisday)}.pkl'


######################################################
# Load ML models for classifying org. perspectives
######################################################

print("Loading models...")
cult_model = joblib.load(cult_model_fp)
relt_model = joblib.load(relt_model_fp)
demog_model = joblib.load(demog_model_fp)


######################################################
# Load vectorizers to keep vocab consistent with training data
######################################################

cult_vec = joblib.load(cult_vec_fp)
relt_vec = joblib.load(relt_vec_fp)
demog_vec = joblib.load(demog_vec_fp)


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
    
    X = vectorizer_model.transform(text) # create TF-IDF-weighted DTM from text
    probabilities = class_model.predict_proba(X)
    
    label = 'no'
    prob_no = probabilities[0][0]
    prob_yes = probabilities[0][1]
    
    # predicted label is one with greater probability
    if probabilities[0][0] < probabilities[0][1]:
        label = 'yes'
        
    return label, prob_yes, prob_no

tqdm.pandas(desc = "Predicting: cultural persp.")
articles[['prediction_cult','prediction_cult_prob_yes','prediction_cult_prob_no']] = articles['text'].progress_apply(lambda sentlist: pd.Series(compute_predictions([' '.join(sent) for sent in sentlist], cult_vec, cult_model)))

tqdm.pandas(desc = "Predicting: relational persp.")
articles[['prediction_relt','prediction_relt_prob_yes','prediction_relt_prob_no']] = articles['text'].progress_apply(lambda sentlist: pd.Series(compute_predictions([' '.join(sent) for sent in sentlist], relt_vec, relt_model)))

tqdm.pandas(desc = "Predicting: demographic persp.")
articles[['prediction_demog','prediction_demog_prob_yes','prediction_demog_prob_no']] = articles['text'].progress_apply(lambda sentlist: pd.Series(compute_predictions([' '.join(sent) for sent in sentlist], demog_vec, demog_model)))


###############################################
# Save DF with predictions to file
###############################################

articles = articles.drop(columns = 'text') # save memory by dropping cleaned text from predictions DF--can merge using col 'file_name'

quickpickle_dump(articles, predicted_fp) # efficiently save to pickle format

print("Saved predictions to file.")

sys.exit() # Close script to be safe