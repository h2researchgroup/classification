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
@description: Preprocesses article data for classifier training purposes. Also creates vectorizers for each one. Saves the labeled, preprocessed data and vectorizers to disk. 
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
from datetime import date # For working with dates & times
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
thisday = date.today().strftime("%m%d%y")

# directory for prepared data: save files here
data_fp = root + 'classification/data/'

# Labeled data
training_cult_raw_fp = data_fp + f'training_cultural_raw_{str(thisday)}.pkl'
training_relt_raw_fp = data_fp + f'training_relational_raw_{str(thisday)}.pkl'
training_demog_raw_fp = data_fp + f'training_demographic_raw_{str(thisday)}.pkl'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = data_fp + f'vectorizer_cult_{str(thisday)}.joblib'
relt_vec_fp = data_fp + f'vectorizer_relt_{str(thisday)}.joblib'
demog_vec_fp = data_fp + f'vectorizer_demog_{str(thisday)}.joblib'

# Vocab of vectorizers (for verification purposes)
cult_vec_feat_fp = data_fp + f'vectorizer_features_cult_{str(thisday)}.csv'
relt_vec_feat_fp = data_fp + f'vectorizer_features_relt_{str(thisday)}.csv'
demog_vec_feat_fp = data_fp + f'vectorizer_features_demog_{str(thisday)}.csv'

# Output
training_cult_prepped_fp = data_fp + f'training_cultural_preprocessed_{str(thisday)}.pkl'
training_relt_prepped_fp = data_fp + f'training_relational_preprocessed_{str(thisday)}.pkl'
training_demog_prepped_fp = data_fp + f'training_demographic_preprocessed_{str(thisday)}.pkl'


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


def collect_article_tokens(article):
    '''
    Collects words from tokenized sentences representing each article.
    
    Args:
        article: list of lists of words (each list is a sentence)
    Returns:
        list: single list of tokens
    '''
    
    tokens = []
    
    for sent in article:
        tokens += [word for word in sent]
        
    return tokens

# Add each word from each article to empty list:
cult_tokens = []; coded_cult['text'].apply(lambda article: cult_tokens.extend([word for word in collect_article_tokens(article)]))
relt_tokens = []; coded_relt['text'].apply(lambda article: relt_tokens.extend([word for word in collect_article_tokens(article)]))
demog_tokens = []; coded_demog['text'].apply(lambda article: demog_tokens.extend([word for word in collect_article_tokens(article)]))


###############################################
# Vectorize texts and save vectorizers to disk
###############################################

# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])

# Use TFIDF weighted DTM because results in better classifier accuracy than unweighted
#vectorizer = CountVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # DTM
vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF

X_cult = vectorizer.fit_transform(cult_tokens)
joblib.dump(vectorizer, open(cult_vec_fp, "wb"))
with open(cult_vec_feat_fp,'w') as f:
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in cultural vectorizer:', len(vectorizer.get_feature_names()))

X_relt = vectorizer.fit_transform(relt_tokens)
joblib.dump(vectorizer, open(relt_vec_fp, "wb"))
with open(relt_vec_feat_fp,'w') as f:
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in relational vectorizer:', len(vectorizer.get_feature_names()))

X_demog = vectorizer.fit_transform(demog_tokens)
joblib.dump(vectorizer, open(demog_vec_fp, "wb"))
with open(demog_vec_feat_fp,'w') as f:
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])

print('Number of features in demographic vectorizer:', len(vectorizer.get_feature_names()))


###############################################
# Save preprocessed text files
###############################################

# Save training data for classifiers: true positives + negatives for each perspective
quickpickle_dump(coded_cult, training_cult_prepped_fp)
quickpickle_dump(coded_relt, training_relt_prepped_fp)
quickpickle_dump(coded_demog, training_demog_prepped_fp)