#!/usr/bin/env python
# coding: utf-8

'''
@title: Train CNN Model for Classifying JSTOR Articles
@authors: Jaren Haber, PhD, Georgetown University; Yoon Sung Hong, Wayfair
@coauthor: Prof. Heather Haveman, UC Berkeley
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: February 2020
@description: Use preprocessed texts and TFIDF vectorizers to build Concurrent Neural Network (CNN) for classifying academic articles into perspectives on organizational theory (yes/no only).
'''


######################################################
# Import libraries
######################################################

import pandas as pd
import numpy as np
import re, csv
from collections import Counter
from datetime import date
from tqdm import tqdm
import os, sys, logging

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# For MLP model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, Input

sys.path.insert(0, "../preprocess/") # For loading functions from files in other directory
from quickpickle import quickpickle_dump, quickpickle_load # custom scripts for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files


######################################################
# Define filepaths
######################################################

cwd = os.getcwd()
root = str.replace(cwd, 'classification/modeling', '')
thisday = date.today().strftime("%m%d%y")

# Directory for prepared data and trained models: save files here
data_fp = root + 'classification/data/'
model_fp = root + 'classification/models/'
logs_fp = model_fp + 'logs/'

logging.basicConfig(
    format='%(asctime)s - %(message)s', 
    filename=logs_fp+'mlp_train_{}.log'.format(thisday), 
    filemode='w', 
    level=logging.INFO)

# Current article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths

# Preprocessed training data
cult_labeled_fp = data_fp + 'training_cultural_preprocessed_022421.pkl'
relt_labeled_fp = data_fp + 'training_relational_preprocessed_022421.pkl'
demog_labeled_fp = data_fp + 'training_demographic_preprocessed_022421.pkl'
orgs_labeled_fp = data_fp + 'training_orgs_preprocessed_022421.pkl'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = model_fp + 'vectorizer_cult_022421.joblib'
relt_vec_fp = model_fp + 'vectorizer_relt_022421.joblib'
demog_vec_fp = model_fp + 'vectorizer_demog_022421.joblib'
orgs_vec_fp = model_fp + 'vectorizer_orgs_022421.joblib'

logging.info("Initialized environment.")


######################################################
# Load data
######################################################

cult_df = quickpickle_load(cult_labeled_fp)
relt_df = quickpickle_load(relt_labeled_fp)
demog_df = quickpickle_load(demog_labeled_fp)
orgs_df = quickpickle_load(orgs_labeled_fp)

# Drop unsure cases: where X_score = 0.5
drop_unsure = True

if drop_unsure:
    cult_df_yes = cult_df[cult_df['cultural_score'] == 1.0]
    cult_df_no = cult_df[cult_df['cultural_score'] == 0.0]
    cult_df = pd.concat([cult_df_yes, cult_df_no])
    
    relt_df_yes = relt_df[relt_df['relational_score'] == 1.0]
    relt_df_no = relt_df[relt_df['relational_score'] == 0.0]
    relt_df = pd.concat([relt_df_yes, relt_df_no])
    
    demog_df_yes = demog_df[demog_df['demographic_score'] == 1.0]
    demog_df_no = demog_df[demog_df['demographic_score'] == 0.0]
    demog_df = pd.concat([demog_df_yes, demog_df_no])
    
    orgs_df_yes = orgs_df[orgs_df['orgs_score'] == 1.0]
    orgs_df_no = orgs_df[orgs_df['orgs_score'] == 0.0]
    orgs_df = pd.concat([orgs_df_yes, orgs_df_no])

    
def collect_article_tokens(article, return_string=False):
    '''
    Collects words from already-tokenized sentences representing each article.
    
    Args:
        article: list of lists of words (each list is a sentence)
        return_string: whether to return single, long string representing article
    Returns:
        tokens: string if return_string, else list of tokens
    '''
    
    tokens = [] # initialize
    
    if return_string:
        for sent in article:
            sent = ' '.join(sent) # make sentence into a string
            tokens.append(sent) # add sentence to list of sentences
        tokens = ' '.join(tokens) # join sentences into string
        return tokens # return string
    
    else:
        for sent in article:
            tokens += [word for word in sent] # add each word to list of tokens
        return tokens # return list of tokens


# Collect articles: Add each article as single str to list of str:
cult_docs = [] # empty list
cult_df['text'].apply(
    lambda article: cult_docs.append(
        collect_article_tokens(
            article, 
            return_string=True)))

relt_docs = [] # empty list
relt_df['text'].apply(
    lambda article: relt_docs.append(
       collect_article_tokens(
            article, 
            return_string=True)))

demog_docs = [] # empty list
demog_df['text'].apply(
    lambda article: demog_docs.append(
        collect_article_tokens(
            article, 
            return_string=True)))

orgs_docs = [] # empty list
orgs_df['text'].apply(
    lambda article: orgs_docs.append(
        collect_article_tokens(
            article, 
            return_string=True)))

logging.info("Loaded data sets.")


######################################################
# Vectorize texts
######################################################

# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])

# Uses TFIDF weighted DTM because results in better classifier accuracy than unweighted
cult_vectorizer = joblib.load(cult_vec_fp, "r+")
X_cult = cult_vectorizer.transform(cult_docs)
logging.info('Number of features in cultural vectorizer:', len(cult_vectorizer.get_feature_names()))
logging.info('Every 1000th word:\n', cult_vectorizer.get_feature_names()[::1000]) # get every 1000th word

relt_vectorizer = joblib.load(relt_vec_fp, "r+")
X_relt = relt_vectorizer.transform(relt_docs)
logging.info('Number of features in relational vectorizer:', len(relt_vectorizer.get_feature_names()))
logging.info('Every 1000th word:\n', relt_vectorizer.get_feature_names()[::1000]) # get every 1000th word

demog_vectorizer = joblib.load(demog_vec_fp, "r+")
X_demog = demog_vectorizer.transform(demog_docs)
logging.info('Number of features in demographic vectorizer:', len(demog_vectorizer.get_feature_names()))
logging.info('Every 1000th word:\n', demog_vectorizer.get_feature_names()[::1000]) # get every 1000th word

orgs_vectorizer = joblib.load(orgs_vec_fp, "r+")
X_orgs = orgs_vectorizer.transform(orgs_docs)
logging.info('Number of features in organizational soc vectorizer:', len(orgs_vectorizer.get_feature_names()))
logging.info('Every 1000th word:\n', orgs_vectorizer.get_feature_names()[::1000]) # get every 1000th word


# check out column order for data once vectorizer has been applied (should be exactly the same as list from previous cell)
test = pd.DataFrame(X_cult.toarray(), columns=cult_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training cultural classifier (after applying cultural vectorizer):', len(list(test)))
logging.info('Every 1000th word:\n', list(test)[::1000])

test = pd.DataFrame(X_relt.toarray(), columns=relt_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training relational classifier (after applying relational vectorizer):', len(list(test)))
logging.info('Every 1000th word:\n', list(test)[::1000])

test = pd.DataFrame(X_demog.toarray(), columns=demog_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training demographic classifier (after applying demographic vectorizer):', len(list(test)))
logging.info('Every 1000th word:\n', list(test)[::1000])

test = pd.DataFrame(X_orgs.toarray(), columns=orgs_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training organizational soc classifier (after applying org-soc vectorizer):', len(list(test)))
logging.info('Every 1000th word:\n', list(test)[::1000])

logging.info("Vectorized predictors.")


######################################################
# Prepare data
######################################################

# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 10% of the entire
# set of labeled data (X_validate, Y_validate)

def resample_data(X_train, Y_train, undersample = False, sampling_ratio = 1.0):
    """
    Balance x_train, y_train for better classifier training.
    
    Args:
        X_train: predictors for classifier
        Y_train: outcomes for classifier
        undersample: boolean for over or undersampling
        sampling_ratio: ratio of minority to majority class
        
        archived/not used:
        sampling_strategy: strategy for resampled distribution
            if oversample: 'majority' makes minority = to majority
            if undersample: 'minority' makes majority = to minority
            
    Returns:
        X_balanced: predictors at balanced ratio
        Y_balanced: outcomes at balanced ratio
    """
    
    if undersample == True:
        undersample = RandomUnderSampler(sampling_strategy=sampling_ratio)
        X_balanced, Y_balanced = undersample.fit_resample(X_train, Y_train)
    else:
        oversample = RandomOverSampler(sampling_strategy=sampling_ratio)
        X_balanced, Y_balanced = oversample.fit_resample(X_train, Y_train)
    
    logging.info(f'Y_train: {Counter(Y_train)}\nY_resample: {Counter(Y_balanced)}')
    
    return X_balanced, Y_balanced

#test_size = 0.2
seed = 43 # for randomizing
sampling_ratio = 1.0 # ratio of minority to majority cases
undersample = False # whether to undersample or oversample

## Cultural
cult_df = cult_df[['text', 'cultural_score']]
logging.info("# cult cases:", str(X_cult.shape[0]))
valueArray = cult_df.values
Y_cult = valueArray[:,1]
Y_cult = Y_cult.astype('float')
logging.info("# cult codes (should match):", str(len(Y_cult)))

## Relational
relt_df = relt_df[['text', 'relational_score']]
logging.info("# relt cases:", str(X_relt.shape[0]))
valueArray = relt_df.values
Y_relt = valueArray[:,1]
Y_relt = Y_relt.astype('float')
logging.info("# relt codes (should match):", str(len(Y_relt)))

## Demographic
demog_df = demog_df[['text', 'demographic_score']]
logging.info("# demog cases:", str(X_demog.shape[0]))
valueArray = demog_df.values
Y_demog = valueArray[:,1]
Y_demog = Y_demog.astype('float')
logging.info("# cult codes (should match):", str(len(Y_demog)))

logging.info("Prepared outcomes.")


######################################################
# Build MLP
######################################################

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

def train_mlp(X, 
              Y, 
              name):
    '''
    Uses keras with droput layers to train MLP model for input data.
    Saves stats to log file and resulting model to disk.
    
    Args:
        X (binary arr): predictors 
        Y (binary arr): outcomes
        name (str): shortened name of perspective we are classifying, e.g. 'relt'
    '''
    
    # Take from global the model folder path, date variable, and resampling settings
    global model_fp, thisday, undersample, sampling_ratio 
    
    # Oversample to desirable ratio
    logging.info('{} perspective: balancing data set for modeling...'.format(name))
    X, Y = resample_data(
        X, 
        Y, 
        undersample=undersample, 
        sampling_ratio=sampling_ratio)

    X.sort_indices()
    # Y.sort_indices()

    n_sample = X.shape[0]
    len_input = X.shape[1]

    cvscores = []

    logging.info('{} perspective: Training Multi-Layer Perceptron (MLP) model in Keras and evaluating with K-Fold Cross-Validation...'.format(name))
    
    for train, test in kfold.split(X, Y):
        #create model
        model = Sequential()

        #add model layers
        # inp = Input(shape=(len_input, 1))
        model.add(Dense(32, input_dim=(len_input), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # fit the keras model on the dataset
        model.fit(X[train], Y[train], epochs=200, batch_size=10)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        logging.info("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    logging.info("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) # Log kfold CV scores
    logging.info(model.summary()) # Log model summary

    model.save(model_fp + "{}_mlp_{}".format(name, thisday)) # Save model
               
    logging.info('{} perspective: MLP model saved.'.format(name))
    
    return


# Prepare data
mlp_data = [(X_cult, Y_cult, "cult"), 
            (X_relt, Y_relt, "relt"), 
            (X_demog, Y_demog, "demog")]
    
# Execute: Train MLP models
for X, Y, name in mlp_data:
    train_mlp(X, Y, name)

sys.close()