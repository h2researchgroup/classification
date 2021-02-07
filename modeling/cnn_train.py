######################################################
# Import libraries
######################################################

import pandas as pd
import numpy as np
import re
from collections import Counter
from datetime import date
from tqdm import tqdm
import os, sys

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')

stemmer = WordNetLemmatizer()

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

import joblib
import csv

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

sys.path.insert(0, "../preprocess/") # For loading functions from files in other directory
from quickpickle import quickpickle_dump, quickpickle_load # custom scripts for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files


######################################################
# Define filepaths
######################################################

thisday = date.today().strftime("%m%d%y")

cwd = os.getcwd()
root = str.replace(cwd, 'classification/modeling', '')

# Directory for prepared data and trained models: save files here
data_fp = root + 'classification/data/'
model_fp = root + 'classification/models/'

# Current article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths

# Preprocessed training data
cult_labeled_fp = data_fp + 'training_cultural_preprocessed_121620.pkl'
relt_labeled_fp = data_fp + 'training_relational_preprocessed_121620.pkl'
demog_labeled_fp = data_fp + 'training_demographic_preprocessed_121620.pkl'

# Model filepaths
cult_model_fp = model_fp + f'classifier_cult_{str(thisday)}.joblib'
relt_model_fp = model_fp + f'classifier_relt_{str(thisday)}.joblib'
demog_model_fp = model_fp + f'classifier_demog_{str(thisday)}.joblib'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts)
cult_vec_fp = model_fp + 'vectorizer_cult_012521.joblib'
relt_vec_fp = model_fp + 'vectorizer_relt_012521.joblib'
demog_vec_fp = model_fp + 'vectorizer_demog_012521.joblib'

cult_df = quickpickle_load(cult_labeled_fp)
relt_df = quickpickle_load(relt_labeled_fp)
demog_df = quickpickle_load(demog_labeled_fp)

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

# For capturing word frequencies, add all words from each article to single, shared list (can't use this to create models)
cult_tokens = []; cult_df['text'].apply(lambda article: cult_tokens.extend([word for word in collect_article_tokens(article)]))
relt_tokens = []; relt_df['text'].apply(lambda article: relt_tokens.extend([word for word in collect_article_tokens(article)]))
demog_tokens = []; demog_df['text'].apply(lambda article: demog_tokens.extend([word for word in collect_article_tokens(article)]))
print()

# Add sentences from each article to empty list:
cult_sents = []; cult_df['text'].apply(
    lambda article: cult_sents.extend(
        [' '.join([word for word in sent]) for sent in article]))
relt_sents = []; relt_df['text'].apply(
    lambda article: relt_sents.extend(
        [' '.join([word for word in sent]) for sent in article]))
demog_sents = []; demog_df['text'].apply(
    lambda article: demog_sents.extend(
        [' '.join([word for word in sent]) for sent in article]))


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


# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])

# Uses TFIDF weighted DTM because results in better classifier accuracy than unweighted
cult_vectorizer = joblib.load(cult_vec_fp, "r+")
X_cult = cult_vectorizer.transform(cult_docs)
print('Number of features in cultural vectorizer:', len(cult_vectorizer.get_feature_names()))
#print(cult_vectorizer.get_feature_names()[::1000]) # get every 1000th word
#print()

relt_vectorizer = joblib.load(relt_vec_fp, "r+")
X_relt = relt_vectorizer.transform(relt_docs)
print('Number of features in relational vectorizer:', len(relt_vectorizer.get_feature_names()))
#print(relt_vectorizer.get_feature_names()[::1000]) # get every 1000th word
#print()

demog_vectorizer = joblib.load(demog_vec_fp, "r+")
X_demog = demog_vectorizer.transform(demog_docs)
print('Number of features in demographic vectorizer:', len(demog_vectorizer.get_feature_names()))
#print(demog_vectorizer.get_feature_names()[::1000]) # get every 1000th word


# check out column order for data once vectorizer has been applied (should be exactly the same as list from previous cell)
test = pd.DataFrame(X_cult.toarray(), columns=cult_vectorizer.get_feature_names())
print('Number of features in preprocessed text for training cultural classifier (after applying cultural vectorizer):', len(list(test)))
#print(list(test)[::1000])
#print()
test = pd.DataFrame(X_relt.toarray(), columns=relt_vectorizer.get_feature_names())
print('Number of features in preprocessed text for training relational classifier (after applying relational vectorizer):', len(list(test)))
#print(list(test)[::1000])
#print()
test = pd.DataFrame(X_demog.toarray(), columns=demog_vectorizer.get_feature_names())
print('Number of features in preprocessed text for training demographic classifier (after applying demographic vectorizer):', len(list(test)))
#print(list(test)[::1000])
print()

######################################################
# Balance x_train, y_train
######################################################

def resample_data(X_train, Y_train, undersample = False, sampling_ratio = 1.0):
    """
    Args:
        X_train: X training data
        Y_train: Y training data
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
    
    print(f'Y_train: {Counter(Y_train)}\nY_resample: {Counter(Y_balanced)}')
    
    return X_balanced, Y_balanced


######################################################
# Prepare training and validation data
######################################################

# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 10% of the entire
# set of labeled data (X_validate, Y_validate)

cult_df = cult_df[['text', 'cultural_score']]
print("Number of cases:", str(X_cult.shape[0]))

valueArray = cult_df.values
Y = valueArray[:,1]
Y = Y.astype('float')
print("Number of codes (should match):", str(len(Y)))

test_size = 0.2
seed = 3
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X_cult, 
    Y, 
    test_size=test_size, 
    random_state=seed)

print(f'Y_train Distribution: {Counter(Y_train).most_common()}')


######################################################
# Oversample to desirable ratio
######################################################

# Use these settings here and below
sampling_ratio = 1.0 # ratio of minority to majority cases
undersample = False # whether to undersample or oversample

X_balanced, Y_balanced = resample_data(
    X_cult, 
    Y, 
    undersample=undersample, 
    sampling_ratio=sampling_ratio)


######################################################
# Build CNN
######################################################

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(66098,1))) #please replace 28 with the actual dimension size, around 60k
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
#save model
model.save("cnn_model")
#######for model prediction output with our other texts ######
#insert text data here to predict label
model.predict(X_test[:4])

sys.close()