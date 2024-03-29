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
from clean_text import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files

# Define stopwords used by JSTOR
jstor_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])


###############################################
# Define file paths
###############################################

cwd = os.getcwd()
root = str.replace(cwd, 'classification/preprocess', '')
thisday = date.today().strftime("%m%d%y")

# directory for prepared data: save files here
data_fp = root + 'classification/data/'
model_fp = root + 'classification/models/'
prepped_fp = root + 'models_storage/preprocessed_texts/'

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
training_cult_prepped_fp = data_fp + f'training_cultural_preprocessed_{str(thisday)}.pkl'
training_relt_prepped_fp = data_fp + f'training_relational_preprocessed_{str(thisday)}.pkl'
training_demog_prepped_fp = data_fp + f'training_demographic_preprocessed_{str(thisday)}.pkl'
training_orgs_prepped_fp = data_fp + f'training_orgs_preprocessed_{str(thisday)}.pkl'


###############################################
# Load data
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
# Preprocess text files
###############################################

def get_maxlen(length, 
               longest, 
               shortest, 
               maxlength, 
               minlength):
    '''
    Compute how many words to return for an article. Will be at least minlength and at most maxlength. 
    Gradate between these based on how long article is relative to longest article in corpus.
    
    Longest article will have (length/discounter) = gap between min and max lengths. 
    Shortest will have (length/discounter) = 0.
    
    Formula: maxlength = minlength + (length/discounter)
    
    Args:
        length (int): number of words in preprocessed article text
        longest (int): longest article in corpus (in # words)
        shortest (int): shortest article in corpus (in # words)
        maxlength (int): maximum number of words to return per article
        minlength (int): minimum number of words to return per article
        
    Returns:
        maxlength (int): how many words to return for this article
    '''
    
    gap = maxlength - minlength # gap between minimum and maximum article lengths = the most we can add to minlength
    
    discounter = (longest-shortest)/gap # how many of these gaps do we have to cover to reach # words in longest article? That's the discounter. Discount by shortest article length to restrict range to between 0 and gap
    
    maxlength = minlength + ((length-shortest)/discounter) # apply the discounter to decide how many "gap-steps" to add for this article. 
    # Apply gap # gap-steps to reach original maxlength.
    
    return int(maxlength)
    

def preprocess_text(article, 
                    shorten = False, 
                    longest = 999999, 
                    shortest = 0, 
                    maxlen = 999999, 
                    minlen = 0):
    '''
    Cleans up articles by removing page marker junk, 
    unicode formatting, and extra whitespaces; 
    re-joining words split by (hyphenated at) end of line; 
    removing numbers (by default) and acronyms (not by default); 
    tokenizing sentences into words using the Apache Lucene Tokenizer (same as JSTOR); 
    lower-casing words; 
    removing stopwords (same as JSTOR), junk formatting words, junk sentence fragments, 
    and proper nouns (the last not by default).
    
    Args:
        article (str): lots of sentences with punctuation etc, often long
        shorten (boolean): if True, shorten sentences to at most maxlen words
        longest (int): number of words in longest article in corpus (get this elsewhere)
        shortest (int): number of words in shortest article in corpus (depends on filtering)
        maxlen (int): maximum number of words to return per article; default is huge number, set lower if shorten == True
        minlen (int): minimum number of words to return per article
        
    Returns:
        list of lists of str: each element of list is a sentence, each sentence is a list of words
    '''
            
    # Remove page marker junk
    article = article.replace('<plain_text><page sequence="1">', '')
    article = re.sub(r'</page>(\<.*?\>)', ' \n ', article)
    
    # Compute maximum length for this article: from minlen to maxlen, gradated depending on longest
    if shorten:
        article_length = len(article.split()) # tokenize (split by spaces) then count # words in article
        
        if article_length > minlen: # if article is longer than minimum length to extract, decide how much to extract
            maxlen = get_maxlen(article_length, 
                                longest, 
                                shortest, 
                                maxlen, 
                                minlen)
        elif article_length <= minlen: # if article isn't longer than minimum length to extract, just take whole thing
            shorten = False # don't shorten
    
    doc = [] # list to hold tokenized sentences making up article
    numwords = 0 # initialize word counter
    
    if shorten:
        while numwords < maxlen: # continue adding words until reaching maxlen
            for sent in article.split('\n'):
                #sent = clean_sent(sent)
                sent = [word for word in clean_sentence_apache(sent, 
                                                               unhyphenate=True, 
                                                               remove_numbers=True, 
                                                               remove_acronyms=False, 
                                                               remove_stopwords=True, 
                                                               remove_propernouns=False, 
                                                               return_string=False) if word != ''] # remove empty strings

                if numwords < maxlen and len(sent) > 0:
                    gap = int(maxlen - numwords)
                    if len(sent) > gap: # if sentence is bigger than gap between current numwords and max # words, shorten it
                        sent = sent[:gap] 
                    doc.append(sent)
                    numwords += len(sent)

                if len(sent) > 0:
                    doc.append(sent)
                    numwords += len(sent)
    
    else: # take whole sentence (don't shorten)
        for sent in article.split('\n'):
            #sent = clean_sent(sent)
            sent = [word for word in clean_sentence_apache(sent, 
                                                           unhyphenate=True, 
                                                           remove_numbers=True, 
                                                           remove_acronyms=False, 
                                                           remove_stopwords=True, 
                                                           remove_propernouns=False, 
                                                           return_string=False) if word != ''] # remove empty strings
            
            if len(sent) > 0:
                doc.append(sent)

    return doc

tqdm.pandas(desc='Cleaning labeled text files...')
coded_cult['text'] = coded_cult['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))
coded_relt['text'] = coded_relt['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))
coded_demog['text'] = coded_demog['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))
coded_orgs['text'] = coded_orgs['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))

tqdm.pandas(desc='Cleaning ALL text files...')
articles['text'] = articles['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))


###############################################
# Detect and parse common multi-word expressions (MWEs)
###############################################

# Notes on gensim.phrases.Phrases module: 
# This module detects MWEs in sentences based on collocation counts. A bigram/trigram needs to occur X number of times together (a 'collocation') relative to Y number of times individually in order to be considered a common MWE.
# Param 'threshold' affects likelihood of forming phrases: how high X needs to be relative to Y. A higher threshold means there will be fewer phrases in the result. 
# The formula: A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, where N is the total vocabulary size. 
# The default threshold is 10.0.

def get_phrased(article, phrase_model):
    '''
    Parse phrases in article using phrase-finding model.
    
    Args:
        article: list of lists of words (each list is a sentence)
    Returns:
        article: same format, with phrases inserted where appropriate
    '''
    
    article = [phrase_model[sent] for sent in article] 
        
    return article

print("Detecting phrases in list of sentences...")

# Add each sentence from each article to empty list, making long list of all sentences:
sent_list = []; articles['text'].apply(lambda article: sent_list.extend([sent for sent in article]))

phrase_finder = Phrases(sent_list, min_count=15, delimiter=b'_', common_terms=jstor_stopwords, threshold=10) 

phraser_fp = model_fp + f'phraser_{str(len(sent_list))}_sents_{str(thisday)}.pkl' # Set phraser filepath
phrase_finder.save(phraser_fp) # save dynamic model (can still be updated)
phrase_finder = phrase_finder.freeze() # Freeze model after saving; more efficient, no more updating

tqdm.pandas(desc='Parsing common phrases...')
coded_cult['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))
coded_relt['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))
coded_demog['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))
coded_orgs['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))

tqdm.pandas(desc='Parsing common phrases in ALL texts...')
articles['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))


###############################################
# Vectorize texts and save vectorizers to disk
###############################################

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

# Add each word from each article to empty list, making long list of all tokens:
cult_tokens = []; coded_cult['text'].apply(lambda article: cult_tokens.extend([word for word in collect_article_tokens(article)]))
relt_tokens = []; coded_relt['text'].apply(lambda article: relt_tokens.extend([word for word in collect_article_tokens(article)]))
demog_tokens = []; coded_demog['text'].apply(lambda article: demog_tokens.extend([word for word in collect_article_tokens(article)]))
orgs_tokens = []; coded_orgs['text'].apply(lambda article: orgs_tokens.extend([word for word in collect_article_tokens(article)]))

# Use TFIDF weighted DTM because results in better classifier accuracy than unweighted
#vectorizer = CountVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # DTM
vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF

X_cult = vectorizer.fit_transform(cult_tokens)
joblib.dump(vectorizer, open(cult_vec_fp, "wb")) # Save DTM
with open(cult_vec_feat_fp,'w') as f: # Save DTM features
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in cultural vectorizer:', len(vectorizer.get_feature_names()))

vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF
X_relt = vectorizer.fit_transform(relt_tokens)
joblib.dump(vectorizer, open(relt_vec_fp, "wb")) # Save DTM
with open(relt_vec_feat_fp,'w') as f: # Save features
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])
    
print('Number of features in relational vectorizer:', len(vectorizer.get_feature_names()))

vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF
X_demog = vectorizer.fit_transform(demog_tokens)
joblib.dump(vectorizer, open(demog_vec_fp, "wb")) # Save DTM
with open(demog_vec_feat_fp,'w') as f: # Save features
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])

print('Number of features in demographic vectorizer:', len(vectorizer.get_feature_names()))

vectorizer = TfidfVectorizer(max_features=100000, min_df=1, max_df=0.8, stop_words=jstor_stopwords) # TFIDF
X_orgs = vectorizer.fit_transform(orgs_tokens)
joblib.dump(vectorizer, open(orgs_vec_fp, "wb")) # Save DTM
with open(orgs_vec_feat_fp,'w') as f: # Save features
    writer = csv.writer(f)
    writer.writerows([vectorizer.get_feature_names()])

print('Number of features in organizational sociology vectorizer:', len(vectorizer.get_feature_names()))


###############################################
# Save preprocessed text files
###############################################

# Save training data for classifiers: true positives + negatives for each perspective
quickpickle_dump(coded_cult, training_cult_prepped_fp)
quickpickle_dump(coded_relt, training_relt_prepped_fp)
quickpickle_dump(coded_demog, training_demog_prepped_fp)
quickpickle_dump(coded_orgs, training_orgs_prepped_fp)

# Save full, preprocessed text data
quickpickle_dump(articles, all_prepped_fp)

print("Saved preprocessed text to file.")

sys.exit() # Close script to be safe