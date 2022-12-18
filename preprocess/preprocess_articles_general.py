#!/usr/bin/env python
# coding: utf-8

'''
@title: Preprocess JSTOR Articles
@author: Jaren Haber, PhD, Dartmouth College
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: jhaber@berkeley.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: December 7, 2020
@description: Preprocesses JSTOR article data for general applications (e.g., word embeddings). Preprocesses filtered article data for sample selection and model training. Does not create vectorizers as embeddings don't use these. Does not retain stopwords or deal with labeled data. Saves the preprocessed data to disk. 
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
from tqdm import tqdm # Shows progress over iterations, including in pandas via `df.progress_apply`
import sys # For terminal tricks
import csv
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
from datetime import date # For working with dates & times
from nltk import sent_tokenize
import enchant # to filter to plain english words
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.phrases import Phrases # for gathering multi-word expressions
import tables
import random
import os; from os import listdir; from os.path import isfile, join

# Custom scripts for working with texts in Python
from clean_text_utils import stopwords_jstor, remove_http, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache, get_maxlen, fix_ngrams, preprocess_text # for preprocessing text
from file_utils import quickpickle_dump, quickpickle_load, write_textlist, read_text # for quick saving & loading to pickle format, reading & writing text lists to .txt files

# Set up for multiprocessing with pandas
#from pandarallel import pandarallel # use via `df.parallel_apply`
#from multiprocessing import cpu_count; cores = cpu_count() # count cores
#pandarallel.initialize(nb_workers = cores-4, verbose = 0) # work quietly


###############################################
#              Define file paths              #
###############################################

cwd = os.getcwd()
root = str.replace(cwd, 'classification/preprocess', '')
thisday = date.today().strftime("%m%d%y")

# paths to metadata, dictionaries, similar objects
data_fp = root + 'classification/data/'
prepped_fp = root + 'models_storage/preprocessed_texts/' # directory for prepared data: save files here
dict_fp = root + 'dictionary_methods/dictionaries/'
meta_fp = root + 'dictionary_methods/code/metadata_combined.h5' # current metadata (dates are wrong)

# paths to article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths


###############################################
#              Load & merge data              #
###############################################

# Load dictionary with words related to organizations
orgs_dict = pd.read_csv(dict_fp + 'core/orgs.csv', delimiter = '\n', 
                        header=None)[0].tolist()

print("Loading & merging datasets...")

# Read full list of articles for new sample selection
#tqdm.pandas(desc='Correcting file paths')
#print('Correcting file paths...')
articles = (pd.read_csv(article_paths_fp, low_memory=False, header=None, names=['file_name']))
articles['file_name'] = articles['file_name'].apply(lambda fp: re.sub('/home/jovyan/work/', root, fp)) # correct filepaths
articles['edited_filename'] = articles['file_name'].apply(lambda fname: fname.split('-')[-1][:-4])

# Read text data from files
tqdm.pandas(desc='Loading ALL text files')
#print('Loading text files...')
articles['text'] = articles['file_name'].progress_apply(lambda fp: read_text(fp, shell = True))

# # combine the data for the correct article dates
dates1 = pd.read_csv(prepped_fp + 'parts-1-3-metadata.csv')
dates1['id'] = dates1.id.apply(lambda url: remove_http(url))
dates2 = pd.read_csv(prepped_fp + 'part-4-metadata.csv')
dates2['id'] = dates2.id.apply(lambda url: remove_http(url))
dates_df = pd.concat([dates1, dates2]) 

# load and prepare full metadata
df_meta = pd.read_hdf(meta_fp).reset_index(drop=False) # extract file name from index
df_meta['edited_filename'] = df_meta['file_name'].apply(lambda fname: fname.split('-')[-1]) # get DOI from file name, e.g. 'journal-article-10.2307_2065002' -> '10.2307_2065002'
df_meta = df_meta[['edited_filename', 'article_name', 'jstor_url', 'abstract', 'journal_title', 'primary_subject', 'year', 'type']] # keep only relevant columns
df_meta['id'] = df_meta.jstor_url.apply(lambda url: remove_http(url, https = True))

# join corrected dates with rest of metadata
# use col 'publicationYear', NOT col 'year'
df_meta = df_meta.merge(dates_df, on = 'id') 
articles = articles.merge(df_meta, on = 'edited_filename', how = 'left', validate='1:1')
articles['doi'] = articles['edited_filename']
articles = articles[['text', 'jstor_url', 'publicationYear', 'article_name', 'doi', 'primary_subject', 'journal_title', 'abstract', 'creator', 'file_name', 'wordCount', 'pageCount']]
articles = articles[~articles['publicationYear'].isna()] # must have year data to keep


###############################################
#  Filter to English words and orgs articles  #
###############################################

enchant_text = ''; orgs_text = '' # defaults to using neither filter

# To filter to English words with enchant, 
# make sure next line is set to 'True' and uncomment line after that one
filter_english = True
enchant_text = '_enchant'

# To filter to articles with at least one word related to organizations, 
# make sure next line is set to 'True' and uncomment line after that one
filter_orgs = True
orgs_text = '_orgdict'

if filter_orgs:
    #print('Filtering for orgs...')
    # Separate by discipline
    soc_articles = articles[articles.primary_subject == 'Sociology']
    mgt_articles = articles[articles.primary_subject == 'Management & Organizational Behavior']
    soc_before_len = len(soc_articles)
    mgt_before_len = len(mgt_articles)
    
    tqdm.pandas(desc='Filtering soc for orgs')
    soc_articles = soc_articles[soc_articles['text'].progress_apply(lambda text: any(term in text for term in orgs_dict))] # (' '.join([token for list in sentlist for token in list]))
    tqdm.pandas(desc='Filtering mgt for orgs')
    mgt_articles = mgt_articles[mgt_articles['text'].progress_apply(lambda text: any(term in text for term in orgs_dict))]
    
    soc_num_removed = int(soc_before_len - len(soc_articles))
    mgt_num_removed = int(mgt_before_len - len(mgt_articles))
    print(f"Removed {soc_num_removed+mgt_num_removed} articles total for missing any word related to orgs.")
    print(f"Breakdown: {soc_num_removed} removed from soc out of {soc_before_len}; {mgt_num_removed} removed from mgt/OB out of {mgt_before_len}.")
    
    # Merge disciplines back together
    articles = pd.concat([soc_articles, mgt_articles])


################################################
#     Clean stopwords, etc. from text files    #
################################################

# Use progress bars with parallelized pandas
#pandarallel.initialize(progress_bar=True, 
#                       nb_workers = cores-4, 
#                       verbose = 1)

tqdm.pandas(desc='Cleaning text files')
#print(' Cleaning text files...')
articles['text'] = articles['text'].progress_apply(
    lambda text: preprocess_text(text, 
                                 filter_english = filter_english,
                                 shorten = False))
                                 #longest = 75000, 
                                 #shortest = 1000, 
                                 #maxlen = 1000, 
                                 #minlen = 500))
          
                
#########################################################################
# Detect & fix multi-word expressions (MWEs) from original dictionaries #
#########################################################################

# Load original dictionaries
cult_orig = pd.read_csv(dict_fp + 'original/cultural_original.csv', delimiter = '\n', 
                        header=None)[0].apply(lambda x: x.replace(',', ' '))
dem_orig = pd.read_csv(dict_fp + 'original/demographic_original.csv', delimiter = '\n', 
                       header=None)[0].apply(lambda x: x.replace(',', ' '))
relt_orig = pd.read_csv(dict_fp + 'original/relational_original.csv', delimiter = '\n', 
                        header=None)[0].apply(lambda x: x.replace(',', ' '))

# Filter dicts to MWEs/bigrams & trigrams
orig_dicts = (pd.concat((cult_orig, dem_orig, relt_orig))).tolist() # full list of dictionaries
orig_ngrams = set([term for term in orig_dicts if len(term.split()) > 1]) # filter to MWEs

# Detect & fix MWEs
tqdm.pandas(desc='Fixing dict MWEs')
#print('Fixing dict MWEs...')
articles['text'] = articles['text'].progress_apply(
    lambda text: fix_ngrams(text, ngrams_list = orig_ngrams))


#########################################################
#         Detect and parse common MWEs from text        #
#########################################################

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

# Add each sentence from each article to empty list, making long list of all sentences:
sent_list = []; articles['text'].apply(lambda article: sent_list.extend([sent for sent in article]))

# Construct phraser
phrase_finder = Phrases(sent_list, min_count=5, delimiter='_', threshold=10) 
phraser_fp = prepped_fp + f'phraser_{str(len(sent_list))}_sents_{str(thisday)}.pkl' # Set phraser filepath
phrase_finder.save(phraser_fp) # save dynamic model (can still be updated)
phrase_finder = phrase_finder.freeze() # Freeze model after saving; more efficient, no more updating

tqdm.pandas(desc='Parsing common phrases in texts')
#print(' Parsing common phrases in texts...')
articles['text'] = articles['text'].progress_apply(
    lambda text: get_phrased(text, phrase_finder))


###############################################
#            Split data by decade             #
###############################################

first_decade = articles[(articles['publicationYear'] >= 1971) & (articles['publicationYear'] <= 1981) ] # 1970s
second_decade = articles[(articles['publicationYear'] >= 1982) & (articles['publicationYear'] <= 1992) ] # 1980s
third_decade = articles[(articles['publicationYear'] >= 1993) & (articles['publicationYear'] <= 2003) ] # 1990s
fourth_decade = articles[(articles['publicationYear'] >= 2004) & (articles['publicationYear'] <= 2014) ] # 2000s


###############################################
#        Save preprocessed text files         #
###############################################

# Use params to define filepaths
all_prepped_fp = prepped_fp + f'filtered{enchant_text}{orgs_text}_preprocessed_texts_ALL_{str(len(articles))}_{str(thisday)}.pkl' # all texts
decade1_fp = prepped_fp + f'filtered{enchant_text}{orgs_text}_preprocessed_texts_1971-1981_{str(len(first_decade))}_{str(thisday)}.pkl' # 1970s
decade2_fp = prepped_fp + f'filtered{enchant_text}{orgs_text}_preprocessed_texts_1982-1992_{str(len(second_decade))}_{str(thisday)}.pkl' # 1980s
decade3_fp = prepped_fp + f'filtered{enchant_text}{orgs_text}_preprocessed_texts_1993-2003_{str(len(third_decade))}_{str(thisday)}.pkl' # 1990s
decade4_fp = prepped_fp + f'filtered{enchant_text}{orgs_text}_preprocessed_texts_2004-2014_{str(len(fourth_decade))}_{str(thisday)}.pkl' # 2000s

# Save full, preprocessed text data
quickpickle_dump(articles, all_prepped_fp) # all texts
quickpickle_dump(first_decade, decade1_fp) # 1970s
quickpickle_dump(second_decade, decade2_fp) # 1980s
quickpickle_dump(third_decade, decade3_fp) # 1990s
quickpickle_dump(fourth_decade, decade4_fp) # 2000s

print("Saved preprocessed text to file.")

sys.exit() # Close script to be safe