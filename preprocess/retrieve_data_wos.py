#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

'''
@title: Train CNN Model for Classifying JSTOR Articles
@authors: Jaren Haber, PhD, Georgetown University; Yoon Sung Hong, Wayfair
@coauthor: Prof. Heather Haveman, UC Berkeley
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date: February 2020
@description: 
'''

######################################################
# Import libraries
######################################################

import pandas as pd
import numpy as np
import re, csv, os
from datetime import date
from tqdm import tqdm
import time, logging
from fuzzywuzzy.fuzz import ratio, partial_ratio # for comparing titles
from time import sleep
import random
random.seed(43)

import sys; sys.path.insert(0, "../preprocess/") # For loading functions from files in other directory
#from clean_text import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache # for preprocessing text
from quickpickle import quickpickle_dump, quickpickle_load # custom scripts for quick saving & loading to pickle format
from text_to_file import write_textlist, read_text # custom scripts for reading and writing text lists to .txt files


######################################################
# API Setup
######################################################

import woslite_client
from woslite_client.rest import ApiException
from pprint import pprint

def load_api_key(path):
    '''
    Loads text from file containing API key.
    '''
    with open(path, 'r') as f:
        for line in f:
            return line.strip()
        
# Configure API key authorization
configuration = woslite_client.Configuration()
configuration.api_key['X-ApiKey'] = load_api_key('wos_api_key.txt')

# Create an instance of the API class
integration_api_instance = woslite_client.IntegrationApi(woslite_client.ApiClient(configuration))
search_api_instance = woslite_client.SearchApi(woslite_client.ApiClient(configuration))
database_id = 'WOK'  # str | Database to search. Must be a valid database ID, one of the following: BCI/BIOABS/BIOSIS/CCC/DCI/DIIDW/MEDLINE/WOK/WOS/ZOOREC. WOK represents all databases.


######################################################
# Define filepaths
######################################################

thisday = date.today().strftime("%m%d%y")

cwd = os.getcwd()
root = str.replace(cwd, 'classification/modeling', '')

# Directory for prepared data and trained models: save files here
data_fp = root + 'classification/data/'
model_fp = root + 'classification/models/'
prepped_fp = root + 'models_storage/preprocessed_texts/'
logs_fp = root + 'classification/logs/'

logging.basicConfig(
    format='%(asctime)s - %(message)s', 
    filename=logs_fp+'retrieve_data_wos_{}.log'.format(thisday), 
    filemode='w', 
    level=logging.INFO)

# Current article lists
article_list_fp = data_fp + 'filtered_length_index.csv' # Filtered index of research articles
article_paths_fp = data_fp + 'filtered_length_article_paths.csv' # List of article file paths
article_names_fp = data_fp + 'filtered_length_article_names.xlsx' # Filtered list of article names and general data, sorted by journal then article name

# Path to predictions for all perspectives
predicted_fp = model_fp + 'predictions_MLP_65365_022621.pkl' # 'predictions_RF_65365_121620.pkl'

# Per-article metadata with year and URL info
meta_fp = root + 'dictionary_methods/code/metadata_combined.h5' 


######################################################
# Load & prepare data
######################################################

# Read in metadata file
df_meta = pd.read_hdf(meta_fp)
df_meta.reset_index(drop=False, inplace=True) # extract file name from index

# For merging purposes, get ID alone from file name, e.g. 'journal-article-10.2307_2065002' -> '10.2307_2065002'
df_meta['edited_filename'] = df_meta['file_name'].apply(lambda x: x[16:]) 
df_meta = df_meta[["edited_filename", "article_name", "jstor_url", "abstract", "journal_title", "primary_subject", "year"]] # keep only relevant columns

df_pred = quickpickle_load(predicted_fp)

# Read in filtered index
df = pd.read_csv(article_list_fp, low_memory=False, header=None, names=["file_name"])
df['edited_filename'] = df['file_name'].apply(lambda x: x[16:]) # New col with only article ID

# Read predictions using latest models
df_pred = quickpickle_load(predicted_fp)

# For consistency across data sources, rename absolute file path to 'file_path' and create shorter 'file_name'
df_pred['file_path'] = df_pred['file_name'] # rename for consistency across files
df_pred['file_name'] = df_pred['file_name'].str.replace(
    '/vol_b/data/jstor_data/ocr/', '').str.replace('.txt', '') # remove folders + file suffix

# Merge meta data, predictions into articles list DF
df = pd.merge(df, df_meta, how='left', on='edited_filename') # meta data
df = pd.merge(df, df_pred, how='right', on='file_name') # predictions

# Rename columns, add new ones for WOS output
df.rename(columns={"year":"year_jstor", "article_name":"article_name_jstor"}, inplace=True) # Differentiate columns from incoming WOS cols
df = df[df['article_name_jstor'].notnull()] # remove any articles without titles
df = df[df['journal_title'].notnull()].reset_index(drop=True) # ditto for journal names (can't search these)
df = df.reindex(columns = df.columns.tolist() + ['year_wos', 'article_name_wos', 'similarity_wos_title']) # add new columns


######################################################
# Call API
######################################################

def get_year_wos(row): 

    '''
    Gets publication year for article using title, using the Scholarly API (which uses Google Scholar). 
    
    Docs: https://github.com/Clarivate-SAR/woslite_py_client
    
    Args:
        row (Series): first element is full title, e.g., 'The Collective Strategy Framework: An Application to Competing Predictions of Isomorphism'; second element is journal title
        
    Returns:
        pub_year (int): year article was published, in four digits (i.e., `19xx` or `20xx`)
    '''
    
    sleeptime = 1 #random.randint(5000,7000)/1000  # set pause for politeness/to avoid getting blocked by API
    title = row[0] # get title #title_col
    journal = row[1] # get journal #journal_col
    
    # Configure query
    title = title.replace("'", "") # remove apostrophes (confuses parser)
    usr_query = f"TI=({title}) AND SO=({journal})" # str | User query for requesting data, ex: TS=(cadmium). The query parser will return errors for invalid queries.
    count = 1  # int | Number of records returned in the request
    first_record = 1  # int | Specific record, if any within the result set to return. Cannot be less than 1 and greater than 100000.
    lang = 'en'  # str | Language of search. This element can take only one value: en for English. If no language is specified, English is passed by default. (optional)
    sort_field = 'PY+D'  # str | Order by field(s). Field name and order by clause separated by '+', use A for ASC and D for DESC, ex: PY+D. Multiple values are separated by comma. (optional)
    
    try:
        # Find record(s) by user query
        api_response = search_api_instance.root_get(database_id, usr_query, count, first_record, lang=lang,
                                                                 sort_field=sort_field)
        
        # Get fields of interest from API response, assign to row
        pub_year = api_response.data[0].source.published_biblio_year[0]
        pub_title = api_response.data[0].title.title[0]
        similarity = ratio(title.lower(), pub_title.lower()) # compare titles
        
        logging.info(f'API record found for: \t"{pub_title}"') # log results
        logging.info(f'JSTOR Title for above:\t"{title}"')
        sleep(sleeptime) # pause
        
        return pd.Series([pub_year, pub_title, similarity])
        
    except Exception as e:
        logging.info("API failed with error: \t%s" % e)
        sleep(sleeptime) # pause
        
        return pd.Series([np.NaN, np.NaN, np.NaN])


# Execute
tqdm.pandas(desc='API -> year...')

try:
    df[['year_wos', 'article_name_wos', 'similarity_wos_title']] = df[['article_name_jstor', 'journal_title']].progress_apply(get_year_wos, axis=1)
except ValueError as e: 
    logging.info(f'Encountered error: {e}')

# Set file path for output
thisday = date.today().strftime("%m%d%y") # update
wos_fp = data_fp + f'merged_wos_{thisday}.csv'

# Save merged data with year, title, match from WOS
df.to_csv(wos_fp, index=False)

sys.exit()