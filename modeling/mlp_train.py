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

# General functions
import pandas as pd
import numpy as np
import re, csv
from collections import Counter
from datetime import date
from tqdm import tqdm
import os, sys, logging

# For sklearn vectorizers and data balancing
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# For MLP modeling
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from keras import backend
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, Input

# Custom pickle and text data functions
sys.path.insert(0, "../preprocess/") # Pass in other directory to load functions
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

# Preprocessed training data: phrased version (unphrased version was 022421)
cult_labeled_fp = data_fp + 'training_cultural_preprocessed_022621.pkl'
relt_labeled_fp = data_fp + 'training_relational_preprocessed_022621.pkl'
demog_labeled_fp = data_fp + 'training_demographic_preprocessed_022621.pkl'
orgs_labeled_fp = data_fp + 'training_orgs_preprocessed_022621.pkl'

# Vectorizers trained on hand-coded data (use to limit vocab of input texts): phrased version (unphrased version was 022421)
cult_vec_fp = model_fp + 'vectorizer_cult_022621.joblib'
relt_vec_fp = model_fp + 'vectorizer_relt_022621.joblib'
demog_vec_fp = model_fp + 'vectorizer_demog_022621.joblib'
orgs_vec_fp = model_fp + 'vectorizer_orgs_022621.joblib'

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
#logging.info('Number of features in cultural vectorizer: {}'.format(len(cult_vectorizer.get_feature_names())))
#logging.info('Every 1000th word:\n{}'.format(cult_vectorizer.get_feature_names()[::1000])) # get every 1000th word

relt_vectorizer = joblib.load(relt_vec_fp, "r+")
X_relt = relt_vectorizer.transform(relt_docs)
#logging.info('Number of features in relational vectorizer: {}'.format(len(relt_vectorizer.get_feature_names())))
#logging.info('Every 1000th word:\n{}'.format(relt_vectorizer.get_feature_names()[::1000])) # get every 1000th word

demog_vectorizer = joblib.load(demog_vec_fp, "r+")
X_demog = demog_vectorizer.transform(demog_docs)
#logging.info('Number of features in demographic vectorizer: {}'.format(len(demog_vectorizer.get_feature_names())))
#logging.info('Every 1000th word:\n{}'.format(demog_vectorizer.get_feature_names()[::1000])) # get every 1000th word

orgs_vectorizer = joblib.load(orgs_vec_fp, "r+")
X_orgs = orgs_vectorizer.transform(orgs_docs)
#logging.info('Number of features in organizational soc vectorizer: {}'.format(len(orgs_vectorizer.get_feature_names())))
#logging.info('Every 1000th word:\n{}'.format(orgs_vectorizer.get_feature_names()[::1000])) # get every 1000th word

'''
# check out column order for data once vectorizer has been applied (should be exactly the same as list from previous cell)
test = pd.DataFrame(X_cult.toarray(), columns=cult_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training cultural classifier (after applying cultural vectorizer): {}'.format(len(list(test))))
logging.info('Every 1000th word:\n{}'.format(list(test)[::1000]))

test = pd.DataFrame(X_relt.toarray(), columns=relt_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training relational classifier (after applying relational vectorizer): {}'.format(len(list(test))))
logging.info('Every 1000th word:\n{}'.format(list(test)[::1000]))

test = pd.DataFrame(X_demog.toarray(), columns=demog_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training demographic classifier (after applying demographic vectorizer): {}'.format(len(list(test))))
logging.info('Every 1000th word:\n{}'.format(list(test)[::1000]))

test = pd.DataFrame(X_orgs.toarray(), columns=orgs_vectorizer.get_feature_names())
logging.info('Number of features in preprocessed text for training organizational soc classifier (after applying org-soc vectorizer): {}'.format(len(list(test))))
logging.info('Every 1000th word:\n{}'.format(list(test)[::1000]))
'''

logging.info("Vectorized predictors.")


######################################################
# Prepare data
######################################################

seed = 43 # for randomizing
sampling_ratio = 1.0 # ratio of minority to majority cases
undersample = False # whether to undersample or oversample

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
    
    logging.info(f'Y_train: {Counter(Y_train)}, Y_resample: {Counter(Y_balanced)}')
    
    return X_balanced, Y_balanced


## Cultural
cult_df = cult_df[['text', 'cultural_score']]
logging.info("# cult cases: {}".format(str(X_cult.shape[0])))
Y_cult = (cult_df.values)[:,1].astype('float')
logging.info("# cult codes (should match): {}".format(str(len(Y_cult))))
#logging.info('{} perspective: balancing data set for modeling...'.format(name))
X_cult, Y_cult = resample_data(X_cult, Y_cult, 
                               undersample=undersample, 
                               sampling_ratio=sampling_ratio)

## Relational
relt_df = relt_df[['text', 'relational_score']]
logging.info("# relt cases: {}".format(str(X_relt.shape[0])))
Y_relt = (relt_df.values)[:,1].astype('float')
logging.info("# relt codes (should match): {}".format(str(len(Y_relt))))
X_relt, Y_relt = resample_data(X_relt, Y_relt, 
                               undersample=undersample, 
                               sampling_ratio=sampling_ratio)

## Demographic
demog_df = demog_df[['text', 'demographic_score']]
logging.info("# demog cases: {}".format(str(X_demog.shape[0])))
Y_demog = (demog_df.values)[:,1].astype('float')
logging.info("# cult codes (should match): {}".format(str(len(Y_demog))))
X_demog, Y_demog = resample_data(X_demog, Y_demog, 
                               undersample=undersample, 
                               sampling_ratio=sampling_ratio)

## Organizational Sociology
orgs_df = orgs_df[['text', 'orgs_score']]
logging.info("# org-soc cases: {}".format(str(X_orgs.shape[0])))
Y_orgs = (orgs_df.values)[:,1].astype('float')
logging.info("# soc codes (should match): {}".format(str(len(Y_orgs))))
X_orgs, Y_orgs = resample_data(X_orgs, Y_orgs, 
                               undersample=undersample, 
                               sampling_ratio=sampling_ratio)

# Assemble predictors and outcomes into array
input_array = [(X_cult, Y_cult, "cult"), 
               (X_relt, Y_relt, "relt"), 
               (X_demog, Y_demog, "demog"), 
               (X_orgs, Y_orgs, "orgs")]

logging.info("Prepared data for modeling.")


######################################################
# Train MLP models
######################################################

num_folds = 10 # number of random splits in k-fold cross-validation: uses (num_folds-1) for training, 1 for scoring
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed) # initialize kfold

def train_model_keras(X, 
                      Y, 
                      name, 
                      algorithm='mlp', 
                      evaluate=True):
    '''
    Uses keras with droput layers to train MLP model for input data. 
    Uses k-fold CV with accuracy metric to evaluate model performance.
    Saves stats to log file and resulting model to disk.
    
    Args:
        X (binary arr): predictors 
        Y (binary arr): outcomes
        name (str): shortened name of perspective we are classifying, e.g. 'relt'
        algorithm (str): whether CNN ('cnn') or MLP ('mlp'; default)
    '''
    
    # Take from global the model folder path, date variable, and random seed
    global model_fp, thisday, seed

    if algorithm.lower() not in ['mlp', 'cnn']:
        logging.error(f'{algorithm} is not an acceptable model type.')
        return
        
    X.sort_indices()
    # Y.sort_indices()
    
    if algorithm=='cnn':
        X = np.array(X.todense())
        Y = np.array(Y)
        X = np.expand_dims(X, axis=2)

    n_sample = X.shape[0]
    len_input = X.shape[1]

    cvscores = []

    logging.info('{} perspective: Training {} model in Keras...'.format(name, algorithm))
    
    if evaluate:
        for train, test in kfold.split(X, Y):    
            model = Sequential() # initialize model

            if algorithm=='mlp':

                #add model layers
                model.add(Dense(32, input_dim=(len_input), activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(16, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1, activation='sigmoid'))

                # compile model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            if algorithm=='cnn':

                #add model layers
                inp = Input(shape=(len_input, 1))
                conv32 = Conv1D(filters=64, kernel_size =10, activation='relu')(inp)
                drop33 = Dropout(0.6)(conv32)
                conv42 = Conv1D(filters=16, kernel_size =10, activation='relu')(drop33)
                drop33 = Dropout(0.6)(conv32)
                pool2 = Flatten()(conv42) # this is an option to pass from 3d to 2d
                out = Dense(1, activation='softmax')(pool2) # the output dim must be equal to the num of class if u use softmax - binary
                model = Model(inp, out)

                # compile model
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

            # fit the keras model on the dataset
            model.fit(X[train], Y[train], epochs=50, batch_size=32)
            scores = model.evaluate(X[test], Y[test], verbose=0)
            logging.info("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)

            logging.info(model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))) # Log model summary

            backend.clear_session() # clear model to avoid clutter

        logging.info(f'{name} perspective: results of model evaluation via K-Fold CV (using keras)')
        logging.info("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
    model = Sequential() # initialize model
    
    if algorithm=='mlp':
            
        #add model layers
        model.add(Dense(32, input_dim=(len_input), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    if algorithm=='cnn':
            
        #add model layers
        inp = Input(shape=(len_input, 1))
        conv32 = Conv1D(filters=64, kernel_size =10, activation='relu')(inp)
        drop33 = Dropout(0.6)(conv32)
        conv42 = Conv1D(filters=16, kernel_size =10, activation='relu')(drop33)
        drop33 = Dropout(0.6)(conv32)
        pool2 = Flatten()(conv42) # this is an option to pass from 3d to 2d
        out = Dense(1, activation='softmax')(pool2) # the output dim must be equal to the num of class if u use softmax - binary
        model = Model(inp, out)
            
        # compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
    # fit the keras model on the dataset
    model.fit(X, Y, epochs=50, batch_size=32)
    
    # Log short model summary
    sum_strlist = []; model.summary(line_length=80, print_fn=lambda x: sum_strlist.append(x)) # Add each line to list of str
    logging.info("\n".join(sum_strlist)) # Join the list as one str, then log
    
    #model.summary(line_length=80, print_fn=logger.info) # Log model summary
    
    model.save(model_fp + "{}_{}_keras_{}".format(name, algorithm, thisday)) # Save model
               
    logging.info('{} perspective: {} model saved.'.format(name, algorithm))
    
    backend.clear_session() # clear models to avoid clutter
    
    return


def log_kfold_output(model,  
                     X, 
                     Y):
    '''
    Estimates the accuracy of model using k-fold CV and logs the accuracy results: averages and std.
    Uses cross_val_predict, which unlike cross_val_score cannot define scoring option/evaluation metric.
    
    Args:
        model (obj): classifier model
        X (binary arr): predictors
        Y (binary arr): outcomes
        
    Source: 
        https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
    '''
       
    # Get kfold results
    cv_results = cross_val_predict(
        model.fit(X, Y), 
        X, 
        Y, 
        cv=kfold, 
        n_jobs=-1) # use all cores = faster
        
    # Log CV results
    logging.info(f'Mean (std):\t {round(cv_results.mean(),4)} ({round(cv_results.std(),4)})')
    logging.info(f'Accuracy:\t {round(accuracy_score(Y, cv_results)), 4}')
    logging.info(f'Confusion matrix:\n{confusion_matrix(Y, cv_results)}')
    logging.info(f'Report:\n{classification_report(Y, cv_results)}')
        
    return


def train_mlp_sklearn(X, 
                      Y, 
                      name, 
                      evaluate=True):
    '''
    Uses sklearn to train MLP model for input data.
    Saves stats to log file and resulting model to disk.
    
    Args:
        X (binary arr): predictors 
        Y (binary arr): outcomes
        name (str): shortened name of perspective we are classifying, e.g. 'relt'
    '''
    
    # Take from global the model folder path, date variable, and random seed
    global model_fp, thisday, seed

    #X.sort_indices()
    # Y.sort_indices()

    #n_sample = X.shape[0]
    #len_input = X.shape[1]

    #cvscores = []

    logging.info('{} perspective: Training Multi-Layer Perceptron (MLP) model in sklearn...'.format(name))
    '''
    mlp = MLPClassifier(max_iter=100, activation='relu') # initialize model
    
    # Set params for GridSearch optimization
    parameter_space = {
        'hidden_layer_sizes': [(50,50), (50,50,2), (50,), (100,100), (100,100,2), (100,)],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'learning_rate': ['constant','adaptive'],
    } 
    
    mlpgrid = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    mlpgrid.fit(X, Y)
    
    logging.info('{} perspective: Best parameters found:\n{}'.format(name, mlpgrid.best_params_))

    # Check out results
    means, stds = mlpgrid.cv_results_['mean_test_score'], mlpgrid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, mlpgrid.cv_results_['params']):
        logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    # Create MLP model with optimized parameters
    mlp = MLPClassifier(**mlpgrid.best_params_).fit(X, Y)'''
    
    mlp = MLPClassifier(random_state=seed, max_iter=200, activation='relu', 
                        alpha=0.0001, hidden_layer_sizes=(50, 50), 
                        learning_rate='adaptive', solver='adam').fit(X, Y)

    logging.info(f'MLP scoring:{mlp.score(X, Y)}')
    
    if evaluate:
        logging.info(f'{name} perspective: results of model evaluation via K-Fold CV (using sklearn)')
        log_kfold_output(model=mlp, X=X, Y=Y)
    
    # Save model
    joblib.dump(mlp, model_fp + "{}_mlp_sklearn_{}.joblib".format(name, thisday))
    
    return
    
    
# Execute: Train MLP models
eval_setting = True # whether to evaluate models using kfold CV (takes time)

for X, Y, name in input_array: # sklearn MLP (optimized) 
    train_mlp_sklearn(X, Y, name, evaluate=eval_setting)

for X, Y, name in input_array: # keras MLP
    train_model_keras(X, Y, name, 'mlp', evaluate=eval_setting)
    
for X, Y, name in input_array: # keras CNN    
    train_model_keras(X, Y, name, 'cnn', evaluate=eval_setting)

    
sys.exit()