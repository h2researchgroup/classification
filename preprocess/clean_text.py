#!/usr/bin/env python
# coding: utf-8

'''
@title: General Functions to Preprocess Academic Articles
@author: Jaren Haber, PhD, Georgetown University
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date_created: Fall 2018
@date_modified: December 14, 2020
@description: Essential functions for text preprocessing. The core function cleans sentences by removing stopwords, proper nouns, etc. Other functions gather proper nouns and create lists of stopwords, punctuation, and unicode.
'''

# Import packages
import re, datetime
import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of stemming words
import os # for working with file trees
import numpy as np
#import spacy 

# Prep dictionaries of English words
from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
#english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
#fname =  "/vol_b/data/data_management/tools/english_words.txt" # Set file path to long english dictionary
#with open(fname, "r") as f:
#    for word in f:
#        english_long.add(word.strip())

#to look for proper nouns when cleaning text
from nltk.tag import pos_tag

#apache tokenizer imports
import lucene
from java.io import StringReader
from org.apache.lucene.analysis.ja import JapaneseAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute


def stopwords_make(extend_stopwords = False):
    """Create stopwords list. 
    
    If extend_stopwords is True, create larger stopword list by joining sklearn list to NLTK list."""
                                                     
    stop_word_list = list(set(stopwords.words("english"))) # list of english stopwords

    # Add dates to stopwords
    for i in range(1,13):
        stop_word_list.append(datetime.date(2008, i, 1).strftime('%B'))
    for i in range(1,13):
        stop_word_list.append((datetime.date(2008, i, 1).strftime('%B')).lower())
    for i in range(1, 2100):
        stop_word_list.append(str(i))
        
    # Add even more stop words:
    if extend_stopwords == True:
        stop_word_list = text.ENGLISH_STOP_WORDS.union(stop_word_list)
                  
    return stop_word_list
                                                     
    
def punctstr_make():
    """Creates punctuations list"""
                    
    punctuations = list(string.punctuation) # assign list of common punctuation symbols
    #addpuncts = ['*','•','©','–','`','’','“','”','»','.','×','|','_','§','…','⎫'] # a few more punctuations also common in web text
    #punctuations += addpuncts # Expand punctuations list
    #punctuations = list(set(punctuations)) # Remove duplicates
    punctuations.remove('-') # Don't remove hyphens - dashes at beginning and end of words are handled separately)
    punctuations.remove("'") # Don't remove possessive apostrophes - those at beginning and end of words are handled separately
    punctstr = "".join([char for char in punctuations]) # Turn into string for regex later

    return punctstr
                                                     
                                                     
def unicode_make():
    """Create list of unicode chars"""
                    
    unicode_list  = []
    for i in range(1000,3000):
        unicode_list.append(chr(i))
    unicode_list.append("_cid:10") # Common in webtext junk
                                                     
    return unicode_list


def gather_propernouns(sentence):
    """ Creates a list of the propernouns in the sentence.
    Args:
        docs: Spacy object of sentence  
    Returns:
        List of proper nouns in the sentence."""
                  
    tagged_sentence = pos_tag(sentence.split())
    # [('James', 'NNP'), ('likes', 'VBZ'), ('apples', 'NNPS')]
    
    propernouns = [word for word,pos in tagged_sentence if pos == 'NNP' or pos == 'NNPS']
    # ['James', 'apples']
 
    return propernouns
#     new_text = []
#     for word in text:
#         if word.pos_ == "PROPN":
#             new_text.append(str(word))
#     print(new_text) # while debugging
#     return new_text


# Initialize Lucene for Apache tokenizer
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

def apache_tokenize(sentence, 
                    lowercase = True):
    '''
    Tokenizes sentences into words using the Apache Lucene Standard Tokenizer (same as JSTOR).
    
    Args:
        sentence: str
        lowercase: binary indicator: whether to lowercase each word
    Returns:
        list of str: each element of list is a word
        
    Requires these packages: 
    lucene
    org.apache.lucene.analysis.standard.StandardAnalyzer
    org.apache.lucene.analysis.standard.StandardTokenizer
    java.io.StringReader
    org.apache.lucene.analysis.tokenattributes.CharTermAttribute
    '''
    
    sent_list = [] # initialize empty list to add words to
    
    tokenizer = StandardTokenizer() # start Tokenizer
    tokenizer.setReader(StringReader(sentence))
    charTermAttrib = tokenizer.getAttribute(CharTermAttribute.class_)
    tokenizer.reset()
    
    if lowercase:
        while tokenizer.incrementToken():
            sent_list.append(charTermAttrib.toString().lower()) #lowercasing
            
        return sent_list
    
    # if not lower-casing:
    while tokenizer.incrementToken():
        sent_list.append(charTermAttrib.toString())
        
    return sent_list
                         


def clean_sentence_apache(sentence, 
                          unhyphenate = True, 
                          lowercase = True, 
                          remove_numbers = False, 
                          remove_acronyms = False, 
                          remove_stopwords = True, 
                          remove_propernouns = False, 
                          return_string = False):
    
    '''
    Cleans up articles by removing unicode formatting and extra whitespaces; 
    re-joining words split by (hyphenated at) end of line; 
    removing numbers (by default) and acronyms (not by default); 
    tokenizing sentences into words using the Apache Lucene Tokenizer (same as JSTOR); 
    lower-casing words; 
    removing stopwords (same as JSTOR), junk formatting words, junk sentence fragments, 
    and proper nouns (the last not by default).
    
    Args:
        sentence (str): sentence that possibly includes spaces and punctuation
        unhyphenate (binary): whether to join any lingering hyphens at end of line (i.e., words ending with '- ')
        lowercase (binary): whether to lower-case each word
        remove_numbers (binary): whether to remove any chars that are digits
        remove_acronyms (binary): whether t
        remove_stopwords (binary): whether to remove stopwords
        remove_propernouns (binary): boolean, removes nouns such as names, etc.
        return_string (binary): return string instead of list of tokens (useful for infersent)  

    Returns:
        list of str: each element of list is a word
    '''
     # Define JSTOR words to remove ("stop words")
    jstor_stop_words = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]) 
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.encode('utf-8').decode('utf-8').replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")

    if unhyphenate:              
        ls = re.findall(r"\w+-\s\w+", sentence)
        if len(ls) > 0:
            ls_new = [re.sub(r"- ", "", word) for word in ls]
            for i in range(len(ls)):
                sentence= sentence.replace(ls[i], ls_new[i])
                
    if remove_numbers:
        #sentence = re.sub(r"\b[0-9]+\b\s*", "", sentence) # remove words made up of numbers
        #sentence = re.sub(r"\b.*[0-9]+\S*\b\s*", "", sentence) # remove words containing numbers
        sentence = re.sub(r"\d+", "", sentence) # remove numbers from anywhere
        
    sentence = re.sub(r"\b[a-zA-Z]\b", "", sentence) #remove any single letter words
    
    if remove_acronyms:
        sentence = re.sub(r"\b[A-Z][A-Z]+\b\s+", "", sentence)
    
    # Tokenize using Apache
    sent_list = apache_tokenize(sentence, lowercase = lowercase)
        
    # Remove same stopwords as JSTOR, also junk formatting words
    if remove_stopwords:
        junk_words = ["colwidth", "colname", "char", "rowsep", "colsep", 
                      "oasis", "pp", "fn", "sec", "pi", "sc", "id"] # define junk/formatting terms to avoid
        stop_words = jstor_stop_words + junk_words # simplest way to avoid stopwords and formatting words: combine them!
        
        sent_list = [word for word in sent_list if 
                     word not in stop_words and 
                     ("valign" not in word)] # one more meddlesome formatting word: "valign"
        
    # Remove common sentences made of formatting (junk) words
    blacklist_sents = ['valign bottom oasis entry oasis entry colname colsep rowsep align char char', 
                       'oasis entry oasis entry colname colsep rowsep align char char', 
                       'oasis entry colname colsep rowsep align char char', 
                  'valign bottom oasis entry colname colsep rowsep align char char', 
                       'colsep rowsep oasis entry align char char', 
                       'oasis entry oasis entry colsep rowsep align char char', 
                       'colsep rowsep oasis entry oasis entry align char char']
    if sent_list in blacklist_sents:
        return('')
        
    # If True, include the proper nouns in stop_words_list
    if remove_propernouns:              
#         doc = nlp(sentence) # Create a document object in spacy
#         proper_nouns = gather_propernouns(doc) # Creates a wordbank of proper nouns we should exclude
        #trying to gather proper nouns by passing in pure sentence in gather_propernouns
        proper_nouns = gather_propernouns(sentence)
        print(proper_nouns)
        # Remove each proper noun from sentence:
        sent_list = [word for word in sent_list if word not in proper_nouns]
        #for term in proper_nouns: # Loop over wordbank
        #    sentence = re.sub(term, "", sentence) # Less effective because removes characters from within terms
    
    if return_string:
        return ' '.join(sent_list) # Return clean, tokenized sentence (string)
    
    return sent_list