#!/usr/bin/env python
# coding: utf-8

# Author: Jaren Haber, PhD Candidate
# Institution (as of this writing): University of California, Berkeley, Dept. of Sociology
# Date created: Fall 2018
# Date last modified: June 6, 2019 
# Source GitHub repo: https://github.com/jhaber-zz/data_management/
# Modified for: https://github.com/h2researchgroup/Computational-Analysis-For-Social-Science
# Description: Essential functions for text preprocessing, e.g. as in training word embeddings or topic models. The core function cleans sentences by removing stopwords, etc.; elementary functions create lists of stopwords, punctuation, and unicode; a wish-list function gathers most common words from a text corpus.

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
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

def stopwords_make(vocab_path_old = "", extend_stopwords = False):
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

    # Add other common stopwords
    stop_word_list.append('00') 
    stop_word_list.extend(['mr', 'mrs', 'sa', 'fax', 'email', 'phone', 'am', 'pm', 'org', 'com', 
                           'Menu', 'Contact Us', 'Facebook', 'Calendar', 'Lunch', 'Breakfast', 
                           'facebook', 'FAQs', 'FAQ', 'faq', 'faqs', '1', '2', '3', '4', '5', '6',
                          '7','8','9','0']) # web stopwords + numbers
    stop_word_list.extend(['el', 'en', 'la', 'los', 'para', 'las', 'san']) # Spanish stopwords
    stop_word_list.extend(['angeles', 'diego', 'harlem', 'bronx', 'austin', 'antonio']) # cities with many charter schools

    # Add state names & abbreviations (both uppercase and lowercase) to stopwords
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 
              'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 
              'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
              'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 
              'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI', 'WV', 'WY', 
              'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 
              'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 
              'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
              'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
              'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
              'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 
              'Vermont', 'Virginia', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming' 
              'carolina', 'columbia', 'dakota', 'hampshire', 'mexico', 'rhode', 'york']
    for state in states:
        stop_word_list.append(state)
    for state in [state.lower() for state in states]:
        stop_word_list.append(state)
        
    # Add even more stop words:
    if extend_stopwords == True:
        stop_word_list = text.ENGLISH_STOP_WORDS.union(stop_word_list)
        
    # If path to old vocab not specified, skip last step and return stop word list thus far
    if vocab_path_old == "":
        return stop_word_list

    # Add to stopwords useless and hard-to-formalize words/chars from first chunk of previous model vocab (e.g., a3d0, \fs19)
    # First create whitelist of useful terms probably in that list, explicitly exclude from junk words list both these and words with underscores (common phrases)
    whitelist = ["Pre-K", "pre-k", "pre-K", "preK", "prek", 
                 "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", 
                 "1st-grade", "2nd-grade", "3rd-grade", "4th-grade", "5th-grade", "6th-grade", 
                 "7th-grade", "8th-grade", "9th-grade", "10th-grade", "11th-grade", "12th-grade", 
                 "1st-grader", "2nd-grader", "3rd-grader", "4th-grader", "5th-grader", "6th-grader", 
                 "7th-grader", "8th-grader", "9th-grader", "10th-grader", "11th-grader", "12th-grader", 
                 "1stgrade", "2ndgrade", "3rdgrade", "4thgrade", "5thgrade", "6thgrade", 
                 "7thgrade", "8thgrade", "9thgrade", "10thgrade", "11thgrade", "12thgrade", 
                 "1stgrader", "2ndgrader", "3rdgrader", "4thgrader", "5thgrader", "6thgrader", 
                 "7thgrader", "8thgrader", "9thgrader", "10thgrader", "11thgrader", "12thgrader"]
    with open(vocab_path_old) as f: # Load vocab from previous model
        junk_words = f.read().splitlines() 
    
    if school_whitelist == True:
        junk_words = [word for word in junk_words[:8511] if ((not "_" in word) 
                                                         and (not any(term in word for term in whitelist)))]
        stop_word_list.extend(junk_words)                                      
        return stop_word_list
        
    junk_words = [word for word in junk_words[:8511] if ((not "_" in word))]
    stop_word_list.extend(junk_words)
                  
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


def get_common_words(tokenized_corpus, max_percentage):
    """Discover most common words in corpus up to max_percentage.
    
    Args:
        Corpus tokenized by words,
        Highest allowable frequency of documents in which a token may appear (e.g., 1-5%)
        
    Returns:
        List of most frequent words in corpus"""
    
    # Code goes here
    # Probably using nltk.CountVectorizer

    
# Create useful lists using above functions:
stop_words_list = stopwords_make()
punctstr = punctstr_make()
unicode_list = unicode_make()

# Instantiate spacy object for part of speech tagging (to remove proper nouns)
#nlp = spacy.load('en', disable=['ner']) # For speed, keep only POS tagging functionality and parsing (which improves accuracy). Does including NER also improve accuracy? 

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




def clean_sentence(sentence, remove_stopwords = True, stopwords_jstor = False, remove_numbers = True, keep_english = False, slow_webclean = False, exclude_words = [], stemming = False, remove_acronyms = True, remove_propernouns = True, unhyphenate = False, return_string = False):
    """Removes numbers, emails, URLs, unicode characters, hex characters, and punctuation from a sentence 
    separated by whitespaces. Returns a tokenized, cleaned list of words from the sentence.
    
    Args: 
        sentence, i.e. string that possibly includes spaces and punctuation
        remove_stopwords: whether to remove stopwords, default True
        remove_numbers: whether to remove any chars that are digits, default True
        keep_english: whether to remove words not in english dictionary, default False; if 'restrictive', keep word only if in NLTK's dictionary of 237K english words; if 'permissive', keep word only if in longer list of 436K english words
        slow_webclean: whether to include advanced sentence cleaning: removing emails, URLs, and unicode and hex chars, default False
        exclude_words: list of words to exclude, may be most common words or named entities, default empty list
        stemming: whether to apply PorterStemmer to each word, default False
        remove_propernouns: boolean, removes nouns such as names, etc., default True 
        unhyphenate: join any lingering hyphens (i.e., words ending with '- ') as for JSTOR text data, default False
        return_string: return string instead of list of tokens (useful for infersent), default False
    Returns: 
        Cleaned & tokenized sentence, i.e. a list of cleaned, lower-case, one-word strings"""
    
    global stop_words_list, punctstr, unicode_list, english_nltk, nlp #, english_long

    # Define stopwords used by JSTOR and implement if called:
    jstor_list_words = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]) #jstor removal words
    if stopwords_jstor:
        stop_words_list = jstor_list_words # Use same stopwords as JSTOR
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")
    
    if unhyphenate:              
        ls = re.findall(r"\w+-\s\w+", sentence)
        if len(ls) > 0:
            ls_new = [re.sub(r"- ", "", word) for word in ls]
            for i in range(len(ls)):
                sentence= sentence.replace(ls[i], ls_new[i])

    if remove_acronyms:
        ls = re.findall(r"\b[A-Z][A-Z]+\b\s+", sentence)
        if len(ls) > 0:
            ls_new = np.repeat("", len(ls))
            for i in range(len(ls)):
                sentence = sentence.replace(ls[i], ls_new[i])
    
    if slow_webclean:
        # Remove hex characters (e.g., \xa0\, \x80):
        sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #replace anything that starts with a hex character 

        # Replace \\x, \\u, \\b, or anything that ends with \u2605
        sentence = re.sub(r"\\x.*|\\u.*|\\b.*|\u2605$", "", sentence)

        # Remove all elements that appear in unicode_list (looks like r'u1000|u10001|'):
        sentence = re.sub(r'|'.join(map(re.escape, unicode_list)), '', sentence)
    
    # If True, include the proper nouns in stop_words_list
    if remove_propernouns:              
#         doc = nlp(sentence) # Create a document object in spacy
#         proper_nouns = gather_propernouns(doc) # Creates a wordbank of proper nouns we should exclude
           
        #trying to gather proper nouns by passing in pure sentence in gather_propernouns
        proper_nouns = gather_propernouns(sentence)
        print(proper_nouns)
        # Remove each proper noun from sentence:
        sentence = " ".join([word for word in sentence.split() if word not in proper_nouns])
        #for term in proper_nouns: # Loop over wordbank
        #    sentence = re.sub(term, "", sentence) # Less effective because removes characters from within terms

    
    sent_list = [] # Initialize empty list to hold tokenized sentence (words added one at a time)
    
    for word in sentence.split(): # Split by spaces and iterate over words
        
        word = word.strip() # Remove leading and trailing spaces
        
        if remove_numbers:
            word = re.sub(r"[0-9]+", "", word) # filter out digits
        
        # Filter out emails and URLs:
        if slow_webclean and ("@" in word or word.startswith(('http', 'https', 'www', '//', '\\', 'x_', 'x/', 'srcimage')) or word.endswith(('.com', '.net', '.gov', '.org', '.jpg', '.pdf', 'png', 'jpeg', 'php'))):
            continue
            
        # Remove punctuation (only after URLs removed):
        word = re.sub(r"["+punctstr+"]+", r'', word).strip("'").strip("-") # Remove punctuations, and remove dashes and apostrophes only from start/end of words
        
        if remove_stopwords and word in stop_words_list: # Filter out stop words
            continue
                            
        # TO DO: Pass in most_common_words to function; write function to find the top 1-5% most frequent words, which we will exclude
        # Remove most common words:
        if word in exclude_words:
            continue
            
        if keep_english == 'restrictive':
            if word not in english_nltk: #Filter out non-English words using shorter list
                continue
            
        if keep_english == 'permissive': 
            if word not in english_long: #Filter out non-English words using longer list
                continue
        
        # Stem word (if applicable):
        if stemming:
            word = ps.stem(word)
        
        sent_list.append(word.lower()) # Add lower-cased word to list (after passing checks)
    
    if return_string:
        return ' '.join(sent_list) # Return clean, tokenized sentence (string)
    
    return sent_list # Return clean, tokenized sentence (list of tokens)



def clean_sentence_apache(sentence, remove_numbers = True, remove_acronyms = True, remove_propernouns = True, unhyphenate = False, return_string = False):
    
    #sentence = ' '.join( [word.lower() for word in sentence.split() if len(word)>1] ) #removing ANY single letter words
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")

    if unhyphenate:              
        ls = re.findall(r"\w+-\s\w+", sentence)
        if len(ls) > 0:
            ls_new = [re.sub(r"- ", "", word) for word in ls]
            for i in range(len(ls)):
                sentence= sentence.replace(ls[i], ls_new[i])
    sentence = re.sub(r"\b[a-zA-Z]\b", "", sentence) #removing any single letter alphabets
    if remove_acronyms:
        sentence = re.sub(r"\b[A-Z][A-Z]+\b\s+", "", sentence)
    if remove_numbers:
        sentence = re.sub(r"\b[0-9]+\b\s*", "", sentence)
#         ls = re.findall(r"\b[0-9]+\b\s*", sentence)
#         if len(ls) > 0:
#             ls_new = np.repeat("", len(ls))
#             for i in range(len(ls)):
#                 sentence = sentence.replace(ls[i], ls_new[i])

    sent_list = []
    
    #tokenizing
    tokenizer = StandardTokenizer()
    tokenizer.setReader(StringReader(sentence))
    charTermAttrib = tokenizer.getAttribute(CharTermAttribute.class_)
    tokenizer.reset()
    while tokenizer.incrementToken():
        sent_list.append(charTermAttrib.toString().lower()) #lowercasing
        
    jstor_list_words = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]) #jstor removal words
    sent_list = [word for word in sent_list if word not in jstor_list_words] #removing stopwords as well as lowercasing all words
        
        
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
    
