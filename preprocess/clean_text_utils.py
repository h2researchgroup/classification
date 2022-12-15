#!/usr/bin/env python
# coding: utf-8

'''
@title: General Functions to Preprocess Academic Articles
@author: Jaren Haber, PhD, Dartmouth College
@coauthors: Prof. Heather Haveman, UC Berkeley; Yoon Sung Hong, Wayfair
@contact: Jaren.Haber@georgetown.edu
@project: Computational Literature Review of Organizational Scholarship
@repo: https://github.com/h2researchgroup/classification/
@date_created: Fall 2018
@date_modified: December 13, 2022
@description: Essential functions for text preprocessing. The core function cleans sentences by removing stopwords, proper nouns, etc. Other functions gather proper nouns and create lists of stopwords, punctuation, and unicode.
'''

###############################################
#                  Initialize                 #
###############################################

# Import packages
import re, datetime
import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
import os # for working with file trees
import numpy as np
from enchant import Dict; check_english_enchant = Dict("en_US")  # dictionary of english words for language filtering 
from nltk.tag import pos_tag #to look for proper nouns when cleaning text

#apache tokenizer imports
import lucene
from java.io import StringReader
from org.apache.lucene.analysis.ja import JapaneseAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute


###############################################
#               Define functions              #
###############################################

def stopwords_general(extend_stopwords = False):
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


def stopwords_jstor(stop = True, junk = True):
    """Define JSTOR words to remove ("stop words"): those used by JSTOR and/or junk formatting words.
    
    Args:
        stop (boolean): whether to return stop words used by JSTOR when creating their ngram files
        junk (boolean): whether to return junk formatting words common in JSTOR's raw OCR text files
    
    Returns:
        combined_stop_words (list of str): words to avoid when dealing with jstor data
    """
    
    # define same stopwords used by JSTOR when creating ngram files
    jstor_stop_words = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]) 
    
    if stop and not junk: # don't combine
        return jstor_stop_words
        
    # define junk/formatting terms to avoid
    junk_words = ['colwidth', 'colname', 'char', 'rowsep', 'colsep', 
                  'oasis', 'pp', 'fn', 'sec', 'pi', 'sc', 'id', 
                  'cyr', 'extcyr', 'caption', 'newcommand', 
                  'normalfont', 'selectfont', 'documentclass', 'aastex', 
                  'declaremathsizes', 'declaretextfontcommand', 
                  'pagestyle', 'xlink:type', 'sub', 'sup', 'nameend', 'pgwide', 
                  'tbody', 'tgroup', 'sup', 'tbfna', 'morerows', 
                  'xlink:href', 'fg.tiff', 'tb.eps', 'df.eps', 'χ', 
                  'xmlns:oasis', 'dtd', 'drd', 'xmlns:oasis', 'http', 
                  'docs.oasis', 'open.org ns'] 
    
    if junk and not stop:
        return junk_words
    
    if stop and junk:
        combined_stop_words = set(list(jstor_stop_words) + junk_words) # simplest way to avoid stopwords and formatting words: combine them!
        return combined_stop_words
                                                     
    
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


def remove_http(url, https = False):
    '''
    modify jstor id to get the link in the form of www.jstor.org/stable/23057056
    '''
    
    if https:
        try: 
            url_cleaned = url.split('https://')[1]
        except:
            url_cleaned = url
            
    else:
        try: 
            url_cleaned = url.split('http://')[1]
        except:
            url_cleaned = url
    
    return url_cleaned


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
                          remove_numbers = True, 
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
        stop_words = stopwords_jstor(stop = True, junk = True) # stopwords for JSTOR and junk formatting words
        sent_list = [word for word in sent_list if 
                     word not in stop_words and 
                     ("valign" not in word)] # one more meddlesome formatting word: "valign"
        
    if not remove_stopwords: # at least remove junk words/residual formatting
        junk_words = stopwords_jstor(stop = False, junk = True)
        sent_list = [word for word in sent_list if 
                     word not in junk_words and 
                     ("valign" not in word)] # one more meddlesome formatting word: "valign"
        
    # Remove common sentences made of formatting (junk) words
    blacklist_sents = ['valign bottom oasis entry oasis entry colname colsep rowsep align char char', 
                       'oasis entry oasis entry colname colsep rowsep align char char', 
                       'oasis entry colname colsep rowsep align char char', 
                       'valign bottom oasis entry colname colsep rowsep align char char', 
                       'colsep rowsep oasis entry align char char', 
                       'oasis entry oasis entry colsep rowsep align char char', 
                       'colsep rowsep oasis entry oasis entry align char char', 
                       'bottom entry', 'align center', 'align left', 
                       'colspec colnum', 'usepackage amsbsy', 'usepackage amsfonts', 
                       'usepackage amssymb', 'usepackage bm', 
                       'usepackage mathrsfs', 'usepackage pifont', 
                       'usepackage stmaryrd', 'usepackage textcomp', 
                       'position float', 'alt version', 'mimetype image', 
                       'italic italic', 'italic ij', 'begin document', 
                       'inline formula', 'entry namest', 'frame topbot', 
                       'orient port', 'list item', 'table wrap', 'tbody top', 
                       'disp formula', 'fig group', 'top entry', 
                       'tex math notation latex', 'usepackage amsmath amsxtra', 
                       'usepackage ot ot fontenc', 'renewcommand rmdefault wncyr', 
                       'renewcommand sfdefault wncyss', 'renewcommand encodingdefault ot', 
                       'end document tex math', 'entry align entry', 'entry align left top', 
                       'align right', 'table wrap foot', 'top break entry', 
                       'table xml exchange table model en', 
                       'exchange table', 'label table label', 'tgroup cols align left', 
                       'disp formula df', 'entry align left top entry', 
                       'fig position float fig type figure', 'fig group', 'graphic ', 
                       'bottom yes entry', 'bottom model entry', 'bottom sd entry', 
                       'entry align left top italic df  italic lt entry',
                       'graphic tb eps', 'bottom mean entry', 
                       'bottom mse entry', 'bottom total entry', 
                       'italic df italic two tailed entry', 'label fig label', 
                       'bottom configuration sets entry', 'bottom continuation rate entry', 
                       'bottom continuation rate other configurations entry', 
                       'italic white boys risk italic', 
                       'bottom italic df italic entry']
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


def fix_ngrams(article, 
               ngrams_list, 
               delimiter = b'_'):
    '''
    Detects and fixes multi-word expressions (MWEs) from ngrams list input. 
    Works with phrases up to three words long (trigrams).
    Returns the input text (article) in same format as input: list of lists of str.
    
    Args:
        article (list): list of lists of words (each list is a sentence)
        ngrams_list (list): list of multi-word expressions, i.e. bigrams and trigrams
        delimiter (str): to join together the words in multi-word expressions
    Returns:
        article_fixed (list): modified list of lists of words (each list is a sentence), with ngrams from list joined with delimiter
    '''
    
    article_fixed = []
    
    for sent in article: # loop over sentences (each a list of words)
        for ngram in ngrams_list:
            sent = re.sub(ngram, ngram.replace(' ', '_'), ' '.join(sent)).split() # replace space in each ngram with delimiter 
        article_fixed.append(sent)
        
    return article_fixed
    

def preprocess_text(article, 
                    filter_english = False,
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
        filter_english (boolean): if True, keep only sentence words that match PyEnchant English dictionary
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
    article = re.sub(r'<.*?>', '', article)
    article = re.sub(r'<body.*\n\s*.*\s*.*>', '', article)
    
    # Filter to English words, if set to do so
    if filter_english:
        '''
        article_filtered = []
        print(article)
        for sent in article:
            sent = [word for word in sent if check_english_enchant.check(word)]
            article_filtered.append(sent)
        article = article_filtered
        '''
        article = ' '.join([word for sent in article.split('\n') for word in sent if check_english_enchant.check(word)])
    
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


def clean_sentence_alt(sentence, 
                       remove_stopwords = True, 
                       keep_english = False, 
                       fast = False, 
                       exclude_words = [], 
                       stemming=False):
    """Alternative sentence cleaner that doesn't use Apache StandardTokenizer, 
    has alternative dictionaries for language filtering, and more easily adapts to non-JSTOR texts.
    Removes numbers, emails, URLs, unicode characters, hex characters, and punctuation from a sentence 
    separated by whitespaces. Returns a tokenized, cleaned list of words from the sentence.
    
    Args: 
        sentence, i.e. string that possibly includes spaces and punctuation
        remove_stopwords: whether to remove stopwords, default True
        keep_english: whether to remove words not in english dictionary, default False; if 'restrictive', keep word only if in NLTK's dictionary of 237K english words; if 'permissive', keep word only if in longer list of 436K english words
        fast: whether to skip advanced sentence cleaning, removing emails, URLs, and unicode and hex chars, default False
        exclude_words: list of words to exclude, may be most common words or named entities, default empty list
        stemming: whether to apply PorterStemmer to each word, default False
    Returns: 
        Cleaned & tokenized sentence, i.e. a list of cleaned, lower-case, one-word strings"""
    
    global stop_words_list, punctstr, unicode_list
    
    # Options for filtering to known English words: NLTK and large English words list
    from nltk.corpus import words # Dictionary of 236K English words from NLTK
    english_nltk = set(words.words()) # Make callable
    english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
    with open("english_words.txt", "r") as f: #path to long english dictionary
        for word in f:
            english_long.add(word.strip())
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")
    
    if not fast:
        # Remove hex characters (e.g., \xa0\, \x80):
        sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #replace anything that starts with a hex character 

        # Replace \\x, \\u, \\b, or anything that ends with \u2605
        sentence = re.sub(r"\\x.*|\\u.*|\\b.*|\u2605$", "", sentence)

        # Remove all elements that appear in unicode_list (looks like r'u1000|u10001|'):
        sentence = re.sub(r'|'.join(map(re.escape, unicode_list)), '', sentence)
    
    sentence = re.sub("\d+", "", sentence) # Remove numbers
    
    sent_list = [] # Initialize empty list to hold tokenized sentence (words added one at a time)
    
    for word in sentence.split(): # Split by spaces and iterate over words
        
        word = word.strip() # Remove leading and trailing spaces
        
        # Filter out emails and URLs:
        if not fast and ("@" in word or word.startswith(('http', 'https', 'www', '//', '\\', 'x_', 'x/', 'srcimage')) or word.endswith(('.com', '.net', '.gov', '.org', '.jpg', '.pdf', 'png', 'jpeg', 'php'))):
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

    return sent_list # Return clean, tokenized sentence