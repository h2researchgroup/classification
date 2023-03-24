#!/usr/bin/env python
# coding: utf-8

'''
@title: Functions for working with files
@author: Jaren Haber, PhD, Dartmouth College
@contact: jhaber@berkeley.edu
@repo: https://github.com/h2researchgroup/classification/
@date created: January 6, 2018
@date modified: December 2022
'''

###############################################
#                   Imports                   #
###############################################

import pandas as pd
import gc # For speeding up loading pickle files ('gc' = 'garbage collector')
import _pickle as cPickle # Optimized version of pickle


###############################################
#               Define functions              #
###############################################         

def quickpickle_load(picklepath):
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    with open(picklepath, 'rb') as loadfile:
        
        gc.disable() # disable garbage collector
        outputvar = cPickle.load(loadfile) # Load from picklepath into outputvar
        gc.enable() # enable garbage collector again
    
    return outputvar


def quickpickle_dump(dumpvar, picklepath):
    '''Very time-efficient way to dump pickle-formatted objects from Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Python object (probably a list of sentences or something similar).
    Output: Filepath to pickled (*.pkl) object.'''

    with open(picklepath, 'wb') as destfile:
        
        gc.disable() # disable garbage collector
        cPickle.dump(dumpvar, destfile) # Dump dumpvar to picklepath
        gc.enable() # enable garbage collector again
    
    return    


def write_textlist(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school().
    Input: Path to file, list of strings
    Output: Nothing (saved to disk)"""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
            

def read_text(file_path, return_string = True, shell = False):
    """Loads text into memory, either as str or as list. Must be assigned to object.
    
    Args: 
        file_path: Path to file (str)
        return_string: boolean indicating whether to return as string format (instead of list)
        shell: boolean indicating if function is called from command line
    
    Returns: 
        str if return_string, else list
    """
    
    if shell: 
        
        with open(file_path, 'r') as file_handler:
            text = file_handler.read()
        
        return text
    
    if return_string:
        
        textstr = '' # empty string
        
        with open(file_path) as file_handler:
            line = file_handler.readline()
            while line:
                textstr += line
                line = file_handler.readline()

        return textstr
        
    else: # return list of text
        
        textlist = [] # empty list
    
        with open(file_path) as file_handler:
            line = file_handler.readline()
            while line:
                textlist.append(line)
                line = file_handler.readline()

        return textlist