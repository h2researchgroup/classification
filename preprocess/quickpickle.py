#!/usr/bin/env python
# coding: utf-8

'''
@author: Jaren Haber, PhD
@date created: January 2018
@repo: https://github.com/jhaber-zz/data_tools
@description: Functions for quickly saving & loading pickle objects to/from file
'''

# Import packages & functions:
import gc # For speeding up loading pickle files ('gc' = 'garbage collector')
import _pickle as cPickle # Optimized version of pickle


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