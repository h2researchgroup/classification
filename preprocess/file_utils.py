#!/usr/bin/env python
# coding: utf-8

'''
@title: Write or read text files
@author: Jaren Haber, PhD, Georgetown University
@contact: Jaren.Haber@georgetown.edu
@repo: https://github.com/h2researchgroup/classification/
@date created: January 6, 2018
@date modified: December 2020
'''

# Import packages & functions:
import pandas as pd


def write_textlist(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school().
    Input: Path to file, list of strings
    Output: Nothing (saved to disk)"""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
    
    return    


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