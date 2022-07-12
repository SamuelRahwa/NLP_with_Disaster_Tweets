#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json

import plotly.express as px
import plotly.graph_objects as go

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from matplotlib.pyplot import figure
matplotlib.rcParams['figure.figsize'] = (22,10)
plt.rcParams.update({'font.size': 18})

import seaborn as sns 
sns.set_style('darkgrid')

from collections import Counter
from wordcloud import WordCloud

import nltk, spacy, re, string, unicodedata, contractions

from spacy_langdetect import LanguageDetector
from spacy.language import Language
#from scispacy.abbreviation import AbbreviationDetector

import pkg_resources
from symspellpy import SymSpell, Verbosity


from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from bs4 import BeautifulSoup
import preprocessor as p 


# # Cleaning

# In[2]:


# Preprocessing Functions


# In[3]:


nlp = spacy.load("en_core_web_lg", disable = ["parser", "ner"])



sym_spell = SymSpell(max_dictionary_edit_distance = 3, prefix_length = 7)

dictionary_path = pkg_resources.resource_filename("symspellpy", 
                                                  "frequency_dictionary_en_82_765.txt")

# term_index is the column of the term and count_index is the
# column of the term frequency

sym_spell.load_dictionary(dictionary_path, term_index = 0, count_index = 1)



def to_lowercase(text):
    return text.lower()

def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def standardize_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_url(text):
    return re.sub(r'https?:\S*', ' ', text)

def expand_contractions(text):
    expanded_words = [] 
    for word in text.split():
        expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)

def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*', ' ', text)
    return re.sub(r'#\S*', ' ', text)

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, ' ', text)

def remove_numbers(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, ' ', text)

def remove_punctuation(text):
    return ''.join([c for c in text if c not in string.punctuation])

def spell_checker(text):
    
    clean_text = []
    
    for word in text.split():
        suggestion = sym_spell.lookup(word, Verbosity.CLOSEST, include_unknown = True)
        clean_text.append(suggestion[0].term)
    return ' '.join(clean_text)

def remove_stopwords(text): 
    
    filtered_sentence = [] 
    
    doc = nlp(text)
    
    for token in doc:
        if token.is_stop == False:
            filtered_sentence.append(token.text)   
            
    return ' '.join(filtered_sentence)


def lemmatize(text):
    
    doc = nlp(text)
    
    lemmatized_text = []
    
    for token in doc:
        lemmatized_text.append(token.lemma_)
        
    return ' '.join(lemmatized_text)


# # Primary Processing Function

def text_preprocesser_nlp(text):
    
    clean = text
    clean = to_lowercase(clean)
    clean = remove_html_tags(clean)
    clean = standardize_accented_chars(clean)
    clean = remove_url(clean)
    clean = expand_contractions(clean)
    clean = remove_mentions_and_tags(clean)
    clean = remove_special_characters(clean)
    clean = remove_numbers(clean)
    clean = remove_punctuation(clean)
    clean = spell_checker(clean)
    clean = remove_stopwords(clean)
    clean = lemmatize(clean)
    
    return clean