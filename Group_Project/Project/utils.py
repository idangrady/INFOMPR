"""
Utils libraries 
 
Construction mapping
"""

# imports
import numpy as np
import torch
import pickle
from torch import Tensor
from torch import FloatTensor, LongTensor
from easydict import EasyDict as edict
print("Libraries Utilis Loaded succussfuly")

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

def saved_load(self, filename:str,filepate: str,  what: str):
    if what=='wb':
        pickle.dump(self, filename, what)
    elif what=='rb':
        return pickle.load(open(filepate, what))


def w_2_onehot(vocab:dict[str,int], words: list)->FloatTensor:
    """ 
    return one hot vertor from a list of words based on their index in the dic vocab
    """
    output =Tensor.zeros((1, len(vocab))).reshape(1,-1) 
    for idx, word in enumerate(words):
        idx_word = vocab[word]
        curr_one_hot = Tensor.zeros((1, len(vocab))).reshape(1,-1)  
        curr_one_hot[idx_word] = 1
        if idx==0:
            output = curr_one_hot
        else: torch.cat((output, curr_one_hot), axis = 0)
        
    # return their one hot matrix. vector
    return output

def word_to_idx(data: list) -> dict:
    """
    Function that maps the data and return a dictionary of words corresponding to their index
    it gets a list
    return:
        dict 1 idx to word
        dict 2 word to idx
    """
    total_letters = [letters for sublist in data for subsublist in sublist for letters in subsublist]
    unique_letters =set(total_letters)
    total_words = [word.replace(',','') for sublist in data for subsublist in sublist for word in subsublist.split()]
    unique_words =list(set(total_words))

    w_2_i = {unique_words[i]:i for i in range(len(unique_words))}
    i_2_w= {i: unique_words[i] for i in range(len(unique_words))}
    print(w_2_i)
    input()
    return (w_2_i, i_2_w)
    