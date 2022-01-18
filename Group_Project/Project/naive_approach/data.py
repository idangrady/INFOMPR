import glob
import struct
import csv
import tensorflow as tf
from tensorflow.core.example import example_pb2

# <s> and </s> are used in the data files to segment the abstracts into sentences.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
BASE_PATH = "/Users/max/downloads/finished_files/chunked/"
TRAINING_PATH = BASE_PATH + "train_*"
VALIDATION_PATH = BASE_PATH + "val_*"
TEST_PATH = BASE_PATH + "test_*"



"""
Function that extracts the articles and abstracts from the finished_files chunked folder

Parameters:
n (int): Number of articles with their abstract that should be fetched. If -1, all available articles and their abstracts will be used
mode (String): Either train, val, or test, depending on which data is needed

Returns:
list[String], list[String]: Returns a tuple list of articles and another list of the corresponding abstracts
"""
def get_data(n, mode): 
    path_mapping = {
      'train': TRAINING_PATH, 
      'val': VALIDATION_PATH, 
      'test': TEST_PATH
    }
    if mode not in path_mapping:
      raise Exception(f"mode needs to be either \"train\", \"val\", or \"test\", not \"{mode}\"")
    path = path_mapping[mode]
    filelist = glob.glob(path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at') # check filelist isn't empty
    filelist = sorted(filelist)
    articles, abstracts = [], []
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(example_str)
        try:
            article_text = tf_example.features.feature['article'].bytes_list.value[0].decode() # the article text was saved under the key 'article' in the data files
            abstract_text = tf_example.features.feature['abstract'].bytes_list.value[0].decode() # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            print("Failed to get article or abstract from example")
            continue
        if len(article_text) == 0:
            print("Found an example with empty article text. Skipping it")
        else:
            cleaned_abstract = clean_s_tags_from_abstract(abstract_text)
            articles.append(article_text)
            abstracts.append(cleaned_abstract)
            if(len(articles) == n):
              return articles, abstracts # in case n != -1, we return n articles and their corresponding abstracts
    return articles, abstracts

def clean_s_tags_from_abstract(abstract):
  cur = 0
  sentences = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sentences.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return " ".join([sentence.strip() for sentence in sentences])

def show_first_n_news_items(n: int):
    articles, abstracts = get_data(n = n, mode = "train")
    for i in range(n):
        print(f"abstract {i}:\n{abstracts[i]}")
        print(f"article {i}:\n{articles[i]}")
