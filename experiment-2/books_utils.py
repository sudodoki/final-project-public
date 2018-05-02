import re
import os
import json
import pandas as pd
def is_meaningful(agent):
  return len(agent['agent']) + \
    len(agent['mod']) + \
    len(agent['speaking']) + \
    len(agent['patient']) + \
    len(agent['poss']) > 0

def longest_name(character):
    names = sorted([name['n'] for name in character['names']],key=len)
    return names[-1]

def book_name_to_annotated_name(book_name, book_char, present_chars, fallback_to_longest=True):
    aliases = set([names['n'] for names in book_char['names']])
    set_intersection = set(present_chars) & set(aliases)
    name = None
    if len(set_intersection) > 0:
        if len(set_intersection) > 1:
            print("WARNING: {} might have multiple aliases: {}".format(longest_name(book_char), list(set_intersection)))

        name = set_intersection.pop()
    else:
        for alias in aliases:
            for present_char in present_chars:
                if alias in present_char or present_char in alias:
                    name = present_char
    return name or ("_" + longest_name(book_char) if fallback_to_longest else None)

class CharacterList:
  def __init__(self, meta_json):
    self.id = meta_json['id']
    self.characters = meta_json['characters']
  
  @property
  def all(self):
    return self.characters

  @property
  def meaningful(self):
    return [character for character in self.all if is_meaningful(character)]

class Book:
  def __init__(self, name, source_folder = './books', book_NLP_folder = './bookNLP_output'):
    self.name = name

    raw_file = os.path.join(os.path.realpath(source_folder), name + '.txt')
    if not os.path.isfile(raw_file):
      raise ValueError("{} does not exist. Either check source_folder arg {} or make sure it contains {}.txt".format(raw_file, source_folder, name))

    abs_book_NLP_folder = os.path.realpath(book_NLP_folder)
    internal_nlp_name = re.sub("\.", "_", name)
    metainfo_file = os.path.join(abs_book_NLP_folder, internal_nlp_name, internal_nlp_name + '.book')
    if not os.path.isfile(metainfo_file):
      raise ValueError("{} does not exist. Either check book_NLP_folder arg {} or make sure it contains {}/{}.book".format(metainfo_file, book_NLP_folder, internal_nlp_name, internal_nlp_name))

    tokens_file = os.path.join(abs_book_NLP_folder, internal_nlp_name, internal_nlp_name + '.tokens')
    if not os.path.isfile(tokens_file):
      raise ValueError("{} does not exist. Either check book_NLP_folder arg {} or make sure it contains {}/{}.tokens".format(tokens_file, book_NLP_folder, internal_nlp_name, internal_nlp_name))

    self.raw_file = raw_file
    self.metainfo_file = metainfo_file
    self.tokens_file = tokens_file

    self._characters = None
    self._text = None
    self._tokens = None

  @property
  def characters(self):
    if (not self._characters):
      with open(self.metainfo_file) as input:
        self._characters = CharacterList(json.load(input))
    return self._characters

  @property
  def text(self):
    if (self._text is None):
      with open(self.raw_file) as input:
        self._text = input.read()
    return self._text

  @property
  def tokens(self):
    if (self._tokens is None):
      self._tokens = pd.read_table(self.tokens_file, sep='\t', error_bad_lines=False, engine="python")
    return self._tokens