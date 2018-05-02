import collections as col
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
import spacy

import books_utils as bu

nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner', 'textcat'])

def extract_relations_from(book, paragraphIds, meaningful_ids):
    relations = col.defaultdict(list)

    for paragraphId in paragraphIds:
        subset = book.tokens[(book.tokens['paragraphId'] == paragraphId)]
        subset_charIds = subset[subset['characterId'].isin(meaningful_ids)]['characterId'].unique()
        if len(subset_charIds) > 1:
            sorted_chars = sorted(subset_charIds)
            for char1 in sorted_chars:
                starting_index = sorted_chars.index(char1) + 1
                for char2 in sorted_chars[starting_index:]:
                    key = str(char1) + '<=>' + str(char2)
                    relations[key].append(paragraphId)
    return relations

def senti_pos(pos):
    if (pos == 'VERB'):
        return wn.VERB
    if (pos == 'NOUN'):
        return wn.NOUN
    if (pos == 'ADJ'):
        return wn.ADJ
    if (pos == 'ADV'):
        return wn.ADV
    return None

def get_avg_sent(word, pos = None):
    synset = list(swn.senti_synsets(word, senti_pos(pos)))[0:5]
    count = len(synset)
    if (count == 0):
        return 0
    total_pos = 0
    total_neg = 0
    for syn in synset:
        total_pos += syn.pos_score()
        total_neg += syn.neg_score()
    return (total_pos - total_neg) / count


def get_relation(book, paragraphIds):
    related_tokens = ' '.join(book.tokens[book.tokens['paragraphId'].isin(paragraphIds)]['lemma'].fillna('').ravel())
    doc = nlp(related_tokens);
    tks = list(doc)
    sent = 0
    for token in tks:
        sent += get_avg_sent(str(token), token.pos_)
    return (sent / len(tks))

def analyze_book(book):
    results = pd.DataFrame(columns=['bookname', 'char_1', 'char_2', 'affinity'])
    meaningful_ids = [char['id'] for char in book.characters.meaningful]
    paragraphIds = book.tokens['paragraphId'].unique()
    
    map_id_to_char = {}
    for id in meaningful_ids:
        map_id_to_char[id] = book.characters.meaningful[meaningful_ids.index(id)]
    
    relations = extract_relations_from(book, paragraphIds, meaningful_ids)
    to_delete = []
    for key, value in relations.items():
        if len(value) < 6:
            to_delete.append(key)
    for key in to_delete:
        del relations[key] 
    for key, parIds in sorted(relations.items(), key=lambda pair: -len(pair[1])):
        id1, id2 = [int(s) for s in key.split("<=>")]
        relation = get_relation(book, parIds) * 50 + 0.5
        results = results.append([{
            'bookname': book.name, 'char_1': id1, 'char_2': id2, 'affinity': relation
        }])
    return results, map_id_to_char

def predict(pr, Xs):
    def get_val(row):
        try:
            return pr[((pr['bookname'] == row['book_name']) & (pr['char_1'] == row['char_1']) & (pr['char_2'] == row['char_2'])) |
                   ((pr['bookname'] == row['book_name']) & (pr['char_2'] == row['char_1']) & (pr['char_1'] == row['char_2']))][0:1]['affinity'][0]
        except:
            return 0.5
    return Xs.apply(get_val, axis=1)

def create_for(books, all_X):
    predictor = pd.DataFrame(columns=['bookname', 'char_1', 'char_2', 'affinity'])
    for book in tqdm(books, desc='Looping over books'):
        single_book_res, map_id_to_char = analyze_book(book)
        subset = all_X[all_X['book_name'] == book.name]
        present_chars = pd.concat([subset['char_1'], subset['char_2']]).unique()
        single_book_res['char_1'] = single_book_res['char_1'].apply(lambda id: bu.book_name_to_annotated_name(book.name, map_id_to_char[id], present_chars))
        single_book_res['char_2'] = single_book_res['char_2'].apply(lambda id: bu.book_name_to_annotated_name(book.name, map_id_to_char[id], present_chars))
        predictor = predictor.append(single_book_res)
    return predictor