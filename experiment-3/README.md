# Extracting character relations from fictional literature

Project pipeline:
          -> characters -> main characters -> char pairs \
raw text                                                  ⟩ windowed spans in paragraphs with chars => relations
          -> paragraphs -> tokens                        /


I. Raw text -> paragraphs

Ideally, splitting should support streaming of files / additional tokenization / lemmatization / name normalization would be cool, like the one done by book-nlp but price of this is lack of flexibility for some things

II. Raw text -> characters

Supposedly, NER with ability to solve coreferences / aliasing used in fictional text
1) can be partially solved by additional data from dbpedia
2) aliasing based on rules (titles, initials, etc)
3) ???

https://pdfs.semanticscholar.org/35d4/af572e687228a8dd2241f85d7a833fcf5e5d.pdf ?
https://arxiv.org/pdf/1608.00646.pdf ?
http://nlp.lsi.upc.edu/papers/sapena07b.pdf ?
https://github.com/dbamman/book-nlp
https://github.com/emdaniels/character-extraction
https://github.com/zfsang/CharacterGo

https://github.com/zfsang/CharacterGo/blob/master/code/Match_label.ipynb
https://github.com/datamade/probablepeople

Original stuff - sparknotes characterlist


### Features

https://aclweb.org/anthology/P/P16/P16-1203.pdf

### 

Approaches to try: framePolarity thingy
https://github.com/google/sling
http://www.ark.cs.cmu.edu/SEMAFOR
https://github.com/Noahs-ARK/open-sesame

Approaches to try: word2vec averaging for words between char1 - char2

drop bad data
udpate characters