from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('./model/GoogleGoogleNews-vectors-negative300.bin.gz', binary=True)


KING = model['king']
MAN = model['king']
WOMAN = model['king']

print("The closest word to KING - MAN + WOMAN is: ",model.wv.most_similar(positive = KING - MAN + WOMAN,topn = 1))

ND = model['New Delhi']
IN = model['India']
EN = model['england']

print("The closest word to PARIS - FRANCE + ENGLAND is: ",model.wv.most_similar(positive = ND-IN+EN,topn = 1))

HO = model['hot']
SU = model['summer']
WI = model['winter']

print("The closest word to HOT - SUMMER + WINTER is: ",model.wv.most_similar(positive = HO-SU+WI,topn = 1))

BR = model['bright']
DA = model['day']
NI = model['night']

print("The closest word to BRIGHT - DAY + NIGHT is: ",model.wv.most_similar(positive = BR-DA+NI,topn = 1))
