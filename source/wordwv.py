# load wv and lookup words

import gensim
import os

# ---------------- load wv ----------------
def load(wd, file):

	os.chdir(wd)

	return wv = gensim.models.KeyedVectors.load("word2vec.model", mmap='r')

# -------------- look up word --------------
def lookup(wv, word):

	p = gensim.parsing.porter.PorterStemmer()

	word = gensim.utils.simple_preprocess(word)

	return wv[p.stem(word)]