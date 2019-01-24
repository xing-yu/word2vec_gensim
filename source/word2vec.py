import sys
import gensim
import json
import os
from gensim.models import KeyedVectors

wd = sys.argv[1]
file = sys.argv[2]

# --------------------- main --------------------
def main(wd, file):

	word2vec(wd, file, load_json, size = 100, window = 5, min_count = 0, workers = 4)

# -------------------- iterator -----------------
class SentencesIterator():
    def __init__(self, generator):

        self.generator = generator

    def __iter__(self):
        # reset the generator
        self.generator = self.generator
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

# --------- load json file into a stream --------
def load_json(file):

	f = open(file, 'r')

	return (json.loads(line)['body'] for line in f if line)

# ------- load csv file into a text stream ------
def load_csv(file):

	f = open(file, 'r')

	return (line.strip().split(',')[1] for line in f if line)

# ------------------- preprocess -----------------
def preprocess(filestream):

	from gensim.parsing.porter import PorterStemmer

	p = PorterStemmer()

	for line in filestream:

		yield p.stem_documents(gensim.utils.simple_preprocess(line))

# ------------------- word2vec --------------------
def word2vec(wd, file, load_func, size = 100, window = 5, min_count = 2, workers = 4):

	os.chdir(wd)

	documents = SentencesIterator(preprocess(load_func(file)))

	model = gensim.models.Word2Vec(documents, size = size, window = window, min_count = min_count, workers = workers)

	# NOTE: this file need to be loaded with
	# wv = gensim.models.KeyedVectors.load("word2vec.model", mmap='r')
	model.wv.save("word2vec.model")

	return model

# ------------------------ call main ----------------
main(wd, file)



