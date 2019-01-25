# load wv and lookup words

import sys
import gensim
import os
import numpy as np

# ---------------- load wv ----------------
def load(file):

	return wv = gensim.models.KeyedVectors.load(file, mmap='r')

# -------------- text to vec ---------------
def text2vec(text, wv):

	p = gensim.parsing.porter.PorterStemmer()

	total = np.zeros((1, 100), dtype = float32)

	word_count = 0

	for word in gensim.utils.simple_preprocess(text):

		word = p.stem(word)

		if word in wv:

			total += wv[word]

			count += 1

		else:

			continue

	if word_count > 0:

		return total/word_count

	else:

		return total

# ----------------- gen features -----------
def gen_word2vec_features(documents_file, word2vec_file, wd, label_file, label_list_file):

	# load label file
	labels = {}

	f = open(label_file, 'r')

	for line in f:

		username = line.strip().split(',')

		label = line.strip().split(',')

		labels[username] = label

	f.close()

	# load word vecors

	os.chdir(wd)

	wv = load(word2vec_file)

	# covert user text to vectors

	f = open(documents_file, 'r')

	label_list = open(label_list_file, 'w')

	ret = None

	for line in f:

		data = line.strip().split(',')

		username = data[0]

		text = data[1]

		if username not in labels:

			continue

		user_vec = text2vec(text, wv)

		if not ret:

			ret = user_vec

		else:

			ret = np.vstack(ret, user_vec)

		label_list.write(labels[username])
		label_list.write('\n')


	f.close()
	label_list.close()

	np.save("features", ret)






