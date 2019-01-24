# This script generates an virtual python3 environment for word2vec training

# makedir virtualpy-word2vec
# cd virtualpy-word2vec
python3 -m venv env
source ./env/bin/activate
pip3 install scipy
pip3 install --upgrade gensim