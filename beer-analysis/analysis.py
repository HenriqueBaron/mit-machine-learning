import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Define os data sets e o caminho
train_path = "data/overall_train.p"
dev_path = "data/overall_dev.p"
test_path = "data/overall_test.p"
train_set = pickle.load(open(train_path, 'rb'))
dev_set = pickle.load(open(dev_path, 'rb'))
test_set = pickle.load(open(test_path, 'rb'))

# Pré-processamento: transforma todos os caracteres em letra minúscula
def preprocess_data(data):
    for indx, sample in enumerate(data):
        text, label = sample['text'], sample['y']
        text = re.sub('\W+', ' ', text).lower().strip()
    return data

train_set = preprocess_data(train_set)
dev_set = preprocess_data(dev_set)
test_set = preprocess_data(test_set)

print(train_set[0])
print()
print(train_set[1])
