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
        data[indx] = text, label
    return data

train_set = preprocess_data(train_set)
dev_set = preprocess_data(dev_set)
test_set = preprocess_data(test_set)

# Retorna algumas informações de exemplo a respeito dos data sets
print("Num train: {}".format(len(train_set)))
print("Num dev: {}".format(len(dev_set)))
print("Num test: {}".format(len(test_set)))
print("Reviews de exemplo:")
print(train_set[0])
print()
print(train_set[1])
print()

# Separa os textos e as labels em dois arrays
trainText = [t[0] for t in train_set]
trainY = [t[1] for t in train_set]

devText = [t[0] for t in dev_set]
devY = [t[1] for t in dev_set]

testText = [t[0] for t in test_set]
testY = [t[1] for t in test_set]

# Pega as palavras "top 1000" que aparecem mais de 5 vezes no texto
max_features = 1000
min_df = 5
countVec = CountVectorizer(min_df = min_df, max_features = max_features)

# Aprende o vocabulário a partir do set de treinamento
countVec.fit(trainText)

# Transforma as listas de reviews em vetores de bag-of-words
trainX = countVec.transform(trainText)
devX = countVec.transform(devText)
testX = countVec.transform(testText)

print("Formato do X de treinamento: {}\n".format(trainX.shape))
print("Formato do vocabulário:\n {}\n".format(np.random.choice(countVec.get_feature_names(), 20)))

# Cria os modelos para ajustar
lr = LogisticRegression()
passAgg = PassiveAggressiveClassifier()
perceptron = Perceptron()

lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("---")

passAgg.fit(trainX, trainY)
print("Passive Aggressive Train:", passAgg.score(trainX, trainY))
print("Passive Aggressive Dev:", passAgg.score(devX, devY))
print("---")

perceptron.fit(trainX, trainY)
print("Perceptron Train:", perceptron.score(trainX, trainY))
print("Perceptron Dev:", perceptron.score(devX, devY))
print("---")
