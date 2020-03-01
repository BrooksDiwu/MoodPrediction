import pandas as pd
import string
import unicodedata
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import sklearn.model_selection as cv
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import tensorflow as tf
plt.style.use("fivethirtyeight")
%matplotlib inline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import accuracy_score 

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import SpatialDropout1D, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

smileyfaces = [':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', ':>', 
               '=]', '8)', '=)']
sadfaces = ['>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', 
            ':[', ':{', '=(','=[', 'D:']
angryfaces = ['>:(', '(╯°□°)╯︵ ┻━┻']
cryingfaces = [":’-(", ":’("]
skepticalfaces = ['>:', '>:/', ':-/', '=/',':L', '=L', ':S', '>.<']
noexpressionfaces = [':|', ':-|', '(｀・ω・´)']
surprisedfaces = ['>:O', ':-O', ':O', ':-o', ':o', '8O', 'O_O', 
                  'o-o', 'O_o', 'o_O', 'o_o', 'O-O']

def cleanText(self, wordSeries):
    '''
    cleans up the text in a series by removing punctuations and replacing 
    eomjis, elipsis, etc. for non unicode text

    Parameters:
    wordSeries: Panda Series of strings

    Return:
    Panda Series of strings
    '''
    def remove_punctuation(x): #removes punctuations
        for char in string.punctuation:
            x = x.replace(char, ' ')
        return x
    for smile in self.smileyfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(smile, 
                                                        ' smileyface '))
    for sad in self.sadfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(sad,
                                                        ' sadface '))
    for angry in self.angryfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(angry,
                                                        ' angryface '))
    for cry in self.cryingfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(cry,
                                                        ' cryingface '))
    for skeptical in self.skepticalfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(skeptical, 
                                                ' skepticalface '))
    for noexp in self.noexpressionfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(noexp,
                                        ' noexpressionfaces '))
    for surprised in self.surprisedfaces:
        wordSeries = wordSeries.apply(lambda x: x.replace(surprised, 
                                        ' surprisedface '))
    wordSeries = wordSeries.apply(lambda x: x.replace('...', 
                                    ' dotdotdot '))
    wordSeries = wordSeries.apply(lambda x: x.replace('!', 
                                    ' exclamatory '))
    wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))
    wordSeries = wordSeries.apply(
                lambda x: ''.join([i for i in x if not i.isdigit()]))
    wordSeries = wordSeries.apply(lambda x: x.lower())
    wordSeries = wordSeries.apply(
                lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))

    return wordSeries

def createTokenizer(self, text):
    '''
    Tokenizes text for the neural network

    Parameters:
    text: Pandas series of strings

    Returns:
    Vectorized and padded text vector
    '''
    tokenizer = Tokenizer(num_words=10000, split = " ")
    tokenizer.fit_on_texts(text)
    textVector = tokenizer.texts_to_sequences(text)
    textVector = pad_sequences(textVector, 154)
    return textVector

def cleanData(self, data, sentimentContent):
    '''
    Cleans the sentiment column of the data as well as tokenizes it

    Parameters:
    data: Dataframe of the data to be fitted
    sentimentContet: String that is the column name of the content
    '''
    data[sentimentContent] = cleanText(data[sentimentContent])

def createModel():
    '''
    Creates the RNN model

    Returns: The model
    '''
    model = Sequential()
    model.add(Embedding(10000, 120, input_length=154))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(60, return_sequences=True, dropout=0.5, 
                   recurrent_dropout=0.5))
    model.add(LSTM(30, return_sequences=False, dropout=0.5, 
                   recurrent_dropout=0.5))
    model.add(Dense(8, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    
    data = pd.read_csv('./data/text_emotionBal.csv')
    #vectorizing data
    cleanData(data, 'content')
    textVector = createTokenizer(data['content'].values)
    #getting labels
    y = pd.get_dummies(data['sentiment']).values
    model = createModel()
    #train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(textVector, y, 
                                                    test_size = 0.25)
    model.fit(Xtrain, Ytrain, epochs=5, batch_size=32, verbose=1)
    model.save('overallRNN2.h5')

    #get accuracy
    preds = model.predict(Xtest)
    actualPred = []
    for i in preds:
        actualPred.append(sents[np.argmax(i)])

    print(accuracy_score(actual, actualPred))
