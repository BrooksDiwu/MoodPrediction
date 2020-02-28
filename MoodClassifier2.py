import pandas as pd
import string
import unicodedata
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
from sklearn.datasets import make_classification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from MoodClassifier import MoodClassifier
from PolarityClassifier import PolarityClassifier
from operator import add

class MoodClassifier2(object):
    def __init__(self):
        self.smileyfaces = [':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)']
        self.sadfaces = ['>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', 
                         ':[', ':{', '=(','=[', 'D:']
        self.angryfaces = ['>:(', '(╯°□°)╯︵ ┻━┻']
        self.cryingfaces = [":’-(", ":’("]
        self.skepticalfaces = ['>:', '>:/', ':-/', '=/',':L', '=L', ':S', '>.<']
        self.noexpressionfaces = [':|', ':-|', '(｀・ω・´)']
        self.surprisedfaces = ['>:O', ':-O', ':O', ':-o', ':o', '8O', 'O_O', 'o-o', 
                               'O_o', 'o_O', 'o_o', 'O-O']
        self.tfidfPolar = None
        self.polarityClassifier = None

    def cleanText(self, wordSeries):
        def remove_punctuation(x):
            for char in string.punctuation:
                x = x.replace(char, ' ')
            return x
        for smile in self.smileyfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(smile, ' smileyface '))
        for sad in self.sadfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(sad,' sadface '))
        for angry in self.angryfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(angry, ' angryface '))
        for cry in self.cryingfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(cry, ' cryingface '))
        for skeptical in self.skepticalfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(skeptical, ' skepticalface '))
        for noexp in self.noexpressionfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(noexp, ' noexpressionfaces '))
        for surprised in self.surprisedfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(surprised, ' surprisedface '))
        wordSeries = wordSeries.apply(lambda x: x.replace('...', ' dotdotdot '))
        wordSeries = wordSeries.apply(lambda x: x.replace('!', ' exclamatory '))
        wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))
        wordSeries = wordSeries.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        wordSeries = wordSeries.apply(lambda x: x.lower())
        wordSeries = wordSeries.apply(lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))

        return wordSeries

    def cleanTextU(self, wordSeries):
        tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
        def remove_punctuation(text):
            return text.translate(tbl)
        for smile in self.smileyfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(smile, ' smileyface '))
        for sad in self.sadfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(sad,' sadface '))
        for angry in self.angryfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(angry, ' angryface '))
        for cry in self.cryingfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(cry, ' cryingface '))
        for skeptical in self.skepticalfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(skeptical, ' skepticalface '))
        for noexp in self.noexpressionfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(noexp, ' noexpressionfaces '))
        for surprised in self.surprisedfaces:
            wordSeries = wordSeries.apply(lambda x: x.replace(surprised, ' surprisedface '))
        wordSeries = wordSeries.apply(lambda x: x.replace('...', ' dotdotdot '))
        wordSeries = wordSeries.apply(lambda x: x.replace('!', ' exclamatory '))
        wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))
        wordSeries = wordSeries.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        wordSeries = wordSeries.apply(lambda x: x.lower())
        wordSeries = wordSeries.apply(lambda x: x.replace('<br >',' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('<br>',' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('`',''))
        wordSeries = wordSeries.apply(lambda x: x.replace(' id ', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace(' im ', ' '))
        wordSeries = wordSeries.apply(lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))

        return wordSeries

    def tokenize(self, documents, unicode):
        if unicode:
            documents = self.cleanTextU(documents)
        else:
            documents = self.cleanText(documents)
        docs = [word_tokenize(content) for content in documents]
        stopwords_=set(stopwords.words('english'))
        def filter_tokens(sent):
            return([w for w in sent if not w in stopwords_])
        docs=list(map(filter_tokens,docs))
        lemmatizer = WordNetLemmatizer()
        docs_lemma = [[lemmatizer.lemmatize(word) for word in words] for words in docs]

        return docs_lemma

    def createTFIDF(self, data, contentCol, encoded = False):
        data['Tokens'] = self.tokenize(data[contentCol], encoded)
        data['Tokens'] = data['Tokens'].apply(lambda x: ' '.join(x))
        corpus = [row for row in data['Tokens']]
        tfidf = TfidfVectorizer()
        document_tfidf_matrix = tfidf.fit_transform(corpus)

        return tfidf, document_tfidf_matrix

    def getLabel(self, data, label):
        return data[label]

    def createRegressor(self, X,y):
        lg = LogisticRegression(max_iter = 5000)
        print("before fit")
        lg.fit(X,y)
        return lg

    def createTokenizer(self, text):
        tokenizer = Tokenizer(num_words = 10000, split = " ")
        tokenizer.fit_on_texts(text)
        textVector = tokenizer.texts_to_sequences(text)
        textVector = pad_sequences(textVector, 120)
        return textVector
    
    def cleanData(self, data, sentimentContent):
        data[sentimentContent] = self.cleanText(data[sentimentContent])
        data[sentimentContent] = self.tokenize(data[sentimentContent], False)
        data[sentimentContent] = data[sentimentContent].apply(lambda x: " ".join(x))
        return self
    
    def createRNNModel(self, input_len):
        model = Sequential()
        model.add(Embedding(10000, 256, input_length=input_len))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()
        return model

    def fitRNNFastLoad(self, positiveModel, negativeModel):

        with open('Logistic_model.pkl', 'rb') as f:
            self.logisticModel = pickle.load(f)

        with open('Logistic_polar_model.pkl', 'rb') as f:
            self.polarityClassifier = pickle.load(f)
        
        self.modelP = load_model(positiveModel)
        self.modelN = load_model(negativeModel)

        self.sent = ['anger', 'happiness', 'joy', 'love', 'neutral', 'sadness', 'surprise', 'worry']
        self.positiveSent = ['happiness', 'joy', 'love', 'neutral', 'surprise']
        self.negativeSent = ['anger', 'neutral', 'sadness', 'surprise', 'worry']

        return self 

    def fitRNNLoad(self, dataPolar, polarContent, polarLabel, dataPositive, dataNegative, 
               sentimentContent, sentimentLabel, positiveModel, negativeModel):

        with open('Logistic_model.pkl', 'rb') as f:
            self.logisticModel = pickle.load(f)


        print("start")
        # self.tfidfPolar, Xpolar = self.createTFIDF(dataPolar, polarContent, True)
        # ypolar = self.getLabel(dataPolar, polarLabel)
        
        # print("creating polarity")
        
        # self.polarityClassifier = self.createRegressor(Xpolar, ypolar)
        with open('Logistic_polar_model.pkl') as f:
            self.polarClassifier = pickle.load(f)
        
        self.modelP = load_model(positiveModel)
        self.modelN = load_model(negativeModel)

        self.sent = ['anger', 'happiness', 'joy', 'love', 'neutral', 'sadness', 'surprise', 'worry']
        self.positiveSent = ['happiness', 'joy', 'love', 'neutral', 'surprise']
        self.negativeSent = ['anger', 'neutral', 'sadness', 'surprise', 'worry']
        #self.positiveSent = ['joy', 'neutral', 'surprise', 'love', 'happiness']
        #self.negativeSent = ['sadness', 'neutral', 'worry', 'surprise', 'anger']


        return self 
    
    #takes in pandas series
    def predict(self, X):
        # tokens = self.tokenize(X, False)
        # tokens = [' '.join(x) for x in tokens]
        # corpus = [row for row in tokens]
        # polarityMTX = self.tfidfPolar.transform(corpus)
        # preds = self.polarityClassifier.predict(polarityMTX)
        preds = self.polarityClassifier.predict(X)
        moodPredictions = []
        logisticPreds = self.logisticModel.predict_proba(X)
        textVector = self.createTokenizer(X.values)

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                logistP = [0, logisticPreds[idx][0], logisticPreds[idx][1], logisticPreds[idx][2],
                           logisticPreds[idx][3], 0, logisticPreds[idx][4], 0]
                pred = self.modelP.predict(textVector[[idx]]) + logistP[idx]
                moodPredictions.append(self.positiveSent[np.argmax(pred)])
                #moodPredictions.append(pred)
            else:
                logistP = [logisticPreds[idx][0], 0, 0, 0, logisticPreds[idx][1], logisticPreds[idx][2],
                           logisticPreds[idx][3], logisticPreds[idx][4]]
                pred = self.modelN.predict(textVector[[idx]]) + logistP[idx]
                moodPredictions.append(self.negativeSent[np.argmax(pred)])
                #moodPredictions.append(pred)
        
        return moodPredictions

    def predict(self, X):
        # tokens = self.tokenize(X, False)
        # tokens = [' '.join(x) for x in tokens]
        # corpus = [row for row in tokens]
        # polarityMTX = self.tfidfPolar.transform(corpus)
        # preds = self.polarityClassifier.predict(polarityMTX)
        preds = self.polarityClassifier.predict(X)
        moodPredictions = []
        logisticPreds = self.logisticModel.predict_proba(X)
        textVector = self.createTokenizer(X.values)

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                logistP = [0, logisticPreds[idx][0], logisticPreds[idx][1], logisticPreds[idx][2],
                           logisticPreds[idx][3], 0, logisticPreds[idx][4], 0]
                pred = self.modelP.predict(textVector[[idx]]) + 10*logisticPreds[idx]
                moodPredictions.append(self.positiveSent[np.argmax(pred)])
                #moodPredictions.append(pred)
            else:
                logistP = [logisticPreds[idx][0], 0, 0, 0, logisticPreds[idx][1], logisticPreds[idx][2],
                           logisticPreds[idx][3], logisticPreds[idx][4]]
                pred = self.modelN.predict(textVector[[idx]]) + 10*logisticPreds[idx]
                moodPredictions.append(self.negativeSent[np.argmax(pred)])
                #moodPredictions.append(pred)
        
        return moodPredictions

    def fitRNN(self, dataPolar, polarContent, polarLabel, dataPositive, dataNegative, 
               sentimentContent, sentimentLabel):
        
        print("start")
        self.tfidfPolar, Xpolar = self.createTFIDF(dataPolar, polarContent, True)
        ypolar = self.getLabel(dataPolar, polarLabel)
        
        print("creating polarity")
        
        self.polarityClassifier = self.createRegressor(Xpolar, ypolar)
        
        print("finished polarity")

        self.cleanData(dataPositive, sentimentContent)
        self.cleanData(dataNegative, sentimentContent)
        
        print("finished cleaning")

        textVectorP = self.createTokenizer(dataPositive[sentimentContent].values)
        textVectorN = self.createTokenizer(dataNegative[sentimentContent].values)
        
        print("finished textVectors")

        self.modelP = self.createRNNModel(textVectorP.shape[1])
        self.modelN = self.createRNNModel(textVectorN.shape[1])

        yP = pd.get_dummies(dataPositive[sentimentLabel]).values
        yN = pd.get_dummies(dataNegative[sentimentLabel]).values
        
        print("finished making dummies")

        self.modelP.fit(textVectorP, yP, epochs = 1, batch_size = 32, verbose = 1)
        self.modelN.fit(textVectorN, yN, epochs = 1, batch_size = 32, verbose = 1)
        
        print("finished fitting")

        self.positiveSent = sorted(dataPositive[sentimentLabel].unique())
        self.negativeSent = sorted(dataNegative[sentimentLabel].unique())

        return self 