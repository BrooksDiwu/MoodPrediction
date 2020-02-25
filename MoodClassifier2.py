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
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        lg = LogisticRegression(max_iter = 1000)
        lg.fit(X,y)
        return lg

    def useTFIDF(self, data, contentCol, tfidf, encoded = False):
        data['Tokens'] = self.tokenize(data[contentCol], encoded)
        data['Tokens'] = data['Tokens'].apply(lambda x: ' '.join(x))
        corpus = [row for row in data['Tokens']]
        document_tfidf_matrix = tfidf.transform(corpus)
        return document_tfidf_matrix

    def splitPositiveNegative(self, data):
        positiveS = ['enthusiasm', 'neutral', 'surprise', 'love', 'fun', 'happiness', 'relief']
        negativeS = ['empty', 'sadness', 'neutral', 'worry', 'hate', 'boredom', 'anger']
        dataP = data[data['sentiment'].isin(positiveS)]
        dataN = data[data['sentiment'].isin(negativeS)]
        dataP['Tokens'] = self.tokenize(dataP['content'], False)
        dataN['Tokens'] = self.tokenize(dataN['content'], False)
        return dataP, dataN

    def getTopN(self, n, reg, X, moods):
        probs = reg.predict_proba(X)
        topN = []
        for prob in probs:
            best_N = list(reversed(np.argsort(prob)))[:n]
            topN.append(best_N)
        topN = np.array(topN)
        topNpred = moods[topN]
        return topNpred

    def createTokenizer(self, text):
        tokenizer = Tokenizer(num_words = 5000, split = " ")
        tokenizer.fit_on_texts(text)
        textVector = tokenizer.texts_to_sequence(text)
        textVector = pad_sequences(textVector)
        return textVector
    
    def cleanData(self, data, sentimentContent):
        data[sentimentContent] = cleanText(data[sentimentContent])
        data[sentimentContent] = tokenize(data[sentimentContent])
        data[sentimentContent] = data[sentimentContent].apply(lambda x: " ".join(x))
        return self
    
    def createRNNModel(self, input_len):
        model = Sequential()
        model.add(Embedding(10000, 256, input_length=input_len))
        model.add(Dropout(0.1))
        model.add(LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(LSTM(256, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(13, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model

    def fitRNN(self, dataPolar, polarContent, polarLabel, dataPositive, dataNegative, 
               sentimentContent, sentimentLabel):
        
        print("start")
        self.tfidfPolar, Xpolar = self.createTFIDF(dataPolar, polarContent, True)
        ypolar = self.getLabel(dataPolar, polarLabel)
        self.polarityClassifier = self.createRegressor(Xpolar, ypolar)
        
        print("finished polarity")

        cleanData(dataPositive, sentimentContent)
        cleanData(dataNegative, sentimentContent)
        
        print("finished cleaning")

        textVectorP = createTokenizer(dataPositive[sentimentContent].values)
        textVectorN = createTokenizer(dataNegative[sentimentContent].values)
        
        print("finished textVectors")

        self.modelP = createRNNModel(textVectorP.shape[1])
        self.modelN = createRNNModel(textVectorN.shape[1])

        yP = pd.get_dummies(dataPositive[sentimentLabel]).values
        yN = pd.get_dummies(dataNegative[sentimentLabel]).values
        
        print("finished making dummies")

        self.modelP.fit(textVectorP, yP, epochs = 10, batch_size = 32, verbose = 1)
        self.modelN.fit(textVectorN, yN, epochs = 10, batch_size = 32, verbose = 1)
        
        print("finished fitting")

        self.positiveSent = sorted(dataPositive[sentimentLabel].unique())
        self.negativeSent = sorted(dataNegative[sentimentLabel].unique())

        return self 
    
    #takes in pandas series
    def predict(self, X):
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        moodPredictions = []
        textVector = createTokenizer(X.values)

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                pred = self.modelP.predict(textVector[idx])
                moodPredictions.append(self.positiveSent[np.argmax(pred)])
            else:
                pred = self.modelN.predict(textVector[idx])
                moodPredictions.append(self.negativeSent[np.argmax(pred)])
        
        return moodPredictions

    def predictN(self, n, X, posMoods, negMoods):
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        moodPredictions = []

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                mtx = self.tfidfPositive.transform([tokens[idx]])
                moodPredictions.append(self.getTopN(n, self.positiveClassifier, mtx, posMoods))
                #moodPredictions.append(self.positiveClassifier.predict(mtx)[0])
            else:
                mtx = self.tfidfNegative.transform([tokens[idx]])
                moodPredictions.append(self.getTopN(n, self.negativeClassifier, mtx, negMoods))
                #moodPredictions.append(self.negativeClassifier.predict(mtx)[0])
        
        return moodPredictions
        
    def score(self, X, y):
        count = 0
        preds = self.predict(X)
        for idx in range(len(y)):
            if y[idx] == preds[idx]:
                count+=1
        return count/len(y)

    def nScore(self, X, y, n, posMoods, negMoods):
        count = 0
        preds = self.predictN(n, X, posMoods, negMoods)
        for idx in range(len(y)):
            if np.array(y)[idx] in preds[idx]:
                count+=1
        return count/len(y)
