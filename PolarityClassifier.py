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

class PolarityClassifier(object):
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

    def getTopN(self, n, reg, X, moods):
        probs = reg.predict_proba(X)
        topN = []
        for prob in probs:
            best_N = list(reversed(np.argsort(prob)))[:n]
            topN.append(best_N)
        topN = np.array(topN)
        topNpred = moods[topN]
        return topNpred

    def fit(self, dataPolar, polarContent, polarLabel):
        self.tfidfPolar, Xpolar = self.createTFIDF(dataPolar, polarContent, True)
        ypolar = self.getLabel(dataPolar, polarLabel)
        self.polarityClassifier = self.createRegressor(Xpolar, ypolar)

        return self    
    
    #takes in pandas series
    def predict(self, X):
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        return preds