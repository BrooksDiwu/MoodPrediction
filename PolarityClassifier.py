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
import pickle

class PolarityClassifier(object):
    '''
    Classifier for polarity of text
    '''
    def __init__(self):
        '''
        initialize the emojis so that they can be replaced with words that can 
        be read by the tokenizer
        '''
        self.smileyfaces = [':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', ':>',
                            '=]', '8)', '=)']
        self.sadfaces = ['>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', 
                         ':[', ':{', '=(','=[', 'D:']
        self.angryfaces = ['>:(', '(╯°□°)╯︵ ┻━┻']
        self.cryingfaces = [":’-(", ":’("]
        self.skepticalfaces = ['>:', '>:/', ':-/', '=/',':L', '=L', ':S', 
                               '>.<']
        self.noexpressionfaces = [':|', ':-|', '(｀・ω・´)']
        self.surprisedfaces = ['>:O', ':-O', ':O', ':-o', ':o', '8O', 'O_O', 
                               'o-o', 'O_o', 'o_O', 'o_o', 'O-O']
        self.tfidfPolar = None
        self.polarityClassifier = None

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

    def cleanTextU(self, wordSeries):
        '''
        cleans up the text in a series by removing punctuations and replacing 
        eomjis, elipsis, etc. for non unicode text

        Parameters:
        wordSeries: Panda Series of strings

        Return:
        Panda Series of strings        
        '''
        tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                            if unicodedata.category(chr(i)).startswith('P'))
        def remove_punctuation(text): #remove punctuations
            return text.translate(tbl)
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
        wordSeries = wordSeries.apply(lambda x: x.replace('<br >',' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('<br>',' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('`',''))
        wordSeries = wordSeries.apply(lambda x: x.replace(' id ', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace(' im ', ' '))
        wordSeries = wordSeries.apply(
            lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))

        return wordSeries

    def tokenize(self, documents, unicode):
        '''
        tokenizes the text column

        Parameters:
        documents: Pandas Series of strings
        unicode: Boolean that says whether the text is unicoded

        Returns:
        Tokenized and Lemmatized text in a 2d list
        '''
        if unicode:
            documents = self.cleanTextU(documents)
        else:
            documents = self.cleanText(documents)
        docs = [word_tokenize(content) for content in documents]
        stopwords_=set(stopwords.words('english')) #remove stopwords
        def filter_tokens(sent):
            return([w for w in sent if not w in stopwords_])
        docs=list(map(filter_tokens,docs))
        lemmatizer = WordNetLemmatizer()
        docs_lemma = [[lemmatizer.lemmatize(word) for word in words] 
                      for words in docs] #lemmatizing the text

        return docs_lemma

    def createTFIDF(self, data, contentCol, encoded = False):
        '''
        Used to create a TFIDF matrix

        Parameters:
        data: dataframe of the data
        contentCol: string that is the name of the column of tokenized text
        enconded: boolean that tells whether the text is unicoded

        Return:
        tfidf: tfidf Vectorizer
        document_tfidf_matrix: the tfidf matrix
        '''
        data['Tokens'] = self.tokenize(data[contentCol], encoded)
        data['Tokens'] = data['Tokens'].apply(lambda x: ' '.join(x))
        corpus = [row for row in data['Tokens']]
        tfidf = TfidfVectorizer()
        document_tfidf_matrix = tfidf.fit_transform(corpus)

        return tfidf, document_tfidf_matrix

    def getLabel(self, data, label):
        '''
        getter function to return the labels

        Parameters:
        data: dataframe of data
        label: string which is column name of labels

        Return:
        pandas series of labels
        '''
        return data[label]

    def createRegressor(self, X,y):
        '''
        Creates logistic regressor fit on X and y

        Parameters:
        X: tfidf matrix
        y: labels

        Return:
        logistic regressor
        '''
        lg = LogisticRegression(max_iter = 1000)
        lg.fit(X,y)
        return lg

    def fit(self, dataPolar, polarContent, polarLabel):
        '''
        Fits the data on the regressor 

        Parameters:
        dataPolar: Dataframe with polarity data
        polarContent: Column name of the content column
        polarLabel: Column name of the polar column
        '''
        self.tfidfPolar, Xpolar = self.createTFIDF(dataPolar, 
                                                   polarContent, True)
        ypolar = self.getLabel(dataPolar, polarLabel)
        self.polarityClassifier = self.createRegressor(Xpolar, ypolar)
        return self    
    
    #takes in pandas series
    def predict(self, X):
        '''
        Predicts the polarity of a piece of text

        Parameters:
        X: pandas series of strings

        Return:
        Predicted polarities of each string
        '''
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        return preds

if __name__ == '__main__':

    dataPolar = readCSV('./data/text_polarity.csv',True)
    polar = PolarityClassifier()
    polar.fit(dataPolar, 5, 0)

    model_path = 'Logistic_polar_model.pkl'
    with open(model_path, 'wb') as f:
    # Write the model to a file.
        pickle.dump(polar, f)