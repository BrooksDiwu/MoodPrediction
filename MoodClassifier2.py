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
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from MoodClassifier import MoodClassifier
from PolarityClassifier import PolarityClassifier
from operator import add

class MoodClassifier2(object):
    '''
    Mood classifier using RNN as well as logistic regression
    '''
    def __init__(self):
        '''
        initialize emojis to be removed
        '''
        self.smileyfaces = [':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', 
                            ':>', '=]', '8)', '=)']
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

    def createTokenizer(self, text):
        '''
        Tokenizes text for the neural network

        Parameters:
        text: Pandas series of strings

        Returns:
        Vectorized and padded text vector
        '''
        tokenizer = Tokenizer(num_words = 10000, split = " ")
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
        data[sentimentContent] = self.cleanText(data[sentimentContent])
        data[sentimentContent] = self.tokenize(data[sentimentContent], False)
        data[sentimentContent] = data[sentimentContent].apply(
            lambda x: " ".join(x))
        return self

    def fitRNNFastLoad(self, moodClass, polarityClass, model):
        '''
        Fits models that have already been saved and pickled

        Parameters:
        moodClass: String with the name of the MoodClassifier pickle
        polarityClass: String with the name of the PolarityClassifier pickle
        model: String with the name of the RNN model
        '''
        with open(moodClass, 'rb') as f:
            self.logisticModel = pickle.load(f)

        with open(polarityClass, 'rb') as f:
            self.polarityClassifier = pickle.load(f)
        
        self.model = load_model(model)

        self.sent = ['anger', 'happiness', 'joy', 'love', 'neutral', 
                     'sadness', 'surprise', 'worry']

        return self 
    
    #takes in pandas series
    def predict(self, X):
        '''
        Predicts the mood of text

        Parameters:
        X: Pandas Series of Strings you want to predict the mood of

        Return:
        List of predicted moods
        '''
        preds = self.polarityClassifier.predict(X)
        moodPredictions = []
        logisticPreds = self.logisticModel.predict_proba(X)
        textVector = self.createTokenizer(X.values)

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                #getting the probabilities for the first MoodClassifier class
                logistP = [0, logisticPreds[idx][0][0], 
                           logisticPreds[idx][0][1], logisticPreds[idx][0][2],
                           logisticPreds[idx][0][3], 0, 
                           logisticPreds[idx][0][4], 0]
                #MoodClassifier with a weight of 3
                pred = (self.model.predict(textVector[[idx]]) +
                         [x * 3 for x in logistP])

                moodPredictions.append(self.sent[np.argmax(pred)])
            else:
                logistP = [logisticPreds[idx][0][0], 0, 0, 0, 
                           logisticPreds[idx][0][1], logisticPreds[idx][0][2],
                           logisticPreds[idx][0][3], logisticPreds[idx][0][4]]
                pred = (self.model.predict(textVector[[idx]]) + 
                        [x * 3 for x in logistP])

                moodPredictions.append(self.sent[np.argmax(pred)])
        
        return moodPredictions

if __name__ == '__main__':
    import sys
    classifier = MoodClassifier2()
    classifier.fitRNNFastLoad('Logistic_model.pkl', 'Logistic_polar_model.pkl'
                              , 'overallRNN2.h5')
    text = pd.Series(sys.argv[1])
    print(classifier.predict(text)[0])
