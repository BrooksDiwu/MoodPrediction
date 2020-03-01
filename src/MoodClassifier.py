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

class MoodClassifier(object):
    '''
    Classifier for the mood of a text
    '''
    def __init__(self):
        '''
        initialize the emojis so that they can be replaced with words that can 
        be read by the tokenizer
        '''
        self.smileyfaces = [':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', 
                            ':>', '=]', '8)', '=)']
        self.sadfaces = ['>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', 
                         ':-[', ':[', ':{', '=(','=[', 'D:']
        self.angryfaces = ['>:(', '(╯°□°)╯︵ ┻━┻']
        self.cryingfaces = [":’-(", ":’("]
        self.skepticalfaces = ['>:', '>:/', ':-/', '=/',':L', '=L', ':S',
                               '>.<']
        self.noexpressionfaces = [':|', ':-|', '(｀・ω・´)']
        self.surprisedfaces = ['>:O', ':-O', ':O', ':-o', ':o', '8O', 'O_O'
                               'o-o', 'O_o', 'o_O', 'o_o', 'O-O']
        self.tfidfPolar = None
        self.polarityClassifier = None
        self.tfidfPositive = None
        self.tfidfNegative = None
        self.positiveClassifier = None
        self.negativeClassifier = None

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

    def getTopN(self, n, reg, X, moods):
        '''
        Gets top N predictions

        Parameters:
        n: number of predictions to get
        reg: regressor to use
        X: pandas series of data to be predicted
        moods: list of moods 

        Returns:
        list of the top N predictions
        '''
        probs = reg.predict_proba(X)
        topN = []
        for prob in probs:
            best_N = list(reversed(np.argsort(prob)))[:n]
            topN.append(best_N)
        topN = np.array(topN)
        topNpred = moods[topN]
        return topNpred

    def fitBalanced(self, dataPolar, polarContent, polarLabel, dataPositive, 
                    dataNegative, sentimentContent, sentimentLabel):
        '''
        Fits three different logistic regressors

        Parameters:
        dataPolar: dataframe of the polarity data
        polarContent: column name of the content in dataPolar
        polarLabel: column name of the labels in dataPolar
        dataPositive: dataframe of the positive sentiment data
        dataNegative: dataframe of the negative sentiment data
        sentimentContent: column name of the contents in the sentiment data
        sentimentLabel: column name of the labels in the sentiment data
        '''
        #creating polarity classifier
        self.tfidfPolar, Xpolar = self.createTFIDF(
                                  dataPolar, polarContent, True)
        ypolar = self.getLabel(dataPolar, polarLabel)
        self.polarityClassifier = self.createRegressor(Xpolar, ypolar)
        dataPositive['Tokens'] = self.tokenize(
                                 dataPositive[sentimentContent], False)
        dataNegative['Tokens'] = self.tokenize(
                                 dataNegative[sentimentContent], False)
        #creating classifiers for both negative and positive
        self.tfidfPositive, Xpositive = self.createTFIDF(
                                        dataPositive, sentimentContent, False)
        self.tfidfNegative, Xnegative = self.createTFIDF(dataNegative, 
                                        sentimentContent, False)
        yPositive = self.getLabel(dataPositive, sentimentLabel)
        yNegative = self.getLabel(dataNegative, sentimentLabel)
        self.positiveClassifier = self.createRegressor(Xpositive, yPositive)
        self.negativeClassifier = self.createRegressor(Xnegative, yNegative)

        return self    
    
    def predict(self, X):
        '''
        Predicts the mood of a piece of text

        Parameters:
        X: Pandas series of strings

        Return:
        List of predictions
        '''
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        #get polarity first
        preds = self.polarityClassifier.predict(polarityMTX) 
        moodPredictions = []

        #use positive/negative regressor based on polarity
        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                mtx = self.tfidfPositive.transform([tokens[idx]])
                moodPredictions.append(self.positiveClassifier.predict(mtx)[0])
            else:
                mtx = self.tfidfNegative.transform([tokens[idx]])
                moodPredictions.append(self.negativeClassifier.predict(mtx)[0])
        
        return moodPredictions

    def predictN(self, n, X, posMoods, negMoods):
        '''
        Predicts the mood of a piece of text

        Parameters:
        n: number of predictions per text
        X: Pandas series of strings
        posMoods: list of positive moods
        negMoods: list of negative moods

        Return:
        2D List of predictions
        '''
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        moodPredictions = []

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                mtx = self.tfidfPositive.transform([tokens[idx]])
                moodPredictions.append(self.getTopN(n, 
                self.positiveClassifier, mtx, posMoods))
            else:
                mtx = self.tfidfNegative.transform([tokens[idx]])
                moodPredictions.append(self.getTopN(n, 
                self.negativeClassifier, mtx, negMoods))
        
        return moodPredictions

    def predict_proba(self, X):
        '''
        Predicts the probability of each mood

        Parameters:
        X: Pandas series of strings

        Return:
        2D list of probabilities of each mood
        '''
        tokens = self.tokenize(X, False)
        tokens = [' '.join(x) for x in tokens]
        corpus = [row for row in tokens]
        polarityMTX = self.tfidfPolar.transform(corpus)
        preds = self.polarityClassifier.predict(polarityMTX)
        moodPredictions = []

        for idx in range(len(preds)):
            if preds[idx] == 4: #value of positives
                mtx = self.tfidfPositive.transform([tokens[idx]])
                moodPredictions.append(
                    self.positiveClassifier.predict_proba(mtx))
            else:
                mtx = self.tfidfNegative.transform([tokens[idx]])
                moodPredictions.append(
                    self.negativeClassifier.predict_proba(mtx))

        return moodPredictions
    
if __name__ == '__main__':
    
    dataPositive = readCSV('./data/text_emotion_positive.csv')
    dataNegative = readCSV('./data/text_emotion_negative.csv')
    dataPolar = readCSV('./data/text_polarity.csv',True)
    mood = MoodClassifier()
    moodB.fitBalanced(dataPolar, 5, 0, dataPositive, dataNegative,
                      'content', 'sentiment')

    model_path = 'Logistic_model.pkl'
    with open(model_path, 'wb') as f:
        # Write the model to a file.
        pickle.dump(moodB, f)
