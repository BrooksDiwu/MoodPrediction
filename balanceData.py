import pandas as pd
import string
import unicodedata
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline

def groupMoods(mood):
    '''
    Groups similar moods

    Parameters:
    mood: String which is the mood

    returns: String which is the mood grouped
    '''
    if mood in ['enthusiasm','fun','relief']:
        return 'joy'
    elif mood in ['hate', 'anger']:
        return 'anger'
    elif mood in ['sadness', 'empty', 'boredom']:
        return 'sadness'
    else:
        return mood

def createBalancedTestSet(data, n):
    '''
    Balances the dataset by undersampling

    Parameters:
    data: DataFrame which inludes all the data
    n: number of samples of each piece of data you want

    returns: DataFrame of data with random values
    '''
    rows = np.random.choice(data.index.values, n)
    return data.loc[rows]

if __name__ == '__main__':
    
    data = pd.read_csv('./data/text_emotion.csv')
    data['sentiment'] = data['sentiment'].apply(lambda x: groupMoods(x))
    data.to_csv('./data/text_emotionBal.csv') #grouped moods

    #getting all data of each mood
    sadness = data[data['sentiment']=='sadness']
    joy = data[data['sentiment']=='joy']
    neutral = data[data['sentiment']=='neutral']
    worry = data[data['sentiment']=='worry']
    surprise = data[data['sentiment']=='surprise']
    love = data[data['sentiment']=='love']
    anger = data[data['sentiment']=='anger']
    happiness = data[data['sentiment']=='happiness']

    #creating balanced datasets
    sadnessTrain = createBalancedTestSet(sadness, 1000)
    joyTrain = createBalancedTestSet(joy, 1000)
    neutralNTrain = createBalancedTestSet(neutral, 1000)
    worryTrain = createBalancedTestSet(worry, 1000)
    surpriseNTrain = createBalancedTestSet(surprise, 1000)
    loveTrain = createBalancedTestSet(love, 1000)
    angerTrain = createBalancedTestSet(anger, 1000)
    happinessTrain = createBalancedTestSet(happiness, 1000)

    #saving one big balanced dataset
    everything = [sadnessTrain, joyTrain, neutralNTrain, worryTrain, 
                  surpriseNTrain, loveTrain, angerTrain, happinessTrain]
    theDF = pd.concat(everything)
    theDF.to_csv('./data/text_emotionBalanced.csv')

    #creating balanced dataset, one being positive, one negative
    sadnessTrain = createBalancedTestSet(sadness, 1000)
    joyTrain = createBalancedTestSet(joy, 2000)
    neutralNTrain = createBalancedTestSet(neutral, 1000)
    neutralPTrain = createBalancedTestSet(neutral, 2000)
    worryTrain = createBalancedTestSet(worry, 1000)
    surpriseNTrain = createBalancedTestSet(surprise, 1000)
    surprisePTrain = createBalancedTestSet(surprise, 2000)
    loveTrain = createBalancedTestSet(love, 2000)
    angerTrain = createBalancedTestSet(anger, 1000)
    happinessTrain = createBalancedTestSet(happiness, 2000)

    negatives = [sadnessTrain, neutralNTrain, worryTrain, surpriseNTrain,
                 angerTrain]
    positives = [joyTrain, neutralPTrain, surprisePTrain, loveTrain, 
                 happinessTrain]
    negativeDF = pd.concat(negatives)
    positiveDF = pd.concat(positives)

    negativeDF.to_csv('./data/text_emotion_negative.csv')
    positiveDF.to_csv('./data/text_emotion_positive.csv')