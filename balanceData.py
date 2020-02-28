import pandas as pd
import string
import unicodedata
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline

def groupMoods(mood):
    if mood in ['enthusiasm','fun','relief']:
        return 'joy'
    elif mood in ['hate', 'anger']:
        return 'anger'
    elif mood in ['sadness', 'empty', 'boredom']:
        return 'sadness'
    else:
        return mood

def createBalancedTestSet(data, n):
    rows = np.random.choice(data.index.values, n)
    return data.loc[rows]

if __name__ == '__main__':
    
    data = pd.read_csv('text_emotion.csv')
    data['sentiment'] = data['sentiment'].apply(lambda x: groupMoods(x))
    data.to_csv('text_emotionBal.csv') #grouped moods

    sadness = data[data['sentiment']=='sadness']
    joy = data[data['sentiment']=='joy']
    neutral = data[data['sentiment']=='neutral']
    worry = data[data['sentiment']=='worry']
    surprise = data[data['sentiment']=='surprise']
    love = data[data['sentiment']=='love']
    anger = data[data['sentiment']=='anger']
    happiness = data[data['sentiment']=='happiness']

    sadnessTrain = createBalancedTestSet(sadness, 1000)
    joyTrain = createBalancedTestSet(joy, 1000)
    neutralNTrain = createBalancedTestSet(neutral, 1000)
    worryTrain = createBalancedTestSet(worry, 1000)
    surpriseNTrain = createBalancedTestSet(surprise, 1000)
    loveTrain = createBalancedTestSet(love, 1000)
    angerTrain = createBalancedTestSet(anger, 1000)
    happinessTrain = createBalancedTestSet(happiness, 1000)

    everything = [sadnessTrain, joyTrain, neutralNTrain, worryTrain, 
                  surpriseNTrain, loveTrain, angerTrain, happinessTrain]
    theDF = pd.concat(everything)
    theDF.to_csv('text_emotionBalanced.csv')

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

    negatives = [sadnessTrain, neutralNTrain, worryTrain, surpriseNTrain, angerTrain]
    positives = [joyTrain, neutralPTrain, surprisePTrain, loveTrain, happinessTrain]
    negativeDF = pd.concat(negatives)
    positiveDF = pd.concat(positives)

    negativeDF.to_csv('text_emotion_negative.csv')
    positiveDF.to_csv('text_emotion_positive.csv')