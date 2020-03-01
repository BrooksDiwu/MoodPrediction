# MoodPrediction

## MoodClassifier.py
This file includes the classifier class for the mood prediction. It includes a fit function and two different predict and score functions, either predicting one mood or predicting the top n moods. The file also includes a main that runs and pickles the classifier to be used later.

## MoodClassifier2.py
This file includes an improved classifier class for mood prediction. It has a fit and predict function that combines the use of the MoodClassifier classifier as well as a RNN to predict the mood of a text. It includes a main that takes in a line of text and will output a mood.

## PolarityClassifier.py
This file includes the classifier that tells the polarity of a certain piece of text. It is used to create the MoodClassifier class. The main pickles the polarity classifier.

## balanceData.py
This file includes the code to balance and split all the data. Run the main to get 4 different CSVs used by the classifiers.

## createRNN.py
This file is used to create the RNN using the data. The main will save the rnn into a file called 'overallRNN.h5'.

## EDA.ipynb
This notebook is used to explore the sentiment data and test different classifiers.

## EDA2.ipynb
This notebook is used to explore the polarity data and test different classifiers.

## Predictor.ipynb
This notebook is used to test out the predictor by putting it through the polarity classifier and then the sentiment classifier.

## PredictorFunction.ipynb
This notebook is the predictor notebook but with everything turned into a function. All of this is combined to create the MoodClassifier class.

## testClass.ipynb
This notebook is used to test the MoodClassifier.

## The Process
The first thing we do is load the data, which is taken from https://www.kaggle.com/kazanova/sentiment140 for the polarity and https://www.kaggle.com/c/sa-emotions for the mood.

### Exploration
After loading the dataset we explore the mood dataset where we find:  
![a](/plots/common_positive.png)  
A couple of graphs outputting the most common negative/neutral/positive words in the dataset
We fit a classifer on this data which gets an accuracy of 33%, recall of 16%, and precision of 27% as well as a confusion matrix that looks like this for the predictions:  
![a](/plots/confusion_mtx.png)  
Obviously there is something wrong when some moods have a 0% chance of being predicted so we then find the counts of each mood in the dataset:  
![a](/plots/countsOfMoods.png)  
The dataset is obviously inbalanced so we do a couple things to balanced the dataset.  
First we will group a couple of the moods together, turning anger and hate into anger, enthusiasm and fun into joy, and add boredom and empty into sadness  
![a](/plots/revisedCountsOfMoods.png)  
We then create a different training dataset for positive and negative moods, with the negative mood dataset including 1000 rows of worry, sadness, neutral, anger, and surprise while the positive mood dataset includes 2000 rows of neutral, surprise, happiness, joy, and love  
Using these datasets we train two logistic regression models, one for positive moods and one for negative moods.  

We then look into the polarity dataset, which includes 1,600,000 rows, 800,000 which are negative and 800,000 which are positive.  
We train a logistic regression model on this data and get a roc curve that looks like:  
![a](/plots/ROC.png)

When using logistic regression on the polarity data we get an accuracy of about 77%, recall of about 80%, and precision of about 76%.  

After creating both a model to determine the polarity and then 2 models to determine the mood after the polarity is determined, we can run the MoodClassifier.  
The classifier outputs an accuracy and recall of about 35% and precision of about 33%, showing we have successfully doubled the effectiveness of our predictor. If we increase the number of moods it can predict it outputs an accuracy of about 50% when predicting 2 moods.  

We then try running MoodClassifier2 and end up with a confusion matrix like this:
![a](/plots/confusionmtx2.png)

and find that MoodClassifier2 gets an accuracy of around 40% with recall and precision also around 40%, improving our model a little.

To reproduce these results download the data as 'text_emotion.csv' for the emotions and 'text_polarity.csv' for the polarities. Then user should run the python files in this order:
1. Run balanceData.py
2. Run PolarityClassifier.py
3. Run MoodClassifier.py
4. Run createRNN.py
5. Run MoodClassifier2.py with the text you want to predict.
