import pandas as pd
import math
import os
from sys import platform

df=pd.read_csv(r'hotelReviews.csv')
df2 = df[df['Rating'] != 3] 
df2.loc[df['Rating'] > 3, 'Rating'] = 'Positive'
df2.loc[df['Rating'] < 3, 'Rating'] = 'Negative'
df2 = df2[~df2["Review"].str.contains('not', na=False)]
df2 = df2[~df2["Review"].str.contains('n\'t', na=False)]
words=['amazing', 'impressive', 'marvelous', 'astounding', 'awesome', 
       'quick', 'wonderful', 'incredible', 'good', 'great', 'neat', 'positive', 
       'nice', 'welcome', 'clean', 'beautiful', 'helpful', 'friendly', 'unique', 
       'creative', 'delightful', 'excellent', 'exciting', 'fabulous', 'fantastic', 
       'fresh', 'pleasant', 'substandard', 'bad', 'poor', 'cheap', 'rough', 'noisy', 
       'atrocious', 'hated', 'yuck', 'terrible', 'awful', 'dirty', 'unacceptable', 
       'ugly', 'abysmal', 'alarming', 'angry', 'annoy', 'appalling', 'rude', 'broken', 
       'damaged', 'horrible', 'bugs', 'nasty', 'mold', 'moldy', 'offensive', 'rotten']

def trainModelAndPredict(data, trainingPercent, inputStr):
    wordsToRemove=[]
    wordFrequencyPositive={}
    wordFrequencyNegative={}
    positiveProb=1
    negativeProb=1
    wordFrequencyPositiveCount=0
    wordFrequencyNegativeCount=0
    rowsForTraining = math.floor(len(data)*(trainingPercent/100))
    dfTrain=data.head(rowsForTraining)
    rowsForTesting = len(data)-rowsForTraining
    dfTest=data.tail(rowsForTesting)
    positiveRows=dfTrain[dfTrain['Rating'] == 'Positive']
    negativeRows=dfTrain[dfTrain['Rating'] == 'Negative']
    probabilityOfYes=len(positiveRows)/len(dfTrain)
    probabilityOfNo=len(negativeRows)/len(dfTrain)
    for i in words:
        for m in positiveRows.Review:
            if i in m:
                wordFrequencyPositiveCount+=1
                wordFrequencyPositive[i]=(wordFrequencyPositiveCount/len(positiveRows))
        wordFrequencyPositiveCount=0
    for i in words:
        for m in negativeRows.Review:
            if i in m:
                wordFrequencyNegativeCount+=1
        wordFrequencyNegative[i]=(wordFrequencyNegativeCount/len(negativeRows))
        wordFrequencyNegativeCount=0
    for i in words:
        if i not in wordFrequencyPositive or i not in wordFrequencyNegative:
            wordsToRemove.append(i)
    updatedWords = [word for word in words if word not in wordsToRemove]
    index=0
    positiveProb=1
    negativeProb=1
    for m in updatedWords:
        if " " + m + " " in inputStr:
            positiveProb*=wordFrequencyPositive.get(m)
        else:
            positiveProb*=1-wordFrequencyPositive.get(m)
        if " " + m + " " in i:
            negativeProb*=wordFrequencyNegative.get(m)
        else:
            negativeProb*=1-wordFrequencyNegative.get(m)
    positiveProb = positiveProb* probabilityOfYes      
    negativeProb = negativeProb* probabilityOfNo 
    if positiveProb > negativeProb:
        return 'positive'
    elif positiveProb < negativeProb:
        return 'negative'

if 'win' in platform:
	os.system('cls')
else:
	os.system('clear')

print('This program predict if a given hotel reviews are positive or negative')

while True:
	inputStr = input("Enter your review (or type exit to exit): ")
	if inputStr == 'exit':
		break
	result=trainModelAndPredict(df2, 80, inputStr)
	print('Your review is predicted to be '+ result)