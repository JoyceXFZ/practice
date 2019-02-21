#tweets airline sentiment analysis

import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from nltk.util import ngrams
import nltk 

df = pd.read_csv('c:/python/Tweets_NAg.csv')
df = df[[u'airline_sentiment', u'text']]
df.loc[:, 'text'] = df.loc[:, 'text'].map(clean_tweet)

udf = pd.read_csv('c:/python/Tweets_Unlabeled.csv')
udf = udf[[u'text']]
udf.loc[:, 'text'] = udf.loc[:,'text'].map(clean_tweet)

TotalNum = df.size/2
TotalNum_Unlabed = udf.size
TestNum = 3000
TrainNum = TotalNum - TestNum

generated_ngrams = ngrams(['TEXT a','TEXT b','TEXT c','TEXT d'],3,pad_left=True,pad_right=True,left_pad_symbol='<s>',right_pad_symbol='</s>')
n4grams=3
Probdist = nltk.KneserNeyProbDist
accus=[]
Gmeans=[]

for iter in range(0,10):
	import random
	random.seed(1212+iter)
	newindex = random.sample(range(0,TotalNum),TotalNum)
	testID = newindex[-TestNum:]
	trainID = newindex[:-TestNum]
	trainID_p = [id for id in trainID if df.loc[id,u'airline_sentiment']=='positive']
	trainID_neg = [id for id in trainID if df.loc[id,u'airline_sentiment']=='negative']
	trainID_neu = [id for id in trainID if df.loc[id,u'airline_sentiment']=='neutral']
	alllist = []

	for i in trainID_p:
		generated_ngrams = ngrams(df.loc[i,'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
		alllist = alllist+list(generated_ngrams)
	freq_dist = nltk.FreqDist(alllist)
	Dist_p = Probdist(freq_dist,1)
	alllist = []
	
	for i in trainID_neg:
		generated_ngrams = ngrams(df.loc[i,'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
	alllist = alllist+list(generated_ngrams)
	freq_dist = nltk.FreqDist(alllist)
	Dist_neg = Probdist(freq_dist,1)
	alllist = []
	for i in trainID_neu:
        generated_ngrams = ngrams(df.loc[i,'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
        alllist = alllist+list(generated_ngrams)
    freq_dist = nltk.FreqDist(alllist)
    Dist_neu = Probdist(freq_dist,1)
    predictLabels=[]
    for i in range(0,TestNum):
		generated_ngrams = ngrams(df.loc[testID[i],'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
		prob_sum_p = 0
		for k in generated_ngrams:
			prob_sum_p += Dist_p.prob(k) 
		generated_ngrams = ngrams(df.loc[testID[i],'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
		prob_sum_neg = 0
		for k in generated_ngrams:
			prob_sum_neg += Dist_neg.prob(k) 
		generated_ngrams = ngrams(df.loc[testID[i],'text'], n4grams, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
		prob_sum_neu = 0
		for k in generated_ngrams:
			prob_sum_neu += Dist_neu.prob(k) 
		if(prob_sum_p>prob_sum_neu and prob_sum_p>prob_sum_neg):
			predictLabels = predictLabels+['positive']
		else:
			if(prob_sum_neg>prob_sum_neu and prob_sum_neg>prob_sum_p):
				predictLabels = predictLabels+['negative']
			else:
				predictLabels = predictLabels+['neutral']
	
	accu=0
	for i in range(0,TestNum):
		if predictLabels[i]==df.loc[testID[i],u'airline_sentiment']:
			accu=accu+1
    accus=accus+[1.0*accu/100]
    predictLabels = predictLabels
    confusionM = confusion_matrix(predictLabels,(df.loc[testID,u'airline_sentiment']))
    Gmeans=Gmeans+[pow(((1.0*confusionM[0,0]/(confusionM[1,0]+confusionM[2,0]+confusionM[0,0]))*(1.0*confusionM[1,1]/(confusionM[1,1]+confusionM[2,1]+confusionM[0,1]))*(1.0*confusionM[2,2]/(confusionM[1,2]+confusionM[2,2]+confusionM[0,2]))), 1.0/3)]

print confusionM
print accus, Gmeans

 