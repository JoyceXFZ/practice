#tweets airline sentiment analysis

import ReadACleanT
from ReadACleanT import clean_tweet
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
 
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

print TotalNum, TestNum, TrainNum, TotalNum_Unlabed

#constructing Doc2Vec model
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

TotalNum = df.size/2
TotalNum_Unlabed = udf.size
TestNum = 3000
TrainNum = TotalNum - TestNum

documents = [TaggedDocument(list(df.loc[i, 'text']), [i]) for i in range (0, TotalNum)]
documents_unlabeled = [TaggedDocument(list(udf.loc[i, 'text']), [i+TotalNum]) for i in range(0, TotalNum_Unlabed)]
documents_all = documents + documents_unlabeled

Doc2VecTrainID = range(0, TotalNum+TotalNum_Unlabed)
random.shuffle(Doc2VecTrainID)
trainDoc = [documents_all[id] for id in Doc2VecTrainID]
Labels = df.loc[:, 'airline_sentiment']

import multiprocessing 
cores = multiprocessing.cpu_count()

model_DM = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores, dm=1, dm_concat=1)
model_DBOW = Doc2Vec(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=cores, dm=0)


#build and train the models 
model_DM.build_vocab(trainDoc)
model_DBOW.build_vocab(trainDoc)

for it in range(1, 10):
	random.shuffle(Doc2VecTrainID)
	trainDoc = [documents_all[id] for id in Doc2VecTrainID]
	model_DM.train(trainDoc, total_examples=model_DM.corpus_count, epochs=model_DM.epochs)
	model_DBOW.train(trainDoc, total_examples=model_DBOW.corpus_count, epochs=model_DBOW.epochs)
	
	
#build the classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

random.seed(1212)
newindex = random.sample(range(0, TotalNum), TotalNum)
testID = newindex[-TestNum:]
trainID = newindex[:-TestNum]

train_targets, train_regressors = zip(*[(Labels[id], list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id])) for id in trainID])
train_regressors = sm.add_constant(train_regressors)
predictor = LogisticRegression(multi_class='multinomial', solver='lbfgs')
predictor.fit(train_regressors, train_targets)


accus1 = []
Gmeans1 = []
test_regressors = [list(model_DM.docvecs[id]) + list(model_DBOW.docvecs[id]) for id in testID]
test_regressors = sm.add_constant(test_regressors)
test_predictions = predictor.predict(test_regressors)
accu1 = 0
for i in range(0, TestNum):
	if test_predictions[i] == df.loc[testID[i], u'airline_sentiment']:
		accu1 = accu1 + 1
accus1 = accus1 + [1.0 * accu1/TestNum]
confusionM = confusion_matrix(test_predictions, (df.loc[testID, u'airline_sentiment']))	
Gmeans1 = Gmeans1+[pow(((1.0*confusionM[0,0]/(confusionM[1,0]+confusionM[2,0]+confusionM[0,0]))*(1.0*confusionM[1,1]/(confusionM[1,1]+confusionM[2,1]+confusionM[0,1]))*(1.0*confusionM[2,2]/(confusionM[1,2]+confusionM[2,2]+confusionM[0,2]))), 1.0/3)]
print confusionM
print Gmeans1, accus1

accus2 = []
Gmeans2 = []
train_predictions = predictor.predict(train_regressors)
accu2 = 0
for i in range(0, len(train_targets)):
	if train_predictions[i] == train_targets[i]:
		accu2 = accu2 + 1
accus2 = accus2 + [1.0 * accu2/len(train_targets)]

confusionM2 = confusion_matrix(train_predictions, train_targets)	
Gmeans2 = Gmeans2 + [pow(((1.0*confusionM2[0,0]/(confusionM2[1,0]+confusionM2[2,0]+confusionM2[0,0]))*(1.0*confusionM2[1,1]/(confusionM2[1,1]+confusionM2[2,1]+confusionM2[0,1]))*(1.0*confusionM2[2,2]/(confusionM2[1,2]+confusionM2[2,2]+confusionM2[0,2]))), 1.0/3)]
print confusionM2
print Gmeans2, accus2

