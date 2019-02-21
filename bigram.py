#bigram NLP

import nltk
import csv
import string 
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords

sentences = []
bigram = Phrases()
with open('C:/python/sentences.csv', 'r') as sentencesfile:
	reader = csv.reader(sentencesfile, delimiter = ',')
	next(reader)
	for row in reader:
		sentence  = [word for word in nltk.word_tokenize(row[4].lower()) if word not in string.punctuation]
		
		sentences.append(sentence)
		bigram.add_vocab([sentence])

print (list(bigram[sentence])[:5])