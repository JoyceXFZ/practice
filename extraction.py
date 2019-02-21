# feature extraction 

measurements = [{'citi': 'Dubai', 'temp': 33.}, {'citi': 'London', 'temp': 12.}, {'citi': 'San Francisco', 'temp': 18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
#print(vec.fit_transform(measurements).toarray())
#print(vec.get_feature_names())

pos_window = [{'word-2' : 'the', 'pos-2' : 'DT', 'word-1' : 'cat', 'pos-1' : 'NN', 'word+1' : 'on', 'pos+1' : 'PP'}]
vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)
#print pos_vectorized
#print(pos_vectorized.toarray())
#print(vec.get_feature_names())

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
#print(vectorizer)

corpus = ['This is the first document.', 'This is the second document.', 'And the third one', 'Is this the first document?']
X = vectorizer.fit_transform(corpus)
#print(X)
analyze = vectorizer.build_analyzer()
#print(analyze("This is a text document to analyze.") == (['this', 'is', 'text', 'document', 'to', 'analyze']))
#print(vectorizer.get_feature_names() == (['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']))
#print(X.toarray())
#print(vectorizer.vocabulary_.get('document'))
#print(vectorizer.transform(['Something completely new.']).toarray())

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print(analyze('Bi-grams are cool') == (['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']))
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
#print(X_2)

feature_index = bigram_vectorizer.vocabulary_.get('is this')
#print(X_2[:, feature_index])

from sklearn.feature_extraction.text import TfidfTransformer
#transformer = TfidfTransformer(smooth_idf = False)
transformer = TfidfTransformer()
#print(transformer)
counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0], [4, 0, 0], [3, 2, 0], [3, 0, 2]]
tfidf = transformer.fit_transform(counts)
#print(tfidf)
#print(tfidf.toarray())

def my_tokenizer(s):
	return s.split()
vectorizer = CountVectorizer(tokenizer = my_tokenizer)
#print(vectorizer.build_analyzer()(u'Some...punctuation!') == (['Some...', 'punctuation!']))


#text tokenization 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
	def __init__ (self):
		self.wnl = WordNetLemmatizer()
	def __call__ (self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vect = CountVectorizer(tokenizer=LemmaTokenizer())
print(vect)

import re
def to_british(tokens):
	for t in tokens:
		t = re.sub(r"(...)our$", r"\1or", t)
		t = re.sub(r"([bt])re$", r"\1er", t)
		t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
		t = re.sub(r"ogue$", "og", t)
		yield t

class CustomVectorizer(CountVectorizer):
	def build_tokenizer(self):
		tokenize = super(CustomVectorizer, self).build_tokenizer()
		return lambda doc: list(to_british(tokenize(doc)))

print(CustomVectorizer().build_analyzer()(u"color colour"))



