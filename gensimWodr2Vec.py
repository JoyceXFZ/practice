# getting started with gensim
import gensim
'''
raw_corpus = ["Human machine interface for lab abc computer applications",
              "A survey of user computer system response time",
			  "The EPS user interface management system",
			  "Relation of user perceived response time to error measurement",
			  "The generation of random binary unordreed trees",
			  "The intersection graph of paths in trees",
			  "Graph minors IV Widths of trees and well quasi ordering",
			  "Graph minors A survey"]
			  
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in raw_corpus]	

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

processed_corpora = [[token for token in text if frequency[token]>1] for text in texts]

from gensim import corpora 
dictionary = corpora.Dictionary(processed_corpora)
#print dictionary
#print dictionary.token2id
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpora]
from gensim import models 
tfidf = models.TfidfModel(bow_corpus)
print tfidf[dictionary.doc2bow("system minors".lower().split())]
'''

#/1
#Word2Vec tutorial
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
model = gensim.models.Word2Vec(min_count=1)

#print (model)
#print (model.wv.vocab)

#/2
#calling word2vec, it will run 2 passes over the sentences 
#build the same model as above 
new_model = gensim.models.Word2Vec(min_count=1)
new_model.build_vocab(sentences)
new_model.train(sentences, total_examples=new_model.corpus_count, epochs=new_model.iter)
#print (new_model)
#print (model.wv.vocab)

#/3 following examples would use the lee corpus 
import os 
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'

class MyText(object):
	def __iter__(self):
		for line in open(lee_train_file):
			yield line.lower().split()
sentences = MyText()

print (sentences)

#model = gensim.models.Word2Vec(sentences, min_count=200)
model = gensim.models.Word2Vec(sentences, workers=1)
model.accuracy('c:/python/questions-words.txt')

#storing and loading models
from tempfile import mkstemp

fs, temp_path = mkstemp("gensim_temp")
model.save(temp_path)

#resuming training 
model = gensim.models.Word2Vec.load(temp_path)
more_sentenses = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentenses, update=True)
model.train(more_sentenses, total_examples=model.corpus_count, epochs=model.iter)
os.close(fs)
os.remove(temp_path)

#use the model
model.most_similar(positive=['human', 'crime'], negative=['party'], topn=1)
model.doesnt_match("input is lunch he sentence cat".split())
print (model.similarity('human', 'party'))
print (model.similarity('tree', 'murder'))
print (model.predict_output_word(['emergency', 'beacon', 'received']))
print (model['tree'])
model_with_loss = gensim.models.Word2Vec(sentences, min_count=1, compute_loss=True, hs=0, sg=1, seed=42)
training_loss = model_with_loss.get_latest_training_loss()
print (training_loss)


from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

def reduce_dimensions(model, plot_in_notebook=True):
	num_dimensions = 2
	
	vectors = []
	labels = []
	for word in model.wv.vocab:
		vectors.append(model[word])
		labels.append(word)
		
	vectors = np.asarray(vectors)
	labels = np.asarray(labels)
	
	vectors = np.asarray(vectors)
	logging.info("starting tSNE dimensionality reduction. this may take some time")
	tsne = TSNE(n_components = num_dimensions, random_state=0)
	vectors = tsne.fit_transform(vectors)
	
	x_vals = [v[0] for v in vectors]
	y_vals = [v[1] for v in vectors]
	
	trace = go.Scatter(
		x = x_vals,
		y = y_vals,
		mode = 'text',
		text = labels)
		
	data = [trace]
	
	logging.info('ALl done, plotting')
	
	if plot_in_notebook:
		init_notebook_mode(connected=True)
		iplot(data, filename='word_embedding-plot')
	else:
		plot(data, filename='word_embedding-plot.html')

reduce_dimensions(model)