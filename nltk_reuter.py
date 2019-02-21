#NLTK reuters
import codecs
import nltk
from nltk.corpus import udhr
import matplotlib.pyplot as plt

#languages = ['English', 'French', 'Italian', 'German_Deutsch', 'Russian', 'Arabic', 'Japanese', 'Chinese', 'Korean', 'Mongolian']
languages = ['English', 'German_Deutsch', 'Spanish', 'Italian', 'Hungarian_Magyar']
cfd = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
plt.show()