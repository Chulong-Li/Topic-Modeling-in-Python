import re
import json
import random
from timeit import default_timer as timer
from pprint import pprint as pprint

from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
import nltk.stem.wordnet
import string
from gensim import corpora, models
import gensim

start = timer()

with open('cinci_data_bdaa.json') as f:
    data = json.load(f)
        
print(timer() - start)

key_set = set()

for post in data: 
    temp_keys = post.keys()
    key_set = key_set.union(temp_keys)
    
pprint(key_set)

type_set = set()

for post in data: 
    post_type = post['type']
    type_set = type_set.union([post_type]) #make sure to include the brackets
    
pprint(type_set)

#pare down the size of the data 
test_data = [data[i] for i in random.sample(xrange(len(data)), 10000)]


#select only the relevant information 

documents = []
for post in test_data: 
    str_list = []
    if post.has_key('caption'): 
        str_list.append(post['caption'])
    if post.has_key('lname'): 
        str_list.append(post['lname'])
    if post.has_key('description'): 
        str_list.append(post['description'])
    if post.has_key('types'): 
        str_list.append(str(post['types'])) #don't forget to include str here! 
    documents.append(' '.join(str_list))
    
pprint(documents[0:3])



stop_words = get_stop_words('en')
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

cleaned_documents = []

for doc in documents:
    doc = doc.lower() #lowercase
    doc = re.sub("http(.*?) ",' ',doc) #remove almost all links
    doc = re.sub("http(.*)",' ',doc) #remove links that came at the end of a doc
    doc = re.sub("u'",'',doc) #remove all the unicode identifiers
    doc = re.sub(r'[{}]'.format(string.punctuation)," ",doc) #remove punctuation
    doc = re.sub('[•\t\n\r\f\v]', ' ', doc) #remove newline characters
    doc = re.sub("  +"," ",doc) #remove multiple spaces
    doc = re.sub(" [a-z0-9] ","",doc) #remove single letter/digit words
    
    tokens = tokenizer.tokenize(doc) 
    stopped_tokens = [i for i in tokens if not i in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stopped_tokens] 
    non_numeric_tokens = [token for token in lemmatized_tokens if not token.isdigit()] #remove just numbers
    longer_than_1_tokens = [token for token in non_numeric_tokens if len(token) > 1] #docs must have >1 word
    cleaned_documents.append(longer_than_1_tokens)

                              

dictionary = corpora.Dictionary(cleaned_documents)
dictionary.filter_extremes(no_below=10, no_above=0.2)
corpus = [dictionary.doc2bow(doc) for doc in cleaned_documents]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))




chunksize = 5000
passes_n = 20
num_topics = 20

start = timer()
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=passes_n,
                                           chunksize = chunksize, alpha = 'auto', eta = 'auto', )

print("time to finish model with {} topics: {}".format(i, timer() - start))

top_topics = ldamodel.top_topics(corpus)
pprint(top_topics)