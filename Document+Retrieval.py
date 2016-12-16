
# coding: utf-8

# # Document retrieval from wikipedia data

# ## Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[ ]:

import graphlab


# In[ ]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load some text data - from wikipedia, pages on people

# In[ ]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[ ]:

people.head()


# In[ ]:

len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for president Obama

# In[ ]:

obama = people[people['name'] == 'Barack Obama']


# In[ ]:

obama


# In[ ]:

obama['text']


# ## Exploring the entry for actor George Clooney

# In[ ]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# # Get the word counts for Obama article

# In[ ]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[ ]:

print obama['word_count']


# ## Sort the word counts for the Obama article

# ### Turning dictonary of word counts into a table

# In[ ]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[ ]:

obama_word_count_table.head()


# In[ ]:

obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[ ]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[ ]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[ ]:

people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Obama article

# In[ ]:

obama = people[people['name'] == 'Barack Obama']


# In[ ]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[ ]:

clinton = people[people['name'] == 'Bill Clinton']


# In[ ]:

beckham = people[people['name'] == 'David Beckham']


# ## Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[ ]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[ ]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[ ]:

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# # Applying the nearest-neighbors model for retrieval

# ## Who is closest to Obama?

# In[ ]:

knn_model.query(obama)


# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  

# ## Other examples of document retrieval

# In[ ]:

swift = people[people['name'] == 'Taylor Swift']


# In[ ]:

knn_model.query(swift)


# In[ ]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[ ]:

knn_model.query(jolie)


# In[ ]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[ ]:

knn_model.query(arnold)

