
# coding: utf-8

# # Predicting sentiment from product reviews
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](/notebooks/Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[ ]:

import graphlab


# In[ ]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[ ]:

products = graphlab.SFrame('amazon_baby.gl/')


# # Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[ ]:

products.head()


# # Build the word count vector for each review

# In[ ]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[ ]:

products.head()


# In[ ]:

graphlab.canvas.set_target('ipynb')


# In[ ]:

products['name'].show()


# # Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[ ]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[ ]:

len(giraffe_reviews)


# In[ ]:

giraffe_reviews['rating'].show(view='Categorical')


# # Build a sentiment classifier

# In[ ]:

products['rating'].show(view='Categorical')


# ## Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[ ]:

# ignore all 3* reviews
products = products[products['rating'] != 3]


# In[ ]:

# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[ ]:

products.head()


# ## Let's train the sentiment classifier

# In[ ]:

train_data,test_data = products.random_split(.8, seed=0)


# In[ ]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# # Evaluate the sentiment model

# In[ ]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[ ]:

sentiment_model.show(view='Evaluation')


# # Applying the learned model to understand sentiment for Giraffe

# In[ ]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[ ]:

giraffe_reviews.head()


# ## Sort the reviews based on the predicted sentiment and explore

# In[ ]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[ ]:

giraffe_reviews.head()


# ## Most positive reviews for the giraffe

# In[ ]:

giraffe_reviews[0]['review']


# In[ ]:

giraffe_reviews[1]['review']


# ## Show most negative reviews for giraffe

# In[ ]:

giraffe_reviews[-1]['review']


# In[ ]:

giraffe_reviews[-2]['review']


# In[ ]:



