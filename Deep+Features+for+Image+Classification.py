
# coding: utf-8

# # Using deep features to build an image classifier
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[ ]:

import graphlab


# In[ ]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load a common image analysis dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set.  

# In[ ]:

image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')


# # Exploring the image data

# In[ ]:

graphlab.canvas.set_target('ipynb')


# In[ ]:

image_train['image'].show()


# # Train a classifier on the raw image pixels
# 
# We first start by training a classifier on just the raw pixels of the image.

# In[ ]:

raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',
                                              features=['image_array'])


# # Make a prediction with the simple model based on raw pixels

# In[ ]:

image_test[0:3]['image'].show()


# In[ ]:

image_test[0:3]['label']


# In[ ]:

raw_pixel_model.predict(image_test[0:3])


# The model makes wrong predictions for all three images.

# # Evaluating raw pixel model on test data

# In[ ]:

raw_pixel_model.evaluate(image_test)


# The accuracy of this model is poor, getting only about 46% accuracy.

# # Can we improve the model using deep features
# 
# We only have 2005 data points, so it is not possible to train a deep neural network effectively with so little data.  Instead, we will use transfer learning: using deep features trained on the full ImageNet dataset, we will train a simple model on this small dataset.

# In[ ]:

len(image_train)


# ## Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)

# In[ ]:

# deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# As we can see, the column deep_features already contains the pre-computed deep features for this data. 

# In[ ]:

image_train.head()


# # Given the deep features, let's train a classifier

# In[ ]:

deep_features_model = graphlab.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')


# # Apply the deep features model to first few images of test set

# In[ ]:

image_test[0:3]['image'].show()


# In[ ]:

deep_features_model.predict(image_test[0:3])


# The classifier with deep features gets all of these images right!

# # Compute test_data accuracy of deep_features_model
# 
# As we can see, deep features provide us with significantly better accuracy (about 78%)

# In[ ]:

deep_features_model.evaluate(image_test)

