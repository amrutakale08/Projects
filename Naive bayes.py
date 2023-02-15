#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Split into training and testing data
x = trump_cleaned['review']
y = trump_cleaned['polarity']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


# In[ ]:


# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x, y)


# In[ ]:


model.score(x_test, y_test)

