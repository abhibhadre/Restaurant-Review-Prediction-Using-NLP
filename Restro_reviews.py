import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t')

features = dataset.iloc[:,0]
labels = dataset.iloc[:,1]


#data cleaning activity

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#lets take the first review for cleanup
print (features[0])
dataset['Review'][0]

#a - z or A - Z

import re

# search for special character set, using re module
# not belonging to a - z or A - Z

review = re.sub('[^a-zA-Z]', ' ',  dataset['Review'][0])

review = review.lower()

review = review.split()

#remove the stopwords
review = [word for word in review if not word in stopwords.words('english')]

#stemming
ps  = PorterStemmer()

review = [ps.stem(word) for word in review]

review = " ".join(review)


corpus = []


for i in range(0, 1000):

    review = re.sub('[^a-zA-Z]', ' ',  dataset['Review'][i])

    review = review.lower()
    
    review = review.split()
    
    #remove the stopwords
    review = [word for word in review if not word in stopwords.words('english')]
    
    #stemming
    ps  = PorterStemmer()
    
    review = [ps.stem(word) for word in review]
    
    review = " ".join(review)
    
    corpus.append(review)


#Tfidf vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

#min_df=5 means "ignore terms/words that appear in less than 5 documents

tfidfv = TfidfVectorizer(min_df = 5)

features = tfidfv.fit_transform(corpus).toarray()


print (tfidfv.get_feature_names())

print (len(tfidfv.get_feature_names()))

#train test split

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

#classification
#svc, knn, nb, logistic regression

#nb based model

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(features_train,labels_train)

print (model_nb.score(features_test, labels_test))

# knn based model
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier()
model_knn.fit(features_train, labels_train)

print (model_knn.score(features_test, labels_test))


#lr based model
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(features_train, labels_train)

print (model_lr.score(features_test, labels_test))


#for a single review


reviewText = 'i lovee this restaurant'

reviewText = tfidfv.transform([reviewText]).toarray()

model_nb.predict(reviewText)




















