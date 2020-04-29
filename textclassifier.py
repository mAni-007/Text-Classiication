import numpy as np 
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')


#Importing the dataset
reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target


#Load_filesfunction(above) generally takes lot of time if working with huge dataset
#So using of pickling method which store file in byte, helps a lot
#Calls as Persistent of Data_sets

#now we have store in byte as a pickle
with open('X.pickle', 'wb') as f:
	pickle.dump(X,f)

with open('y.pickle', 'wb') as f:
	pickle.dump(y,f)
#to read it, we unpickle it
with open('X.pickle', 'rb') as f:
	X = pickle.load(f)
with open('y.pickle', 'rb') as f:
	y = pickle.load(f)
#-------------------------------------------

#preprocessing the data

corpus = []
for i in range(0,len(X)):
	review = re.sub(r'\W',' ',str(X[i]))
	review = re.sub(r'\s+[a-z]\s+',' ',review)
	review = re.sub(r'\[a-z]\s+',' ',review)
	review = re.sub(r'\s+',' ', review)
	review = review.lower()
	corpus.append(review)


#Creating the model...Using technique BOW(bag of words)


#max_feature  = will select the only 2000 words, min_df = will remove all those word which occure less than 3 or equal to 3
#max_df  = it is in percentage so word occure more than 60% of the time will be removed such as the,in,a,for all of it
from sklearn.feature_extraction.text import CounVectorizer
vectorizer = CounVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
x = vectorizer.fit_transform(corpus).toarray()
print x

#based on the coefi=fie
#Converting the model into TF-IDF model(much better than BOW model)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
x = transformer.fit_transform(x).toarray()




#Divinding the dataset into two part one for training and other for testing the dataset
#out of 2000 dataset inclduing posi and neg

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = testrain_test_split(x,y,test_size = 0.2, random_state = 0)



#Logistic regression :it calculate the value of the coefficient
#based on the coefficient new sentene are given points,if greater than 0.5 it is posi ortherwise neg

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)


#time the check the model performance
#confusion matrix form a matrix coulmn as true value and and row as predicted value,


sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)
accuracy = (cm[0][0]+ cm[1][1])*(100/4)


sample = ['You are a nice person man, have a good day']
sample = vectorizer.transform(sample).toarray()
print(classifier.predict())