#a. Loading data
import timeit
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')


#part 1

#vectorize

vocabulary_size_list=[10,100,1000,10000,50000,100000,None]

print("Vocabulary size that is the max features considered are:",vocabulary_size_list)

accuracyWithOutStopWords=[]
accuracyWithStopWords=[]
vocabularyWithOutStopWords=[]
vocabularyWithStopWords=[]

print("\nTraining Knn model using data without stop words")
for vocab_size in vocabulary_size_list:
    vectorizer = CountVectorizer(max_features=vocab_size,stop_words="english")
    X_train = vectorizer.fit_transform(data_train.data)
    print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}') 
    size=len(vectorizer.get_feature_names_out())
    vocabularyWithOutStopWords.append(size)
    X_test = vectorizer.transform(data_test.data)
    y_train, y_test = data_train.target, data_test.target
    nnclassifier=KNeighborsClassifier(n_neighbors=1)
    nnclassifier=nnclassifier.fit(X_train,y_train) 
    pred=nnclassifier.predict(X_test)
    acc=accuracy_score(y_test,pred)
    accuracyWithOutStopWords.append(acc)
    print("The accuracy of the sklearn KNN classifier",acc,"which is ",round(acc*100,3),"%\n")


print("\nTraining Knn model using data with stop words")
for vocab_size in vocabulary_size_list:
    vectorizer = CountVectorizer(max_features=vocab_size)
    X_train = vectorizer.fit_transform(data_train.data)
    print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}')
    size=len(vectorizer.get_feature_names_out())
    vocabularyWithStopWords.append(size)    
    X_test = vectorizer.transform(data_test.data)
    y_train, y_test = data_train.target, data_test.target
    nnclassifier=KNeighborsClassifier(n_neighbors=1)
    nnclassifier=nnclassifier.fit(X_train,y_train)
    pred=nnclassifier.predict(X_test)
    acc=accuracy_score(y_test,pred)
    accuracyWithStopWords.append(acc)
    print("The accuracy of the sklearn KNN classifier",acc,"which is ",round(acc*100,3),"%\n")

print("When data containing stop words is used for vocabulary sizes 10 and 100, it provides a bit higher accuracy when compared to Knn trained with the same vocabulary size but with stop words removed. But when the vocabulary size is increased by more than 100, knn trained with data with no stop words perform better. And according to the accuracy received, it is good to go with a vocabulary size of 10,000 and the data with stop words removed to get accuracy close to 44.6% with n_neighbour=1. when increasing the vocabulary size in this setting doesn't improve the accuracy much. So, removing stop words plays an important role in improving accuracy.")

#part 2

#plot 1
# Importing the required libraries  
  
# Creating the X and Y dataset  
x = vocabularyWithStopWords
y = arr = np.repeat(accuracyWithStopWords[6], repeats = 7)
    
# Plotting the X and Y data  
plt.plot(x, y, label="model with stop-word in data")  
    
x_ = vocabularyWithOutStopWords
y_ = arr = np.repeat(accuracyWithOutStopWords[6], repeats = 7)
    
# Plotting the x1 and y1 data  
plt.plot(x_, y_, '-.',label="model without stop-word in data")  
    
plt.xlabel("Vocabulary size")  
plt.ylabel("Accuracy score")  
plt.title("Vocabulary vs Accuracy plot for full vocabulary")  
leg = plt.legend(loc='center')
plt.show()  


#plot 2

# Importing the required libraries  
import matplotlib.pyplot as plt  
import numpy as np  
  
# Creating the X and Y dataset  
x = vocabularyWithStopWords
y = accuracyWithStopWords
    
# Plotting the X and Y data  
plt.plot(x, y, label="model with stop-word in data")  
    
x_ = vocabularyWithOutStopWords
y_ = accuracyWithOutStopWords
    
# Plotting the x1 and y1 data  
plt.plot(x_, y_, '-.',label="model without stop-word in data")  
    
plt.xlabel("Vocabulary size")  
plt.ylabel("Accuracy score")  
plt.title("Vocabulary vs accuracy plot")  
leg = plt.legend(loc='lower right')
plt.show()  


#part 3

# tf-idf transform
print("Training KNN Model with TF-IDF transformer")
vocabulary_size_list=[10,100,1000,10000,100000,None]

print("Vocabulary size that is the max features considered are:",vocabulary_size_list)

tfidfaccuracyWithOutStopWords=[]
tfidfaccuracyWithStopWords=[]
tfidfvocabularySizeWithOutStopWords=[]
tfidfvocabularySizeWithStopWords=[]

print("Using data without stop words\n")
for vocab_size in vocabulary_size_list:
    vectorizer = CountVectorizer(max_features=vocab_size,stop_words="english")
    X_train_v1 = vectorizer.fit_transform(data_train.data)
    X_test_v1 = vectorizer.transform(data_test.data)
    size=len(vectorizer.get_feature_names_out())
    tfidfvocabularySizeWithOutStopWords.append(size)
    #tfidf transformer
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train_v1).toarray()
    X_test = tfidf_transformer.transform(X_test_v1).toarray()
    
    y_train, y_test = data_train.target, data_test.target
    nnclassifier=KNeighborsClassifier(n_neighbors=1)
    nnclassifier=nnclassifier.fit(X_train,y_train) 
    pred=nnclassifier.predict(X_test)
    acc=accuracy_score(y_test,pred)
    tfidfaccuracyWithOutStopWords.append(acc)
    print("The accuracy of the sklearn KNN classifier",acc,"which is ",round(acc*100,3),"% with vocabulary size ",{len(vectorizer.get_feature_names_out())},"\n")
    
print("Using data with stop words\n")
for vocab_size in vocabulary_size_list:
    vectorizer = CountVectorizer(max_features=vocab_size)
    X_train = vectorizer.fit_transform(data_train.data)  
    X_test = vectorizer.transform(data_test.data)
    size=len(vectorizer.get_feature_names_out())
    tfidfvocabularySizeWithStopWords.append(size)
    
    #tfidf transformer
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train).toarray()
    X_test = tfidf_transformer.transform(X_test).toarray()
    
    y_train, y_test = data_train.target, data_test.target
    nnclassifier=KNeighborsClassifier(n_neighbors=1)
    nnclassifier=nnclassifier.fit(X_train,y_train)
    pred=nnclassifier.predict(X_test)
    acc=accuracy_score(y_test,pred)
    tfidfaccuracyWithStopWords.append(acc)
    print("The accuracy of the sklearn KNN classifier",acc,"which is ",round(acc*100,3),"% with vocabulary size ",{len(vectorizer.get_feature_names_out())},"\n")
    
#plot tf-idf
# Importing the required libraries  
import matplotlib.pyplot as plt  
import numpy as np  
  
# Creating the X and Y dataset  
x = tfidfvocabularySizeWithStopWords
y = tfidfaccuracyWithStopWords
    
# Plotting the X and Y data  
plt.plot(x, y, label="model with tf-idf transformer with stop-word in data")  
    
x_ = tfidfvocabularySizeWithOutStopWords
y_ = tfidfaccuracyWithOutStopWords
    
# Plotting the x1 and y1 data  
plt.plot(x_, y_, '-.',label="model with tf-idf transformer without stop-word in data")  
    
plt.xlabel("Vocabulary size")  
plt.ylabel("Accuracy score")  
plt.title("Vocabulary vs Accuracy plot with tf-idf transformer")  
leg = plt.legend(loc='lower right')
plt.show()  

print("When the tf-idf transformer is used after the count vectorizer, it performs more or less similarly with data containing stop words vs. without stop words. In the final graph we can see the 3 model accuracy is similar and the lines overlap. But, when considering the accuracy it performs better than the model without tfidf transformer.")