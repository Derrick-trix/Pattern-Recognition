#a. Loading data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import timeit
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import statistics
from statistics import mode

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')
#vectorize
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}')
X_test = vectorizer.transform(data_test.data)
y_train, y_test = data_train.target, data_test.target
print(X_test.shape)
print(y_test.shape)

#b. Dummy classifier 
clf=DummyClassifier()
clf=clf.fit(X_train,y_train)
pred=clf.predict(X_test)
#most frequent class approach
a=mode(y_train)
predMostFrequent=[a for i in range(1,7533)] 

#c. timeit - default_timer()
startTime=timeit.default_timer()
pred=clf.predict(X_test)
time_taken=timeit.default_timer()-startTime
print("Time taken for baseline to predict:",time_taken)

#d. Accuracy for baseline
acc=accuracy_score(y_test,pred)
print("The accuracy of the baseline model is:",acc,"which is ",round(acc*100,3),"%")

#e and f. simple nearest neighbor approach
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

nn_predictions = []

X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()[:500]
y_test2 = y_test[:500]

start_time2 = timeit.default_timer()
nearest_neighbor_indices = np.zeros(X_test_dense.shape[0], dtype=int)
min_distances = np.zeros(X_test_dense.shape[0])

for i, x_test in enumerate(X_test_dense):
    distances = np.linalg.norm(X_train_dense - x_test, axis=1)

    nearest_neighbor_indices[i] = np.argmin(distances)
    min_distances[i] = distances[nearest_neighbor_indices[i]]

nn_predictions = y_train[nearest_neighbor_indices]
end_time2 = timeit.default_timer()

accuracy2 = accuracy_score(y_test2, nn_predictions)
computation_time2 = end_time2 - start_time2

print("own NN Classifier Accuracy:", round(accuracy2*100,3),"%")
print("Test data set considered = 500")
print("Computation time for own NN Classifier:", computation_time2)


#g.Sklearn NN
#h.Sklearn NN classifier accuracy and computation time
nnclassifier=KNeighborsClassifier(n_neighbors=1)
nnclassifier=nnclassifier.fit(X_train,y_train)

startTime=timeit.default_timer()
pred=nnclassifier.predict(X_test)
time_taken=timeit.default_timer()-startTime


print("Time taken for sklearn KNN classifier:",time_taken)
acc=accuracy_score(y_test,pred)
print("The accuracy of the sklearn KNN classifier with 1 NN:",acc,"which is ",round(acc*100,3),"%")