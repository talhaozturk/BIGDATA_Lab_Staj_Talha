from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import os
import string
import numpy as np
import sklearn.datasets

path = 'data\\'
data = []
file_names = np.array([])
target_names = ["ekonomi","magazin","saglik","siyasi","spor"]
target = np.array([], dtype=np.int64)
x = -1

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        #target_names.append(subdir[5:])
        shakes = open(file_path, 'r')
        text = shakes.read()
        text = ''.join(ch for ch in text if ch not in string.punctuation).replace('\n', '')
        data.append(text)
        file_names = np.append(file_names,file)
        if subdir[5:] == "ekonomi":
            x = 0
        elif subdir[5:] == "magazin":
            x = 1
        elif subdir[5:] == "saglik":
            x = 2
        elif subdir[5:] == "siyasi":
            x = 3
        else:
            x = 4
        target = np.append(target,x)

# 1150haber dataset
dataset = sklearn.datasets.base.Bunch(data=data, filenames = file_names, target_names= target_names, target=target)

# class labels
y_data = dataset.target

# binary counting
binary_vectorizer = CountVectorizer(binary=True)    #binary
X_data_binary = binary_vectorizer.fit_transform(dataset.data)

# term frequency
tf_vectorizer = CountVectorizer()  #tf
X_data_tf = tf_vectorizer.fit_transform(dataset.data)

# term frequency - inverse document frequency
tfidf_transformer = TfidfTransformer()  #td-idf
X_data_tfidf = tfidf_transformer.fit_transform(X_data_tf)

# data split
#X_train, X_test, y_train, y_test = train_test_split(X_data_binary, y_data, test_size=0.4)  # default test size = 0.25, binary
#X_train, X_test, y_train, y_test = train_test_split(X_data_tf, y_data, test_size=0.4)  # default test size = 0.25, tf
#X_train, X_test, y_train, y_test = train_test_split(X_data_tfidf, y_data, test_size=0.4)  # default test size = 0.25, tfidf

# 10 fold cross validation
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X_data_tf)
print(kf)
for train_index, test_index in kf.split(X_data_tf):
   X_train, X_test = X_data_tf[train_index], X_data_tf[test_index]
   y_train, y_test = y_data[train_index], y_data[test_index]

#classfier = svm.LinearSVC.fit_transform(X_train, y_train)
classifier_MNB = MultinomialNB().fit(X_train, y_train)
predicted_multinomialNB = classifier_MNB.predict(X_test)
#print(np.mean(predicted_multinomialNB == y_test))

acc = accuracy_score(y_test, predicted_multinomialNB, normalize=False) ###10 avg mi en iyisi mi
print("accuracy: ", round(acc,3))

print(metrics.classification_report(y_test, predicted_multinomialNB, target_names=dataset.target_names))
# Create the SVC model
#svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear').fit(X_train, y_train)

# Fit the data to the SVC model
#svc_model.fit(X_train, y_train)



#svm, liblinearsvm, LR, multinominal naive bais, k-nn