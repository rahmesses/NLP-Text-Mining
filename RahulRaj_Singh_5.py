#!/usr/bin/env python
# coding: utf-8

# In[35]:


## Module Import

import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

## Files Preprocessing

print("Preprocessing & Feature Engineering\n")

pos_files = os.listdir('E:/Fall 2019/Python/Lab NLTK/5_lab/pos/pos')
neg_files = os.listdir('E:/Fall 2019/Python/Lab NLTK/5_lab/neg/neg')

pos_corpus = []
neg_corpus = []

swlist = stopwords.words('english')

for f in pos_files:
    fname = 'E:/Fall 2019/Python/Lab NLTK/5_lab/pos/pos/' + f
    fh = open(fname, 'r', encoding="utf8")
    content = fh.read()
    processed_content = re.sub(r'[-!@#$%^&*()/\,.?":{}|<>]', ' ', content.lower())
    # print(f,':',content)
    # print(processed_content)
    words = word_tokenize(processed_content)
    # print(words)

    pos_words = nltk.pos_tag(words)
    # print(pos_words)

    clean_words = []
    for w in pos_words:
        if w in swlist or len(w[0]) <= 3 or w[1] not in ('JJ', 'JJR', 'JJS'):
            continue
        clean_words.append(w[0])

    # print(pos_words)
    # print(clean_words)

    # post_tag
    pos_content = ' '.join(clean_words)
    pos_corpus.append(pos_content)

#print(pos_corpus)

for f in neg_files:
    fname_neg = 'E:/Fall 2019/Python/Lab NLTK/5_lab/neg/neg/' + f
    fh = open(fname_neg, 'r', encoding="utf8")
    content_neg = fh.read()
    processed_content_neg = re.sub(r'[-!@#$%^&*/\(),.?":{}|<>]', ' ', content_neg.lower())
    # print(f,':',content_neg)
    # print(processed_content_neg)
    words_neg = word_tokenize(processed_content_neg)
    # print(words_neg)

    neg_words = nltk.pos_tag(words_neg)
    # print(neg_words)

    clean_words_neg = []
    for w in neg_words:
        if w in swlist or len(w[0]) <= 3 or w[1] not in ('JJ', 'JJR', 'JJS'):
            continue
        clean_words_neg.append(w[0])

    # print(neg_words)
    # print(clean_words_neg)

    #neg_tag
    neg_content = ' '.join(clean_words_neg)
    neg_corpus.append(neg_content)

#print(neg_corpus)

## Feature Engineering

#vectorizer = TfidfVectorizer()

vectorizer = TfidfVectorizer(ngram_range=(1,1))

corpus = pos_corpus + neg_corpus
X = vectorizer.fit_transform(corpus)

#print(vectorizer.get_feature_names())
#print(len(pos_corpus))
#print(len(neg_corpus))
#print(len(vectorizer.get_feature_names()))
#print(corpus[0])
#print(vectorizer.get_feature_names()[122])
#print(X[0,:])
y = [1]*len(pos_corpus) + [0]*len(neg_corpus)


#print(y)

## Linear Regression

print("Model Training and Validation\n")

print("Logistic Regression\n")

lr_model = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2019)
lr_model.fit(X_train,y_train)
score = lr_model.score(X_test,y_test)
print('holdout score:', score)

#cross validation

scores = cross_val_score(lr_model, X, y, cv = 10)
print('10 run scores', scores, ',', np.mean(scores),',',np.std(scores))

## Predictions

pred_Log = lr_model.predict(X_test)
print("Logistic Accuracy:", accuracy_score(y_test,pred_Log, normalize = True))

# ROC Curve

fpr_Log = dict()
tpr_Log = dict()
roc_auc_Log = dict()

fpr_Log, tpr_Log, _ = roc_curve(y_test, pred_Log)
roc_auc_Log = auc(fpr_Log,tpr_Log)

print("ROC curve for Logistic Regression\n")

plt.figure()
lw = 2
plt.plot(fpr_Log, tpr_Log, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_Log)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

## Random Forest

print("Random Forest\n")

rf = RandomForestRegressor(n_estimators = 50, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
pred_RF = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(pred_RF - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:\n', round(np.mean(errors), 2), 'degrees.')

# ROC Curve Random Forest

fpr_RF = dict()
tpr_RF = dict()
roc_auc_RF = dict()

fpr_RF, tpr_RF, _ = roc_curve(y_test, pred_RF)
roc_auc_RF = auc(fpr_RF,tpr_RF)

plt.figure()
lw = 2
plt.plot(fpr_RF, tpr_RF, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_RF)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()


## SVM

print("Support Vector Machine\n")
svc_model = LinearSVC(random_state=1)
pred_SVM = svc_model.fit(X_train, y_train).predict(X_test)
print("SVC Accuracy:", accuracy_score(y_test,pred_SVM, normalize = True))

# ROC Curve SVM

fpr_SVM = dict()
tpr_SVM = dict()
roc_auc_SVM = dict()

fpr_SVM, tpr_SVM, _ = roc_curve(y_test, pred_SVM)
roc_auc_SVM = auc(fpr_SVM,tpr_SVM)

plt.figure()
lw = 2
plt.plot(fpr_SVM, tpr_SVM, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_SVM)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - SVM')
plt.legend(loc="lower right")
plt.show()

print("Model training and validation completed\n")

##########################################################################################################

## New input text

New_text = input("Please Enter a Review\n")

clean_words_New = []

processed_content_New = re.sub(r'[-!@#$%^&*/\(),.?":{}|<>]', ' ', New_text.lower())

words_New = word_tokenize(processed_content_New)

pos_words_New = nltk.pos_tag(words_New)

for w in pos_words_New:
    if w in swlist or len(w[0]) <= 3 or w[1] not in ('JJ', 'JJR', 'JJS'):
        continue
    clean_words_New.append(w[0])
    
pos_content_New = ' '.join(clean_words_New)

corpus_New = list(pos_content_New.split(","))

#print(corpus_New)

X = vectorizer.transform(corpus_New)
#print(X)

### I am using Logistic Regression to predict

#pred_SVM_New = svc_model.fit(X_train, y_train).predict(X)
pred_Log_New = lr_model.predict(X)

if pred_Log_New == 1:
    print("\nThe review is positive")
else:
    print("\nThe review is negative")




# In[ ]:




