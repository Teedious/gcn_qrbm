from rbm import BernoulliRBM
# from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
import numpy as np
from random import shuffle
from gcn.train import *
from gcn.utils import *
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

dataset = 'pubmed'

categories = {'citeseer':6,'cora':7,'pubmed':3}
t = np.array(range(1,categories[dataset]+1)).reshape((-1,1))
enc = OneHotEncoder()
enc.fit(t)


a = Trainer(dataset=dataset)
a.train()


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

x = a.get_predictions()
x_train = x[train_mask]
x_test = x[test_mask]
x_val = x[val_mask]

y_train = enc.inverse_transform(y_train[train_mask]).reshape((-1,))
y_test = enc.inverse_transform(y_test[test_mask]).reshape((-1,))
y_val = enc.inverse_transform(y_val[val_mask]).reshape((-1,))

b = np.zeros_like(x_train)
b[np.arange(len(x_train)), x_train.argmax(1)] = 1

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0,learning_rate=0.1,n_iter=100,n_components=7, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

rbm_features_classifier.fit(x_train,y_train)

y_pred = rbm_features_classifier.predict(x_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, y_pred)))


print()