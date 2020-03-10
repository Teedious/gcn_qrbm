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
from collections import defaultdict
from RBM.PCD import PCDRBM

from sklearn.preprocessing import OneHotEncoder

dataset = 'citeseer'
outfile_name = 'metrics/{}_metrics_.txt'.format(dataset)
quantum = False
save_file = False

dataset_categories = {
'citeseer':6,
'cora':7,
'pubmed':3,
'random_100_1000_2000_0p1_5':5,
'random_500_500_10_0p500_5':5,
'random_100_500_100_0p900_10':10,
'random_10000_500_100_0p1_5':5,
'random_100000_1000_100_0p1_5':5,
'random_20000_20000_200_0p1_5':5,
}
t = np.array(range(1,dataset_categories[dataset]+1)).reshape((-1,1))
enc = OneHotEncoder()
enc.fit(t)

class Const:
    def __init__(self,estimation,enc):
        self.estimation = np.zeros_like(estimation)
        self.estimation[np.arange(len(estimation)), estimation.argmax(1)] = 1
        self.estimation = enc.inverse_transform(self.estimation).reshape((-1,))
        
    def predict(self, x):
        return self.estimation

    def fit(self,x,y):
        pass
    

def create_estimators(x_test):
    logistic1 = linear_model.LogisticRegression()
    logistic2 = linear_model.LogisticRegression()
    logistic3 = linear_model.LogisticRegression()
    logistic4 = linear_model.LogisticRegression()
    logistic5 = linear_model.LogisticRegression()

    lr = 0.01
    iterations = 1000
    num_hidden = 100
    # lr = 0.01
    # iterations = 10
    # num_hidden = 100
    batch_size = 100

    rbm1 = BernoulliRBM(learning_rate=lr, n_iter=iterations, n_components=num_hidden, verbose=True, batch_size=batch_size, op_mode = BernoulliRBM.op_mode_classic)
    rbm2 = BernoulliRBM(learning_rate=lr, n_iter=iterations, n_components=num_hidden, verbose=True, batch_size=batch_size, op_mode = BernoulliRBM.op_mode_simulate_quantum)
    rbm3 = BernoulliRBM(learning_rate=lr, n_iter=iterations, n_components=num_hidden, verbose=True, batch_size=batch_size, op_mode = BernoulliRBM.op_mode_quantum)
    rbm4 = PCDRBM(
        visible = dataset_categories[dataset], 
        hidden = num_hidden, 
        particles = -1, 
        iterations = iterations, 
        epochs = 1, 
        step=lr, 
        weight_decay=lr/iterations,
        )
    rbm5 = PCDRBM(
        visible = dataset_categories[dataset], 
        hidden = num_hidden, 
        particles = -1, 
        iterations = iterations, 
        epochs = 1, 
        step=lr, 
        weight_decay=lr/iterations,
        )

    pipe1 = Pipeline(steps=[('rbm1', rbm1), ('logistic1', logistic1)])
    pipe2 = Pipeline(steps=[('rbm2', rbm2), ('logistic2', logistic2)])
    pipe3 = Pipeline(steps=[('rbm3', rbm3), ('logistic3', logistic3)])
    pipe5 = Pipeline(steps=[('rbm5', rbm5), ('logistic5', logistic5)])


    estimators = []
    estimators.append(["GCN",Const(x_test,enc)])
    estimators.append(["GCN_LOG",logistic4])
    estimators.append(["GCN_CRBM_LOG",pipe1])
    # estimators.append(["GCN_SQRBM_LOG",pipe2])
    if quantum:
        estimators.append(["GCN_QRBM_LOG",pipe3])

    estimators.append(["GCN_PCDRBM",PCDRBM(
        visible = dataset_categories[dataset], 
        hidden = num_hidden, 
        particles = -1, 
        iterations = iterations, 
        epochs = 1, 
        step=lr, 
        weight_decay=lr/iterations,
        )])
    estimators.append(["GCN_PCDRBM_LOG",pipe5])

    return estimators



def save_metrics(dataset,metrics_list):
    cat_string = 'Category'
    header = {cat_string}
    to_save = defaultdict(dict)

    for i in range(1,dataset_categories[dataset]+1):
        to_save[i][cat_string]=i

    sum = 0
    for category in metrics_list[0][1].keys():
        if category in {'accuracy','macro avg','weighted avg'}:
                continue
        sum += metrics_list[0][1][category]['support']

    for m in metrics_list:
        for category in m[1].keys():
            if category in {'accuracy','macro avg','weighted avg'}:
                continue
            
            for metr in m[1][category].keys():
                col_name = "{}_{}".format(str(m[0]),str(metr))
                header.add(col_name)
                if metr == 'support':
                    to_save[int(category)][col_name] = m[1][category][metr]/sum
                else:
                    to_save[int(category)][col_name] = m[1][category][metr]

    with open(outfile_name, mode='w') as file:
        file.write(", ".join(str(x) for x in sorted(list(header))))
        for category in sorted(to_save.keys()):
            row = []
            for col in sorted(list(header)):
                row.append(to_save[category][col])

            file.write("\n")
            file.write(",".join("{:9.4f}".format(x) for x in row))



def compare_on_dataset(dataset,save_metrics_to_file):
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

    estimators = create_estimators(x_test)

    metrics_list = []
    for estimator in estimators:
        print("fitting {:12}".format(estimator[0]))
        estimator[1].fit(x_train,y_train)
        
        y_pred = estimator[1].predict(x_test)
        if len(y_pred.shape)>1:
            temp = np.zeros_like(y_pred)
            temp[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
            y_pred = enc.inverse_transform(temp).reshape((-1,))

        print("{:12} metrics:\n{}\n".format(estimator[0],metrics.classification_report(y_test, y_pred)))
        metrics_list.append([estimator[0],metrics.classification_report(y_test, y_pred,output_dict=True)])

    if save_metrics_to_file:
        save_metrics(dataset,metrics_list)

compare_on_dataset(dataset,save_metrics_to_file=save_file)