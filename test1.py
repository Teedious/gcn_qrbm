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
import subprocess


from sklearn.preprocessing import OneHotEncoder

save_file = False
test_training_data = False
enc = None

dataset_categories = {
    'citeseer': 6,
    'cora': 7,
    'pubmed': 3,
    'random_100_1000_2000_0p1_5': 5,
    'random_500_500_10_0p500_5': 5,
    'random_100_500_100_0p900_10': 10,
    'random_10000_500_100_0p1_5': 5,
    'random_100000_1000_100_0p1_5': 5,
    'random_20000_20000_200_0p1_5': 5,
}


class Const:
    def __init__(self, enc):
        self.enc = enc

    def predict(self, x):
        self.estimation = np.zeros_like(x)
        self.estimation[np.arange(len(x)), x.argmax(1)] = 1
        self.estimation = enc.inverse_transform(self.estimation).reshape((-1,))
        return self.estimation

    def fit(self, x, y):
        pass


def create_estimators(x_test, dataset):
    logistic1 = linear_model.LogisticRegression()
    logistic2 = linear_model.LogisticRegression()
    logistic3 = linear_model.LogisticRegression()
    logistic4 = linear_model.LogisticRegression()
    logistic5 = linear_model.LogisticRegression()
    logistic7 = linear_model.LogisticRegression()

    lr = 0.1
    iterations = 50
    classic_iterations = 5000
    num_hidden = 10
    batch_size = 1000
    weight_decay = 0

    rbm1 = BernoulliRBM(learning_rate=lr, n_iter=classic_iterations, n_components=num_hidden,
                        verbose=True, batch_size=batch_size, op_mode=BernoulliRBM.op_mode_classic)
    rbm2 = BernoulliRBM(learning_rate=lr, n_iter=iterations, n_components=num_hidden,
                        verbose=True, batch_size=batch_size, op_mode=BernoulliRBM.op_mode_simulate_quantum)
    rbm3 = BernoulliRBM(learning_rate=lr, n_iter=iterations, n_components=num_hidden,
                        verbose=True, batch_size=batch_size, op_mode=BernoulliRBM.op_mode_quantum)
    rbm4 = PCDRBM(visible=dataset_categories[dataset], hidden=num_hidden, particles=-1,
                  iterations=classic_iterations, epochs=1, step=lr, weight_decay=weight_decay)
    rbm5 = PCDRBM(visible=dataset_categories[dataset], hidden=num_hidden, particles=-1,
                  iterations=classic_iterations, epochs=1, step=lr, weight_decay=weight_decay)
    rbm6 = PCDRBM(visible=dataset_categories[dataset], hidden=num_hidden, particles=-1,
                  iterations=iterations, epochs=1, step=lr, weight_decay=weight_decay, op_mode=PCDRBM.op_mode_simulate_quantum)
    rbm7 = PCDRBM(visible=dataset_categories[dataset], hidden=num_hidden, particles=-1,
                  iterations=iterations, epochs=1, step=lr, weight_decay=weight_decay, op_mode=PCDRBM.op_mode_simulate_quantum)

    pipe1 = Pipeline(steps=[('rbm1', rbm1), ('logistic1', logistic1)])
    pipe2 = Pipeline(steps=[('rbm2', rbm2), ('logistic2', logistic2)])
    pipe3 = Pipeline(steps=[('rbm3', rbm3), ('logistic3', logistic3)])
    pipe5 = Pipeline(steps=[('rbm5', rbm5), ('logistic5', logistic5)])
    pipe7 = Pipeline(steps=[('rbm5', rbm7), ('logistic5', logistic7)])

    estimators = []
    estimators.append(["GCN", Const(enc)])
    # estimators.append(["GCN->LOG", logistic4])
    # estimators.append(["GCN->old_RBM->LOG", pipe1])
    # estimators.append(["GCN->old_sim_Q_RBM->LOG", pipe2])
    # # estimators.append(["GCN->old_Q_RBM->LOG", pipe3])

    # estimators.append(["GCN->old_RBM", rbm1])
    # estimators.append(["GCN->new_RBM", rbm4])
    estimators.append(["GCN->new_RBM->LOG", pipe5])
    # estimators.append(["GCN->new_sim_Q_RBM", rbm6])
    # estimators.append(["GCN->new_sim_Q_RBM->LOG", pipe7])

    return estimators


def save_metrics(dataset, metrics_list, metrics_result_file):
    cat_string = 'Category'
    header = {cat_string}
    to_save = defaultdict(dict)

    for i in range(1, dataset_categories[dataset]+1):
        to_save[i][cat_string] = i

    sum = 0
    for category in metrics_list[0][1].keys():
        if category in {'accuracy', 'macro avg', 'weighted avg'}:
            continue
        sum += metrics_list[0][1][category]['support']

    for m in metrics_list:
        for category in m[1].keys():
            if category in {'accuracy', 'macro avg', 'weighted avg'}:
                continue

            for metr in m[1][category].keys():
                col_name = "{}_{}".format(str(m[0]), str(metr))
                header.add(col_name)
                if metr == 'support':
                    to_save[int(category)][col_name] = m[1][category][metr]/sum
                else:
                    to_save[int(category)][col_name] = m[1][category][metr]

    with open(metrics_result_file, mode='w', encoding="utf-8") as file:
        file.write(", ".join(str(x) for x in sorted(list(header))))
        for category in sorted(to_save.keys()):
            row = []
            for col in sorted(list(header)):
                row.append(to_save[category][col])

            file.write("\n")
            file.write(",".join("{:9.4f}".format(x) for x in row))


def save_metrics2(dataset, metrics_list, metrics_result_file):
    lines = []

    sum = 0
    for category in metrics_list[0][1].keys():
        if category in {'accuracy', 'macro avg', 'weighted avg'}:
            continue
        sum += metrics_list[0][1][category]['support']

    for metric_element in metrics_list:
        cur_estimator = metric_element[0]
        cur_metric_report = metric_element[1]
        for cur_category in cur_metric_report:
            if cur_category in {'accuracy', 'macro avg', 'weighted avg'}:
                continue
            cur_category_dict = cur_metric_report[cur_category]
            for cur_metric in cur_category_dict:
                cur_value = cur_category_dict[cur_metric]
                if cur_metric == 'support':
                    cur_value /= sum
                lines.append("\n{:8d}, {:16}, {:10}, {:9.4f}".format(
                    int(cur_category), cur_estimator, cur_metric, cur_value))

    with open(metrics_result_file, mode='w', encoding="utf-8") as file:
        file.write("{:8}, {:16}, {:10}, {:9}".format(
            'Category', 'estimator', 'metric', 'value'))
        file.writelines(lines)


def fit_and_test(estimators, x_train, y_train, x_test, y_test):
    metrics_list = []
    for estimator in estimators:
        if "Q_RBM" in estimator[0] and x_train.shape[0] > 5000:
            print("skipping dataset for estimator {} - too many nodes ({})".format(
                estimator[0], x_train.shape[0]))
            continue
        print("fitting {:12}".format(estimator[0]))
        estimator[1].fit(x_train, y_train)

        y_pred = estimator[1].predict(x_test)
        if len(y_pred.shape) > 1:
            temp = np.zeros_like(y_pred)
            temp[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
            y_pred = enc.inverse_transform(temp).reshape((-1,))

        print("{:12} metrics:\n{}\n".format(
            estimator[0], metrics.classification_report(y_test, y_pred)))
        metrics_list.append([estimator[0], metrics.classification_report(
            y_test, y_pred, output_dict=True)])
    return metrics_list


def to_ones(x_train, x_test):
    _x_train = np.zeros_like(x_train)
    _x_train[np.arange(len(x_train)), x_train.argmax(1)] = 1

    _x_test = np.zeros_like(x_test)
    _x_test[np.arange(len(x_test)), x_test.argmax(1)] = 1

    return _x_train, _x_test


def compare_on_dataset(dataset, metrics_result_file, test_training_data):
    a = Trainer(dataset=dataset)
    a.train()

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
        dataset)

    x = a.get_predictions()
    x_train = x[train_mask]
    x_test = x[test_mask]
    x_val = x[val_mask]

    y_train = enc.inverse_transform(y_train[train_mask]).reshape((-1,))
    y_test = enc.inverse_transform(y_test[test_mask]).reshape((-1,))
    y_val = enc.inverse_transform(y_val[val_mask]).reshape((-1,))

    estimators = create_estimators(x_test, dataset)

    if test_training_data:
        x_test = x_train
        y_test = y_train

    x_train, x_test = to_ones(x_train, x_test)

    metrics_list = fit_and_test(estimators, x_train, y_train, x_test, y_test)

    if None is not metrics_result_file:
        save_metrics2(dataset, metrics_list, metrics_result_file)


def test(dataset_list, test_training_data_list):
    global enc

    if save_file:
        results_dir = os.path.abspath("./metrics/results")
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        dir_idx = len(os.listdir(results_dir))+1
        dir_name = "{}/result{}".format(results_dir, dir_idx)

        csv_dir = "{}/csv".format(dir_name)
        pdf_dir = "{}/pdf".format(dir_name)
        tex_dir = "{}/tex".format(dir_name)

        r_script_file = os.path.abspath("./metrics/loher.r")

        os.makedirs(csv_dir)
        os.makedirs(pdf_dir)
        os.makedirs(tex_dir)

    for test_training_data in test_training_data_list:
        for dataset in dataset_list:
            t = np.array(
                range(1, dataset_categories[dataset]+1)).reshape((-1, 1))
            enc = OneHotEncoder()
            enc.fit(t)
            print("""








################################################################################
#              Testing {:20}                    #
################################################################################
""".format(dataset))

            if save_file:
                csv_file = "{}/{}_train_{}_metric_results.txt".format(
                    csv_dir, dataset, test_training_data)
                compare_on_dataset(
                    dataset, metrics_result_file=csv_file, test_training_data=test_training_data)
                subprocess.call([
                    'RScript.exe',
                    '--vanilla',
                    r_script_file,
                    csv_file,
                    dir_name,
                    "{}_{}".format(
                        dataset, "train" if test_training_data else "test"),
                ])
            else:
                compare_on_dataset(
                    dataset, metrics_result_file=None, test_training_data=test_training_data)
                print('skipping file creation')


def test_all():
    test(dataset_categories.keys(), [False, True])


test(['random_500_500_10_0p500_5'], [True])
# test_all()
