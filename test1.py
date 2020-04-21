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
from d_wave_client import upper_diagonal_blockmatrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("pgf")
#_ = plt.rcParams["figure.figsize"]
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'xtick.bottom':False,
    'xtick.labelbottom':False,
    'xtick.top':True,
    'xtick.labeltop':True,
    'axes.axisbelow':False,
    'figure.dpi': 70,
    #'figure.figsize':(1.4, _[1]*1.4/_[0]),
})


import subprocess
from sklearn.preprocessing import OneHotEncoder


def write_file_with_header(file_name, data):
    file_name = file_name.replace("->","_")
    with open(file_name, mode='w', encoding="utf-8") as file:
        file.write("{:8}, {:16}, {:10}, {:9}".format(
            'Category', 'estimator', 'metric', 'value'))
        file.writelines(data)

def as_csv_file(file_name):
    file_ending = "csv"
    return "{}.{}".format(file_name,file_ending)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        ret = np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
        return ret

class Const:
    def __init__(self, enc):
        self.enc = enc

    def predict(self, x):
        self.estimation = np.zeros_like(x)
        self.estimation[np.arange(len(x)), x.argmax(1)] = 1
        self.estimation = self.enc.inverse_transform(
            self.estimation).reshape((-1,))
        return self.estimation

    def fit(self, x, y):
        pass


class Tester:
    def __init__(self, save_file=False, pdf=False, create_images=False, tikz=False, enc=None, lr=0.1, q_iterations=5, c_iterations=50, rbm_hidden=10):
        self.save_file = save_file
        self.pdf = pdf
        self.create_images = create_images
        self.tikz = tikz
        self.enc = enc
        self.lr = lr
        self.q_iterations = q_iterations
        self.c_iterations = c_iterations
        self.rbm_hidden = rbm_hidden
        self.dataset_categories = {
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

    def rbm(self, factory, op_mode):
        iterations = self.c_iterations if op_mode < BernoulliRBM.op_mode_quantum else self.q_iterations
        if factory is BernoulliRBM:
            return BernoulliRBM(tester=self, learning_rate=self.lr, n_iter=iterations, n_components=self.rbm_hidden,
                                verbose=True, batch_size=1000, op_mode=op_mode)
        elif factory is PCDRBM:
            return PCDRBM(tester=self, visible=self.dataset_categories[self.cur_dataset], hidden=self.rbm_hidden, particles=-1,
                          iterations=iterations, epochs=10000, step=self.lr, weight_decay=0, op_mode=op_mode)

    def create_estimators(self, x_test):

        logistic1 = linear_model.LogisticRegression()
        logistic2 = linear_model.LogisticRegression()
        logistic3 = linear_model.LogisticRegression()
        logistic4 = linear_model.LogisticRegression()
        logistic5 = linear_model.LogisticRegression()
        logistic6 = linear_model.LogisticRegression()
        logistic7 = linear_model.LogisticRegression()
        logistic8 = linear_model.LogisticRegression()
        logistic9 = linear_model.LogisticRegression()

        o_rbm_cl = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_classic)
        o_rbm_sa = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_simulated_annealing)
        o_rbm_qu = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_quantum)
        o_rbm_qs = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_qbsolv)
        n_rbm_cl = self.rbm(PCDRBM, BernoulliRBM.op_mode_classic)
        n_rbm_sa = self.rbm(PCDRBM, BernoulliRBM.op_mode_simulated_annealing)
        n_rbm_qu = self.rbm(PCDRBM, BernoulliRBM.op_mode_quantum)
        n_rbm_qs = self.rbm(PCDRBM, BernoulliRBM.op_mode_qbsolv)

        o_rbm_cl_p = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_classic)
        o_rbm_sa_p = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_simulated_annealing)
        o_rbm_qu_p = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_quantum)
        o_rbm_qs_p = self.rbm(BernoulliRBM, BernoulliRBM.op_mode_qbsolv)
        n_rbm_cl_p = self.rbm(PCDRBM, BernoulliRBM.op_mode_classic)
        n_rbm_sa_p = self.rbm(PCDRBM, BernoulliRBM.op_mode_simulated_annealing)
        n_rbm_qu_p = self.rbm(PCDRBM, BernoulliRBM.op_mode_quantum)
        n_rbm_qs_p = self.rbm(PCDRBM, BernoulliRBM.op_mode_qbsolv)

        o_rbm_cl_pipe = Pipeline(steps=[('rbm',o_rbm_cl_p),('log_regression',logistic2)])
        o_rbm_sa_pipe = Pipeline(steps=[('rbm',o_rbm_sa_p),('log_regression',logistic3)])
        o_rbm_qu_pipe = Pipeline(steps=[('rbm',o_rbm_qu_p),('log_regression',logistic4)])
        o_rbm_qs_pipe = Pipeline(steps=[('rbm',o_rbm_qs_p),('log_regression',logistic5)])
        n_rbm_cl_pipe = Pipeline(steps=[('rbm',n_rbm_cl_p),('log_regression',logistic6)])
        n_rbm_sa_pipe = Pipeline(steps=[('rbm',n_rbm_sa_p),('log_regression',logistic7)])
        n_rbm_qu_pipe = Pipeline(steps=[('rbm',n_rbm_qu_p),('log_regression',logistic8)])
        n_rbm_qs_pipe = Pipeline(steps=[('rbm',n_rbm_qs_p),('log_regression',logistic9)])

        estimators = []
        # estimators.append(["GCN", Const(self.enc)])
        # estimators.append(["LogisticRegression", logistic1])

        # estimators.append(['GCN->o_rbm_cl',o_rbm_cl])
        # # estimators.append(['GCN->o_rbm_sa',o_rbm_sa])
        # # estimators.append(['GCN->o_rbm_qu',o_rbm_qu])
        # # estimators.append(['GCN->o_rbm_qs',o_rbm_qs])
        # # estimators.append(['GCN->n_rbm_cl',n_rbm_cl])
        # # estimators.append(['GCN->n_rbm_sa',n_rbm_sa])
        # # estimators.append(['GCN->n_rbm_qu',n_rbm_qu])
        # # estimators.append(['GCN->n_rbm_qs',n_rbm_qs])

        # estimators.append(['classicalRBM',o_rbm_cl_pipe])
        # estimators.append(['GCN--nrbmcl--LOG',n_rbm_cl_pipe])
        # ########################-#######-####################
        estimators.append(['quantumSimualtedRBMsa',o_rbm_sa_pipe])
        # estimators.append(['GCN--nrbmsa--LOG',n_rbm_sa_pipe])
        # ########################-#######-####################
        # estimators.append(['quantumSimualtedRBMts',o_rbm_qs_pipe])
        # # estimators.append(['GCN--nrbmqs--LOG',n_rbm_qs_pipe])

        # estimators.append(['quantumRBM',o_rbm_qu_pipe])
        # estimators.append(['GCN->n_rbm_qu->LOG',n_rbm_qu_pipe])

        return estimators

    cdict = {
        'red':  (( .0  ,  .0,  .0),
                 ( .20 ,  .6,  .6),
                 ( .37 ,  .2,  .2),
                 ( .43 ,  .2,  .2),
                 ( .495,  .9,  .9),
                 ( .5  , 1. , 1. ),
                 ( .505,  .9,  .9),
                 ( .55 ,  .0,  .0),
                 ( .62 ,  .6,  .6),
                 ( .7  , 1. , 1. ),
                 ( .78 , 1. , 1. ),
                 ( .84 ,  .8,  .8),
                 ( .9  , 1. , 1. ),
                 (1.   , 1. , 1. )),

        'green':(( .0  ,  .0,  .0),
                 ( .20 ,  .0,  .0),
                 ( .37 ,  .2,  .2),
                 ( .43 ,  .5,  .5),
                 ( .495,  .9,  .9),
                 ( .5  , 1. , 1. ),
                 ( .505,  .9,  .9),
                 ( .55 , 1. , 1. ),
                 ( .62 ,  .6,  .6),
                 ( .7  , 1. , 1. ),
                 ( .78 ,  .5,  .5),
                 ( .84 ,  .0,  .0),
                 ( .9  ,  .0 , .0),
                 (1.   ,  .6 , .6)),

        'blue': (( .0  ,  .0,  .0),
                 ( .20 ,  .6,  .6),
                 ( .37 ,  .8,  .8),
                 ( .43 , 1. , 1. ),
                 ( .495, 1. , 1. ),
                 ( .5  , 1. , 1. ),
                 ( .505,  .9,  .9),
                 ( .55 ,  .0,  .0),
                 ( .62 ,  .0,  .0),
                 ( .7  ,  .0,  .0),
                 ( .78 ,  .0,  .0),
                 ( .84 ,  .0,  .0),
                 ( .9  ,  .0 , .0),
                 (1.   , 1.  ,1. )),
    }
    my_colormap = LinearSegmentedColormap('my_colormap',cdict,N=1024)


    def save_weights_img(self, B, C, W, i):
        if self.save_file and self.create_images:
            img_dir = "{}/{}".format(self.plt_dir, self.cur_dataset)
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)
            Q = upper_diagonal_blockmatrix(
                B, C, W)
            _max = .8# max(0, np.amax(Q))
            _min = -3#min(0, np.amin(Q))
            _mid = 0
            
            fig, ax = plt.subplots()

            ticks = list(range(Q.shape[0]))
            ax.tick_params(length=0)
            ax.set_xlabel("Node j")
            ax.xaxis.set_label_position('top') 
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_ylabel("Node i")
            ax.invert_yaxis()
            heatmap = ax.imshow(Q, cmap=self.my_colormap, interpolation='nearest', clim=(
                _min, _max), norm=MidpointNormalize(_min, _max, _mid))
            heatmap = ax.imshow(Q, cmap=self.my_colormap, interpolation='nearest', clim=(
                _min, _max), norm=MidpointNormalize(_min, _max, _mid))

            cbar = plt.colorbar(heatmap)
            cbar.ax.set_ylabel('Node bias / Connection value',labelpad=5, rotation=90)

            file_str = '{}/{}_{}_{}_{}'.format(
                    img_dir,
                    self.cur_dataset,
                    self.cur_estimator.replace(">", "+"),
                    "train" if self.cur_test_train else "test",
                    i)
            plt.savefig(file_str+".png")
            if self.tikz:
                plt.savefig(file_str+".pgf")
            plt.clf()
            plt.close(fig)

    def get_iter_str(self,estimator):
        iter_str = "{}+{}".format(self.gcn.epochs,"{}")
        if estimator == "GCN":
            return iter_str.format(0)
        elif "rbm_q" in estimator or "rbm_sa" in estimator or "quantum" in estimator:
            return iter_str.format(self.q_iterations)
        else:
            return iter_str.format(self.c_iterations)


    def save_metrics2(self, dataset, metrics_list, metrics_result_file):
        lines = []
        sum = 0
        for category in metrics_list[0][1].keys():
            if category in {'accuracy', 'macro avg', 'weighted avg'}:
                continue
            sum += metrics_list[0][1][category]['support']

        for metric_element in metrics_list:
            estimator_lines = []
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
                    estimator_lines.append(lines[-1])
            estimator_file = "{}_{}_{}iter".format(metrics_result_file,cur_estimator,self.get_iter_str(cur_estimator))
            #write_file_with_header(as_csv_file(estimator_file),lines)

        write_file_with_header(as_csv_file(metrics_result_file),lines)

    def fit_and_test(self, estimators, x_train, y_train, x_test, y_test):
        metrics_list = []
        for estimator in estimators:
            self.cur_estimator = estimator[0]
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
                y_pred = self.enc.inverse_transform(temp).reshape((-1,))

            print("{:12} metrics:\n{}\n".format(
                estimator[0], metrics.classification_report(y_test, y_pred)))
            metrics_list.append([estimator[0], metrics.classification_report(
                y_test, y_pred, output_dict=True)])
        return metrics_list

    def to_ones(self, x_train, x_test):
        _x_train = np.zeros_like(x_train)
        _x_train[np.arange(len(x_train)), x_train.argmax(1)] = 1

        _x_test = np.zeros_like(x_test)
        _x_test[np.arange(len(x_test)), x_test.argmax(1)] = 1

        return _x_train, _x_test

    def compare_on_dataset(self, dataset, metrics_result_file, test_training_data):
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
            dataset)
        self.gcn = Trainer(dataset=dataset)
        self.gcn.train()

        x = self.gcn.get_predictions()
        x_train = x[train_mask]
        x_test = x[test_mask]
        x_val = x[val_mask]

        y_train = self.enc.inverse_transform(
            y_train[train_mask]).reshape((-1,))

        y_test = self.enc.inverse_transform(y_test[test_mask]).reshape((-1,))
        y_val = self.enc.inverse_transform(y_val[val_mask]).reshape((-1,))

        if test_training_data:
            x_test = x_train
            y_test = y_train

        # x_train, x_test = self.to_ones(x_train, x_test)

        estimators = self.create_estimators(x_test)

        metrics_list = self.fit_and_test(
            estimators, x_train, y_train, x_test, y_test)

        if self.save_file:
            self.save_metrics2(dataset, metrics_list, metrics_result_file)

    def test(self, dataset_list, test_training_data_list):

        if self.save_file:
            results_dir = os.path.abspath("./metrics/results")
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            dir_name = "{}/final".format(results_dir)

            csv_dir = "{}/csv".format(dir_name)
            pdf_dir = "{}/pdf".format(dir_name)
            tex_dir = "{}/tex".format(dir_name)
            self.plt_dir = "{}/plt".format(dir_name)

            r_script_file = os.path.abspath("./metrics/plot_gen.r")

            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(tex_dir, exist_ok=True)
        csv_file = None

        for test_training_data in test_training_data_list:
            self.cur_test_train = test_training_data
            for dataset in dataset_list:
                self.cur_dataset = dataset
                t = np.array(
                    range(1, self.dataset_categories[dataset]+1)).reshape((-1, 1))
                self.enc = OneHotEncoder()
                self.enc.fit(t)
                print("""








################################################################################
#                            Testing {:30}            #
################################################################################
""".format(dataset))

                if self.save_file:
                    dataset_dir = "{}/{}".format(csv_dir, dataset)
                    os.makedirs(dataset_dir, exist_ok=True)

                    dir_idx = len(os.listdir(dataset_dir))+1

                    csv_file = "{}/{}-{}-{}".format(
                        dataset_dir, dataset, "train" if self.cur_test_train else "test", dir_idx) if self.save_file else None

                self.compare_on_dataset(
                    dataset, metrics_result_file=csv_file, test_training_data=self.cur_test_train)

                if self.pdf:
                    subprocess.call([
                        'RScript.exe',
                        '--vanilla',
                        r_script_file,
                        as_csv_file(csv_file),
                        dir_name,
                        "{}-{}".format(
                            dataset, "train" if self.cur_test_train else "test"),
                    ])
                else:
                    print('skipping pdf creation')

    def test_all(self):
        self.test(self.dataset_categories.keys(), [False, True])
