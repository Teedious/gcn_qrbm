from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

class Trainer:

    def __init__(self, 
                 dataset='cora', 
                 network_model='gcn', 
                 learning_rate=0.1, 
                 epochs=200,
                 hidden=16,
                 dropout=0.5,
                 weight_decay=5e-4,
                 early_stopping=10,
                 max_degree=3):
        super().__init__()

        self.dataset=dataset
        self.network_model=network_model
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.hidden=hidden
        self.dropout=dropout
        self.weight_decay=weight_decay
        self.early_stopping=early_stopping
        self.max_degree=max_degree
        
        # Set random seed
        self.seed = 123
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    # Define model evaluation function
    def evaluate(self,features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def train(self):
        # Settings
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string('dataset', self.dataset, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
        flags.DEFINE_string('model', self.network_model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
        flags.DEFINE_float('learning_rate', self.learning_rate, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', self.epochs, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', self.hidden, 'Number of units in hidden layer 1.')
        flags.DEFINE_float('dropout', self.dropout, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', self.weight_decay, 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_integer('early_stopping', self.early_stopping, 'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('max_degree', self.max_degree, 'Maximum Chebyshev polynomial degree.')

        # Load data
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

        # Some preprocessing
        features = preprocess_features(features)
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        self.model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        self.sess = tf.Session()


        # Init variables
        self.sess.run(tf.global_variables_initializer())

        cost_val = []

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = self.sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = self.evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_duration = self.evaluate(features, support, y_test, test_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
