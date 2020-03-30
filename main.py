from test1 import Tester

t = Tester(save_file=True, create_images=True)
t.dataset_categories = {
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

t.test(dataset_list=['citeseer', 'cora', 'random_500_500_10_0p500_5', 'random_100_500_100_0p900_10', 'random_100_1000_2000_0p1_5'],
       test_training_data_list=[False, True])

# t.test_all()
