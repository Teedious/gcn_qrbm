from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from gcn.train import *
from gcn.utils import *
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

dataset = 'random_20000_20000_200_0p1_5'
outfile_name = 'metrics/{}_metrics_1.txt'.format(dataset)

dataset_categories = {'citeseer':6,'cora':7,'pubmed':3,'random_100_1000_2000_0p1_5':5,'random_500_500_10_0p500_5':5,'random_100_500_100_0p900_10':10,'random_10000_500_100_0p1_5':5,
'random_100000_1000_100_0p1_5':5,
'random_20000_20000_200_0p1_5':5,
}
t = np.array(range(1,dataset_categories[dataset]+1)).reshape((-1,1))
enc = OneHotEncoder()
enc.fit(t)



def save_metrics(dataset,metrics_list):
    cat_string = 'Category'
    header = {cat_string}
    to_save = defaultdict(dict)


    for i in range(1,dataset_categories[dataset]+1):
        to_save[i][cat_string]=i

    for m in metrics_list:
        for category in m[1].keys():
            if category in {'accuracy','macro avg','weighted avg'}:
                continue
            
            for metr in m[1][category].keys():
                col_name = "{}_{}".format(str(m[0]),str(metr))
                header.add(col_name)
                to_save[int(category)][col_name] = m[1][category][metr]

    # for i in range(dataset_categories[dataset]):
    #     to_save['category'][i]=i

    # for m in metrics_list:
    #     for category in m[1].keys():
    #         if category in {'accuracy','macro avg','weighted avg'}:
    #             continue
            
    #         for metr in m[1][category].keys():
    #             to_save["{}_{}".format(str(m[0]),str(metr))][category] = m[1][category][metr]
    with open(outfile_name, mode='w') as file:
        file.write(", ".join(str(x) for x in sorted(list(header))))
        for category in sorted(to_save.keys()):
            row = []
            for col in sorted(list(header)):
                row.append(to_save[category][col])

            file.write("\n")
            file.write(",".join("{:9.4f}".format(x) for x in row))


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

x_train = features[train_mask]
x_test = features[test_mask]

y_train = enc.inverse_transform(y_train[train_mask]).reshape((-1,))
y_test = enc.inverse_transform(y_test[test_mask]).reshape((-1,))

logistic = linear_model.LogisticRegression(solver='newton-cg')

logistic.fit(x_train,y_train)

y_pred = logistic.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

save_metrics(dataset,[["LOG",metrics.classification_report(y_test, y_pred,output_dict=True)]])