from csv import DictReader
from collections import defaultdict 
from pprint import pprint
import os

category="Category"
estimator="estimator"
metric="metric"
value="value"

def make_header(list_of_rows):
    header = ["Estimator"]
    known = set()

    for rows in list_of_rows:
        for row in rows:
            est = row[estimator]
            if est not in known:
                known.add(est)
                header.append(est.strip())
    return header

def add_row(list_of_rows, dataset, header):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    counts_key = "{estimator}-{metric}"
    num_cat = 0
    ret_row = [dataset]
    for rows in list_of_rows:
        for row in rows:
            if num_cat < int(row[category]):
                num_cat = int(row[category])
            counts[counts_key.format(estimator=row[estimator], metric=row[metric])] += 1
            sums[row[estimator]][row[metric]] += float(row[value])

    for estimator_ in header[1:]:
        s = sums[estimator_]["f1-score"]
        c = counts[counts_key.format(estimator=estimator_, metric="f1-score")]
        ret_row.append(s/c)
    return ret_row

def getListOfRowsFromFile(files_dir):
    list_of_rows = []
    for file_name in os.listdir(files_dir):
        rows = []
        with open("{}/{}".format(files_dir,file_name)) as file:
            header = [h.strip() for h in file.readline().split(',')]
            reader = DictReader(file,fieldnames=header)
            for row in reader:
                for key in row:
                    row[key] = row[key].strip()
                rows.append(row)
        list_of_rows.append(rows)
    return list_of_rows

def replace_in_all_files(directory, old, new):
    for file_name in os.listdir(directory):
        abs_file = ("{}/{}".format(directory,file_name))
        file_content = None
        with open(abs_file) as input_file:
            file_content = input_file.read()
        with open(abs_file, 'w') as output_file:
            output_file.write(file_content.replace(old,new))

def create_table():
    datasets = [
       ("CiteSeer",'citeseer'),
       ("Cora",'cora'),
       ("Pubmed",'pubmed'),
       ("Random 1",'random_100_1000_2000_0p1_5'),
       ("Random 2",'random_500_500_10_0p500_5'),
       ("Random 3",'random_100_500_100_0p900_10'),
    ]
    top_dir = "C:/Users/Florian/Google Drive/source/python/Bachelor/metrics/results/final/csv"
    table = []
    header = None
    for dataset,folder_name in datasets:
        dataset_dir = "{}/{}".format(top_dir,folder_name)

        # replace_in_all_files(dataset_dir,"quantumSimualtedRBMsa","simQuantumRBM 1")
        # replace_in_all_files(dataset_dir,"quantumSimualtedRBMts","simQuantumRBM 2")
        list_of_rows= getListOfRowsFromFile(dataset_dir)
        if header is None:
            header = make_header(list_of_rows)
            table.append(header)

        

        table.append(add_row(list_of_rows, dataset, header))
    return table


def table_to_tabular(table):
    table = list(map(list, zip(*table)))
    tabular = "\\begin{tabular}{l" + "c"*(len(table[0])-1) +"}\n"
    for i,row in enumerate(table):
        tabular_row = ""
        for entry in row:
            if isinstance(entry,float):
                entry = "{:.3f}".format(entry)
            tabular_row +=" {:20} &".format(entry)
        tabular += "{} \\\\ \n".format(tabular_row[:-1].strip())
        tabular += "\\toprule \n" if i is 0 else ""
    tabular += "\\bottomrule\n\\end{tabular}"
    return tabular

print(table_to_tabular(create_table()))
