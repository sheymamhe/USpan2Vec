import numpy as np
import timeit
import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from spmf import Spmf
import random

labels, sentences = [], []
with open("output.txt") as f:
    spmf = Spmf("USpan", input_filename="DataBase_HUSRM.txt",
                output_filename="output.txt",
                spmf_bin_location_dir="C:/Users/HP/Downloads/",
                arguments=[35,4])
    spmf.run()
with open('output.txt','r') as source:
    data = [(random.random(), line) for line in source]
data.sort()
with open('output.txt','w') as target:
    for _, line in data:
        target.write(line)


### represent a trans in form of items and fis,
### joint-training: learn trans vectors using Doc2Vec (PV-DBOW) from both items and fis
### use SVM as classifier

### variables ###
data_name = "output.txt"
path = data_name
size_train = 0.8
use_train = "fix" # cv, fix
pattern = "fis" # fis, cs
tune_svm_trans = False # True, False
dim_d2v = 128
para_minSup = [0.002]
n_run = 10

### functions ###
# load data file in form of items and fis
def load_item_fis(file_name):
    labels, sentences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
                labels.append(label)
                sentences.append(content.rstrip().split(" "))
    return sentences, labels
# add a label to each sentence
def labelizeSentences(sentences):
    label_sentences = []
    for idx, val in enumerate(sentences):
        label = "t_{}".format(idx)
        label_sentences.append(TaggedDocument(val, [label]))
    return label_sentences

print("start time: {}".format(datetime.datetime.now()))
start_time = timeit.default_timer()
for minSup in para_minSup:
    print("d2v_it_pattern_joi_classify, data: {}, train: {}, pattern: {}, minSup={}".format(
        data_name, use_train, pattern, minSup))
    # load data in the form of items and fis
    data_path = path
    data_X, data_y = load_item_fis(data_path)
    if use_train == "fix":
        if data_name == "snippets":
            n_train = 10060
        if data_name == "dblp":
            n_train = 61479
        if data_name == "mr":
            n_train = 7108
        else:
            n_train = int(len(data_y) * size_train)
    # find the longest trans
    ws_max = len(max(data_X, key=len))
    # assign a label to each sentence
    data_sen_X = labelizeSentences(data_X)
    all_acc, all_mic, all_mac = [], [], []
    for run in range(n_run):
        with open('output.txt', 'r') as source:
            data = [(random.random(), line) for line in source]
        data.sort()
        with open('output.txt', 'w') as target:
            for _, line in data:
                target.write(line)
        print("run={}".format(run))
        # learn trans vectors using Doc2Vec (PV-DBOW)
        # d2v_dbow = Doc2Vec(size=dim_d2v, window=ws_max, min_count=0, workers=16, dm=0, iter=50, dbow_words=1)
        d2v_dbow = Doc2Vec(size=dim_d2v, min_count=0, workers=16, dm=0, iter=50)
        d2v_dbow.build_vocab(data_sen_X)
        d2v_dbow.train(data_sen_X, total_examples=d2v_dbow.corpus_count, epochs=d2v_dbow.iter)
        data_vec = [d2v_dbow.docvecs[idx] for idx in range(len(data_sen_X))]
        del d2v_dbow  # delete unneeded model memory

        if use_train == "cv":
            # generate train and test vectors using 10-fold CV
            train_vec, test_vec, train_y, test_y = \
                train_test_split(data_vec, data_y, test_size=0.1, random_state=run, stratify=data_y)
        if use_train == "fix":
            train_vec, train_y = data_vec[:n_train], data_y[:n_train]
            test_vec, test_y = data_vec[n_train:], data_y[n_train:]
        if tune_svm_trans == True:
            # tune parameters for SVM on train data
            C_range = 10. ** np.arange(-1, 3, 1)
            param_grid = dict(C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, cv=cv)
            grid.fit(train_vec, train_y)
            print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
            svm_d2v = svm.LinearSVC(C=grid.best_params_["C"])
        else:
            svm_d2v = svm.LinearSVC()
        # classify test data
        svm_d2v.fit(train_vec, train_y)
        test_pred = svm_d2v.predict(test_vec)
        acc = accuracy_score(test_y, test_pred)
        mic = f1_score(test_y, test_pred, pos_label=None, average="micro")
        mac = f1_score(test_y, test_pred, pos_label=None, average="macro")
        all_acc.append(acc)
        all_mic.append(mic)
        all_mac.append(mac)
        # obtain accuracy and F1-score
        print("d2v_it_pattern accuracy: {}".format(acc))
        print("d2v_it_pattern micro: {}".format(mic))
        print("d2v_it_pattern macro: {}".format(mac))

    print("minSup: {}, d2v_it_pattern avg accuracy: {} ({})".format(
        minSup, np.round(np.average(all_acc), 4), np.round(np.std(all_acc), 3)))
    print("minSup: {}, d2v_it_pattern avg micro: {} ({})".format(
        minSup, np.round(np.average(all_mic), 4), np.round(np.std(all_mic), 3)))
    print("minSup: {}, d2v_it_pattern avg macro: {} ({})".format(
        minSup, np.round(np.average(all_mac), 4), np.round(np.std(all_mac), 3)))

print("runtime: {}".format(timeit.default_timer() - start_time))
