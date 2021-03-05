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
### represent a trans in form of items and fis,
### learn trans vectors using Doc2Vec (PV-DBOW) from items and learn trans vectors using Doc2Vec (PV-DBOW) from fis
### take average of two trans vectors
### use SVM as classifier
with open("output.txt") as f:
    spmf = Spmf("USpan", input_filename="DataBase_HUSRM.txt",
                output_filename="output.txt",
                spmf_bin_location_dir="C:/Users/HP/Downloads/",
                arguments=[35,4])
    spmf.run()
with open('output.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('output.txt','w') as target:
    for _, line in data:
        target.write( line )

### variables ###
data_name = "output.txt"
path = "output.txt"
size_train = 0.8
use_train = "fix" # cv, fix
pattern = "fis" # fis, cs
w_it = 1
w_fis = 1
tune_svm_trans = False # True, False
dim_d2v = 128
para_minSup = [0.002]
n_run = 10

### functions ###
# load data file in form of items
def load_items(file_name):
    labels, sentences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
                labels.append(label)
                sentences.append(content.rstrip().split(" "))
    return sentences, labels
# load data file in form of frequent itemsets
def load_fis(file_name):
    labels, sentences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
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
# load data in the form of items
data_path = path
data_it_X, data_it_y = load_items(data_path)
# assign a label to each sentence
data_sen_it = labelizeSentences(data_it_X)
for minSup in para_minSup:
    print("d2v_i_d2v_f_classify, data: {}, train: {}, pattern: {}, w_it: {}, w_fis: {}, minSup={}".format(
        data_name, use_train, pattern, w_it, w_fis, minSup))
    # load data in the form of frequent itemsets
    data_path = path
    data_fis_X, data_fis_y = load_fis(data_path)
    if use_train == "fix":
        if data_name == "snippets":
            n_train = 10060
        if data_name == "dblp":
            n_train = 61479
        if data_name == "mr":
            n_train = 7108
        else:
            n_train = int(len(data_fis_y) * size_train)
    # assign a label to each sentence
    data_sen_fis = labelizeSentences(data_fis_X)
    all_acc, all_mic, all_mac = [], [], []
    for run in range(n_run):
        print("run={}".format(run))
        # learn trans vectors using Doc2Vec (PV-DBOW) from items
        d2v_it = Doc2Vec(size=dim_d2v, min_count=0, workers=16, dm=0, iter=50)
        d2v_it.build_vocab(data_sen_it)
        d2v_it.train(data_sen_it, total_examples=d2v_it.corpus_count, epochs=d2v_it.iter)
        data_it_vec = [d2v_it.docvecs[idx] for idx in range(len(data_sen_it))]
        del d2v_it  # delete unneeded model memory
        # learn trans vectors using Doc2Vec (PV-DBOW) from frequent itemsets
        d2v_fis = Doc2Vec(size=dim_d2v, min_count=0, workers=16, dm=0, iter=50)
        d2v_fis.build_vocab(data_sen_fis)
        d2v_fis.train(data_sen_fis, total_examples=d2v_fis.corpus_count, epochs=d2v_fis.iter)
        data_fis_vec = [d2v_fis.docvecs[idx] for idx in range(len(data_sen_fis))]
        del d2v_fis  # delete unneeded model memory
        # take weighted average of trans vectors
        data_it_vec = np.array(data_it_vec).reshape(len(data_it_vec), dim_d2v)
        data_fis_vec = np.array(data_fis_vec).reshape(len(data_fis_vec), dim_d2v)
        data_vec = (w_it * data_it_vec + w_fis * data_fis_vec) / 2
        if use_train == "cv":
            # generate train and test vectors using 10-fold CV
            train_vec, test_vec, train_y, test_y = \
                train_test_split(data_vec, data_fis_y, test_size=0.1, random_state=run, stratify=data_fis_y)
        if use_train == "fix":
            train_vec, train_y = data_vec[:n_train], data_fis_y[:n_train]
            test_vec, test_y = data_vec[n_train:], data_fis_y[n_train:]
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
        print("accuracy: {}".format(acc))
        print("micro: {}".format(mic))
        print("macro: {}".format(mac))

    print("minSup: {}, avg accuracy: {} ({})".format(
        minSup, np.round(np.average(all_acc), 4), np.round(np.std(all_acc), 3)))
    print("minSup: {}, avg micro: {} ({})".format(
        minSup, np.round(np.average(all_mic), 4), np.round(np.std(all_mic), 3)))
    print("minSup: {}, avg macro: {} ({})".format(
        minSup, np.round(np.average(all_mac), 4), np.round(np.std(all_mac), 3)))

print("runtime: {}".format(timeit.default_timer() - start_time))
