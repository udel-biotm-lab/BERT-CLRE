from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="binary", help='default:binary, possible other options:{chemprot}')
parser.add_argument('--fold_number',type=int, default=10, help='')
parser.add_argument('--step',type=str, default=2, help='')
parser.add_argument('--task_name',type=str, default='', help='')
args = parser.parse_args()

results = dict()
results["f1 score"] = 0
results["recall"] = 0
results["precision"] = 0
results["specificity"] = 0

results_macro = dict()
results_macro["f1 score"] = 0
results_macro["recall"] = 0
results_macro["precision"] = 0
results_macro["specificity"] = 0
for fi in range(1,args.fold_number+1):


    if args.fold_number==0:
        testdf = pd.read_csv(args.answer_path+'test.tsv', sep="\t", index_col=None)
        preddf = pd.read_csv(args.output_path+'test_results.tsv', sep="\t", header=None)
    else:
        testdf = pd.read_csv(args.answer_path+str(fi)+'/test.tsv', sep="\t", index_col=None)
        preddf = pd.read_csv(args.output_path+str(fi)+'/test_results.tsv', sep="\t", header=None)

    # binary
    if args.task == "binary":
        pred = [preddf.iloc[i].tolist() for i in preddf.index]
        pred_class = [np.argmax(v) for v in pred]
        pred_prob_one = [v[1] for v in pred]

        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])

        results["f1 score"] += f[1]
        results["recall"] += r[1]
        results["precision"] += p[1]
        #results["specificity"] = r[0]
        pm,rm,fm,sm = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"],average='macro')
        results_macro["f1 score"] += fm
        results_macro["recall"] += rm
        results_macro["precision"] += pm

# chemprot
# micro-average of 5 target classes
# see "Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction (Mehryary, 2018)" for details
if args.task == "DDI" or args.task == "ChemProt":
    testdf = pd.read_csv(args.answer_path+'test.tsv', sep="\t", index_col=None)
    preddf = pd.read_csv(args.output_path+'test_results.tsv', sep="\t", header=None)

    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]

    if args.task == "DDI":
        label_list=["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", 'DDI-false']
        str_to_int_mapper = dict()
        for i,v in enumerate(label_list):
            str_to_int_mapper[v] = i

        test_answer = [str_to_int_mapper[v] for v in testdf["label"]]


        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, labels=[0,1,2,3], average="micro")
        results = dict()
        results["f1 score"] = f
        results["recall"] = r
        results["precision"] = p
    elif args.task == "ChemProt":
        label_list=["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]
        str_to_int_mapper = dict()
        for i,v in enumerate(label_list):
            str_to_int_mapper[v] = i
        test_answer = [str_to_int_mapper[v] for v in testdf["label"]]


        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, labels=[0,1,2,3,4], average="micro")
        results = dict()
        results["f1 score"] = f
        results["recall"] = r
        results["precision"] = p

output_results=args.task_name+'_results_'+args.step+'.txt'

if args.fold_number==0:
    fold_num=1
else:
    fold_num=args.fold_number
with open(output_results, 'w') as file_object:
    file_object.write('Epoch '+args.step)
    file_object.write('\n')
    for k,v in results.items():
        print("{:11s} : {:.2%}".format(k,v/fold_num))
        file_object.write("{:11s} : {:.2%}".format(k,v/fold_num))
        file_object.write('\n')

print('Macro average results:')
for k,v in results_macro.items():
    print("{:11s} : {:.2%}".format(k,v/fold_num))