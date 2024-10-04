import numpy as np
import os
import argparse
import pandas as pd
from  sklearn.metrics import f1_score, log_loss, roc_auc_score as AUC, confusion_matrix
from sklearn.metrics import roc_auc_score
import glob

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='FF-ALL', help='dataset_name')

    return parser.parse_args()


def ff_metrics(testset):
    result=dict()
    temp_set=dict()
    for k,j in enumerate(['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']):
        d=testset[k*140:(k+1)*140]
        temp_set[j]=d

    for i in ['Deepfakes','NeuralTextures','FaceSwap','Face2Face','all']:
        if i!='all':
            rs=test_metric(temp_set[i]+temp_set['Origin'])
        else:
            rs=test_metric(testset) 
        result[i]=rs
    return result

def test_metric(testset):
    video_labels=[]
    video_preds=[]
    for i in testset:
        video_preds.append(i['pred'])
        video_labels.append(i['label'])
    video_thres,video_acc,video_f1=acc_f1_eval(video_labels,video_preds)
    video_auc=AUC(video_labels,video_preds)
    video_log_loss = log_loss(video_labels, video_preds, labels=[0, 1])
    rs={'video_acc':video_acc,'video_threshold':video_thres,'video_auc':video_auc,'video_f1':video_f1, 'video_log_loss':video_log_loss}
    return rs

def acc_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    acc=np.mean((preds>=thres)==labels)
    return thres,acc

def acc_f1_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    thres_result = (preds>=thres)==labels
    acc=np.mean(thres_result)
    f1 = f1_score(labels, thres_result)
    return thres,acc,f1

def basic_eval(labels, preds,result_root):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    thres_result = preds>=thres
    real_basic = confusion_matrix(labels, labels, labels=[0,1])
    pred_basic = confusion_matrix(labels, thres_result, labels=[0,1])
    tn, fp, fn, tp = pred_basic.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return str(real_basic.ravel()), str(pred_basic.ravel()),precision,recall


def produce_result(labels, predicts, result_root):
    real_basic, pred_basic,precision,recall = basic_eval(labels, predicts, result_root)
    video_thres, video_acc, video_f1=acc_f1_eval(labels,predicts)
    video_auc=roc_auc_score(labels,predicts)
    return video_acc, video_auc, pred_basic,precision,recall

def get_ff_result(videos, predicts, labels,result_root):
    fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures','FF-ALL']
    result_grps = {'Deepfakes':{'labels':[],'predicts':[]}, 'Face2Face':{'labels':[],'predicts':[]}, \
        'FaceSwap':{'labels':[],'predicts':[]}, 'NeuralTextures':{'labels':[],'predicts':[]}, \
            'original_sequences':{'labels':[],'predicts':[]}}

    for video, predict, label in zip(videos, predicts, labels):
        if 'original_sequences' in video:
            result_grps['original_sequences']['labels'].append(label)
            result_grps['original_sequences']['predicts'].append(predict)
        else:
            fake_type = video.split('/')[1]
            result_grps[fake_type]['labels'].append(label)
            result_grps[fake_type]['predicts'].append(predict)
    
    results = []
    labels_r = result_grps['original_sequences']['labels']
    predicts_r = result_grps['original_sequences']['predicts']
    for fake_type in fake_types:
        if fake_type == 'FF-ALL':
            labels_f, predicts_f = [], []
            for fake_type_1 in fake_types[:-1]:
                labels_f += result_grps[fake_type_1]['labels']
                predicts_f += result_grps[fake_type_1]['predicts']
            labels_rf = labels_r + labels_f
            predicts_rf = predicts_r + predicts_f
        else:
            labels_rf = labels_r + result_grps[fake_type]['labels']
            predicts_rf = predicts_r + result_grps[fake_type]['predicts']

        video_acc, video_auc, pred_basic,precision,recall = produce_result(labels_rf, predicts_rf,result_root)
        result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, fake_type)
        results.append(result)
    return results

def final_scores(result_root='', result_file=''):
    if result_file == '':
        result_files = glob.glob(result_root+'*.csv')
    else:
        result_files = [result_file]
        result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []
    for result_file in result_files:
        results = pd.read_csv(result_file)
        label = results['label'].values
        predict = results['predict'].values
        video = results['video'].values
        dataset_name = os.path.basename(result_file)[:-4]
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)
    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            file.write('\n')
    return video_auc, video_acc
