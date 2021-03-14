import re
from nltk.metrics import distance

import torch
import torch.nn.functional as F


def text_accuracy(preds, labels, converter, max_length=25):
    batch_size = preds.size(0)
    preds_str = prediction_string(preds, converter, max_length=max_length)
    correct = count_text_correct(preds_str, labels)
    avg = correct / batch_size

    return avg


def count_text_correct(preds_str, labels):
    n_correct = 0
    for pred, gt in zip(preds_str, labels):
#         print(f'log before: pred:{pred} == gt:{gt}')
        
        pred, pred_eos = clean_text(pred, eos_str='[s]')
        gt, gt_eos = clean_text(gt, eos_str='[s]')
#         print(f'log after: pred:{pred} == gt:{gt}')
        
        if pred == gt:
            n_correct += 1

    return n_correct


def text_norm_distance(preds, labels, converter, max_length=25):
    batch_size = preds.size(0)
    preds_str = prediction_string(preds, converter, max_length=max_length)
    sum_norm_ed = count_norm_distance(preds_str, labels)
    avg = sum_norm_ed / batch_size
    return avg


def count_norm_distance(preds_str, labels):
    norm_ed = 0
    for pred, gt in zip(preds_str, labels):
        pred, pred_eos = clean_text(pred, eos_str='[s]')
        
        gt, gt_eos = clean_text(gt, eos_str='[s]')

        ned = normalized_distance(pred, gt)
        norm_ed += ned

    return norm_ed


def normalized_distance(pred_text, gt_text):
    ned = 0
    # ICDAR2019 Normalized Edit Distance
    if len(gt_text) == 0 or len(pred_text) == 0:
        ned = 0
    elif len(gt_text) > len(pred_text):
        ned = 1 - distance.edit_distance(pred_text, gt_text) / len(gt_text)
    else:
        ned = 1 - distance.edit_distance(pred_text, gt_text) / len(pred_text)

    return ned


def text_confidence_scores(preds, labels, converter, max_length=25):
    batch_size = preds.size(0)
    preds_prob = prediction_probability(preds)
    preds_str = prediction_string(preds, converter, max_length=max_length)
    confidence_list = batch_confidence_scores(preds_str, preds_prob, labels)
    return confidence_list


def batch_confidence_scores(preds_str, preds_prob, labels):
    confidence_list = []
    for gt, pred, pred_prob in zip(labels, preds_str, preds_prob):
        pred, pred_eos = clean_text(pred, eos_str='[s]')
        gt, gt_eos = clean_text(gt, eos_str='[s]')

        cscore = confidence_score(pred_prob, pred_eos)
        confidence_list.append(cscore)

    return confidence_list


def confidence_score(pred_max_prob, pred_eos):
    try:
        pred_max_prob = pred_max_prob[:pred_eos]
        confidence = pred_max_prob.cumprod(dim=0)[-1]
    except ValueError:
        # for empty pred case, when prune after "end of sentence" token ([s])
        confidence = 0

    return confidence


def prediction_index(preds):
    _, preds_index = preds.max(2)
    return preds_index


def prediction_probability(preds):
    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    return preds_max_prob


def prediction_string(preds, converter, max_length=25):
    batch_size = preds.size(0)
    length = torch.IntTensor([max_length] * batch_size)

    preds_index = prediction_index(preds)
    preds_string = converter.decode(preds_index, length)
    return preds_string


def clean_text(text, eos_str='[s]'):
    text, text_eos = prune_eos(text, eos_str=eos_str)
    text = case_insensitive_filter(text)
    return text, text_eos


def prune_eos(text, eos_str='[s]'):
    text_eos = text.find(eos_str)
    text = text[:text_eos]
    return text, text_eos


def case_insensitive_filter(text):
    text = text.lower()
    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
    text = re.sub(out_of_alphanumeric_case_insensitve, '', text)
    return text
