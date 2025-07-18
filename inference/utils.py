import json
import os
import re

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def extract_characters_regex(s, choices):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ""
    return matches[0]


def calculate_bleu_scores(gt_captions, pred_captions):
    """
    计算给定真实描述和预测描述的 BLEU-1, BLEU-2, BLEU-4 分数及平均值。

    Args:
        gt_captions (list of str): 包含所有真实描述的列表。
        pred_captions (list of str): 包含所有模型预测描述的列表。

    Returns:
        dict: 一个包含 'BLEU-1', 'BLEU-2', 'BLEU-4', 和 'Avg_BLEU' 的字典。
    """
    if len(gt_captions) != len(pred_captions):
        raise ValueError("The number of ground truth captions and predicted captions must be the same.")

    # 初始化一个平滑函数，以避免当 n-gram 不匹配时分数为 0
    chencherry = SmoothingFunction()

    # 用于累加每个样本的分数
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []

    for gt, pred in zip(gt_captions, pred_captions):
        # 将句子分割成单词列表 (tokenization)
        reference_tokens = gt.lower().split()
        candidate_tokens = pred.lower().split()

        # NLTK 的 sentence_bleu 期望参考答案是一个列表的列表
        # 因为一个候选可以有多个参考答案，这里我们只有一个
        list_of_references = [reference_tokens]

        # 计算 BLEU-1 (weights=(1, 0, 0, 0))
        bleu_1 = sentence_bleu(list_of_references, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        bleu_1_scores.append(bleu_1)

        # 计算 BLEU-2 (weights=(0.5, 0.5, 0, 0)) - 注意是累积的
        bleu_2 = sentence_bleu(list_of_references, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        bleu_2_scores.append(bleu_2)
        
        # 计算 BLEU-4 (weights=(0.25, 0.25, 0.25, 0.25)) - 标准 BLEU
        bleu_4 = sentence_bleu(list_of_references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        bleu_4_scores.append(bleu_4)

    # 计算整个数据集的平均分数
    avg_bleu_1 = np.mean(bleu_1_scores)
    avg_bleu_2 = np.mean(bleu_2_scores)
    avg_bleu_4 = np.mean(bleu_4_scores)
    
    # 计算三个指标的均值
    avg_of_bleus = np.mean([avg_bleu_1, avg_bleu_2, avg_bleu_4])

    return {
        "BLEU-1": avg_bleu_1,
        "BLEU-2": avg_bleu_2,
        "BLEU-4": avg_bleu_4,
        "Avg_BLEU": avg_of_bleus
    }

def computeIoU(bbox1, bbox2, return_iou=False):
    """
    计算两个 HBB 的 IoU。
    输入 bbox 格式: [x_min, y_min, x_max, y_max]
    """
    # 增加一个除零保护
    if not all(isinstance(c, (int, float)) for c in bbox1) or not all(isinstance(c, (int, float)) for c in bbox2):
        # 如果坐标不是数字，无法计算
        return (0.0, 0.0, 1.0) if return_iou else 0.0

    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # 确保坐标是 x_min <= x_max
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x3, x4 = min(x3, x4), max(x3, x4)
    y3, y4 = min(y3, y4), max(y3, y4)

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    
    # +1 是因为处理离散像素坐标，如果你的坐标是归一化的，就不需要 +1
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area
        
    if return_iou:
        return iou, intersection_area, union_area
    else:
        return iou

def parse_bbox_from_string(bbox_str):
    """
    从 "{<x><y><w><h>}" 格式的字符串中解析出坐标列表。
    如果解析失败，返回 None。
    """
    if not isinstance(bbox_str, str) or "PARSE_ERROR" in bbox_str:
        return None
    
    # 使用你之前的eval.py中的正则表达式来提取数字
    integers = re.findall(r'\d+', bbox_str)
    
    if len(integers) == 4:
        # 将字符串列表转换为整数列表
        return [int(num) for num in integers]
    else:
        # 如果没有找到4个数字，说明格式错误
        return None

def calculate_iou_accuracy(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    计算视觉定位任务的准确率 (X2)。

    Args:
        gt_boxes (list of str): 真实边界框字符串列表。
        pred_boxes (list of str): 预测边界框字符串列表。
        iou_threshold (float): 判断定位是否成功的 IoU 阈值。

    Returns:
        float: 定位准确率 (0.0 到 1.0 之间)。
    """
    if len(gt_boxes) != len(pred_boxes):
        raise ValueError("The number of ground truth boxes and predicted boxes must be the same.")

    successful_predictions = 0
    total_predictions = len(gt_boxes)
    
    if total_predictions == 0:
        return 0.0

    for gt_str, pred_str in zip(gt_boxes, pred_boxes):
        # 1. 解析字符串得到坐标列表
        gt_bbox = parse_bbox_from_string(gt_str)
        pred_bbox = parse_bbox_from_string(pred_str)

        # 2. 如果任一解析失败，则认为此次预测失败
        if gt_bbox is None or pred_bbox is None:
            continue # 跳过，不计入成功数

        # 3. 计算 IoU
        try:
            iou_score, _, _ = computeIoU(gt_bbox, pred_bbox, return_iou=True)
        except Exception as e:
            print(f"Error computing IoU for gt: {gt_bbox}, pred: {pred_bbox}. Error: {e}")
            iou_score = 0.0 # 计算出错也算失败

        # 4. 判断是否成功
        if iou_score >= iou_threshold:
            successful_predictions += 1
            
    # 5. 计算最终准确率
    accuracy = successful_predictions / total_predictions
    return accuracy

def normalize_answer(text):
    """
    对答案进行标准化处理：转小写、去标点、去首尾空格。
    """
    if not isinstance(text, str):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 去除常见的标点符号
    text = re.sub(r'[^\w\s]', '', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def calculate_vqa_accuracy(gt_answers, pred_answers):
    """
    计算开放式 VQA 任务的准确率 (X3)。

    Args:
        gt_answers (list of str): 真实答案字符串列表。
        pred_answers (list of str): 预测答案字符串列表。

    Returns:
        float: VQA 准确率 (0.0 到 1.0 之间)。
    """
    if len(gt_answers) != len(pred_answers):
        raise ValueError("The number of ground truth answers and predicted answers must be the same.")

    correct_predictions = 0
    total_predictions = len(gt_answers)

    if total_predictions == 0:
        return 0.0

    for gt, pred in zip(gt_answers, pred_answers):
        # 1. 标准化真实答案和预测答案
        normalized_gt = normalize_answer(gt)
        normalized_pred = normalize_answer(pred)

        # 如果标准化后为空，则认为不匹配
        if not normalized_gt or not normalized_pred:
            continue
        
        # 2. 进行完全匹配比较
        if normalized_gt == normalized_pred:
            correct_predictions += 1
            
    # 3. 计算最终准确率
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_mme_rs_accuracy(gt_answers, pred_answers):
    """
    计算 MME-RealWorld-RS 任务的准确率，假设预测答案已被解析。

    Args:
        gt_answers (list of str): 
            真实答案字母列表，例如 ['C', 'A', 'B', 'E']。
        pred_answers (list of str): 
            已经过 extract_characters_regex 解析后的预测答案字母列表，
            例如 ['C', 'B', 'PARSE_ERROR', 'E']。

    Returns:
        float: MME-RealWorld-RS 的准确率 (0.0 到 1.0 之间)。
    """
    if len(gt_answers) != len(pred_answers):
        raise ValueError("The number of ground truth answers and predicted answers must be the same.")

    correct_predictions = 0
    total_predictions = len(gt_answers)

    if total_predictions == 0:
        return 0.0

    for gt, pred in zip(gt_answers, pred_answers):
        # 确保比较时大小写一致
        gt_upper = str(gt).upper().strip()
        pred_upper = str(pred).upper().strip()
        
        # 直接进行字符串比较
        # 如果预测结果是'PARSE_ERROR'或空字符串，则肯定不等于真值，自动算错
        if gt_upper == pred_upper:
            correct_predictions += 1
            
    # 计算最终准确率
    accuracy = correct_predictions / total_predictions
    return accuracy