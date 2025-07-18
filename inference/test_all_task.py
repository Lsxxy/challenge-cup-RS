import json
import re
import os
from tqdm import tqdm
import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalData
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator # <--- 导入 accelerate
from torch.utils.data import DataLoader, Dataset
# 确保你的 utils.py 文件包含了我们之前讨论的所有评测函数
from utils import extract_characters_regex, calculate_bleu_scores, calculate_iou_accuracy, calculate_vqa_accuracy, calculate_mme_rs_accuracy

def load_all_test_data():
    # ... (这个函数保持不变) ...
    tasks = []

    # 1. 加载 VRSBench 数据
    # VRSBench_EVAL_Cap.json
    with open('/root/Documents/code/FM9G4B-V/data/VRSBench/VRSBench_EVAL_Cap.json', 'r') as f:
        for item in json.load(f):
            tasks.append({
                'task_id': item['question_id'],
                'image_path': item['image_id'],
                'task_type': 'vrs_caption',
                'prompt_info': {'question': item['question']},
                'ground_truth': item['ground_truth']
            })

    # VRSBench_EVAL_referring.json
    with open('/root/Documents/code/FM9G4B-V/data/VRSBench/VRSBench_EVAL_referring.json', 'r') as f:
        for item in json.load(f):
            tasks.append({
                'task_id': item['question_id'],
                'image_path': item['image_id'],
                'task_type': 'vrs_referring',
                'prompt_info': {'description': item['question']},
                'ground_truth': item['ground_truth']
            })

    # VRSBench_EVAL_vqa.json
    with open('/root/Documents/code/FM9G4B-V/data/VRSBench/VRSBench_EVAL_vqa.json', 'r') as f:
        for item in json.load(f):
            tasks.append({
                'task_id': item['question_id'],
                'image_path': item['image_id'],
                'task_type': 'vrs_vqa',
                'prompt_info': {'question': item['question']},
                'ground_truth': item['ground_truth']
            })

    # 2. 加载 MME-RealWorld-RS 数据
    with open('/root/Documents/code/FM9G4B-V/data/MME/MME_RealWorld.json', 'r') as f: # 假设所有MME任务在一个文件里
        for item in json.load(f):
            # 只处理 Remote Sensing 子任务
            if item.get('Subtask') == 'Remote Sensing':
                tasks.append({
                    'task_id': item['Question_id'],
                    'image_path': item['Image'],
                    'task_type': 'mme_vqa',
                    'prompt_info': {
                        'question': item['Text'],
                        'choices': item['Answer choices'] # 注意：这里直接是列表，不是字符串
                    },
                    'ground_truth': item['Ground truth'] # 'A', 'B', 'C'...
                })
            
    return tasks

def build_prompt_for_task(task):
    """辅助函数：为单个任务构建 prompt"""
    prompt = ""
    if task['task_type'] == 'vrs_caption':
        prompt = "Please describe this image in detail."
    
    elif task['task_type'] == 'vrs_vqa':
        question = task['prompt_info']['question']
        prompt = f"Answer the following question with a short word or phrase: {question}"
    
    elif task['task_type'] == 'vrs_referring':
        description = task['prompt_info']['description']
        prompt = f"What is the bounding box for the object described as \"{description}\"? Provide the coordinates in the format {{<x_min><y_min><x_max><y_max>}}."
        
    elif task['task_type'] == 'mme_vqa':
        question = task['prompt_info']['question']
        # 注意：MME的数据格式中 choices 已经是列表了
        choices_str = "\n".join(task['prompt_info']['choices'])
        sys_prompt = "Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option."
        prompt = f"{question}\n{choices_str}\n{sys_prompt}\nThe best answer is:"
        
    return prompt

def calculate_final_scores(results):
    # ... (这个函数基本保持不变，但从 results 中提取数据的方式需要调整) ...
    # --- 1. 分离不同任务的结果 ---
    vrs_caption_results = [r for r in results if r['task_type'] == 'vrs_caption']
    vrs_referring_results = [r for r in results if r['task_type'] == 'vrs_referring']
    vrs_vqa_results = [r for r in results if r['task_type'] == 'vrs_vqa']
    mme_vqa_results = [r for r in results if r['task_type'] == 'mme_vqa']

    # --- 2. 计算各项指标 ---
    # VRSBench
    gt_captions = [r['ground_truth'] for r in vrs_caption_results]
    pred_captions = [r['model_output'] for r in vrs_caption_results]
    bleu_results = calculate_bleu_scores(gt_captions, pred_captions)
    # 比赛要求是三个指标的均值
    X1 = bleu_results['Avg_BLEU']

    gt_boxes = [r['ground_truth'] for r in vrs_referring_results]
    pred_boxes = [r['model_output'] for r in vrs_referring_results]
    X2 = calculate_iou_accuracy(gt_boxes, pred_boxes, iou_threshold=0.5)
    
    gt_vqa = [r['ground_truth'] for r in vrs_vqa_results]
    pred_vqa = [r['model_output'] for r in vrs_vqa_results]
    X3 = calculate_vqa_accuracy(gt_vqa, pred_vqa)

    # MME-RealWorld-RS
    # 注意：这里需要传入包含 choices 的完整结果
    X1_mme = calculate_mme_rs_accuracy(mme_vqa_results)


    # --- 3. 计算最终得分 (根据比赛文档) ---
    S1 = X1 * 25 + X2 * 25 + X3 * 50
    S2 = X1_mme * 100
    S_final = (S1 + S2) / 2

    # --- 4. 打印报告 ---
    print("\n" + "="*40)
    print("------ Stage 1 Performance Report ------")
    print("="*40)
    print(f"VRSBench Score (S1): {S1:.2f} / 100.0")
    print(f"  - Caption (X1_avg_bleu = {X1:.3f}): {X1 * 25:.2f} / 25")
    print(f"    (Details: BLEU-1: {bleu_results['BLEU-1']:.3f}, BLEU-2: {bleu_results['BLEU-2']:.3f}, BLEU-4: {bleu_results['BLEU-4']:.3f})")
    print(f"  - Referring (X2_acc@0.5 = {X2:.3f}): {X2 * 25:.2f} / 25")
    print(f"  - VQA (X3_acc = {X3:.3f}): {X3 * 50:.2f} / 50")
    print("-" * 20)
    print(f"MME-RealWorld-RS Score (S2): {S2:.2f} / 100.0")
    print(f"  - VQA Accuracy (X1_mme = {X1_mme:.3f})")
    print("-" * 20)
    print(f"Final Combined Score (S): {S_final:.2f} / 100.0")
    print("="*40)

    return S_final


#使用vllm进行分布式预测################################################################################################
def run_vllm_inference(llm, tokenizer, tasks, image_base_path_dict, sampling_params):
    """
    使用 vLLM 对所有任务进行统一的、并行的推理。
    """
    # 1. 准备 vLLM 需要的输入
    prompts_for_vllm = []
    multi_modal_data_for_vllm = []
    valid_tasks = [] # 只保留图像加载成功的任务

    print("Preparing inputs for vLLM...")
    for task in tqdm(tasks, desc="Loading images and building prompts"):
        # 拼接完整的图像路径
        if task['task_type'] == 'mme_vqa':
            full_image_path = os.path.join(image_base_path_dict['mme_vqa'], task['image_path'])
        else:
            full_image_path = os.path.join(image_base_path_dict[task['task_type']], task['image_path'])

        try:
            image = Image.open(full_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {full_image_path}. Skipping task {task['task_id']}.")
            continue
        
        # 获取文本 prompt
        text_prompt = build_prompt_for_task(task)
        
        # 构建 vLLM 需要的最终 prompt (包含图像占位符)
        # 你的模型使用的占位符可能是 <image> 或其他，请根据模型文档确认
        final_prompt = f"<image>\n{text_prompt}"
        
        # 使用 apply_chat_template 格式化
        formatted_prompt = tokenizer.apply_chat_template(
            conversation=[{'role': 'user', 'content': final_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        
        prompts_for_vllm.append(formatted_prompt)
        multi_modal_data_for_vllm.append(MultiModalData(image=image))
        valid_tasks.append(task)

    # 2. 执行 vLLM 并行推理
    print(f"\nRunning inference on {len(prompts_for_vllm)} tasks with vLLM...")
    outputs = llm.generate(
        prompts_for_vllm, 
        sampling_params, 
        multi_modal_data=multi_modal_data_for_vllm
    )
    print("Inference completed.")

    # 3. 解析和整理输出结果
    results = []
    print("Parsing outputs...")
    for i, output in enumerate(tqdm(outputs, desc="Parsing results")):
        task = valid_tasks[i]
        raw_output = output.outputs[0].text
        
        # 复用你之前的解析逻辑
        if task['task_type'] == 'mme_vqa':
            choices_list = task['prompt_info']['choices']
            parsed_answer = extract_characters_regex(raw_output, choices_list)
        elif task['task_type'] == 'vrs_referring':
            match = re.search(r'\{?<?(\d+)>?<?(\d+)>?<?(\d+)>?<?(\d+)>?\}?', str(raw_output))
            if match:
                parsed_answer = f"{{<{match.group(1)}><{match.group(2)}><{match.group(3)}><{match.group(4)}>}}"
            else:
                parsed_answer = "PARSE_ERROR"
        else: # for caption and vrs_vqa
            parsed_answer = str(raw_output).strip()
        
        # 保存格式化的结果
        results.append({
            'task_id': task['task_id'],
            'task_type': task['task_type'],
            'model_output': parsed_answer,
            'ground_truth': task['ground_truth'],
            'choices': task['prompt_info'].get('choices', [])
        })
        
    return results


if __name__ == '__main__':
    # --- 1. 定义超参数 ---
    MODEL_FILE = '/root/Documents/code/FM9G4B-V/model/'
    # 设置你想使用的GPU数量
    TENSOR_PARALLEL_SIZE = 4 
    
    # --- 2. 初始化 vLLM 模型和 Tokenizer ---
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_FILE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        dtype='bfloat16', # 与你之前的 torch_dtype 一致
        # 如果遇到显存不足，可以尝试降低这个值
        gpu_memory_utilization=0.90 
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FILE, trust_remote_code=True)
    print("vLLM engine is ready.")

    # --- 3. 定义采样参数 ---
    sampling_params = SamplingParams(
        temperature=0.0, # 对于评测，通常使用确定性采样
        top_p=1.0,
        max_tokens=256 # 限制最大生成长度
    )

    # --- 4. 读取和准备数据 ---
    print("Loading test data...")
    test_data = load_all_test_data()
    print(f"Loaded {len(test_data)} tasks in total.")

    image_base_path_dict = {
        'vrs_caption': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
        'vrs_referring': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
        'vrs_vqa': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
        'mme_vqa': '/root/Documents/code/FM9G4B-V/data/MME'
    } 

    # --- 5. 执行 vLLM 分布式推理 ---
    all_inference_results = run_vllm_inference(llm, tokenizer, test_data, image_base_path_dict, sampling_params)
    
    # --- 6. 保存和计算最终分数 (这部分逻辑完全复用) ---
    with open('/root/Documents/code/FM9G4B-V/data/inference_results_vllm.json', 'w') as f:
        json.dump(all_inference_results, f, indent=4)
    print(f"Aggregated inference results saved to 'inference_results_vllm.json'. Total items: {len(all_inference_results)}")

    print("Calculating final scores...")
    score = calculate_final_scores(all_inference_results)

















#单卡使用的inference和主函数##########################################################################################
# def run_batch_inference(model, tokenizer, tasks, image_base_path_dict, batch_size=8):
#     """
#     对所有任务进行统一的、批量的推理。
#     """
#     results = []
    
#     # 开启推理模式，关闭梯度计算以加速并节省内存
#     with torch.inference_mode():
#         # 手动创建批次进行循环
#         for i in tqdm(range(0, len(tasks), batch_size), desc="Running Batch Inference"):
#             # 1. 获取当前批次的任务
#             batch_tasks = tasks[i:i+batch_size]
            
#             # --- 2. 批量构建模型输入 (msgs) ---
#             batch_msgs = []
#             for task in batch_tasks:
#                 # 拼接完整的图像路径
#                 # 注意：MME的 image_path 已经是 "remote_sensing/..." 的形式了
#                 if task['task_type'] == 'mme_vqa':
#                     full_image_path = os.path.join(image_base_path_dict['mme_vqa'], task['image_path'])
#                 else:
#                     full_image_path = os.path.join(image_base_path_dict[task['task_type']], task['image_path'])

#                 try:
#                     image = Image.open(full_image_path).convert('RGB')
#                 except FileNotFoundError:
#                     print(f"Warning: Image not found at {full_image_path}. Skipping task {task['task_id']}.")
#                     # 添加一个错误标记，方便后续跳过
#                     batch_msgs.append("IMAGE_NOT_FOUND")
#                     continue
                
#                 # 构建单个任务的 prompt
#                 prompt = build_prompt_for_task(task)
                
#                 # 构建符合 model.chat 格式的输入
#                 batch_msgs.append([{'role': 'user', 'content': [image, prompt]}])

#             # 过滤掉图像加载失败的样本
#             valid_batch_tasks = [task for task, msg in zip(batch_tasks, batch_msgs) if msg != "IMAGE_NOT_FOUND"]
#             valid_batch_msgs = [msg for msg in batch_msgs if msg != "IMAGE_NOT_FOUND"]
            
#             if not valid_batch_msgs:
#                 continue

#             # --- 3. 批量模型推理 ---
#             try:
#                 # 这是根据 chat.py 确定的真实批量调用方式！
#                 raw_outputs = model.chat(
#                     image=None,
#                     msgs=valid_batch_msgs,
#                     tokenizer=tokenizer,
#                     max_new_tokens=256 # 限制输出长度，可以根据任务调整
#                 )
#             except Exception as e:
#                 # 捕获到任何异常
#                 print("\n" + "="*50)
#                 print(f"FATAL ERROR: An exception occurred during model inference for the batch starting at index {i}.")
#                 print(f"Stopping the program.")
#                 print(f"Error Type: {type(e).__name__}")
#                 print(f"Error Details: {e}")
#                 print("="*50)
#                 # 重新抛出异常，使程序终止
#                 raise e
            
#             # raw_outputs 现在是一个列表，长度等于 valid_batch_msgs 的长度

#             # --- 4. 批量解析和保存结果 ---
#             for task_idx, task in enumerate(valid_batch_tasks):
#                 raw_output = raw_outputs[task_idx]
                
#                 # 解析输出
#                 if task['task_type'] == 'mme_vqa':
#                     choices_list = task['prompt_info']['choices']
#                     parsed_answer = extract_characters_regex(raw_output, choices_list)
#                 elif task['task_type'] == 'vrs_referring':
#                     match = re.search(r'\{?<?(\d+)>?<?(\d+)>?<?(\d+)>?<?(\d+)>?\}?', str(raw_output))
#                     if match:
#                         parsed_answer = f"{{<{match.group(1)}><{match.group(2)}><{match.group(3)}><{match.group(4)}>}}"
#                     else:
#                         parsed_answer = "PARSE_ERROR"
#                 else: # for caption and vrs_vqa
#                     parsed_answer = str(raw_output).strip()

#                 # 保存格式化的结果
#                 results.append({
#                     'task_id': task['task_id'],
#                     'task_type': task['task_type'],
#                     'model_output': parsed_answer,
#                     'ground_truth': task['ground_truth'],
#                     'choices': task['prompt_info'].get('choices', []) # 为MME评测保存choices
#                 })
                
#     return results
# if __name__ == '__main__':
#     # 1.定义模型与分词器
#     model_file = '/root/Documents/code/FM9G4B-V/model'
#     print("Loading model and tokenizer...")
#     model = AutoModel.from_pretrained(
#         model_file, 
#         trust_remote_code=True,
#         attn_implementation='flash_attention_2', 
#         torch_dtype=torch.bfloat16
#     ).eval().cuda()
#     tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
#     print("Model and tokenizer loaded.")

#     # 2.读取所有test_data
#     print("Loading test data...")
#     test_data = load_all_test_data()
#     print(f"Loaded {len(test_data)} tasks in total.")

#     # 3.获取test_data对应的图像主路径
#     image_base_path_dict = {
#         'vrs_caption': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'vrs_referring': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'vrs_vqa': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'mme_vqa': '/root/Documents/code/FM9G4B-V/data/MME' # MME的Image字段包含了子目录
#     }

#     # 4.获取模型对test_data的结果
#     # BATCH_SIZE可以根据你的GPU显存进行调整，从一个较小的值开始尝试，如 4 或 8
#     BATCH_SIZE =24
#     results = run_batch_inference(model, tokenizer, test_data, image_base_path_dict, batch_size=BATCH_SIZE)
    
#     # [可选] 将中间结果保存到文件，方便调试或重复计算分数
#     with open('/root/Documents/code/FM9G4B-V/inference/inference_results.json', 'w') as f:
#         json.dump(results, f, indent=4)
#     print("Inference results saved to inference_results.json")

#     # 5.获取测试结果
#     # 如果是从文件加载，可以: with open('inference_results.json', 'r') as f: results = json.load(f)
#     print("Calculating final scores...")
#     score = calculate_final_scores(results)
##############################################################################################################



#这个Dataset，collect_fn,inference函数，主函数，都是accelerator+分布式预测的东西############################################################
# class InferenceTaskDataset(Dataset):
#     def __init__(self, tasks, image_base_path_dict):
#         self.tasks = tasks
#         self.image_base_path_dict = image_base_path_dict

#     def __len__(self):
#         return len(self.tasks)

#     def __getitem__(self, idx):
#         task = self.tasks[idx]
#         task_type = task['task_type']
        
#         # 拼接图像路径
#         if task_type == 'mme_vqa':
#             full_image_path = os.path.join(self.image_base_path_dict['mme_vqa'], task['image_path'])
#         else:
#             full_image_path = os.path.join(self.image_base_path_dict[task_type], task['image_path'])

#         try:
#             image = Image.open(full_image_path).convert('RGB')
#         except FileNotFoundError:
#             # 返回 None，让 collate_fn 过滤掉
#             print(f"Warning: Image not found at {full_image_path}. Skipping.")
#             return None

#         prompt = build_prompt_for_task(task)
#         # 将所有信息打包返回
#         return {'image': image, 'prompt': prompt, 'original_task': task}

# def collate_fn_for_chat(batch):
#     # 过滤掉加载失败的样本
#     batch = [item for item in batch if item is not None]
#     if not batch:
#         return None

#     # 不需要堆叠张量，因为 model.chat 接收的是 PIL Image 列表
#     images = [item['image'] for item in batch]
#     prompts = [item['prompt'] for item in batch]
#     original_tasks = [item['original_task'] for item in batch]
    
#     # 构建符合 model.chat 批量输入的 msgs
#     batch_msgs = [[{'role': 'user', 'content': [img, pmt]}] for img, pmt in zip(images, prompts)]
    
#     return {'msgs': batch_msgs, 'original_tasks': original_tasks}

# def run_distributed_inference(model, tokenizer, dataloader, accelerator):
#     results = []
    
#     # 将模型和 dataloader 交给 accelerator 管理
#     model, dataloader = accelerator.prepare(model, dataloader)
    
#     with torch.inference_mode():
#         for batch in tqdm(dataloader, desc=f"Inference on Process {accelerator.process_index}"):
#             if batch is None:
#                 continue
            
#             batch_msgs = batch['msgs']
#             original_tasks = batch['original_tasks']
            
#             # 模型推理
#             # 注意：在 accelerator.prepare 之后，model 可能是被包装过的
#             # 我们需要调用 accelerator.unwrap_model 来获取原始模型以调用 .chat
#             unwrapped_model = accelerator.unwrap_model(model)
#             raw_outputs = unwrapped_model.chat(
#                 image=None,
#                 msgs=batch_msgs,
#                 tokenizer=tokenizer,
#                 max_new_tokens=256
#             )
            
#             # 解析和保存结果
#             for task_idx, task in enumerate(original_tasks):
#                 raw_output = raw_outputs[task_idx]
                
#                 # 解析输出
#                 if task['task_type'] == 'mme_vqa':
#                     choices_list = task['prompt_info']['choices']
#                     parsed_answer = extract_characters_regex(raw_output, choices_list)
#                 elif task['task_type'] == 'vrs_referring':
#                     match = re.search(r'\{?<?(\d+)>?<?(\d+)>?<?(\d+)>?<?(\d+)>?\}?', str(raw_output))
#                     if match:
#                         parsed_answer = f"{{<{match.group(1)}><{match.group(2)}><{match.group(3)}><{match.group(4)}>}}"
#                     else:
#                         parsed_answer = "PARSE_ERROR"
#                 else: # for caption and vrs_vqa
#                     parsed_answer = str(raw_output).strip()

#                 # 保存格式化的结果
#                 results.append({
#                     'task_id': task['task_id'],
#                     'task_type': task['task_type'],
#                     'model_output': parsed_answer,
#                     'ground_truth': task['ground_truth'],
#                     'choices': task['prompt_info'].get('choices', []) # 为MME评测保存choices
#                 })
                
#     # --- 关键：在多卡环境下收集所有进程的结果 ---
#     all_results = accelerator.gather_for_metrics(results)
    
#     return all_results

# if __name__ == '__main__':
#     # 1. 初始化 Accelerator
#     accelerator = Accelerator()
    
#     # 2. 定义模型与分词器 (在所有进程中都执行)
#     model_file = '/root/Documents/code/FM9G4B-V/model'
#     # 只在主进程打印加载信息
#     if accelerator.is_main_process:
#         print("Loading model and tokenizer...")
        
#     # 模型加载方式不变
#     model = AutoModel.from_pretrained(
#         model_file, 
#         trust_remote_code=True,
#         attn_implementation='flash_attention_2', 
#         torch_dtype=torch.bfloat16
#     ) # 注意：这里先不 .cuda()，让 accelerator 来处理设备分配
#     tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    
#     if accelerator.is_main_process:
#         print("Model and tokenizer loaded.")

#     # 3. 读取和准备数据 (在所有进程中都执行)
#     if accelerator.is_main_process:
#         print("Loading test data...")
#     test_data = load_all_test_data()
#     if accelerator.is_main_process:
#         print(f"Loaded {len(test_data)} tasks in total.")

#     image_base_path_dict = {
#         'vrs_caption': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'vrs_referring': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'vrs_vqa': '/root/Documents/code/FM9G4B-V/data/VRSBench/Images_val',
#         'mme_vqa': '/root/Documents/code/FM9G4B-V/data/MME' # MME的Image字段包含了子目录
#     } 

#     # 4. 创建 Dataset 和 DataLoader
#     BATCH_SIZE_PER_DEVICE = 10 # 这是每张卡的 batch size
#     dataset = InferenceTaskDataset(test_data, image_base_path_dict)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE_PER_DEVICE,
#         shuffle=False,
#         num_workers=4,
#         collate_fn=collate_fn_for_chat
#     )

#     # 5. 执行分布式推理
#     all_inference_results = run_distributed_inference(model, tokenizer, dataloader, accelerator)
    
#     # --- 关键：只有主进程负责后续的计算和打印 ---
#     if accelerator.is_main_process:
#         # [可选] 保存聚合后的结果
#         with open('inference_results_all_gpus.json', 'w') as f:
#             json.dump(all_inference_results, f, indent=4)
#         print(f"Aggregated inference results saved. Total items: {len(all_inference_results)}")

#         # 6. 获取测试结果
#         print("Calculating final scores...")
#         score = calculate_final_scores(all_inference_results)
###############################################################################################


