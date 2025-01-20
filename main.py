import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import evaluate


# 加载数据集
def load_custom_dataset():
    all_data = []
    for item in load_dataset('json', data_files='sft_dataset.json')['train']:
        conversations = item["conversations"]
        if len(conversations) >= 2:
            human_input = conversations[0]["value"]
            gpt_response = conversations[1]["value"]
            all_data.append({
                "input": human_input,
                "output": gpt_response
            })
    return Dataset.from_list(all_data)


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("D:\LLMProject\Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("D:\LLMProject\Qwen2.5-0.5B")

# 加载数据集
dataset = load_custom_dataset()


# 数据预处理函数
def preprocess_function(examples):
    all_inputs = []
    all_labels = []
    if 'conversations' in examples:
        for item in zip(examples['conversations']):
            conversations = item[0]
            if len(conversations) >= 2:
                human_input = conversations[0]["value"]
                gpt_response = conversations[1]["value"]
                input_ids = tokenizer.encode(human_input, truncation=True, padding='max_length')
                label_ids = tokenizer.encode(gpt_response, truncation=True, padding='max_length')
                all_inputs.append(input_ids)
                all_labels.append(label_ids)
    return {
        'input_ids': all_inputs,
        'attention_mask': [[1] * len(ids) for ids in all_inputs],
        'labels': all_labels
    }



tokenized_dataset = dataset.map(preprocess_function, batched=True)


# 定义评估指标
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    accuracy = metric.compute(predictions=predictions, references=torch.tensor(labels))
    return accuracy


# 训练参数设置
training_args = TrainingArguments(
    output_dir='sft_qwen',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)


# 开始训练
trainer.train()


# 保存模型
trainer.save_model('fine_tuned_qwen2.5_0.5B')