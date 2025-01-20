import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import evaluate


# 加载数据集
def load_custom_dataset():
    dataset = load_dataset('json', data_files='sft_dataset.json')
    print(dataset)
    return dataset

dataset = load_custom_dataset()
tokenizer = AutoTokenizer.from_pretrained("D:\LLMProject\Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("D:\LLMProject\Qwen2.5-0.5B")

# 数据预处理函数
def preprocess_function(examples):
    all_inputs = []
    all_labels = []
    if 'conversations' in examples:
        for item in zip(examples['conversations']):
            conversations = item[0]
            human_input = ""
            gpt_response = ""
            if len(conversations) >= 2:
                human_input = conversations[0]["value"]
                gpt_response = conversations[1]["value"]
            input_ids = tokenizer.encode(human_input, truncation=True, padding='max_length')
            label_ids = tokenizer.encode(gpt_response, truncation=True, padding='max_length')
            all_inputs.append(input_ids)
            all_labels.append(label_ids)
    result = {
        'input_ids': all_inputs,
        'attention_mask': [[1] * len(ids) for ids in all_inputs],
        'labels': all_labels
    }
    # 检查返回值是否包含所需的键
    required_keys = ['input_ids', 'attention_mask', 'labels']
    for key in required_keys:
        if key not in result:
            raise KeyError(f"Missing key {key} in preprocess_function return value")
    return result


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 检查 tokenized_dataset['train'] 是否包含所需的键
required_keys = ['input_ids', 'attention_mask', 'labels']
if 'train' in tokenized_dataset:
    for key in required_keys:
        if key not in tokenized_dataset['train'].features:
            raise KeyError(f"Missing key {key} in tokenized_dataset['train']")
else:
    raise KeyError("Missing 'train' split in tokenized_dataset")


# 重新构建为标准的Dataset实例
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }


custom_train_dataset = CustomDataset(
    tokenized_dataset['train']['input_ids'],
    tokenized_dataset['train']['attention_mask'],
    tokenized_dataset['train']['labels']
)

tokenized_dataset = DatasetDict({'train': custom_train_dataset})

metric = evaluate.load("accuracy")  # level


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    accuracy = metric.compute(predictions=predictions, references=torch.tensor(labels))
    return accuracy

args_train = TrainingArguments(
    output_dir='./sft_qwen',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args_train,
    train_dataset=tokenized_dataset['train'],
    compute_metrics=compute_metrics
)


# 开始训练
trainer.train()


# 保存模型
trainer.save_model('sft_qwen2.5_0.5B')
