from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import samples
from json import dumps as json_dumps


model_name = "deepseekr1-1.5b"
tokenizers = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


print("Model loaded successfully")

with open("datasets.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line = json_dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("Dataset saved successfully")


from datasets import load_dataset
dataset = load_dataset("json", data_files="datasets.jsonl", split="train")
print("数据数量", len(dataset))


train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
print("训练集数量", len(train_dataset))
print("测试集数量", len(test_dataset))

def tokenizer_function(examples):
    text = [f"{prompt}\n{completion}"for prompt ,completion in zip(examples["prompt"], examples["completion"])]
    tokens = tokenizers(text, padding="max_length", truncation=True, max_length=2048)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenizer_function, batched=True)


from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
print("模型加载成功")



from peft import get_peft_model,LoraConfig,TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("模型加载成功")


from transformers import TrainingArguments,Trainer


training_ars = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    save_steps=1000,
    logging_steps=10,
    logging_dir="./logs",
    run_name="deepseekr1-1.5b-lora"
)

print("参数加载成功")

trainer = Trainer(
    model=model,
    args=training_ars,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizers
)

print("训练开始")
trainer.train()
trainer.save_model("./output/deepseekr1-1.5b-lora")
print("训练结束")

#保存全量模型
model.save_pretrained("./output/deepseekr1-1.5b-lora")
print("模型保存成功")
print("训练结束")



