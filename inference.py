from transformers import AutoModelForCausalLM, AutoTokenizer

#加载调好的模型
model_name = "output/deepseekr1-1.5b-lora"
tokenizers = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")



#构建推理pipeline

from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizers,
    device="cuda"
)
prompt = "tell me some singing skills"
result = pipe(prompt, max_new_tokens=512, num_workers=1)
print("开始回答-----", result[0]["generated_text"])


