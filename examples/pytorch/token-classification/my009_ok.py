from transformers import AutoModelForTokenClassification,AutoTokenizer,pipeline
model = AutoModelForTokenClassification.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
ner = pipeline('ner', model=model, tokenizer=tokenizer)
result = ner("江苏警方通报特斯拉冲进店铺")
print(result)
