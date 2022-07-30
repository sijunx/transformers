from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-chinanews-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-chinanews-chinese')
text_classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
result = text_classification("北京上个月召开了两会")
print(result)
