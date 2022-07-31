from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline

mode_name = 'liam168/trans-opus-mt-en-zh'

model = AutoModelWithLMHead.from_pretrained(mode_name)

tokenizer = AutoTokenizer.from_pretrained(mode_name)

translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)

result = translation('I like to study Data Science and Machine Learning.', max_length=400)

print(result)
