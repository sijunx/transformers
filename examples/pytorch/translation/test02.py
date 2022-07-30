import torch

from transformers import (
  EncoderDecoderModel,
  AutoTokenizer
)
PRETRAINED = "raynardj/wenyanwen-chinese-translate-to-ancient"
# PRETRAINED = "/Users/xusijun/Documents/MY_NLP_001/wenyanwen-chinese-translate-to-ancient"

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
model = EncoderDecoderModel.from_pretrained(PRETRAINED)

def inference(text):
    tk_kwargs = dict(
      truncation=True,
      max_length=128,
      padding="max_length",
      return_tensors='pt')

    inputs = tokenizer([text,],**tk_kwargs)
    with torch.no_grad():
        return tokenizer.batch_decode(
            model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=3,
            bos_token_id=101,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ), skip_special_tokens=True)

text = "今天天气特别冷，开始下雨了"

result = inference(text)
print(result)
