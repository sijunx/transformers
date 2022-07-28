from transformers import AutoModelForQuestionAnswering, BertTokenizer
import torch
# 参照链接：https://blog.csdn.net/weixin_43324905/article/details/123974973
model = AutoModelForQuestionAnswering.from_pretrained('wptoux/albert-chinese-large-qa')
tokenizer = BertTokenizer.from_pretrained('wptoux/albert-chinese-large-qa')

# inputs = tokenizer.encode("测试数据，喜欢吃苹果")
question = "喜欢吃什么"
context = "我喜欢吃苹果，你喜欢吃什么"

# inputs = tokenizer(question, context, max_length=218, truncation="only_second", return_overflowing_tokens=True, stride=False)

question, doc = "我喜欢吃什么" , "我喜欢吃苹果，你喜欢吃什么"
encoding = tokenizer.encode_plus(text = question,text_pair = doc,  verbose=False)
inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs)  #input tokens
print("tokens: ", tokens)

outputs = model(input_ids=torch.tensor([inputs]),
                token_type_ids=torch.tensor([sentence_embedding]))
#BertForQuestionAnswering返回一个QuestionAnsweringModelOutput对象。
#由于将BertForQuestionAnswering的输出设置为start_scores, end_scores，
# 因此返回的QuestionAnsweringModelOutput对象被强制转换为字符串的元组('start_logits', 'end_logits')，从而导致类型不匹配错误。
# torch.argmax(dim)会返回dim维度上张量最大值的索引
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)
# start_index = torch.argmax(start_scores)
# end_index = torch.argmax(end_scores)
#print("start_index:%d, end_index %d"%(start_index, end_index))
answer = ' '.join(tokens[start_index:end_index + 1])
print(answer)
# 每次执行的结果不一致，这里因为模型没有经过训练，所以效果不好，输出结果不佳
