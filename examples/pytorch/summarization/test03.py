import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""
# article_text = "该公司表示，将删除的视频中有一些视频说，经批准的疫苗是危险的，会导致自闭症、癌症或不孕不育。该政策包括终止抗疫苗影响者的账户。科技巨头因没有采取更多措施来应对其网站上的虚假健康信息而受到批评。7月，美国总统乔·拜登（JoeBiden）表示，社交媒体平台在很大程度上造成了人们对通过传播错误信息接种疫苗的怀疑，并呼吁他们解决这个问题。谷歌旗下的YouTube表示，自去年实施禁止传播新冠肺炎疫苗错误信息的内容以来，其平台上已有13万个视频被删除。在一篇博客文章中，该公司表示，它看到有关新冠肺炎疫苗接种的虚假说法“蔓延到有关疫苗的一般错误信息中”。这项新政策涵盖了长期以来获得批准的疫苗，如麻疹或乙型肝炎疫苗。“我们正在YouTube上扩大我们的医疗错误信息政策，对目前使用的疫苗制定新的指导方针，这些疫苗经地方卫生当局和世界卫生组织批准并确认是安全有效的，”该帖子提到世界卫生组织。"
article_text = "用户在好特卖超时购买了购买了汽水等饮品，产生了2元的超重费。最终因商家缺货，进行了部分退单，但是超重费却没有退。故用户发起了退款申请，商家拒接表示这是平台收的，需要找平台。扣除退货的物品，已不超重，为何要收取超重费，不合理。建议缺货后退款后，平台减去相应的增重配送费"
#
# model_name = "csebuetnlp/mT5_m2o_chinese_simplified_crossSum"
# model_name = "/Users/zard/Documents/nlp002/mT5_m2o_chinese_simplified_crossSum"
# model_name = "/Users/xusijun/Documents/MY_NLP_001/mT5_multilingual_XLSum"
model_name = "/Users/xusijun/Documents/MY_NLP_001/mT5_m2o_chinese_simplified_crossSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
