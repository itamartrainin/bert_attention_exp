#%%
from transformers import BertForSequenceClassification
import pandas as pd

#%%
df = pd.read_csv(r'C:\Data\quora-question-pairs\test.csv')

#%%
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

#%%
encoded_dict = tokenizer.encode_plus(df["question1"][0], df["question2"][0], max_length=310, pad_to_max_length=True, 
                      return_attention_mask=True, return_tensors='pt', truncation=True)

#%%
input_ids = encoded_dict['input_ids']
token_type_ids = encoded_dict["token_type_ids"]
attention_masks = encoded_dict['attention_mask']

#%%
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    output_attentions=True,
    output_hidden_states=True
)

#%%
res = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)

#%%
ret = res[0]
att = res[1]
hidden = res[2]

#%%


#%%
print(ret.shape)
print(att.shape)
print(hidden.shape)