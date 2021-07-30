import numpy as np 

def create_bert_tokens(tokenizer, df, text_col, MAX_LEN = 512, truncate = True):
    input_ids = []
    if truncate:
        print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    for sent in df[text_col]:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        if truncate:
            if len(encoded_sent) < MAX_LEN:
                encoded_sent = np.pad(encoded_sent, (0, MAX_LEN - len(encoded_sent)), 'constant', constant_values=0).tolist()
            else:
                encoded_sent = encoded_sent[: MAX_LEN]
        input_ids.append(encoded_sent)
    return (input_ids)

def create_attention_mask(bert_tokens):
    attention_masks = []
    for sent in bert_tokens:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

