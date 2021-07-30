import nltk

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def Check_milestone(text):
    tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9]*|')
    token = [str.lower(i) for i in tokenizer.tokenize(text) if len(i) > 0]
    index_m = [i for i,j in enumerate(token) if j == 'milestone']
    marker = 0
    if len(index_m) > 0:
        for i in index_m:
            for j in range(max(0, i-2), min(i+2, len(token))):
                if token[j].find('billing') >= 0:
                    for k in range(j, min(j+2, len(token))):
                        if isfloat(token[k]):
                            marker += 1
    if marker >= 1:
        return 1
    else:
        return 0

def Check_event(text, key_word_list):
    tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9]*|')
    token = [str.lower(i) for i in tokenizer.tokenize(text) if len(i) > 0]
    index_m = [i for i,j in enumerate(token) if j in key_word_list]
    if len(index_m) > 0:
        return 1
    else:
        return 0