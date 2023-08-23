#%%
import pandas as pd
import spacy
from collections import defaultdict, Counter
from similar_docs import get_similar_docs
import time

def format_time(d):
    b = "0" + str(d % 3600 // 60) if d % 3600 // 60 < 10 else str(d % 3600 // 60)
    c = "0" + str(d % 60) if d % 60 < 10 else str(d % 60)
    return str(d // 3600) + ":" + b + ":" + c

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_messages(keyword, topn = 0):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_json("../vast3/public/data.json")

    df = df[(df["label"]==1.0) | (df["label"].isnull())]

    df["time"] = df["datetime"]
    df["datetime"] = df["datetime"].apply(format_time)

    df = df.set_index("datetime")

    df['message'] = df['message'].astype(str)
    df['message'] = df['message'].str.lower()
    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].str.lower()
    df_keyword = df[df["Text"].str.contains(keyword)][['message','ID']]
    if topn == 0:
        df_keyword = df_keyword.drop_duplicates(subset=['message'])
        df_keyword.to_csv(f'./assets/data/{keyword}.txt')
        return df_keyword
    # Group by hour
    df_grouped = df.groupby(df.index.map(lambda x: x[:2])).agg({'Text': list, 'ID': 'first'})
    
    cooccurring_words_by_hour = defaultdict(list)
    for hour, text_list in df_grouped['Text'].items():
        # Combine all text in the list
        combined_text = " ".join(text_list)
        
        # 英文分词
        doc = nlp(combined_text)

        cooccurring_tokens = []
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and token.text!= keyword and len(token.text) > 2:
                cooccurring_tokens.append(token.text)

        # 统计词频
        cooccurring_word_freq = Counter(cooccurring_tokens)
        sorted_cooccurring_word_freq = sorted(cooccurring_word_freq.items(), key=lambda x: x[1], reverse=True)

        # 取出共现频率最高的单词
        top_cooccurring_words = [word[0] for word in sorted_cooccurring_word_freq[:topn]]
        cooccurring_words_by_hour[hour] = top_cooccurring_words

    df_all = pd.DataFrame(columns=['message', 'ID'])
    for hour, top_words in cooccurring_words_by_hour.items():
        df["datetime_format"] = df.index
        df["hour"] = df["datetime_format"].apply(lambda x: int(x.split(':')[0]))
        # Filter rows where the hour is equal to 17
        df_hour = df[df["hour"] == int(hour)]
        df_hour = df_hour.drop(columns=['hour'])
        result_df = df_hour[df_hour['Text'].str.contains('|'.join(top_words))][['message', 'ID']]
        df_all = pd.concat([df_all,result_df])
        print(len(df_all))
    df_all = df_all.drop_duplicates(subset=['message'])
    df_all = df_all.sort_values(by='ID')
    return df_all

# %%
def get_messages_geo():
    # Load your JSON data
    df = pd.read_json("../vast3/public/data.json")

    df = df[(df["label"]==1.0) | (df["label"].isnull())]

    df["time"] = df["datetime"]
    df["datetime"] = df["datetime"].apply(format_time)

    df = df.set_index("datetime")

    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].str.lower()
    
    # 找出df["latitude"]不为null的数据
    df = df[df["latitude"].notnull()][["message", "ID"]]
    df.reset_index(drop=True, inplace=True)
    df.to_csv(f'./assets/data/geo_data.csv')
    return df

def get_final_messages(keyword):
    df_search = get_messages(keyword)
    if len(df_search)>90:
        df_search = df_search.sample(n=90)
        
    question='\n'.join(list(df_search['message']))
    df_geo = get_messages_geo()
    df_geo_filtered = get_similar_docs(question, df_search_context=df_geo, data_name ="geo", k=10)

    df_all = pd.concat([df_search,df_geo_filtered])
    df_all.drop_duplicates(subset=['message','ID'], inplace=True)
    df_all = df_all.sort_values(by='ID')
    df_all.to_csv(f'./assets/data/{keyword}.txt')
    return df_all

#%%
def extract_top_words(topn):
    
    # Load your JSON data
    df = pd.read_json("../vast3/public/data.json")

    df = df[(df["label"]==1.0) | (df["label"].isnull())]

    df["time"] = df["datetime"]
    df["datetime"] = df["datetime"].apply(format_time)

    df = df.set_index("datetime")

    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].str.lower()

    # Combine all messages into one text for word frequency analysis
    all_messages_text = ' '.join(df['Text'])
    
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # 英文分词
    doc = nlp(all_messages_text)

    tokens = []
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN'):
            if token.lemma_ != "-PRON-":
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

    # 统计词频
    word_freq = Counter(tokens)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # 取出排名前n的单词
    top_words = [word[0] for word in sorted_word_freq[:topn]]
    return top_words

# top_words = extract_top_words(50)
# for word in top_words:
#     get_final_messages(word)
#     time.sleep(5)
