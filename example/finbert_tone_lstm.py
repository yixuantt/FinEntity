import json
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from model.lstm_model import get_data,train,get_keras_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import os
from minepy import MINE
import math
def min_max(data):
    ma = max(data)
    mi = min(data)
    for ind,item in enumerate(data):
        data[ind] = (float(data[ind])-float(mi))/(float(ma) - float(mi))
        
def print_stats(mine):
    print ("MIC", mine.mic())
    
def get_tone_corr(crypto):
    excelname = './crypto_data/'+crypto+'_corr.xls'
    if os.path.exists(excelname):
        print("***load the data from disk***")
        df = pd.read_excel(excelname,index_col = 0)
    else:
        f1 = open('./news_data/news_entity/20230105_entity.json', 'r', encoding='utf8')
        f2 = open('./news_data/news_entity/20230106to20230201_entity.json', 'r', encoding='utf8')
        f3 = open('./news_data/news_entity/20220520to20221130_entity.json', 'r', encoding='utf8')
        content1 = f1.read() 
        content2 = f2.read() 
        content3 = f3.read() 
        raw1 = json.loads(content1)
        raw2 = json.loads(content2)
        raw3 = json.loads(content3)
        raw1.extend(raw2)
        raw1.extend(raw3)
        raw = raw1

        print("***load the data***")
        value_dict = {'btc':['btc','bitcoin','xbt'],'doge':['dogecoin','doge'],'ltc':['ltc','litecoin'],'eth':['ethereum','eth'],'xrp':['xrp','ripple']}
        bitcoin = value_dict.get(crypto)

        bit_sentiment = []
        for item in raw:
            sub_entity_list = []
            sub_sentiment_list = []
            sub_list={}
            en_list = item['entity']
            for index,en in enumerate(en_list):
                if en.lower() in bitcoin:
                    sub_entity_list.append(en.lower())
                    sub_sentiment_list.append(item['sentiment'][index])
            if len(sub_entity_list)==0:
                continue
            sub_list['content']=item['content']
            sub_list['date']=item['date']
            bit_sentiment.append(sub_list)


        for ind, subitem in enumerate(bit_sentiment):
            c = subitem['content']
            c = c.replace('\nAD',' ')
            c = c.replace('\n',' ')
            clist = c.split('.')
            find_list = []
            for s in clist:
                if any(substring.lower() in s.lower() for substring in bitcoin):
                    find_list.append(s)
            bit_sentiment[ind]['btc'] = find_list
            date_list=[]
        for ind,item in enumerate(bit_sentiment):
            date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
            date_key = time.strftime("%Y-%m-%d",date_time)
            date_list.append(date_key)

        date_list=list(set(date_list))
        date_list.sort(reverse=True)
        df_sen = pd.DataFrame(columns=['content'], index=date_list)

        for ind,item in enumerate(bit_sentiment):
            date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
            date_key = time.strftime("%Y-%m-%d",date_time)
            index = date_key
            if index not in date_list:
                continue
            if df_sen.loc[index,'content'] is np.nan:
                df_sen.loc[index,'content'] = item['btc']
            else:
                c = df_sen.loc[index,'content']
                c.extend(item['btc'])
                df_sen.loc[index,'content'] = c

        # load the finbert-tone
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

        df1 = pd.DataFrame(columns=['label'], index=date_list)
        for idx,data in df_sen.iterrows(): 
            if len(data['content']) > 0:
                content = data['content']
                for i,c in enumerate(content):
                    if len(c)>512:
                        content[i] = c[:512]  
                result = nlp(content)
                count = 0;
                for i,r in enumerate(result):
                    label = result[i]['label']
                    if label == 'Negative':
                        count = count - 1  
                    if label == 'Positive':
                        count = count + 1  
                df1.loc[idx,'label'] = count

        filename = './news_data/'+crypto+'_USD.csv'
        df_bit = pd.read_csv(filename)
        if set(['Vol.','Change %']).issubset(df_bit.columns):
            df_bit = df_bit.drop(['Vol.','Change %'],axis=1)
        for idx,data in df_bit.iterrows(): 
            a = data['Date']
            old_time = time.strptime(a, "%m/%d/%Y")
            new_time = time.strftime("%Y-%m-%d", old_time)
            df_bit.loc[idx,'Date'] = new_time
            
        df_bit = df_bit.applymap(lambda x: x.replace(',','') if isinstance(x,str) else x)
        df_bit = df_bit.applymap(lambda x: x.replace('K','') if isinstance(x,str) else x)
        df_bit = df_bit.applymap(lambda x:x if not '%' in str(x) else x.replace('%',''))

        df_bit.set_index("Date", inplace=True)
        df = pd.concat([df_bit, df1], axis=1)
        df = df.dropna(axis=0,how='all')
        ##save
        df.to_excel(excelname)
        
    
    if set(['Vol.','Change %']).issubset(df.columns):
        df = df.drop(['Vol.','Change %'],axis=1)
    # MIC
    emotion = []
    price = []
    for idx,data in df.iterrows(): 
        if np.isnan(data['label']):  
            continue
        else:
            price.append(data['Price'])
            emotion.append(data['label'])
    min_max(price)
    min_max(emotion)  
    
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(emotion, price)
    print_stats(mine)
    
    
def get_tone_lstm_data():
    f1 = open('./news_data/news_entity/20230105_entity.json', 'r', encoding='utf8')
    f2 = open('./news_data/news_entity/20230106to20230201_entity.json', 'r', encoding='utf8')
    f3 = open('./news_data/news_entity/20220520to20221130_entity.json', 'r', encoding='utf8')
    content1 = f1.read() 
    content2 = f2.read() 
    content3 = f3.read() 
    raw1 = json.loads(content1)
    raw2 = json.loads(content2)
    raw3 = json.loads(content3)
    raw1.extend(raw2)
    raw1.extend(raw3)
    raw = raw1

    print("***load the data***")
    bitcoin = ['btc','bitcoin','xbt']
    bit_sentiment = []
    for item in raw:
        sub_entity_list = []
        sub_sentiment_list = []
        sub_list={}
        en_list = item['entity']
        for index,en in enumerate(en_list):
            if en.lower() in bitcoin:
                sub_entity_list.append(en.lower())
                sub_sentiment_list.append(item['sentiment'][index])
        if len(sub_entity_list)==0:
            continue
        sub_list['content']=item['content']
        sub_list['date']=item['date']
        bit_sentiment.append(sub_list)
    bitcoin = 'bitcoin'
    btc = 'btc'
    for ind, subitem in enumerate(bit_sentiment):
        c = subitem['content']
        c = c.replace('\nAD',' ')
        c = c.replace('\n',' ')
        clist = c.split('.')
        find_list = []
        for s in clist:
            if s.lower().find(bitcoin.lower()) > 0 or s.lower().find(btc.lower())>0:
                find_list.append(s)
        subitem['btc'] = find_list
        date_list=[]
    for ind,item in enumerate(bit_sentiment):
        date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
        date_key = time.strftime("%Y-%m-%d",date_time)
        date_list.append(date_key)
    
    date_list=list(set(date_list))
    date_list.sort(reverse=True)
    df_sen = pd.DataFrame(columns=['content'], index=date_list)

    for ind,item in enumerate(bit_sentiment):
        date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
        date_key = time.strftime("%Y-%m-%d",date_time)
        index = date_key
        if index not in date_list:
            continue
        if df_sen.loc[index,'content'] is np.nan:
            df_sen.loc[index,'content'] = item['btc']
        else:
            c = df_sen.loc[index,'content']
            c.extend(item['btc'])
            df_sen.loc[index,'content'] = c
    
    # load the finbert-tone
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    df2 = pd.DataFrame(columns=['Positive','Negative','Neutral'], index=date_list)
    df2 = df2.fillna(0)
    for idx,data in df_sen.iterrows(): 
        if len(data['content']) > 0:
            content = data['content']
            for i,c in enumerate(content):
                if len(c)>512:
                    content[i] = c[:512]  
            result = nlp(content)
            count = 0;

            for i,r in enumerate(result):
                label = result[i]['label']
                df2.loc[idx,label] = df2.loc[idx,label] + 1
        pos = df2.loc[idx,'Positive']
        neg = df2.loc[idx,'Negative']
        neu = df2.loc[idx,'Neutral']
        count = pos + neg + neu
        if count !=0:
            df2.loc[idx,'Positive'] = pos/count
            df2.loc[idx,'Negative'] = neg/count
            df2.loc[idx,'Neutral'] = neu/count
    df_bit = pd.read_csv('./news_data/BTC_USD.csv')
    
    #process date and min max normalization
    for idx,data in df_bit.iterrows(): 
        a = data['Date']
        old_time = time.strptime(a, "%m/%d/%Y")
        new_time = time.strftime("%Y-%m-%d", old_time)
        df_bit.loc[idx,'Date'] = new_time
    df_bit = df_bit.applymap(lambda x: x.replace(',',''))
    df_bit = df_bit.applymap(lambda x: x.replace('K',''))

    df_bit.set_index("Date", inplace=True)
    df = pd.concat([df_bit, df2], axis=1)
    df = df.drop(['Vol.','Change %'],axis=1)
    df = df.fillna(value=0)

    df = df.drop(df[(df.Positive == 0)& (df.Neutral==0)& (df.Positive==0)].index)
    df = df.dropna(axis=0,how='all')  

    df.to_excel('./crypto_data/toned_lstm.xls')
    
def tone_lstm():
    df = pd.read_excel(io='./crypto_data/toned_lstm.xls',index_col = 0)
    print("***features of LSTM with sequence-level sentiment***")
    print(df.tail())
    
    #training
    lookback=5
    n_features=df.shape[1]
    x_train, x_val, y_train, y_val,x_test,y_test = get_data(df,lookback,n_features)

    model = get_keras_model(lookback,n_features)

    train(model,lookback,n_features,x_train,y_train,x_val,y_val)
    y_pred = model.predict(x_test)

    print("RMSE(seq-Sentiment):",np.sqrt(mean_squared_error(y_pred,y_test)))
    print("")
    print("r2_score(seq-Sentiment):",r2_score(y_pred,y_test))