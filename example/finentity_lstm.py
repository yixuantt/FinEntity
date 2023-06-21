import json
import pandas as pd
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from model.lstm_model import get_data,train,get_keras_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as ticker
from minepy import MINE
import numpy as np
def print_stats(mine):
    print ("MIC", mine.mic())
    
def min_max(data):
    ma = max(data)
    mi = min(data)
    for ind,item in enumerate(data):
        data[ind] = (data[ind]-mi)/(ma - mi)
        
def entity_corr(crypto):
     #load the data
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
    value_dict = {'btc':['btc','bitcoin','xbt'],'doge':['dogecoin','doge'],'ltc':['ltc','litecoin'],'eth':['ethereum','eth'],'xrp':['xrp','ripple']}
    bitcoin = value_dict.get(crypto)
    print("*** entity_dict:",bitcoin,"***")
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
        sub_list['entity']=sub_entity_list
        sub_list['sentiment']=sub_sentiment_list
        sub_list['date']=item['date']
        bit_sentiment.append(sub_list)
    bit_sentiment.sort(key=lambda x: x['date'], reverse=True)
    print("from ",bit_sentiment[len(bit_sentiment)-1]['date']," to ",bit_sentiment[0]['date'])
    
    filename = './news_data/'+crypto+'_USD.csv'
    df = pd.read_csv(filename)
    if set(['Vol.','Change %']).issubset(df.columns):
        df = df.drop(['Vol.','Change %'],axis=1)
    #process date and min max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    for idx,data in df.iterrows(): 
        a = data['Date']
        old_time = time.strptime(a, "%m/%d/%Y")
        new_time = time.strftime("%Y-%m-%d", old_time)
        df.loc[idx,'Date'] = new_time

    
    df = df.applymap(lambda x: x.replace(',','') if isinstance(x,str) else x)
    df = df.applymap(lambda x: x.replace('K','') if isinstance(x,str) else x)
    df = df.applymap(lambda x:x if not '%' in str(x) else x.replace('%',''))
    scaled_data = scaler.fit_transform(df.iloc[:,1:])
    scaled_data
    for idx,data in df.iterrows(): 
        leng = len(data)
        for i in range(leng):
            if i !=0:
                data[i]=scaled_data[idx][i-1]             

    # count emotion  positive:+1 negative:-1 neutral:0
    emotion_dict={}
    for indx,item in enumerate(bit_sentiment):
        date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
        date_key = time.strftime("%Y-%m-%d",date_time)
        if date_key in emotion_dict:
            old_emo = emotion_dict[date_key]
        else:
            old_emo = 0
        for senti in item['sentiment']:
            if senti == 'Positive':
                old_emo = old_emo + 1
            if senti == 'Negative':
                old_emo = old_emo - 1

        emotion_dict[date_key] = old_emo
    emo_list = list(emotion_dict.values())
    emo_max = max(emo_list)
    emo_min = min(emo_list)
    for key in emotion_dict:
        x = emotion_dict[key]
        nor = (x-emo_min)/(emo_max-emo_min)
        emotion_dict[key] = nor
    date_list = list(emotion_dict.keys())
    price_list= []
    for idx,data in df.iterrows(): 
        this_date = data['Date']
        if this_date in date_list:
            price_list.append(data['Price'])

    emotion = list(emotion_dict.values())
    start_date = bit_sentiment[len(bit_sentiment)-1]['date']
    end_date = bit_sentiment[0]['date']
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(date_list,emotion,color = '#1c9099',label="emotion")
    ax.plot(date_list,price_list,color = '#fec44f',label="btc-price")
    tick_spacing = 30
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    for lobj in ax.get_xticklabels():
        lobj.set_rotation(35)
        lobj.set_size(8)
    plt.xlabel("date")
    plt.ylabel("price/emotion")
    emo = crypto+'-price'
    plt.legend(loc = "best" ,labels=['emotion',emo])
    plt.show()
    

    # Calculate MIC
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(emotion, price_list)

    print_stats(mine)

def finentity_lstm_data():
        #load the data
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
    #for bitcoin
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
        sub_list['entity']=sub_entity_list
        sub_list['sentiment']=sub_sentiment_list
        sub_list['date']=item['date']
        bit_sentiment.append(sub_list)

    date_list=[]
    for ind,item in enumerate(bit_sentiment):
        count_sen = len(item['sentiment'])
        if(count_sen==0):
            continue
        date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
        date_key = time.strftime("%Y-%m-%d",date_time)
        date_list.append(date_key)

    date_list=list(set(date_list))
    date_list.sort(reverse=True)
    df_sen = pd.DataFrame(columns=['pos', 'neg', 'neu'], index=date_list)
    df_sen = df_sen.fillna(value=0)

    for ind,item in enumerate(bit_sentiment):
        date_time = time.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
        date_key = time.strftime("%Y-%m-%d",date_time)
        index = date_key
        if index not in date_list:
            continue
        count_neg = item['sentiment'].count('Negative')
        count_pos = item['sentiment'].count('Positive')
        count_neu = item['sentiment'].count('Neutral')
        pos = df_sen.loc[index,'pos']
        neg = df_sen.loc[index,'neg']
        neu = df_sen.loc[index,'neu']
        df_sen.loc[index,'pos'] = pos + count_pos
        df_sen.loc[index,'neg'] = neg + count_neg
        df_sen.loc[index,'neu'] = neu + count_neu
    df_sen.to_excel('./crypto_data/finentity_lstm.xls')
    
def entity_lstm():    
    df_sen = pd.read_excel(io='./crypto_data/finentity_lstm.xls',index_col = 0)
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
    df = pd.concat([df_bit, df_sen], axis=1)
    df = df.drop(['Vol.','Change %'],axis=1)
    df = df.fillna(value=0)
    drop_idx_list=[]
    for idx,data in df.iterrows(): 
        senti_sum = data['pos']+data['neg']+data['neu']
        if(senti_sum == 0 ):
            drop_idx_list.append(idx)
            continue
        df.loc[idx,'pos'] = data['pos']/senti_sum
        df.loc[idx,'neg'] = data['neg']/senti_sum
        df.loc[idx,'neu'] = data['neu']/senti_sum

    df = df.drop(labels=drop_idx_list,axis=0)
      
    print("***features of LSTM with entity-level sentiment***")
    print(df.tail())
    
    lookback=5
    n_features=df.shape[1]
    x_train, x_val, y_train, y_val,x_test,y_test = get_data(df,lookback,n_features)
    print("***predict with entity-level sentiment***")
    model = get_keras_model(lookback,n_features)
    train(model,lookback,n_features,x_train,y_train,x_val,y_val)
    y_pred = model.predict(x_test)
    # plt.figure(figsize=(10,5))
    # plt.plot( y_test_no, color='red', label='Real values', alpha=0.5)
    # plt.plot( y_pred, color='blue', label='Predicted values(Sentiment)', alpha=1)
    # plt.plot( y_pred_no, color='green', label='Predicted values', alpha=1)
    # plt.legend()
    # plt.show()

    print("RMSE(entity-Sentiment):",np.sqrt(mean_squared_error(y_pred,y_test)))
    print("")
    print("r2_score(entity-Sentiment):",r2_score(y_pred,y_test))