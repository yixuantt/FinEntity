# FinEntity

A Dataset for **entity-level sentiment** classification.

Please note that some hyper-parameters may affect the performance, which can vary among different tasks/environments/software/hardware; thus, careful tunning is required. 

## Instruction
In the financial domain, conducting entity-level sentiment analysis is crucial for accurately assessing the sentiment directed toward a specific financial entity. To our knowledge, no publicly available dataset currently exists for this purpose. In this work, we introduce an entity-level sentiment classification dataset, called **FinEntity**, that annotates sentiment (positive, neutral, and negative) of individual financial entities in financial news. The dataset construction process is well-documented in the paper.
Additionally, we benchmark several pre-trained models (BERT, FinBERT, etc.) and ChatGPT on entity-level sentiment classification and find out that fine-tuning pre-trained models outperform ChatGPT. In a case study, we demonstrate the practical utility of using FinEntity in monitoring cryptocurrency markets. 

## Project Path 

```
├─bert_crf_train.ipynb     ---BERT-CRF training  
│  bert_softmax_train.ipynb ---BERT training  
│  config.py                ---Model config for BERT-CRF and BERT  
│  mic.ipynb                ---Calculate the maximum information coefficient  
│  model_bert_crf           ---A fine-tuning pre-trained BERT-CRF model   
│  openai.ipynb             ---Train for chatgpt 3.5  
│  predict_by_crf.ipynb     ---Extract entities from financial news  
│  price_predict.ipynb      ---Use Case in Cryptocurrency Market  
│  readme.md  
│  
├─crypto_data 
│  │  btc_corr.xls          ---Mic data for Bitcoin   
│  │  doge_corr.xls         ---Mic data for Dogecoin   
│  │  eth_corr.xls          ---Mic data for Ethereum  
│  │  finentity_lstm.xls    ---Data for LSTM with entity-level sentiment   
│  │  toned_lstm.xls        ---Data for LSTM with sequence-level sentiment   
│  └─ xrp_corr.xls   
│  
├─data  
│  └─ FinEntity.json        ---FinEntity data  
│                               
├─data_process  
│  │  fleiss_kappa.ipynb    ---process the data with fleiss kappa
│  └─ Jaccard_vote.ipynb    ---process the data with Jaccard similarity  
│   
├─example  
│  │  finbert_tone_lstm.py  ---LSTM use FinBert-tone  
│  └─ finentity_lstm.py     ---LSTM use fine-tuning FinBert-CRF  
│                  
├─model  
│  │  bert_crf.py           ---Model for BERT-CRF  
│  │  lstm_model.py         ---Model for LSTM  
│  │  
│  └─layers  
│    └─  crf.py             ---CRF layer  
│    
├─news_data   
│  │  20230105.json              ---News from 20230105  
│  │  20230106to20230201.json    ---News from 20230106 to 20230201  
│  │  BTC_USD.csv           ---OHLC of Bitcoin  
│  │  doge_USD.csv          ---OHLC of Dogecoin  
│  │  eth_USD.csv           ---OHLC of Ethereum  
│  │  lstm_no.csv         
│  │  lstm_sentiment.csv  
│  │  ltc_USD.csv           ---OHLC of Litecoin  
│  │  news20220520to20221130.json  
│  │  xrp_USD.csv           ---OHLC of Ripple  
│  │  
│  │
│  ├─news_entity            ---Entities recognized  
│  │  │  20220520to20221130_entity.json  
│  │  │  20230105_entity.json
│  │  └─ 20230106to20230201_entity.json
│  │
│  └─news_url               ---News with url  
│          20220520to20221130.xlsb
│          20230105.xlsb
│          20230106to20230201.xlsb
│
├─sequence_aligner          ---Data preprocessing   
│  │  alignment.py 
│  │  containers.py  
│  │  dataset.py  
│  │  labelset.py 
│  └─  __init__.py   
│   
│  
├─util                      ---Model Training Tool Module  
│  │  adversairal.py  
│  │  process.py  
│  │  train.py   
│  └─  __init__.py 
   
```
## License
FinEntity is licensed under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)
  
