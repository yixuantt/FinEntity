import torch
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import config
from model.bert_crf import BertCrfForNer as BertNER
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from model.bert_crf import BertCrfForNer
from transformers import get_linear_schedule_with_warmup
from torch import cuda
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from util.process import ids_to_labels,Metrics,Metrics_e
from seqeval.scheme import BILOU
from util.adversairal import FGM 
from transformers import BertTokenizerFast
from sequence_aligner.dataset import PredictDatasetCRF,PredictDatasetBySeq
from sequence_aligner.containers import TraingingBatch,PredictBatch
from transformers import BertTokenizerFast
from sequence_aligner.labelset import LabelSet
from torch.utils.data import DataLoader

def train_epoch(e,model, data_loader,optimizer,scheduler,device):
    model.train()
    fgm = FGM(model)
    losses = 0.0
    for step, d in enumerate(data_loader):
        step += 1
        input_ids = d["input_ids"]
        attention_mask = d["attention_masks"].type(torch.uint8)
        targets = d["labels"]
        inputs = {
            'input_ids':input_ids.to(device),
            'attention_mask':attention_mask.to(device),
            'labels':targets.to(device)
        }
        outputs = model(
            **inputs
        )
        loss = outputs[0] 
        losses += loss.item()
        loss.backward()
        
        #fgm
        fgm.attack() 
        loss_adv = model( **inputs)[0]
        loss_adv.backward() 
        fgm.restore() 
    
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("Epoch: {}, train Loss:{:.4f}".format((e+1), losses/step))
    return losses/step

def valid_epoch(e,model, data_loader,device,label_set):
    model.eval()
    y_true, y_pred = [], []
    losses = 0
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            y_true_sub, y_pred_sub = [], []
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].type(torch.uint8)
            targets = d["labels"]
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
                'labels':targets.to(device)
            }
            outputs = model(
                **val_input
            )
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, d['attention_masks'].to(device))
            tags = tags.squeeze(0).cpu().numpy().tolist()
            out_label_ids = d['labels'].cpu().numpy().tolist()
            for i, label in enumerate(out_label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if m == -1:
                        continue
                    temp_1.append(label_set.ids_to_label[out_label_ids[i][j]])
                    temp_2.append(label_set.ids_to_label[tags[i][j]])
                y_true.append(temp_1)
                y_pred.append(temp_2)
    report=classification_report(y_true, y_pred, mode='strict', scheme=BILOU)
    print(report)
    
        
def valid_epoch_not_crf(e,model, val_loader,device,label_set):
    model.eval()
    trues, preds = [], []
    losses = 0
    with torch.no_grad():
        for step, d in enumerate(val_loader):
            sub_preds, sub_trues = [],[]
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].type(torch.uint8)
            targets = d["labels"]
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
                'labels':targets.to(device)
            }
            outputs = model(
                **val_input
            )
            tmp_eval_loss, logits = outputs[:2]
            
            sub_preds =np.argmax(logits.cpu().numpy(), axis=2).reshape(-1).tolist()
            sub_trues = d["labels"].detach().cpu().numpy().reshape(-1).tolist()
            # data process
            gold_labeled,pred_labeled = ids_to_labels(label_set,sub_trues,sub_preds)
            trues.append(gold_labeled)
            preds.append(pred_labeled)
    report=classification_report(trues, preds, mode='strict', scheme=BILOU)
    print(report)

def predict4entity2sequence(model,data,device,tokenizer,label_set,save_list,nlp):
    if len(data) ==0:
        return
    dataset = PredictDatasetBySeq(data=data, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
    if len(dataset) ==0:
        return
    pred_loader = DataLoader(dataset, batch_size=16, collate_fn=PredictBatch, shuffle=True)
    result_entity,result_sentiment = predict(model, pred_loader,device,label_set,tokenizer)
    if len(data)>512:
        data = data[:511]
    result = nlp(data)
    sub_save = {}
    sub_save['text'] = data
    sub_save['entity'] = result_entity
    sub_save['ent_sentiment'] = result_sentiment
    sub_save['seq_sentiment'] = result[0]['label']
    save_list.append(sub_save)
    
def predict4news(model,data,device,tokenizer,label_set,save_list):
    dataset = PredictDatasetCRF(data=data, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
    pred_loader = DataLoader(dataset, batch_size=16, collate_fn=PredictBatch, shuffle=True)
    result_entity,result_sentiment = predict(model, pred_loader,device,label_set,tokenizer)
    sub_save = {}
    sub_save['date'] = data["date_publish"]
    sub_save['content'] = data["content"]
    sub_save['entity'] = result_entity
    sub_save['sentiment'] = result_sentiment
    save_list.append(sub_save)
    
def predict(model, data_loader,device,label_set,tokenizer):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            y_pred_sub = []
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].bool()
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
            }
            logits = model(
                **val_input
            )[0]
            tags = model.crf.decode(logits, d['attention_masks'].to(device))
            tags = tags.squeeze(0).cpu().numpy().tolist()
            y_pred = []
            result_entity = []
            result_sentiment = []
            
            for i,tag in enumerate(tags):
                temp = []
                sub_tag = tags[i]
                content = tokenizer.decode([token for token in input_ids[i] if token!=0])
                conten_list = tokenizer.tokenize(content)
                id_to_decode = []
                for j, item in enumerate(sub_tag):
                    if(attention_mask[i][j]==True):
                        temp.append(label_set.ids_to_label[tags[i][j]])
                        
                        t = label_set.ids_to_label[tags[i][j]]
     
                        ind = input_ids[i][j].item()
                        if t != "O":
                            if t.startswith("B"):
                                id_to_decode.append(ind)
                            if t.startswith("I"):
                                id_to_decode.append(ind)
                            if t.startswith("L"):
                                id_to_decode.append(ind)
                                
                        if t!="O" and t.startswith("U"):
                            result_entity.append(tokenizer.decode(ind))
                            result_sentiment.append(t.strip('U-'))
                            
                        if t!="O" and t.startswith("L") and j-1>0 and label_set.ids_to_label[tags[i][j-1]] != "O":
                            tokens = tokenizer.convert_ids_to_tokens([token for token in id_to_decode if token!=0])
                            string = tokenizer.convert_tokens_to_string(tokens)
                            result_entity.append(string)
                            id_to_decode = []
                            result_sentiment.append(t.strip('L-'))
                        
                y_pred.append(temp)
            return result_entity,result_sentiment