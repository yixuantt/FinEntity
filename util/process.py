from sequence_aligner.labelset import LabelSet
from collections import Counter
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import matplotlib.pyplot as plt
import re
import numpy as np
def find_comp_symbols(raw):
    string = "~!@#$%^&*()_+-*/<>,.[]\/"
    total_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for item in subset:
            value = item.get('value')
            for i in string:
                if i in value:
                    total_count = total_count + len(item.get('annotated_by'))
                    print(value)         
    return total_count

def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0):
    x = np.arange(len(labels)) * tick_step
    group_num = len(datas)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width)
    plt.ylabel('Scores')
    plt.title('multi datasets')
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels)
    plt.show()

    
    
def cal_entities(raw):
    total_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for item in subset:
            total_count = total_count + len(item.get('annotated_by'))
            
    return total_count

def cal_entities_final(raw):
    total_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        total_count = total_count + len(subset)
            
    return total_count

def cal_same_entities(raw):
    same_count = 0
    same_list = []
    for i,val in enumerate(raw):
        annotater = len(val["seen_by"])
        subset = val.get("annotations")
        same_sub_List = []
        for item in subset:
            if annotater == len(item.get('annotated_by')):
                same_count = same_count + len(item.get('annotated_by'))
                same_sub_List.append(item)               
        if len(same_sub_List)>0:
            text = {}
            space_list = []
            text["example_id"] = val["example_id"]
            text["seen_by"] = val["seen_by"]
            text["content"] = val.get("content")
            text["annotations"] = same_sub_List
            same_list.append(text)
    return same_count,same_list
    
def cal_annotator(raw):
    label_set = {'Neutral':0,'Positive':1,'Negative':2}
    name_set = {}
    annotator = np.zeros((12,3),dtype=np.int)
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for item in subset:
            label = item.get("tag")
            sub_annotator = item.get('annotated_by')
            for anno in sub_annotator:
                name_key = anno.get('annotator')
                name_key = name_key.split('@')[0]
                if name_key in name_set:
                    index = int(label_set[label])
                    annotator[name_set[name_key]][index] = annotator[name_set[name_key]][index] + 1
                else:
                    leng = len(name_set)
                    name_set[name_key] = int(leng)
                    index = int(label_set[label])
                    annotator[leng][index] = 1
    return name_set,label_set,annotator
            
def cal_space(raw):
    ## Calculate spaces
    space_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for item in subset:
            s = item.get("value")
            if s.startswith(' '):
                space_count = space_count + 1
            if s.endswith(' '):
                space_count = space_count + 1
    return space_count

def remove_space(raw):
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for index,item in enumerate(subset): 
            s = item.get("value")
            if s.startswith(' '):           
                print(s)
                item['value'] = s.strip()
                item['start'] = item['start'] + 1
            if s.endswith(' '):
                print(s)
                item['value'] = s.strip()
                item['end'] = item['end'] -1 
def cal_inconsistency(raw):
    tag_count = 0
    inconsistentCount = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        tag_count = tag_count + len(subset)
        subinconsistent = 0
        j=0
        for item in subset:
            for n in subset[j+1:]:
                if n.get("value") == item.get("value") and n.get("start")==item.get("start"):
                    subinconsistent = subinconsistent+ 1
            if j+1==len(subset):
                break
            j=j+1
        inconsistentCount = subinconsistent + inconsistentCount
    inconsistency = inconsistentCount / tag_count
    return inconsistency

def cal_inaccuracy_entity(raw):
    tag_count = 0
    inconsistentCount = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        tag_count = tag_count + len(subset)
        subinconsistent = 0
        j=0
        for item in subset:
            for n in subset[j+1:]:
                if n.get("value") in item.get("value"):
                    if n.get("value") !=item.get("value") and n.get("start")>item.get("start") and n.get("start")<item.get("end"):
                        subinconsistent = subinconsistent+1
                if item.get("value") in n.get("value"):
                    if item.get("value") !=n.get("value") and item.get("start")>n.get("start") and item.get("start")<n.get("end"):
                        subinconsistent = subinconsistent+1

            j=j+1
            if j+1==len(subset):
                break
        inconsistentCount = subinconsistent + inconsistentCount
    inconsistency = inconsistentCount / tag_count
    return inconsistency

def proess_data(raw):
    tag_count = 0
    for i, val in enumerate(raw):
        subset = val.get("annotations")
        tag_count = tag_count + len(subset)
        removeList=[]
        for j, item in enumerate(subset):
            for m, n in enumerate(subset, j):
                if n.get("value") == item.get("value") and n.get("start")==item.get("start"):
                    if len(n.get("annotated_by")) < len(item.get("annotated_by")):
                        removeList.append(n)
                    if len(n.get("annotated_by")) > len(item.get("annotated_by")):
                        removeList.append(item)
        for r in removeList:
            if r in subset:
                subset.remove(r)

def ids_to_labels(label_set,gold,pred):
    label_dict = label_set.ids_to_label
    gold_labeled = []
    pred_labeled = []
    for i, val in enumerate(gold):
        if val == -100:
            continue
        in_pred = pred[i]
        gold_labeled.append(label_dict[val])
        pred_labeled.append(label_dict[in_pred])
    return gold_labeled,pred_labeled


def plot_feature_final(raw):
    ## count
    feature_dict ={}
    entity_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for j,item in enumerate(subset):
            entity_count = entity_count + 1
            tag = item['tag'] 
            if tag in feature_dict.keys():
                feature_dict[tag] = feature_dict[tag] + 1
            else:
                feature_dict[tag] = 1
    print("all entities :",entity_count)
    ## plot 
    labels = []
    sizes = []
    sort = sorted(feature_dict.items(), key = lambda kv:(kv[1], kv[0]))
    feature_dict = dict(sort)
    for key in feature_dict:
        print(key," tag count: ",feature_dict[key])
        labels.append(key)
        sizes.append(feature_dict[key])
    explode = None
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels,  autopct='%.2f%%',colors=["#4682B4", "#B0C4DE", "#ADD8E6"],
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend()
    plt.show()
    
def plot_feature(raw):
    ## count
    feature_dict ={}
    entity_count = 0
    for i,val in enumerate(raw):
        subset = val.get("annotations")
        for j,item in enumerate(subset):
            entity_count = entity_count + len(item.get("annotated_by")) 
            tag = item['tag'] 
            if tag in feature_dict.keys():
                feature_dict[tag] = feature_dict[tag] + len(item.get("annotated_by")) 
            else:
                feature_dict[tag] = len(item.get("annotated_by")) 
    print("all entities :",entity_count)
    ## plot 
    labels = []
    sizes = []
    sort = sorted(feature_dict.items(), key = lambda kv:(kv[1], kv[0]))
    feature_dict = dict(sort)
    for key in feature_dict:
        print(key," tag count: ",feature_dict[key])
        labels.append(key)
        sizes.append(feature_dict[key])
    explode = None
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels,  autopct='%.2f%%',colors=["#4682B4", "#B0C4DE", "#ADD8E6"],
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend()
    plt.savefig('Demo_official.jpg')
    plt.show()

def Metrics_e(y_pred,y_true):
    report=classification_report(y_true, y_pred)
    print(report)
    return report

class Metrics(object):

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)     #将已知标签和预测的标签拼成列表
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)    #标签集合（不重复）
        self.correct_tags_number = self.count_correct_tags()    #某个tag预测正确的次数
        self.predict_tags_counter = Counter(self.predict_tags)  #预测标签各个标签的个数
        self.golden_tags_counter = Counter(self.golden_tags)    #源文件标签各个标签的个数

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
    #计算每个tag的准确率
        precision_scores = {}
        for tag in self.tagset:
            #correct_tags_number为tag计算正确的次数     predict_tags_counter所有predict_tag的个数
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                                    max(1e-10, self.predict_tags_counter[tag])

        return precision_scores

    def cal_recall(self):
    #计算每个tag的召回率
        recall_scores = {}
        for tag in self.tagset:
            #correct_tags_number为tag计算正确的次数     golden_tags_counter为tag出现的总次数
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                                 max(1e-10, self.golden_tags_counter[tag])
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        print()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def count_correct_tags(self):
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):  #zip()用于将数据打包成元组
            if gold_tag == predict_tag:    #每当匹配的情况就将correct_dict[gold_tag]加一
                if gold_tag not in correct_dict:   #若该标签不在已知标签字典中，将其加入
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)   #标准标签的总数

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]  #标准文件各个标签的个数
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("The original total number of tags is {}，and{} O tags are removed，accounting for{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size + 1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
