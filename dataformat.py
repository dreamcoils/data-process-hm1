import time

import pandas as pd
from tqdm import *

data_path = "./data/199801_clear (1).txt"
stopwd_path = "./data/cn_stopwords.txt"
stop_words = pd.read_csv(stopwd_path)
exclude_term = ["u", "c", "m"]


def getFormatData():
    start = time.time()
    data_list = []
    res_list = []
    topic = ""
    raw_data = open(data_path, encoding='gbk')

    for line in tqdm(raw_data):
        if line == '\n':
            continue
        read_list = line.split("  ")
        cur_topic = read_list[0]
        str_list = cur_topic.split("-")
        cur_topic = str_list[0] + str_list[1] + str_list[2]

        if cur_topic != topic:
            topic = cur_topic
            if res_list:
                add_list = res_list.copy()
                data_list.append(add_list)
            res_list = []

        del read_list[0]
        for tmp in read_list:
            cur_list = tmp.split("/")
            if cur_list[0] != "\n" and cur_list[0] not in stop_words.values and cur_list[1] not in exclude_term:
                res_list.append(cur_list[0])
    data_list.append(res_list)
    raw_data.close()
    print("Data clean phase is over! Take [ %s ] seconds." % str(time.time() - start))
    return data_list


'''
数据准备：计算文档总数，平均文档长度，统计词频
total_wd_tf：每个单词在整个库中的词频（该词在多少个文本中出现）
single_wd_tf_list：统计单个文本中各单词的词频
max_tf：单词在文档中最大出现数量
'''
def getFeatureData():
    data_list = getFormatData()
    start = time.time()
    doc_count = len(data_list)
    total_word_count = 0
    total_wd_tf = {}
    single_wd_tf = []
    max_tf = {}
    diff_wds = []
    for tmp_list in data_list:
        total_word_count += len(tmp_list)
        single_tf = {}
        for wd in tmp_list:
            if single_tf.get(wd):
                single_tf[wd] += 1
            else:
                single_tf[wd] = 1
            diff_wds.append(wd)
        single_wd_tf.append(single_tf)

        for k, v in single_tf.items():
            if total_wd_tf.get(k):
                total_wd_tf[k] += 1
            else:
                total_wd_tf[k] = 1
            if max_tf.get(k):
                max_tf[k] = max(max_tf[k], v)
            else:
                max_tf[k] = v
    avg_len = total_word_count / doc_count
    diff_wds = list(set(diff_wds))

    print("Data conversion phase is over! Take [ %s ] seconds." % str(time.time() - start))
    print("The total number of documents is ：%d" % doc_count)
    print("The average length of documents is ：%d" % avg_len)
    print("The total number of words is ：%d" % len(diff_wds))
    return doc_count, single_wd_tf, max_tf, total_wd_tf, diff_wds



