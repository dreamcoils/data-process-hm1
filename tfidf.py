import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import *
import time


def tf(f, max_f, K=0.5):
    return K + (1 - K) * f / max_f


def idf(N, nt):
    return math.log(N / nt)


# 生成单词索引字典，帮助完成词向量的构建
def getWordDict(diff_wds):
    wd_idx_dict = {}
    idx = 0
    for i in diff_wds:
        wd_idx_dict[i] = idx
        idx += 1
    return wd_idx_dict


# 计算各文本中每个单词的tfidf，并生成文本词向量
def getWordVectors(doc_count, single_wd_tf_list, max_tf, total_wd_tf, diff_wds):
    vectors = []
    wd_idx_dict = getWordDict(diff_wds)
    for i in tqdm(range(doc_count)):
        cur_tf_dict = single_wd_tf_list[i]
        vector = np.zeros(len(diff_wds), dtype=float)
        for k, v in cur_tf_dict.items():
            idx = wd_idx_dict[k]
            vector[idx] = tf(v, max_tf[k]) * idf(doc_count, total_wd_tf[k])
        vectors.append(vector)

    # 3. 词频向量降维-svd
    start = time.time()
    svd = TruncatedSVD(3000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    svd_vectors = lsa.fit_transform(vectors)
    print("SVD phase is over. Take [ %s ] seconds." % str(time.time() - start))
    return svd_vectors


# 余弦相似度
def cosine_similarity(x, y):
    x = np.mat(x)
    y = np.mat(y)
    num = float(x * y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
