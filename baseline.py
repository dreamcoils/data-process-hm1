import time
import numpy as np
from tqdm import *

from dataformat import getFeatureData
from tfidf import cosine_similarity, getWordVectors

doc_count, single_wd_tf_list, max_tf, total_wd_tf, diff_wds = getFeatureData()
svd_vectors = getWordVectors(doc_count, single_wd_tf_list, max_tf, total_wd_tf, diff_wds)

start = time.time()
matrix = np.eye(doc_count, dtype=float)
for i in tqdm(range(doc_count)):
    for j in range(i + 1, doc_count):
        res = cosine_similarity(svd_vectors[i], svd_vectors[j])
        matrix[i][j] = res
print("Similarity calculating is finished. Take [ %s ] seconds.：" + str(time.time() - start) + "秒")

print("finish!")