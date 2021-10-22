import multiprocessing
import time
import numpy as np
from multiprocessing import Pool, Manager
from dataformat import getFeatureData
from tfidf import cosine_similarity, getWordVectors


# 用多进程的方法优化baseline
def calculate_cosine(svd_vectors, start, end, cur, doc_count, res_dict):
    print("process " + str(cur) + " start ! ")
    for i in range(start, end):
        vector = np.zeros(len(svd_vectors[0]))
        for j in range(i + 1, doc_count):
            similarity = cosine_similarity(svd_vectors[i], svd_vectors[j])
            vector[j] = similarity
        res_dict[i] = vector
    print("Process" + str(cur) + " is finished!")


def main_process():
    cpu_count = multiprocessing.cpu_count()
    doc_count, single_wd_tf_list, max_tf, total_wd_tf, diff_wds = getFeatureData()
    matrix = np.eye(doc_count, doc_count)
    svd_vectors = getWordVectors(doc_count, single_wd_tf_list, max_tf, total_wd_tf, diff_wds)
    res_dict = Manager().dict()

    print('Matrix parallel calculation starting...')
    startTime = time.time()
    p = Pool(cpu_count - 1)
    end = 0
    for cur in range(cpu_count):
        start = end
        end = start + doc_count // cpu_count
        if cur == cpu_count - 1:
            end = doc_count
        p.apply_async(calculate_cosine, args=(svd_vectors, start, end, cur, doc_count, res_dict))
    print('Waiting for all sub processes done...')
    p.close()
    p.join()

    for i in range(doc_count):
        matrix[i] = res_dict[i]
        matrix[i][i] = 1
    print('All subprocesses done. Take [ %s ] seconds.' % str(time.time() - startTime))


if __name__ == '__main__':
    main_process()
