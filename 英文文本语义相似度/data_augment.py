import random
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

def data_Augment(df, seed, prob=0.2):
    def discard(lines, discard_prob):
        # 随机丢弃产生数据
        lines_discard = []
        for index, line_list in enumerate(lines):
            tmp_lines = []
            for _, tmp_line in enumerate(line_list[:-1]):
                # 随机丢弃一些词
                words = tmp_line.strip().split()
                words_index = []  # 随机丢弃的索引
                words_new = []
                for _ in range(int(discard_prob*len(words))):
                    random_num = random.randint(0, len(words)-1)
                    words_index.append(random_num)
                
                if index == 0:
                    print("随机舍弃的索引：", words_index)
                for k in range(len(words)):
                    if k in words_index:
                        continue
                    words_new.append(words[k])
                tmp_line = ' '.join(words_new)
        
                tmp_lines.append(tmp_line)
            tmp_lines.append(line_list[-1])
            if index == 0:
                print("随机舍弃：", tmp_lines)  # 打印一个样本出来看一下
            lines_discard.append(tmp_lines)
        lines_discard = lines + lines_discard
        return lines_discard

    def insert(lines, insert_prob):
        # 随机插入产生数据
        lines_insert = []
        for index, line_list in enumerate(lines):
            tmp_lines = []
            for _, tmp_line in enumerate(line_list[:-1]):
                # 随机插入一些词
                words = tmp_line.strip().split()
                words_new = words
                words_index = []
                words_insert = []
                for _ in range(int(insert_prob*len(words))):
                    random_num_1 = random.randint(0, len(words_new)-1) # 要插入的位置
                    random_num_2 = random.randint(0, len(words)-1) # 要插入的词的索引
                    words_index.append(random_num_1)
                    words_insert.append(random_num_2)
                    words_new.insert(random_num_1,words[random_num_2])
                if index == 0:
                    print("插入词的位置", words_index)
                    print("待插入词的索引", words_insert)
                tmp_line = ' '.join(words_new)
                tmp_lines.append(tmp_line)
            tmp_lines.append(line_list[-1])
            if index == 0:
                print("随机插入：", tmp_lines)  # 打印一个样本出来看一下
            lines_insert.append(tmp_lines)
        lines_insert = lines + lines_insert
        return lines_insert

    def exchange(lines, exchange_prob):
        # 随机交换一些词的位置
        lines_exchange = []
        for index, line_list in enumerate(lines):
            tmp_lines = []
            for _, tmp_line in enumerate(line_list[:-1]):
                # 随机交换一些词
                words = tmp_line.strip().split()
                words_new = words
                for _ in range(int(exchange_prob*len(words))):
                    words_exchange = random.randint(0, len(words_new)-1)  # 要交换的位置
                    words_be_exchange = random.randint(0, len(words_new)-1)  # 要被交换的位置
                    tmp_exchange = words_new[words_exchange]
                    words_new[words_exchange] = words_new[words_be_exchange]
                    words_new[words_be_exchange] = tmp_exchange
                tmp_line = ' '.join(words_new)
                tmp_lines.append(tmp_line)
            tmp_lines.append(line_list[-1])
            if index == 0:
                print("随机交换：", tmp_lines)  # 打印一个样本出来看一下
            lines_exchange.append(tmp_lines)
        lines_exchange = lines + lines_exchange
        return lines_exchange

    def syn(lines, syn_prob):
        # 随机同义替换一些词
        lines_syn = []
        for index, line_list in enumerate(lines):
            tmp_lines = []
            for _, tmp_line in enumerate(line_list[:-1]):
                # 随机同义替换一些词
                words = tmp_line.strip().split()
                words_index = []  # 随机同义替换的索引
                words_new = words
                for _ in range(int(syn_prob*len(words))):
                    words_index.append(random.randint(0, len(words)-1))
                if index == 0:
                    print("随机同义词替换的索引：", words_index)
                for k in range(len(words_index)):
                    word_set = wn.synsets(words[words_index[k]])
                    word_set = [word.lemma_names() for word in word_set]
                    if word_set != []:
                        word_set = word_set[0]  # 选出第一组同义词
                        word_choose_index = random.randint(0, len(word_set)-1)
                        words_new[words_index[k]] = word_set[word_choose_index]
                tmp_line = ' '.join(words_new)
                tmp_lines.append(tmp_line)
            tmp_lines.append(line_list[-1])
            if index == 0:
                print("随机同义词替换：", tmp_lines)  # 打印一个样本出来看一下
            lines_syn.append(tmp_lines)
        lines_syn = lines + lines_syn
        return lines_syn

    df_columns = df.columns
    lines = []
    for index, line in df.iterrows():
        line_array = np.array(line)
        line_list = line_array.tolist()
        if index == 0:
            print("原始样本：", line_list)  # 打印一个样本出来看一下
        lines.append(line_list)

    lines_discard = discard(lines, prob)
    lines_insert = insert(lines, prob)
    lines_exchange = exchange(lines, prob)
    lines_syn = syn(lines, prob)

    times = 6 # multiple of the expansion of the original data
    lines_multi = []
    for _ in range(times):
        lines_multi = lines_multi + lines
    lines = lines_multi + lines_discard + lines_insert + lines_exchange + lines_syn
    random.shuffle(lines)

    df = pd.DataFrame(lines)
    df.columns = df_columns
    df = df.reset_index(drop=True)
    return df