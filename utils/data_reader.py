import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re, math
import random
from random import randint
from collections import Counter
from random import shuffle
import pprint
pp = pprint.PrettyPrinter(indent=1)
import torch
import torch.utils.data as data
# from torch.autograd import Variable
from utils import config
#import config
import pickle
import os
import json
from collections import defaultdict

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "PAD", 2: "EOS", 3: "SOS"} 
        #self.index2word = {}
        self.n_words = 4 # Count default tokens
        #vocab_file = open("data_simp/vocab_mask.txt", encoding="utf-8")
        #vocab_file = open("data_simp/vocab.txt", encoding="utf-8")
        vocab_file = open(config.vocab_file, encoding="utf-8")
        vocab_data = vocab_file.readlines()
        vocab_file.close()
        print(len(vocab_data))
        for index, line in enumerate(vocab_data):
            word = line.rstrip()
            self.index2word[index + 4] = word
            self.word2index[word] = index + 4
            self.n_words += 1

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, node_dic, node_emb, node_emb2, vocab, per=0):
        """Reads source and target sequences from txt files."""
        self.src = []
        self.trg = [] 
        self.trg2 = []
        self.node_index = []
        self.node_emb = []
        self.node_emb2 = []
        self.max_len_sent = 1
        self.max_len_words = 0
        self.max_len_answer = 0
        for d in data:
            if(len(d[0].split(' ')) > self.max_len_words): 
                self.max_len_words = len(d[0].split(' ')) 
            if(len(d[1].split(' ')) > self.max_len_words): 
                self.max_len_words = len(d[1].split(' '))
            if(len(d[1].split(' ')) > self.max_len_answer): 
                self.max_len_answer = len(d[1].split(' '))
            self.src.append([d[0]])
            self.trg.append(d[1])
            self.trg2.append(d[1])
            idx = node_dic[d[0]]
            self.node_index.append(idx)
            self.node_emb.append(node_emb[idx])
            if not os.path.exists(config.data_dir + "/node_embeddings_phrases.p"):
                self.node_emb.append(0)
            else:
                self.node_emb.append(node_emb[idx])
            if not os.path.exists(config.data_dir + "/node_embeddings_phrases_def.p"):
                self.node_emb2.append(0)
            else:
                self.node_emb2.append(node_emb2[idx])
        self.vocab = vocab
        self.num_total_seqs = len(data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["input_txt"] = self.src[index]
        item["target_txt"] = self.trg[index]
        item["target_txt2"] = self.trg2[index]
        item["node_index"] = self.node_index[index]
        item["node_emb"] = self.node_emb[index]
        item["node_emb2"] = self.node_emb2[index]
        item["input_batch"] = self.preprocess(self.src[index]) 
        item["target_batch"] = self.preprocess(self.trg[index], anw=True) 
        item["target_batch2"] = self.preprocess(self.trg2[index], anw=True)  
        if config.pointer_gen:
            item["input_ext_vocab_batch"], item["article_oovs"] = self.process_input(item["input_txt"])
            item["target_ext_vocab_batch"] = self.process_target(item["target_txt"], item["article_oovs"])
        return item

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr.split()] + [config.EOS_idx]
        else:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in ' '.join(arr).split()]
        return torch.LongTensor(sequence)


    # for know I ignore unk
    def process_input(self, input_txt):
        seq = []
        oovs = []
        for word in ' '.join(input_txt).strip().split():
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            #elif:
                #seq.append(config.UNK_idx)
            else:
                if word not in oovs:
                    oovs.append(word)
                #seq.append(self.vocab.n_words + oovs.index(word))
                seq.append(config.UNK_idx)
        
        seq = torch.LongTensor(seq)
        return seq, oovs

    def process_target(self, target_txt, oovs):
        # seq = [self.word2index[word] if word in self.word2index and self.word2index[word] < self.output_vocab_size else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
        seq = []
        for word in target_txt.strip().split():
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            elif word in oovs:
                #seq.append(self.vocab.n_words + oovs.index(word))
                seq.append(config.UNK_idx)
            else:
                seq.append(config.UNK_idx)
        seq.append(config.EOS_idx)
        seq = torch.LongTensor(seq)
        return seq

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    data.sort(key=lambda x: len(x["input_batch"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    input_batch, input_lengths = merge(item_info['input_batch'])
    target_batch, target_lengths = merge(item_info['target_batch'])
    target_batch2, target_lengths2 = merge(item_info['target_batch2'])

    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)
    target_batch2 = target_batch2.transpose(0, 1)
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    target_lengths2 = torch.LongTensor(target_lengths2)

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()
        target_batch2 = target_batch2.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()
        target_lengths2 = target_lengths2.cuda()

    d = {}
    d["input_batch"] = input_batch
    d["target_batch"] = target_batch
    d["target_batch2"] = target_batch2
    d["input_lengths"] = input_lengths
    d["target_lengths"] = target_lengths
    d["target_lengths2"] = target_lengths2
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]
    d["node_index"] = item_info["node_index"]
    d["node_emb"] = torch.FloatTensor(item_info["node_emb"])
    d["node_emb2"] = torch.FloatTensor(item_info["node_emb2"])
    if config.USE_CUDA:
        d["node_emb"] = d["node_emb"].cuda()
        d["node_emb2"] = d["node_emb2"].cuda()


    if 'input_ext_vocab_batch' in item_info:
        input_ext_vocab_batch, _ = merge(item_info['input_ext_vocab_batch'])
        target_ext_vocab_batch, _ = merge(item_info['target_ext_vocab_batch'])
        input_ext_vocab_batch = input_ext_vocab_batch.transpose(0, 1)
        target_ext_vocab_batch = target_ext_vocab_batch.transpose(0, 1)
        if config.USE_CUDA:
            input_ext_vocab_batch = input_ext_vocab_batch.cuda()
            target_ext_vocab_batch = target_ext_vocab_batch.cuda()
        d["input_ext_vocab_batch"] = input_ext_vocab_batch
        d["target_ext_vocab_batch"] = target_ext_vocab_batch
        if "article_oovs" in item_info:
            d["article_oovs"] = item_info["article_oovs"]
            d["max_art_oovs"] = max(len(art_oovs) for art_oovs in item_info["article_oovs"])
    return d 

def prepare_data_seq():
 
    train_name = open(config.data_dir + "/train_name.txt", "r").readlines()
    train_def = open(config.data_dir + "/train_def.txt", "r").readlines()
    valid_name = open(config.data_dir + "/valid_name.txt", "r").readlines()
    valid_def = open(config.data_dir + "/valid_def.txt", "r").readlines()
    test_name = open(config.data_dir + "/test_name.txt", "r").readlines()
    test_def = open(config.data_dir + "/test_def.txt", "r").readlines()
    with open(config.data_dir + "/graph.json", 'r') as f:
        graph = json.load(f)
    train = {}
    valid = {}
    test = {}
    for i in range(len(train_name)):
        train[len(train)] = [[train_name[i][:-1],train_def[i][:-1]]]
    for i in range(len(valid_name)):
        valid[len(valid)] = [[valid_name[i][:-1],valid_def[i][:-1]]]
    for i in range(len(test_name)):
        test[len(test)] = [[test_name[i][:-1],test_def[i][:-1]]]
    vocab = Lang()
    print("Vocab_size %s " % vocab.n_words) 
    adj_lists = defaultdict(set)
    node_dic = {}
    for l in train_name:
        node_dic[l.replace("\n","")] = len(node_dic) 
    for l in valid_name:
        node_dic[l.replace("\n","")] = len(node_dic) 
    for l in test_name:
        node_dic[l.replace("\n","")] = len(node_dic)
    for k in graph:
        node1 = node_dic[k]
        for n in graph[k]:
            node2 = node_dic[n]
            adj_lists[node1].add(node2)
            adj_lists[node2].add(node1)
   
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(config.save_path+'dataset.p', "wb") as f:
        pickle.dump([train,valid,test,vocab], f)
        print("Saved PICKLE")
    with open(config.save_path+'/node_dic.json','w') as f:
        json.dump(node_dic,f)
    if not os.path.exists(config.data_dir + "/node_embeddings_phrases.p"):
        return train, valid, test, vocab, adj_lists, node_dic, node_dic, node_dic
    with open(config.data_dir + "/node_embeddings_phrases.p", "rb") as f:
        node_emb = pickle.load(f)
    with open(config.data_dir + "/node_embeddings_phrases_def.p", "rb") as f:
        node_emb2 = pickle.load(f)
    return train, valid, test, vocab, adj_lists, node_dic, node_emb, node_emb2

class Data:
    def __init__(self):
        self.train, self.valid, self.test, self.vocab, self.graph, self.node_dic, self.node_emb, self.node_emb2 = prepare_data_seq()
        self.type = {'train': self.train, 'valid': self.valid, 'test': self.test} 
        for k in self.node_dic:
            if self.node_dic[k] not in self.graph:
                self.graph[self.node_dic[k]] = set()
                    
    def get_len_dataset(self,split):
        return len(self.type[split])

    def get_all_data(self,batch_size):
        tr = []
        val = []
        test = []
        total = []
        for i in range(len(self.train)):
            #if i > (len(self.train) // 2):
                #break
            for p in self.train[i]:
                tr.append(p)
                total.append(p)
        for i in range(len(self.valid)):
            for p in self.valid[i]:
                val.append(p)
                total.append(p)
        for i in range(len(self.test)):
            for p in self.test[i]:
                test.append(p) 
                total.append(p)
        
        dataset_train = Dataset(tr, self.node_dic, self.node_emb, self.node_emb2, self.vocab)
        dataset_total = Dataset(total, self.node_dic, self.node_emb, self.node_emb2, self.vocab)
        data_loader_all = torch.utils.data.DataLoader(dataset=dataset_total,
                                                batch_size=len(dataset_total),
                                                shuffle=False, collate_fn=collate_fn)
        data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True, collate_fn=collate_fn)


        dataset_val = Dataset(val, self.node_dic, self.node_emb, self.node_emb2, self.vocab)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,collate_fn=collate_fn)  

        dataset_test = Dataset(test, self.node_dic, self.node_emb, self.node_emb2, self.vocab)
        data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,collate_fn=collate_fn)           
        return data_loader_all, data_loader_tr, data_loader_val, data_loader_test

    