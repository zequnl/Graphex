import os
import json
import numpy as np
from biobert_embedding.embedding import BiobertEmbedding
import pickle

# prepare data for global semantic embeddings
file_list = os.listdir("data/")
for fi in file_list:
    if fi.find(".txt") != -1:
        continue  
    train_names = open("data/" + fi + "/train_name.txt", "r").readlines()
    valid_names = open("data/" + fi +  "/valid_name.txt", "r").readlines()
    test_names = open("data/" + fi + "/test_name.txt", "r").readlines()
    train_defs = open("data/" + fi + "/train_def.txt", "r").readlines()
    valid_defs = open("data/" + fi +  "/valid_def.txt", "r").readlines()
    test_defs = open("data/" + fi + "/test_def_gen.txt", "r").readlines()
    graph = json.load(open("data/" + fi + "/graph.json", "r"))
    l = []
    dic = {}
    dicr = {}
    for k in train_names:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    for k in valid_names:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    for k in test_names:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    with open("data/" + fi + "/phrase_dic.p", "wb") as f:
        pickle.dump(dic, f)
    with open("data/" + fi + "/reversed_dic.p", "wb") as f:
        pickle.dump(dicr, f)
    with open("data/" + fi + "/phrase_vocab.p", "wb") as f:
        pickle.dump(l, f)
    f1 = open("data/" + fi + "/graph.txt", "w")
    for n in graph:
        n1 = dic[n]
        for k in graph[n]:
            n2 = dic[k]
            f1.write(str(n1))
            f1.write(" ")
            f1.write(str(n2))
            f1.write("\n")
    f1.close()
    l = []
    dic = {}
    dicr = {}
    for k in train_defs:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    for k in valid_defs:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    for k in test_defs:
        key = k[:-1]
        dicr[len(dicr)] = key
        dic[key] = len(dic)
        l.append(key)
    with open("data/" + fi + "/phrase_dic_def.p", "wb") as f:
        pickle.dump(dic, f)
    with open("data/" + fi + "/reversed_dic_def.p", "wb") as f:
        pickle.dump(dicr, f)
    with open("data/" + fi + "/phrase_vocab_def.p", "wb") as f:
        pickle.dump(l, f)
