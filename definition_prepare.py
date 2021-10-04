from utils.data_reader import Data
from model.transformer import Transformer
from model.common_layer import evaluate_transformer as evaluate
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import numpy as np 
import pickle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
config.data_dir = "data_full/"
config.save_path = "save_transformer/"
config.save_path_dataset = "save_transformer/"
config.vocab_file = "data/vocab.txt"
config.emb_file = "vectors/embeddings.txt"
p = Data()
train_len = len(open(config.data_dir + "/train_name.txt", "r").readlines())
valid_len = len(open(config.data_dir + "/valid_name.txt", "r").readlines())
test_len = len(open(config.data_dir + "/test_name.txt", "r").readlines())
config.train_len = train_len
config.valid_len = train_len + valid_len
config.total_len = train_len + valid_len + test_len
data_loader_all, data_loader_tr, data_loader_val, data_loader_test = p.get_all_data(batch_size=config.batch_size)
model = Transformer(p.vocab)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))
best_ppl = 50000
cnt = 0
best_model = model.state_dict()
for e in range(config.epochs):
    print("Epoch", e)
    p, l, c = [],[], []
    pbar = tqdm(enumerate(data_loader_tr),total=len(data_loader_tr))
    for i, d in pbar:
        torch.cuda.empty_cache()
        loss, ppl, total_loss = model.train_one_batch(d)
        l.append(loss)
        p.append(ppl)
        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
        torch.cuda.empty_cache()
        
    loss,ppl_val,bleu_score_b = evaluate(model,data_loader_val,model_name=config.model,ty="valid", verbose=False)
    if(ppl_val <= best_ppl):
        best_ppl = ppl_val
        cnt = 0
        best_model = model.state_dict()
        model.save_model(best_ppl,e,0,0,bleu_score_b,0)
    else: 
        cnt += 1
    if(cnt > 10): 
        break
        
model.load_state_dict(best_model)
file_list = os.listdir("data/")
for di in file_list:
    if di == ".ipynb_checkpoints" or di == "vocab.txt":
        continue
    config.data_dir = "data/" + di
    config.save_path = "save_transformer/" + di + "/"
    config.save_path_dataset = "save_transformer/" + di + "/"
    p = Data()
    train_len = len(open(config.data_dir + "/train_name.txt", "r").readlines())
    valid_len = len(open(config.data_dir + "/valid_name.txt", "r").readlines())
    test_len = len(open(config.data_dir + "/test_name.txt", "r").readlines())
    config.train_len = train_len
    config.valid_len = train_len + valid_len
    config.total_len = train_len + valid_len + test_len
    data_loader_all, data_loader_tr, data_loader_val, data_loader_test = p.get_all_data(batch_size=config.batch_size)
    pbar = tqdm(enumerate(data_loader_test),total=len(data_loader_test))
    loss,ppl,bleu_score_b = evaluate(model,data_loader_test,model_name=config.model,ty='test',verbose=False,log=True, result_file="data/" + di + "/test_def_gen.txt", ref_file="results/" + di + "/ref_transformer_full.txt", case_file="results/" + di + "/case_transformer_full.txt")
    
            
 



