import os
import json
import numpy as np
from biobert_embedding.embedding import BiobertEmbedding
import pickle

# pretrained biobert embeddings
f = open("data/vocab.txt", "r")
f2 = open("vectors/embeddings.txt", "w")
lines = f.readlines()
biobert = BiobertEmbedding(model_path="../biobert_v1.1_pubmed_pytorch_model")
for i,l in enumerate(lines):
    if i % 100 == 0:
        print(i)
    word_embeddings = biobert.word_vector(l[:-1])
    f2.write(l[:-1])    
    for e in word_embeddings[0]:
        f2.write(" ")
        f2.write(str(e)[7:-1])
    f2.write("\n")
f2.close()
