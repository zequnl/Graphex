## Graphine: A Dataset for Graph-aware Terminology Definition Generation

This is the implementation of the Graphex model in our EMNLP 2021 paper:

**Graphine: A Dataset for Graph-aware Terminology Definition Generation**. 

Zequn Liu, Shukai Wang, Yiyang Gu, Ruiyi Zhang, Ming Zhang* and Sheng Wang*

https://arxiv.org/abs/2109.04018

Please cite our paper when you use this code in your work.

## Data preparation

1. Our [**Graphine**](https://zenodo.org/record/5320310#.YVlnnZrP02w) dataset are released. Download the dataset and split each DAG into ```train_name.txt```,```train_def.txt```,```valid_name.txt```,```valid_def.txt```,```test_name.txt```,```test_def.txt```. (Since the dataset is large, we suggest you to choose part of it to conduct experiments.) Put the splited data in /data/, each DAG is a directory.

2. Generate the BioBERT embeddings and the data for global semantic embedding:
 ```console
❱❱❱ python embeddings_prepare.py
```
The BioBERT embeddings ```vectors/embeddings.txt``` will be generated.

Use a pretrained Transformer model to generate the replacement of definitions for the test set given their terminologies. We use the union of the training sets in all DAGs to train the Transformer and use the union of the validation sets to early stop. We put these data in /data_full/. ```definition_prepare.sh``` is the script for training and inference.```data/DAG_NAME/test_def_gen.txt``` will be generated. Then:

```console
❱❱❱ python data_prepare.py
```
The data for global semantic embedding of each DAG ```data/DAG_NAME/phrase_dic.p```, ```data/DAG_NAME/reversed_dic.p```, ```data/DAG_NAME/phrase_vocab.p``` , ```data/DAG_NAME/graph.txt``` will be generated.


## Training

1. Encoding global semantic via graph propagation: Use [**Content-Aware-Node2Vec**](https://github.com/SotirisKot/Content-Aware-Node2Vec) and the preprocessed data ```data/DAG_NAME/phrase_dic.p```, ```data/DAG_NAME/reversed_dic.p```, ```data/DAG_NAME/phrase_vocab.p``` , ```data/DAG_NAME/graph.txt``` to generate the global semantic embeddings for terminologies ```data/DAG_NAME/node_embeddings_phrases.p``` , ```data/DAG_NAME/node_embeddings_phrases_def.p```.

2. Fusing local and global semantic for definition generation. ```experiments.sh``` is the script for training and inference.

