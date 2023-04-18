# sentiment-classification-gnn
This is a course project for [CSC2516](https://artsci.calendar.utoronto.ca/course/csc413h1) Winter 2023. 
We aimed at exploring the application of two different Graph Neural Network(GNN) in solving aspect-based sentiment analysis tasks(ABAS).

## Abstract
Aspect-based sentiment analysis (ABSA) is a highly popular NLP task that aims to identify polarity towards a specific aspect. Recently, the combination of graph with traditional network has become a useful technique for solving ABSA. In this paper, we have implemented Graph Convolutional Network (GCN) and Graph Attention Network (GAT) for ABSA with the objective of comparing the performance between the two models. First, we compare the performance of GAT and GCN with two different classifier heads on Semeval tasks from 2014 to 2016. Then, we conduct a novel analysis on the effect of multiple numbers of aspects on the two models. Finally, we analyze the results and discuss the limitations. 

## Requirements
- Python 3.9
- PyTorch 1.13.0
- SpaCy 3.5.1
- Numpy 1.23.3

## Usage
1. Install SpaCy package and language models with
   ```
   pip install spacy
   ```
   ```
   python -m spacy download en
   ```
2. Generate graph data with ```python dependency_graph.py```
3. Download pretrained GloVe embeddings with [Glove](https://nlp.stanford.edu/projects/glove/) and extract glove.840B.300d.txt into glove/.
4. Train with command, optional arguments could be found in train.py.
```python train.py --model_name asgcn --dataset rest14```


   
## Reference
The code in this repository is  dependent on [ASGCN](https://github.com/GeneZC/ASGCN) and[GAT](https://github.com/gordicaleksa/pytorch-GAT)

- Zhang, Chen and Li, Qiuchi and Song, Dawei. 
"Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks". 
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) 
2019 Nov. Hong Kong, China. Association for Computational Linguistics. https://www.aclweb.org/anthology/D19-1464. 10.18653/v1/D19-1464",
pages 4560-4570.
- Gordić, Aleksa. Gordić2020PyTorchGAT. 2020. GitHub.https://github.com/gordicaleksa/pytorch-GAT.


