# sentiment-classification-gnn
This is a course project for [CSC2516](https://artsci.calendar.utoronto.ca/course/csc413h1) Winter 2023. 
We aimed at exploring the application of two different Graph Neural Network(GNN) in solving aspect-based sentiment analysis tasks(ABAS).

## Abstract
Aspect-based sentiment analysis(ABAS) is one of the most popular NLP techniques that identifying polarity towards a specific aspect, and combining graph with traditional network becomes an essential techniques when solving ABAS. In this paper, we implemented two graph-related deep-learning architectures --Graph Convolutional Network(GCN) and Graph Attention Network(GAT)-- in nowadays aspect-based sentiment analysis.We aimed to compare the performance between the two models. First, we tuned the parameter on common benchmarks Semeval tasks from 2014 to 2016 to explore the best-performed model. Then we introduce the analysis on the effect of multiple number of aspects on two models, which, to our knowledge, is a novel approach in the comparison. Finally, we analyzed the difference between the results and discussed the limitations, which will help us have a better knowledge about Graph Neural Network on ABAS and its possible improvement direction in the future.

## Requirements
- Python 3.6
- PyTorch 1.0.0
- SpaCy 2.0.18
- Numpy 1.15.4

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
4. Train with command, optional arguments could be found in train.py. Our model has been pretrained so the current models are with the best performances.
```python train.py --model_name asgcn --dataset rest14```


   
## Reference
The code in this repository is  dependent on [ASGCN](https://github.com/GeneZC/ASGCN) and[GAT](https://github.com/gordicaleksa/pytorch-GAT)

- Zhang, Chen and Li, Qiuchi and Song, Dawei. 
"Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks". 
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) 
2019 Nov. Hong Kong, China. Association for Computational Linguistics. https://www.aclweb.org/anthology/D19-1464. 10.18653/v1/D19-1464",
pages 4560-4570.
- Gordić, Aleksa. Gordić2020PyTorchGAT. 2020. GitHub.https://github.com/gordicaleksa/pytorch-GAT.


