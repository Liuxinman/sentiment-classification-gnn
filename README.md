# sentiment-classification-gnn
This is a course project for CSC2516, aimed at exploring the application of GNN in solving sentiment analysis tasks. The code in this repository is heavily dependent on [ASGCN](https://github.com/GeneZC/ASGCN).

## Resources

1. [ATAE-LSTM](https://aclanthology.org/D16-1058.pdf) [[code](https://paperswithcode.com/paper/attention-based-lstm-for-aspect-level)]
2. [ASGCN](https://aclanthology.org/D19-1464.pdf) [[code](https://github.com/GeneZC/ASGCN)]
3. [BiGCN](https://aclanthology.org/2020.emnlp-main.286.pdf) [[code](https://aclanthology.org/2020.emnlp-main.286.pdf)]
4. [TD-GAT](https://aclanthology.org/D19-1549.pdf) [[code](https://github.com/gordicaleksa/pytorch-GAT)] (没找到TD-GAT，但是有GAT的代码可参考)

## TODOs

-   [x] 跑通ASGCN (配置环境，benchmark rest14) (3.28 ~ 4.1)
-   [x] 研究ASGCN embedding部分的代码，提供如何改为pretrained bert的思路 (4.2 ~ 4.7)(Ruixuan, Xinman)
-   [ ] ~~将glove/lstm embedding替换为pretrained bert (4.2 ~ 4.7)(Ruixuan, Xinman)~~
-   [x] 将conv aggregate改为attention（GCN -> GAT）(4.2 ~ 4.7)(Xinrong)
-   [ ] experiment
-   [ ] report

## Experiment

Architecture: GCN / GAT

classifier head: 1. 用所有的hidden states / 2. 只用aspect的hidden states

#### Hyperparameter tuning:

-   Learning rate (0.01, 0.001, 0.0001, 0.00001)
-   Hidden_dim (25, 50, 100, 300)
-   attention heads (1, 4, 8, 16)

#### Ablation study

1.   GCN with classifier1
2.   GCN with classifier2
3.   GAT with classifier1
4.   GAT with classifier2

#### Investigate the effect of multiple aspects

1.   script

#### 分工

-   xinrong: GAT hyperparameter
-   Ruixuan: GCN hyperparameter, ablation study
-   Xinman: multiple aspects

## Meetings

-   [x] 4.2 Sunday 11:00am
-   [x] 4.8 Saturday 11:00am
-   [x] 4.13 Thursday 11:00am
-   [ ] 4.16 Sunday 10:00am
