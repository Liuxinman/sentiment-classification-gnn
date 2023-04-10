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
-   [x] 将glove/lstm embedding替换为pretrained bert (4.2 ~ 4.7)(Ruixuan, Xinman)
-   [x] 将conv aggregate改为attention（GCN -> GAT）(4.2 ~ 4.7)(Xinrong)
-   [ ] Train GCN, GAT, ATAE-LSTM
-   [ ] Test

## Meetings

-   [x] 4.2 Sunday 11:00am
-   [ ] 4.8 Saturday 11:00am
