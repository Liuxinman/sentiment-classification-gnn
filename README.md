# sentiment-classification-gnn

## Resources

1. [ATAE-LSTM](https://aclanthology.org/D16-1058.pdf) [[code](https://paperswithcode.com/paper/attention-based-lstm-for-aspect-level)]
2. [ASGCN](https://aclanthology.org/D19-1464.pdf) [[code](https://github.com/GeneZC/ASGCN)]
3. [BiGCN](https://aclanthology.org/2020.emnlp-main.286.pdf) [[code](https://aclanthology.org/2020.emnlp-main.286.pdf)]
4. [TD-GAT](https://aclanthology.org/D19-1549.pdf) [[code](https://github.com/gordicaleksa/pytorch-GAT)] (没找到TD-GAT，但是有GAT的代码可参考)

## TODOs

-   [ ] 跑通ASGCN (配置环境，benchmark rest14) (3.28 ~ 4.1)
-   [ ] 研究ASGCN embedding部分的代码，提供如何改为pretrained bert的思路
-   [ ] 将glove/lstm embedding替换为pretrained bert
-   [ ] 将conv aggregate改为attention （GCN -> GAT）
-   [ ] Train GCN, GAT, ATAE-LSTM
-   [ ] Test

## Meetings

-   [ ] 4.1 Saturday / 4.2 Sunday