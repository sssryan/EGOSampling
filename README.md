# EGOSampling

We provide detailed implmentations of plug-in of our EGOSampling on different kinds of graph neural networks on OGB-MAG.

Before using EGOSampling to scale each model, you need to run '''python preprocess_ogbn_mag.py''' to generate the graph format which EGOSampling needs.

For HGT, run '''python ogbn_mag.py --n_layers 4 --prev_norm --last_norm --use_RTE''' in /OGB-MAG/HGT/pyHGT.
For the GCN and GAT model, you can simply add '''--conv name gcn/gat'''.

For R-GCN, we modify the code of official implementation of R-GCN from ogb leaderboard.
Run '''python rgcn.py''' in /OGB-MAG/R-GCN.
