#!/bin/bash

train_flemme --config /home/guoqingzhang/flemme/resources/pcd/medshapenet/train_pointnet_clm.yaml
train_flemme --config /home/guoqingzhang/flemme/resources/pcd/medshapenet/train_pointnet2_clm.yaml
train_flemme --config /home/guoqingzhang/flemme/resources/pcd/medshapenet/train_dgcnn_clm.yaml
train_flemme --config /home/guoqingzhang/flemme/resources/pcd/medshapenet/train_pointtrans_clm.yaml