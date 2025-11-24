#!/bin/bash
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/skspsdf/train_edm_condition_config.yaml
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/spsdf/train_edm_condition_config.yaml
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/sksdf/train_edm_condition_config.yaml
python train_gen.py --categories brain
python train_gen.py --categories bladder
python train_gen.py --categories colon
python train_gen.py --categories coronary_artery_left_d
python train_gen.py --categories coronary_artery_right_d
python train_gen.py --categories duodenum
python train_gen.py --categories gallbladder
python train_gen.py --categories liver
python train_gen.py --categories pancreas
python train_gen.py --categories skull
python train_gen.py --categories spleen
python train_gen.py --categories stomach
python train_gen.py --categories trachea
python train_gen.py --categories uterus
python train_ae.py
