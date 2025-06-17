#!/bin/bash
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_ddpm.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_resunet_ddpm.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_cddpm_add.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_cddpm_inj_add_to_time.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_cddpm_inj_bias.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_cddpm_inj_gate_bias.yaml
train_flemme --config /home/wlsdzyzl/project/flemme/resources/img/mnist/different_condition_injection/train_unet_cddpm_inj_ca.yaml
