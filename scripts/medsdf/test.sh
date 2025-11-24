#!/bin/bash
python test_gen.py --categories brain --ckpt logs_gen/GEN_2025_11_14__17_58_39/ckpt_last.pt
python test_gen.py --categories bladder --ckpt logs_gen/GEN_2025_11_14__20_52_05/ckpt_last.pt
python test_gen.py --categories colon --ckpt logs_gen/GEN_2025_11_15__00_04_37/ckpt_last.pt
python test_gen.py --categories coronary_artery_left_d --ckpt logs_gen/GEN_2025_11_15__03_11_21/ckpt_last.pt
python test_gen.py --categories coronary_artery_right_d --ckpt logs_gen/GEN_2025_11_15__06_19_54/ckpt_last.pt
python test_gen.py --categories duodenum --ckpt logs_gen/GEN_2025_11_15__09_27_27/ckpt_last.pt
python test_gen.py --categories gallbladder --ckpt logs_gen/GEN_2025_11_15__12_46_53/ckpt_last.pt
python test_gen.py --categories liver --ckpt logs_gen/GEN_2025_11_15__15_49_37/ckpt_last.pt
python test_gen.py --categories pancreas --ckpt logs_gen/GEN_2025_11_15__19_00_30/ckpt_last.pt
python test_gen.py --categories skull --ckpt logs_gen/GEN_2025_11_15__22_02_35/ckpt_last.pt
python test_gen.py --categories spleen --ckpt logs_gen/GEN_2025_11_16__01_12_52/ckpt_last.pt
python test_gen.py --categories stomach --ckpt logs_gen/GEN_2025_11_16__04_13_28/ckpt_last.pt
python test_gen.py --categories trachea --ckpt logs_gen/GEN_2025_11_16__07_18_22/ckpt_last.pt
python test_gen.py --categories uterus --ckpt logs_gen/GEN_2025_11_16__10_21_17/ckpt_last.pt
python test_ae.py --ckpt /home/wlsdzyzl/project/diffusion-point-cloud/logs_ae/AE_2025_11_14__11_15_16/ckpt_0.004129_134000.pt
