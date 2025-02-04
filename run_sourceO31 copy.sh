python train_centralDA_source.py --domain_s 'dslr' --domain_u 'amazon' --model_name VITs --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'CosineAnnealingLR' --cycles 50  --control_name "2_sup-ft-fix_15_0.3_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020  --backbone_arch vit-small 
python train_centralDA_source.py --domain_s 'amazon' --domain_u 'webcam' --model_name VITs --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'CosineAnnealingLR' --cycles 50   --control_name "2_sup-ft-fix_15_0.3_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020   --backbone_arch vit-small 
python train_centralDA_source.py --domain_s 'webcam' --domain_u 'amazon' --model_name VITs --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'CosineAnnealingLR' --cycles 50   --control_name "2_sup-ft-fix_15_0.3_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020  --backbone_arch vit-small 


