python train_centralDA_sourceOC.py --data_name 'OfficeCaltech' --domain_s 'webcam' --data_name_unsup 'OfficeCaltech' --domain_u 'dslr'  --model_name resnet50 --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'ExponentialLR' --cycles 50   --control_name "1_sup-ft-fix_8_0.5_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020  
python train_centralDA_sourceOC.py --data_name 'OfficeCaltech' --domain_s 'dslr' --data_name_unsup 'OfficeCaltech' --domain_u 'webcam'  --model_name resnet50 --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'ExponentialLR' --cycles 50   --control_name "1_sup-ft-fix_8_0.5_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020  




python train_centralDA_sourceOC.py --data_name 'DomainNet' --domain_s 'clipart' --data_name_unsup 'DomainNet' --domain_u 'infograph'  --model_name resnet50 --d_mode new --device cuda:0 --var_lr 0.03  --scheduler_name  'ExponentialLR' --cycles 50   --control_name "1_sup-ft-fix_8_0.5_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020  