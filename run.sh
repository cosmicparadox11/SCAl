python train_classifier_baseline.py --data_name CIFAR10 --model_name resnet18 --data_mode old --control_name 5000_sim-ft-bl_100_0.1_iid_5-5_0.5_1
python train_classifier_baseline.py --data_name CIFAR10 --model_name resnet18 --data_mode new --control_name 5000_sim-ft-bl_100_0.1_iid_5-5_0.5_1

python train_classifier_ssfl.py --data_name CIFAR10 --model_name resnet18 --data_mode old --cycles 150 --control_name 50000_sup_100_0.1_iid_5-5_0.5_1
python train_classifier_ssfl.py --data_name CIFAR10 --model_name resnet18 --data_mode new --cycles 150 --control_name 50000_sup_100_0.1_iid_5-5_0.5_1