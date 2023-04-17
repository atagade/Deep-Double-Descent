k ?= $(shell bash -c 'read -p "k: " k; echo $$k')

resnet:
	python train.py --dataset cifar100 --model resnet18 --width $(k) --num_gpus 1