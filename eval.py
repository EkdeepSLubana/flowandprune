import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import torch.optim as optim
import os
import shutil
from models import *
from pruner import *
from ptflops import get_model_complexity_info
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="architecture model to be analyzed", default='vgg', choices=['vgg', 'mobilenet', 'resnet', 'resnet-56'])
parser.add_argument("--num_classes", help="number of classes in the dataset", default='100')
parser.add_argument("--model_path", help="path to model", default='0')
parser.add_argument("--data_path", help="path to dataset", default='CIFAR100')
parser.add_argument("--download", help="download CIFAR10/CIFAR100", default='False')
parser.add_argument("--pruned", help="is the model to be analyzed a pruned model?", default='False', choices=['True', 'False'])
parser.add_argument("--train_acc", help="evaluate train accuracy", default='False', choices=['True', 'False'])
parser.add_argument("--test_acc", help="evaluate test accuracy", default='False', choices=['True', 'False'])
parser.add_argument("--flops", help="calculate flops in a model", default='False', choices=['True', 'False'])
parser.add_argument("--compression", help="calculate compression ratio for model", default='False', choices=['True', 'False'])
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = int(args.num_classes)
criterion = nn.CrossEntropyLoss()
	
######### Functions to evaluate different properties #########
# Accuracy
def cal_acc(net, use_loader):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(use_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	return correct / total

# FLOPs 
def cal_flops(net):
	with torch.cuda.device(0):
		flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
		print('	FLOPs: {:<8}'.format(flops))

# Compression Ratio
def cal_compression_ratio(net_path, model):
	temp_path = "./temp_models/"
	base_model = create_model(name=model, is_pruned=False, num_classes=num_classes)
	if os.path.exists(temp_path):
		shutil.rmtree(temp_path)
	os.mkdir(temp_path)
	state = {'net': base_model.state_dict()}
	torch.save(state, temp_path+'temp_base.pth')
	base_size = os.path.getsize(temp_path+'temp_base.pth')
	model_size = os.path.getsize(net_path)
	print("	Compression ratio: {:.3}".format(base_size / model_size))
	shutil.rmtree(temp_path)

# Create model for evaluation
def create_model(name, is_pruned, num_classes=num_classes):
	print('num_classes:', num_classes)
	if(name == 'vgg'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(VGG_p(cfg_p, num_classes=num_classes))
		else:
			net = torch.nn.DataParallel(VGG(num_classes=num_classes))

	elif(name == 'mobilenet'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(MobileNet_p(cfg_p[0], cfg_p[1:], num_classes=num_classes))
		else:
			net = torch.nn.DataParallel(MobileNet(num_classes=num_classes))

	elif(name == 'resnet-34'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(ResPruned(cfg_p, num_classes=num_classes))
		else:
			net = torch.nn.DataParallel(ResNet34(num_classes=num_classes))

	elif(name == 'resnet-18'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(ResPruned(cfg_p, num_blocks=[2,2,2,2], num_classes=num_classes))
		else:
			net = torch.nn.DataParallel(ResNet18(num_classes=num_classes))

	elif(name == 'resnet-56'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(ResPruned_cifar(cfg_p, num_classes=num_classes))
		else:
			net = torch.nn.DataParallel(ResNet56(num_classes=num_classes))
	return net


######### Print model name #########
print((args.model).upper())

######### Dataloader #########
if(args.train_acc == 'True' or args.test_acc == 'True'):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	transform_test = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	if(args.data_path=='CIFAR100'):
		trainset = torchvision.datasets.CIFAR100(root='./../datasets/cifar100', train=True, download=(args.download=='True'), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
		testset = torchvision.datasets.CIFAR100(root='./../datasets/cifar100', train=False, download=(args.download=='True'), transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
	elif(args.data_path=='CIFAR10'):
		num_classes = 10
		trainset = torchvision.datasets.CIFAR10(root='./../datasets/cifar10', train=True, download=(args.download=='True'), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
		testset = torchvision.datasets.CIFAR10(root='./../datasets/cifar10', train=False, download=(args.download=='True'), transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
	else:
		trainset = datasets.ImageFolder(root=data_path+'/train', transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
		testset = datasets.ImageFolder(root=data_path+'/test', transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

######### Load network or create new #########
if(args.data_path=='CIFAR10'):
       	num_classes = 10
if(args.train_acc == 'True' or args.test_acc == 'True' or args.flops=='True'):
	net_dict = torch.load(args.model_path)
	net = create_model(name=args.model, is_pruned=(args.pruned=='True'), num_classes=num_classes)
	net.load_state_dict(net_dict['net'])
	
######### FLOPs evaluation #########
if(args.flops == 'True'):
	cal_flops(net)

######### Compression ratio evaluation #########
if(args.compression == 'True'):
	cal_compression_ratio(net_path=args.model_path, model=args.model)

######### Train accuracy evaluation #########
if(args.train_acc == 'True'):
	acc = cal_acc(net, use_loader=trainloader)
	print("	Train accuracy: {:.2%}".format(acc))

######### Test accuracy evaluation #########
if(args.test_acc == 'True'):
	acc = cal_acc(net, use_loader=testloader)
	print("	Test accuracy: {:.2%}".format(acc))