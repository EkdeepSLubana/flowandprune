import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import *
from config import *
from imp_estimator import cal_importance
from ptflops import get_model_complexity_info
import pickle as pkl
import os
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model to be trained", default='vgg', choices=['vgg', 'mobilenet', 'resnet-34', 'resnet-18', 'resnet-56'])
parser.add_argument("--seed", help="create a new backbone model by setting a different seed", default='0')
parser.add_argument("--data_path", help="path to dataset", default='CIFAR100')
parser.add_argument("--num_classes", help="number of classes in the dataset", default='100')
parser.add_argument("--T", help="temperature to smoothen the landscape", default='5')
parser.add_argument("--download", help="download the standard datasets?", default='False')
parser.add_argument("--track_stats", help="track train/test accuracy for later analysis", default='True', choices=['True', 'False'])
args = parser.parse_args()

######### Setup #########
torch.manual_seed(int(args.seed))
cudnn.deterministic = True
cudnn.benchmark = False
device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if(device == 'cuda'):
		print("Backend:", device)
else:
	raise Exception("Please use a cuda-enabled GPU.")

if not os.path.isdir('pretrained'):
	os.mkdir('pretrained')
if not os.path.isdir('stats'):
	os.mkdir('stats')
if not os.path.isdir('stats/base_models'):
	os.mkdir('stats/base_models')

pretrained_root = 'pretrained/'
data_path = args.data_path # path for dataloaders 
base_sched, base_epochs, wd = base_sched_iter, base_epochs_iter, wd

num_classes = int(args.num_classes)
T = float(args.T) # temperature to smooth the loss, if desired
track_stats = (args.track_stats == 'True')

######### Dataloaders #########
transform = transforms.Compose(
	[transforms.RandomCrop(32, padding=4),
	 transforms.RandomHorizontalFlip(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])
transform_test = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])

if(data_path=='CIFAR100'):
	trainset = torchvision.datasets.CIFAR100(root='./../datasets/cifar100', train=True, download=(args.download=='True'), transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR100(root='./../datasets/cifar100', train=False, download=(args.download=='True'), transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
elif(data_path=='CIFAR10'):
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

######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Optimizers #########
def get_optimizer(net, lr, wd):
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
	return optimizer

######### Training functions #########
# Training
def train(net, T=1.0):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs) / T
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
	stat['train'].append(100*(correct/total))

# Testing
def test(net, T=1.0):
	global cfg_state
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs) / T
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	stat['test'].append(100*(correct/total))
	# Save checkpoint.
	global best_acc
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {'net': net.state_dict()}
		torch.save(state, pretrained_root+'{mod_name}'.format(mod_name=args.model) + '_temp_' + str(T) + '_seed_' + args.seed +'.pth')
		best_acc = acc

# Create model for evaluation#net = torch.nn.DataParallel(VGG())
def create_model(name, num_classes=num_classes):
	print('num_classes:', num_classes)
	if(name == 'vgg'):
		net = torch.nn.DataParallel(VGG(num_classes=num_classes))
	elif(name == 'mobilenet'):
		net = torch.nn.DataParallel(MobileNet(num_classes=num_classes))
	elif(name == 'resnet-56'):
		net = torch.nn.DataParallel(ResNet56(num_classes=num_classes))
	elif(name == 'resnet-34'):
		net = torch.nn.DataParallel(ResNet34(num_classes=num_classes))
	elif(name == 'resnet-18'):
		net = torch.nn.DataParallel(ResNet18(num_classes=num_classes))
	return net

######### Determine model, load, and train #########
net = create_model(name=args.model, num_classes=num_classes)

# Train 
print("\n------------------ Training base model ------------------\n")
best_acc = 0
lr_ind = 0
epoch = 0
optimizer = get_optimizer(net, lr=base_sched[lr_ind], wd=wd)
stat = {'train':[], 'test':[]}
while(lr_ind < len(base_sched)):
	optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
	print("\n--learning rate is {}".format(base_sched[lr_ind]))
	for n in range(base_epochs[lr_ind]):
		print('\nEpoch: {}'.format(epoch))
		train(net, T=T)
		test(net, T=T)
		epoch += 1
	lr_ind += 1
print("Accuracy of trained model (best checkpoint): {:.2%}".format(best_acc / 100))

# Print FLOPs in base model 
with torch.cuda.device(0):
	flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
	print('{:<30}  {:<8}\n'.format('FLOPs in base model: ', flops))

if(track_stats):
	stats_loc = './stats/base_models/'
	stats_loc += (args.model + '_')
	stats_loc += 'seed_' + args.seed + '_'
	stats_loc += 'temp_' + str(int(T))
	stats_loc += '.pkl'

	with open(stats_loc, 'wb') as f:
		pkl.dump(stat, f)