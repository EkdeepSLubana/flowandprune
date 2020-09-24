import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import *
from pruner import *
from config import *
from imp_estimator import cal_importance, model_params, model_grads
from ptflops import get_model_complexity_info
import copy
import numpy as np
import pickle as pkl
import os
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
# Arguments to fiddle with experiment in the paper.
parser.add_argument("-m", "--model", help="model to be pruned", default='vgg', choices=['vgg', 'mobilenet', 'resnet-56', 'resnet-34', 'resnet-18'])
parser.add_argument("--seed", help="setup the random seed", default='0')
parser.add_argument("--pretrained_path", help="path to pretrained model, if a pretrained model is to be pruned", default='0')
parser.add_argument("--data_path", help="path to dataset", default='CIFAR100')
parser.add_argument("--download", help="download the standard datasets?", default='False')
parser.add_argument("--pruning_type", help="train nets with which pruning method", choices=['mag_based', 'loss_based', 'biased_loss', 'grasp', 'grad_preserve', 'tfo', 'fisher', 'nvidia', 'biased_loss_tracked', 'l1', 'bn'])
parser.add_argument("--prune_percent", help="percentage to prune")
parser.add_argument("--n_rounds", help="number of rounds to perform pruning in")
parser.add_argument("--T", help="temperature to smoothen the landscape", default='5')
parser.add_argument("--grasp_T", help="temperature to smoothen the landscape", default='200')
parser.add_argument("--imp_samples", help="number of samples for importance estimation", default='2560')
parser.add_argument("--track_stats", help="track train/test accuracy for later analysis", default='False', choices=['True', 'False'])
# Extra arguments for experimental purposes.
parser.add_argument("--num_classes", help="number of classes in the dataset", default='100')
parser.add_argument("--moment", help="momentum for optimizer", default='0.9')
parser.add_argument("--warmup_epochs", help="warmup the model by training for few epochs", default='0')
parser.add_argument("--thresholds", help="define manual thresholds for pruning rounds", default='default')
parser.add_argument("--lr_prune", help="learning rate for initial pruning and training phase", default='0.1')
parser.add_argument("--preserve_moment", help="preserve momentum for pruned model", default='False')
parser.add_argument("--track_importance", help="track importance", default='False')
parser.add_argument("--track_batches", help="how many batches to track importance for (from the end)", default='20')
parser.add_argument("--alpha", help="momentum for tracking importance", default='default')
parser.add_argument("--use_init", help="use distance from initialization", default='False')
parser.add_argument("--use_l2", help="use l2 norm with a standard importance metric", default='False')
parser.add_argument("--bypass", help="bypass tracking settings for pruning pretrained models", default='False')
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
if not os.path.isdir('pruned_nets'):
	os.mkdir('pruned_nets')
if not os.path.isdir('stats'):
	os.mkdir('stats')
if not os.path.isdir('stats/pruned_models'):
	os.mkdir('stats/pruned_models')
pretrained_root = 'pretrained/'
pruned_root = 'pruned_nets/'

######### Setup the framework from argparser variables #########
model_name = args.model # model architecture
data_path = args.data_path # path for dataloaders 
pretrained_path = args.pretrained_path
num_classes = int(args.num_classes) # number of classes 
use_pretrained = (pretrained_path!='0')

n_rounds = int(args.n_rounds) # number of rounds for pruning
p_total = int(args.prune_percent) / 100 # how much to prune
thresholds = args.thresholds # choose manual thresholds instead of a pruning schedule
preserve_moment = (args.preserve_moment == 'True') # preserve momentum over pruning rounds?
pruning_type = args.pruning_type # how to prune

pruned_sched, pruned_epochs = pruned_sched_iter, pruned_epochs_iter # training settings for the final model
wd = wd # weight decay
warmup_epochs = int(args.warmup_epochs)
moment = float(args.moment) # momentum for training
lr_prune = float(args.lr_prune) # learning rate for the pruning + training phase
alpha = 0.8 if (args.alpha == 'default') else float(args.alpha) # momentum for importance tracking
T = float(args.T) # temperature to smooth the loss, if desired
grasp_T = float(args.grasp_T) # temperature for hessian-gradient products
if(pruning_type == 'grad_preserve'): 
	grasp_T = 1
imp_samples = 0 if track_importance else int(args.imp_samples)
track_stats = (args.track_stats == 'True')

### Following options are provided for experimental purposes ###
track_importance = (args.track_importance == 'True') # track importance over last several minibatches?
use_l2 = (args.use_l2 == 'True') # use parameter magnitude along with another importance measure?
use_init = (args.use_init == 'True') # use distance from initialization along with another importance measure?
if(pruning_type == 'biased_loss'): # Bias loss-based measure towards small distance from init, instead of plain magnitude?
	if(args.bypass == 'True'):
		use_init = True

if(pruning_type == 'biased_loss_tracked'): # Best settings when using tracked version of biased loss measure.
	if(args.bypass == 'False'):
		use_init, use_l2, preserve_moment, track_importance = False, True, True, True
track_batches = int(args.track_batches) if track_importance else 0 # number of batches to track importance over

print("\n------------------ Setup For Pruning ------------------\n")
print("Model to be pruned:", model_name)
print("Pruning using:", args.pruning_type)
print("Learning rate for pruning + training:", lr_prune)
print("Momentum:", moment)
print("Weight decay:", wd)
print("Number of pruning rounds:", n_rounds)
print("Targeted pruning ratio: {:.2%}".format(p_total))
print("Track importance:", track_importance)
print("Momentum for tracking importance: {:.2}".format(alpha))
print("# of Batches for tracking importance:", track_batches)
print("Use L2:", use_l2)
print("Use distance from initialization:", use_init)
print("Preserve momentum over pruning rounds:", preserve_moment)
print("Pretrained model:", use_pretrained)
print("Temperature for training model:", T)
if(pruning_type == 'grad_preserve' or pruning_type == 'grasp'): 
	print("GraSP calculation Temperature:", T)
print("Warmup epochs:", warmup_epochs)
print("Number of samples for importance estimation:", imp_samples)
print("Track train/test stats:", track_stats)

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

######### Importance trackers (for experimental purposes) #########
def cal_metrics(net, pruning_type=None, list_base=None, alpha=1.0, is_init=False, use_l2=True, params_init=None):
	if(is_init):
		params = model_params(net)
		list_metrics = [torch.zeros_like(params[3*ind+1]) for ind in range(int((len(params)-2)/3))]
		return list_metrics
	else:
		list_new = cal_importance(net, pruning_type, trainloader=None)
		if(use_l2):
			params = model_params(net)
			params = [p.reshape(p.shape[0], -1) for p in params]
			if (params_init == None):
				list_l2 = [params[3*ind].abs().detach().clone() for ind in range(int((len(params)-2)/3))]
			else:
				list_l2 = [(params[3*ind] - params_init[3*ind]).abs().detach().clone() for ind in range(int((len(params)-2)/3))]
			list_new = [(l_new * l_l2).sum(dim=1) for l_new, l_l2 in zip(list_new, list_l2)]
		list_metrics = [(alpha * l_new + (1 - alpha) * l_old) for l_new, l_old in zip(list_new, list_base)]
		return list_metrics

def shrink_metrics(list_base, order_layerwise, prune_ratio, orig_size):
	list_metrics = []
	for ind, (key, val) in enumerate(order_layerwise.items()):
		list_metrics.append(list_base[ind][np.sort(val[prune_ratio[ind]:])])
	return list_metrics

######### Initialization / training functions #########
# Create model
def create_model(name, num_classes=num_classes):
	print('num_classes:', num_classes)
	if(name == 'vgg'):
		net = torch.nn.DataParallel(VGG(num_classes=num_classes))
	elif(name == 'mobilenet'):
		net = torch.nn.DataParallel(MobileNet(num_classes=num_classes))
	elif(name == 'resnet-56'):
		net = torch.nn.DataParallel(ResNet56(num_classes=num_classes))
	elif(name == 'resnet-34'): # Not used in paper
		net = torch.nn.DataParallel(ResNet34(num_classes=num_classes))
	elif(name == 'resnet-18'): # Not used in paper
		net = torch.nn.DataParallel(ResNet18(num_classes=num_classes))
	return net

# Training
def train(net, alpha=1.0, T=1.0, track_batches=10, track_importance=False, pruning_type=None, mode=None):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	thresh = len(trainloader) - track_batches
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
		# Track importance
		if(track_importance and batch_idx >= thresh):
			global list_metrics
			list_metrics = cal_metrics(net, pruning_type=pruning_type, list_base=list_metrics, alpha=alpha, use_l2=use_l2, params_init=params_init)
	stat[mode]['train'].append(100*(correct/total))

# Testing
def test(net, T=1.0, save=False, warmup=False, mode=None):
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
	stat[mode]['test'].append(100*(correct/total))
	# Save best checkpoint
	if(save):
		global best_acc
		if(warmup):
			acc = 100.*correct/total
			if acc > best_acc:
				print('Saving..')
				state = {'net': net.state_dict()}
				torch.save(state, pretrained_root+'{mod_name}_warmup_to_{num}_temp_{T}'.format(mod_name=model_name, num=str(warmup_epochs), T=T)+'_'+args.seed+'.pth')
				best_acc = acc
		else:
			global cfg_state
			acc = 100.*correct/total
			if acc > best_acc:
				print('Saving..')
				state = {'net': net.state_dict(), 'cfg': cfg_state}
				torch.save(state, pruned_root+'{mod_name}_{type}_{num}_temp_{T}_rounds_{rounds}_seed'.format(mod_name=model_name, type=pruning_type, num=str(100-p_cumulative), T=T, rounds=n_rounds) + '_' + args.seed + '.pth')
				best_acc = acc

# Calculate accuracy on a dataloader
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
	return 100.*(correct / total)

### Initialize model ###
net = create_model(name=model_name, num_classes=num_classes)

### Use pretrained model ###
if(use_pretrained):
	print("\n------------------ Loading pretrained model ------------------\n")
	net_dict = torch.load(pretrained_path)
	net.load_state_dict(net_dict['net'])

# Print FLOPs in base model 
with torch.cuda.device(0):
	flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
	print('{:<30}  {:<8}\n'.format('FLOPs in base model: ', flops))
	print('{:<30}  {:<8}'.format('Parameters in base model: ', params))

######### Initialize the setup for pruning #########
### Pruning ratios for each round ###
if(thresholds == 'default'):
	p_ratios = p_total / (np.arange(1, n_rounds+1, 1) * (p_total) +  n_rounds * (1 - p_total))
else:
	p_list = thresholds
	for char_remov in ['[', ' ', ']']:
		p_list = p_list.replace(char_remov, '')
	p_list = p_list.split(',')
	for ind, thresh in enumerate(p_list):
		p_list[ind] = (float(thresh) / 100)
	p_list = np.array(p_list)
	p_ratios = [p_list[0]]
	for i in range(1, p_list.shape[0]):
		p_ratios.append((p_list[i] - p_list[i-1]) / (1 - p_list[i-1]))
	p_ratios = np.array(p_ratios)
	n_rounds = p_ratios.shape[0]

### Track importance estimates, if desired ###
if(track_importance):
	list_metrics = cal_metrics(net, is_init=True, use_l2=use_l2)

### Store initial parameters for later use, if distance from init is to be used ###
if(use_init):
	params_init = model_params(net)
	params_init = [p.reshape(p.shape[0], -1) for p in params_init]
else:
	params_init = None

### Base model ###
net_p = copy.deepcopy(net)
base_size = cal_size(net, name=model_name).sum()
p_cumulative = 100
optimizer = optim.SGD(net_p.parameters(), lr=lr_prune, momentum=moment, weight_decay=1e-4)
stat = {'warmup': {'train':[], 'test':[]},
		'pruning_and_training': {'train':[], 'test':[]},
		'final_training': {'train':[], 'test':[]}
		}

### Warmup training, if desired ###
if(warmup_epochs > 0):
	print("\n------------------ Warmup training for the base model ------------------\n")
	lr_ind = 0
	best_acc = 0
	print("\n--learning rate is {}".format(lr_prune))
	for epoch in range(warmup_epochs-1): # subtract 1 because pruning+training performs one round of warmup by default
		print('\nEpoch: {}'.format(epoch))
		train(net_p, T=T, track_importance=False, mode='warmup')
		test(net_p, T=T, save=True, warmup=True, mode='warmup')

######### Pruning + Training process begins here #########
for n_iter in range(n_rounds):
	print("\n------------------ Pruning round: {iter} ------------------\n".format(iter=n_iter+1))

	### Train ###
	train(net_p, alpha=alpha, T=T, track_batches=track_batches, track_importance=track_importance, pruning_type=pruning_type, mode='pruning_and_training')
	test(net_p, T=T, save=False, mode='pruning_and_training')

	### Amount to be pruned ###
	prune_iter = p_ratios[n_iter] * 100
	p_cumulative = int(p_cumulative * (1 - p_ratios[n_iter]))

	### Estimate importance ###
	# vgg
	if(model_name=='vgg'):
		imp_order = np.array([[],[],[]]).transpose()
		list_imp = list_metrics if(track_importance) else cal_importance(net_p, pruning_type, trainloader, num_stop=imp_samples, num_of_classes=num_classes, T=T, grasp_T=grasp_T, params_init=params_init if use_init else None)
		for ind, l_index in enumerate([2, 5, 9, 12, 16, 19, 23, 26, 30, 33]):
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([l_index],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)

	# MobileNet-V1
	elif(args.model=='mobilenet'):
		imp_order = np.array([[],[],[]]).transpose()
		list_imp = list_metrics if(track_importance) else cal_importance(net_p, pruning_type, trainloader, num_stop=imp_samples, num_of_classes=num_classes, T=T, grasp_T=grasp_T, params_init=params_init if use_init else None)
		ind = 0
		nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
		imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
		ind+=1
		for l_index in range(13):
			imp_prev = imp_order[imp_order[:,0] == ind-1, 2]
			nlist = [np.linspace(0, imp_prev.shape[0]-1, imp_prev.shape[0]), imp_prev]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(),
											nlist[1].tolist()]).transpose()), 0)
			ind+=1
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
			ind+=1

	# ResNet-56
	elif(args.model=='resnet-56'):
		imp_order = np.array([[],[],[]]).transpose()
		list_imp = list_metrics if(track_importance) else cal_importance(net_p, pruning_type, trainloader, num_stop=imp_samples, num_of_classes=num_classes, T=T, grasp_T=grasp_T, params_init=params_init if use_init else None)
		for ind in range(57):
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
			imp_order[imp_order[:,0] == 0, 2] = 1e7

	# ResNet-34 (Not used in paper)
	elif(args.model=='resnet-34'):
		imp_order = np.array([[],[],[]]).transpose()
		list_imp = list_metrics if(track_importance) else cal_importance(net_p, pruning_type, trainloader, num_stop=imp_samples, num_of_classes=num_classes, T=T, grasp_T=grasp_T, params_init=params_init if use_init else None)
		for ind in range(36):
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
			imp_order[imp_order[:,0] == 0, 2] = 1e7

	# ResNet-18 (Not used in paper)
	elif(args.model=='resnet-18'):
		imp_order = np.array([[],[],[]]).transpose()
		list_imp = list_metrics if(track_importance) else cal_importance(net_p, pruning_type, trainloader, num_stop=imp_samples, num_of_classes=num_classes, T=T, grasp_T=grasp_T, params_init=params_init if use_init else None)
		for ind in range(20):
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
											nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
			imp_order[imp_order[:,0] == 0, 2] = 1e7


	### Prune using estimated importance ###
	# VGG
	if(model_name=='vgg'):
		# Prune network
		imp_order = constrain_ratios(imp_order, vgg_size(net))
		orig_size = vgg_size(net_p)
		order_layerwise, prune_ratio = vgg_order_and_ratios(imp_order, prune_iter / 100)
		net_p, cfg_state = vgg_pruner(net_p, order_layerwise, prune_ratio, orig_size, num_classes=num_classes)	
		# Basic checks
		print("\nPruned: {:.2%}".format(1 - vgg_size(net_p).sum() / base_size))
		print("Pruned Architecture:", orig_size - prune_ratio)
		print("Accuracy without fine-tuning: {:.2%}".format(cal_acc(net_p.eval(), testloader) / 100))

		# Update optimizer (preserve momentum, if desired)
		optimizer = vgg_optimizer(base_optimizer=optimizer, lr=lr_prune, net_p=net_p, imp_order=order_layerwise, prune_ratio=prune_ratio, 
									orig_size=orig_size, preserve_momentum=preserve_moment, moment=moment)

		# Prune the original init model to match dimensions with the current model
		if(use_init):
			net, _ = vgg_pruner(net, order_layerwise, prune_ratio, orig_size, init=True)
			params_init = model_params(net)
			params_init = [p.reshape(p.shape[0], -1) for p in params_init]
		
		# Clean up
		if(track_importance):
			list_metrics = cal_metrics(net_p, is_init=True, use_l2=use_l2)

	# MobileNet-V1
	elif(model_name=='mobilenet'):
		# Prune network
		imp_order = constrain_ratios(imp_order, mobile_size(net))
		orig_size = mobile_size(net_p)
		order_layerwise, prune_ratio = mobile_order_and_ratios(imp_order, prune_iter / 100)
		net_p, cfg_state = mobile_pruner(net_p, order_layerwise, prune_ratio, orig_size, num_classes=num_classes)	
		# Basic checks
		print("\nPruned: {:.2%}".format(1 - mobile_size(net_p).sum() / base_size))
		print("Pruned Architecture:", orig_size - prune_ratio)
		print("Accuracy without fine-tuning: {:.2%}".format(cal_acc(net_p.eval(), testloader) / 100))

		# Update optimizer (preserve momentum, if desired)
		optimizer = mobile_optimizer(base_optimizer=optimizer, lr=lr_prune, net_p=net_p, imp_order=order_layerwise, prune_ratio=prune_ratio, 
									orig_size=orig_size, preserve_momentum=preserve_moment, moment=moment)

		# Prune the original init model to match dimensions with the current model
		if(use_init):
			net, _ = mobile_pruner(net, order_layerwise, prune_ratio, orig_size)
			params_init = model_params(net)
			params_init = [p.reshape(p.shape[0], -1) for p in params_init]
		
		# Clean up
		if(track_importance):
			list_metrics = cal_metrics(net_p, is_init=True, use_l2=use_l2)

	# ResNet-56
	elif(model_name=='resnet-56'):
		# Prune network
		orig_size = res_size_cifar(net_p)
		res_iter = 1 - (p_cumulative / 100) * (base_size.sum() / orig_size.sum())
		order_layerwise, prune_ratio = res_order_and_ratios_cifar(imp_order, res_iter)
		prune_ratio[0] = 0
		for i, n in enumerate(orig_size - np.array(prune_ratio)):
			if(n == 0):
				prune_ratio[i] -= 1 
		net_p, order_zeros = res_pruner_cifar(net_p, order_layerwise, prune_ratio, orig_size, num_classes=int(args.num_classes))
		net_p, cfg_state = skip_or_prune_cifar(net_p, order_zeros, prune_ratio, orig_size)
		cfg_zero = cfg_res_zero_cifar(prune_ratio, orig_size)
		net_zero = torch.nn.DataParallel(ResPruned_cifar(cfg_zero, num_classes=num_classes))

		# Basic checks
		print("\nPruned: {:.2%}".format(1 - res_size_cifar(net_zero).sum() / base_size.sum()))
		print("Pruned Architecture:", orig_size - prune_ratio)
		print("Accuracy without fine-tuning: {:.2%}".format(cal_acc(net_p.eval(), testloader) / 100))

		# Update optimizer (preserve momentum, if desired)
		optimizer = res_cifar_optimizer(base_optimizer=optimizer, lr=lr_prune, net_p=net_p, imp_order=order_layerwise, prune_ratio=prune_ratio, 
									orig_size=orig_size, preserve_momentum=preserve_moment, moment=moment)

		# Prune the original init model to match dimensions with the current model
		if(use_init):
			net, order_zeros = res_pruner_cifar(net, order_layerwise, prune_ratio, orig_size, num_classes=num_classes)
			net, cfg_state = skip_or_prune_cifar(net, order_zeros, prune_ratio, orig_size)
			params_init = model_params(net)
			params_init = [p.reshape(p.shape[0], -1) for p in params_init]
		
		# Clean up
		if(track_importance):
			list_metrics = cal_metrics(net_p, is_init=True, use_l2=use_l2)

	# ResNet-34 (Not used in paper)
	elif(model_name=='resnet-34'):
		# Prune network
		orig_size = res_size(net_p)
		res_iter = 1 - (p_cumulative / 100) * (base_size.sum() / orig_size.sum())
		order_layerwise, prune_ratio = res_order_and_ratios(imp_order, res_iter)
		prune_ratio[0] = 0
		for i, n in enumerate(orig_size - np.array(prune_ratio)):
			if(n == 0):
				prune_ratio[i] -= 1 
		net_p, order_zeros = res_pruner(net_p, order_layerwise, prune_ratio, orig_size, num_classes=num_classes)
		net_p, cfg_state = skip_or_prune(net_p, order_zeros, prune_ratio, orig_size)
		cfg_zero = cfg_res_zero(prune_ratio, orig_size)
		net_zero = torch.nn.DataParallel(ResPruned(cfg_zero, num_classes=num_classes))

		# Basic checks
		print("\nPruned: {:.2%}".format(1 - res_size(net_zero).sum() / base_size.sum()))
		print("Pruned Architecture:", orig_size - prune_ratio)
		print("Accuracy without fine-tuning: {:.2%}".format(cal_acc(net_p.eval(), testloader) / 100))

		# Update optimizer (preserve momentum, if desired)
		optimizer = res_optimizer(base_optimizer=optimizer, lr=lr_prune, net_p=net_p, imp_order=order_layerwise, prune_ratio=prune_ratio, 
									orig_size=orig_size, preserve_momentum=preserve_moment, moment=moment)

		# Prune the original init model to match dimensions with the current model
		if(use_init):
			net, order_zeros = res_pruner(net, order_layerwise, prune_ratio, orig_size, num_classes=num_classes)
			net, cfg_state = skip_or_prune(net, order_zeros, prune_ratio, orig_size)
			params_init = model_params(net)
			params_init = [p.reshape(p.shape[0], -1) for p in params_init]
		
		# Clean up
		if(track_importance):
			list_metrics = cal_metrics(net_p, is_init=True, use_l2=use_l2)

	# ResNet-18 (Not used in paper)
	elif(model_name=='resnet-18'):
		# Prune network
		num_blocks=[2,2,2,2]
		orig_size = res_size(net_p)
		res_iter = 1 - (p_cumulative / 100) * (base_size.sum() / orig_size.sum())
		order_layerwise, prune_ratio = res_order_and_ratios(imp_order, res_iter, arch_type=18)
		prune_ratio[0] = 0
		for i, n in enumerate(orig_size - np.array(prune_ratio)):
			if(n == 0):
				prune_ratio[i] -= 1 
		net_p, order_zeros = res_pruner(net_p, order_layerwise, prune_ratio, orig_size, arch_type=18, num_classes=num_classes)
		net_p, cfg_state = skip_or_prune(net_p, order_zeros, prune_ratio, orig_size, arch_type=18)
		cfg_zero = cfg_res_zero(prune_ratio, orig_size, num_blocks=num_blocks)
		net_zero = torch.nn.DataParallel(ResPruned(cfg_zero, num_blocks=num_blocks, num_classes=num_classes))

		# Basic checks
		print("\nPruned: {:.2%}".format(1 - res_size(net_zero).sum() / base_size.sum()))
		print("Pruned Architecture:", orig_size - prune_ratio)
		print("Accuracy without fine-tuning: {:.2%}".format(cal_acc(net_p.eval(), testloader) / 100))

		# Update optimizer (preserve momentum, if desired)
		optimizer = res_optimizer(base_optimizer=optimizer, lr=lr_prune, net_p=net_p, imp_order=order_layerwise, prune_ratio=prune_ratio, 
									orig_size=orig_size, preserve_momentum=preserve_moment, moment=moment, arch_type=18)

		# Prune the original init model to match dimensions with the current model
		if(use_init):
			net, order_zeros = res_pruner(net, order_layerwise, prune_ratio, orig_size, arch_type=18, num_classes=num_classes)
			net, cfg_state = skip_or_prune(net, order_zeros, prune_ratio, orig_size, arch_type=18)
			params_init = model_params(net)
			params_init = [p.reshape(p.shape[0], -1) for p in params_init]
		
		# Clean up
		if(track_importance):
			list_metrics = cal_metrics(net_p, is_init=True, use_l2=use_l2)

######### Process final pruned model #########
### FLOPs ###
with torch.cuda.device(0):
	flops, params = get_model_complexity_info(net_p, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
	print('{:<30}  {:<8}'.format('FLOPs in pruned model: ', flops))

### Final training ###
print("\n------------------ Training the final pruned model ------------------\n")
lr_ind = 0
best_acc = 0
epoch = 0
pruned_epochs[0] -= (n_rounds + warmup_epochs)
while(lr_ind < len(pruned_sched)):
	print("\n--Training at {lr} learning rate for {n} epochs".format(lr=pruned_sched[lr_ind], n=pruned_epochs[lr_ind]))
	optimizer.param_groups[0]['lr'] = pruned_sched[lr_ind]
	for n in range(pruned_epochs[lr_ind]):
		print('\nEpoch: {}'.format(epoch))
		train(net_p, T=T, track_importance=False, mode='final_training')
		test(net_p, T=T, save=True, mode='final_training')
		epoch += 1
	lr_ind += 1		
print("Accuracy of pruned model (best checkpoint): {:.2%}".format(best_acc / 100))

### Save train/test stats ###
if(track_stats):
	stats_loc = './stats/pruned_models/'
	stats_loc += (model_name + '_')
	stats_loc += 'seed_' + args.seed + '_'
	if use_pretrained:
		stats_loc += 'pretrained_' 
	stats_loc += args.pruning_type + '_'
	stats_loc += 'pruned_' + args.prune_percent + '_'
	stats_loc += 'rounds_' + str(n_rounds) + '_'
	stats_loc += 'temp_' + str(int(T))
	stats_loc += '.pkl'

	with open(stats_loc, 'wb') as f:
		pkl.dump(stat, f)
