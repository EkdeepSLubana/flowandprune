import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Some helper functions ###
# Model parameters
def model_params(model):
	params = []
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		params.append(param)
	return params

# Parameter-wise gradients as a list 
def model_grads(model):
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		grads.append(0. if param.grad is None else param.grad + 0.)
	return grads

# Randomly sampled data for hessian-gradient product based measures
def grasp_data(dataloader, n_classes, n_samples):
	datas = [[] for _ in range(n_classes)]
	labels = [[] for _ in range(n_classes)]
	mark = dict()
	dataloader_iter = iter(dataloader)
	while True:
		inputs, targets = next(dataloader_iter)
		for idx in range(inputs.shape[0]):
			x, y = inputs[idx:idx+1], targets[idx:idx+1]
			category = y.item()
			if len(datas[category]) == n_samples:
				mark[category] = True
				continue
			datas[category].append(x)
			labels[category].append(y)
		if len(mark) == n_classes:
			break

	X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
	return X, y

# Gradient calculator 
def cal_grad(net, trainloader, num_stop=5000, T=1):
	net.eval()
	num_data = 0  # count the number of datum points in the dataloader
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for inputs, targets in trainloader:
		if(num_data >= num_stop):
			break
		net.zero_grad()
		tmp_num_data = inputs.size(0)
		outputs = net(inputs.to(device)) / T
		loss = criterion(outputs, targets.to(device))
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.detach().clone() * float(tmp_num_data) for gbase1, g1 in zip(gbase, gradsH) ]
		num_data += float(tmp_num_data)

	gbase = [gbase1 / num_data for gbase1 in gbase]

	return gbase

# Hessian-gradient product calculator 
def cal_hg(net, trainloader, n_samples=100, n_classes=100, T=200):
	net.eval()
	d_in, d_out = grasp_data(trainloader, n_classes, n_samples)
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	hgbase = [torch.zeros(p.size()).to(device) for p in base_params]  
	### gbase
	tot_samples = 0
	for num_class in tqdm(range(n_classes)):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs) / T
		loss = criterion(outputs, targets)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.detach().clone() for gbase1, g1 in zip(gbase, gradsH) ]
	gbase = [gbase1 / n_classes for gbase1 in gbase]

	### Hg
	for num_class in tqdm(range(n_classes)):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs) / T 
		loss = criterion(outputs, targets)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=True)
		gnorm = 0
		for i in range(len(gbase)):
			gnorm += (gbase[i] * gradsH[i]).sum()
		gnorm.backward()
		Hg = model_grads(net)
		### update
		hgbase = [hgbase1 + hg1.detach().clone() for hgbase1, hg1 in zip(hgbase, Hg)]

	hgbase = [hgbase1.reshape(hgbase1.shape[0], -1) / n_classes for hgbase1 in hgbase]
	return hgbase

### Importance measures used in the paper ###
# Magnitude-based importance (L2 norm)
def cal_importance_mag(net):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [l_params[3*ind].norm(2,1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Loss-preservation based importance
def cal_importance_loss_based(net, trainloader, num_stop=5000, T=1):
	gvec = model_grads(net) if(trainloader == None) else cal_grad(net, trainloader, num_stop=num_stop, T=T)
	gvec = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind] * gvec[3*ind]).abs().sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Loss-preservation measure biased towards removing small magnitude parameters
# Similar to SNIP applied to filters (Refer Lee et al., (ICLR, 2019)) 
def cal_importance_biased_loss(net, trainloader, num_stop=5000, T=1, params_init=None):
	gvec = model_grads(net) if(trainloader == None) else cal_grad(net, trainloader, num_stop=num_stop, T=T)
	gvec = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(params_init == None):
		list_imp = [(l_params[3*ind].pow(2) * gvec[3*ind]).abs().sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	else:
		list_imp = [((l_params[3*ind] - params_init[3*ind]).abs() * (l_params[3*ind] * gvec[3*ind]).abs()).sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Increase in Gradient-norm based importance (equal to GraSP when T=200)
def cal_importance_grasp(net, trainloader, n_samples=100, n_classes=100, T=200):
	hgvec = cal_hg(net, trainloader, n_samples=n_samples, n_classes=n_classes, T=T)
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind] * hgvec[3*ind]).sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Gradient-norm preservation based importance 
def cal_importance_grad_preserve(net, trainloader, n_samples=100, n_classes=100, T=1):
	hgvec = cal_hg(net, trainloader, n_samples=n_samples, n_classes=n_classes, T=T)
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind] * hgvec[3*ind]).sum(dim=1).abs().detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### Other loss-preservation based measures (included for experimental purposes; not used in paper) ###
# Tracked version for biased loss-preservation measure; Returns |thetag| as a parameter-wise list to the caller.
def cal_importance_biased_loss_tracked(net, trainloader, num_stop=5000, T=1):
	gvec = model_grads(net) if(trainloader == None) else cal_grad(net, trainloader, num_stop=num_stop, T=T)
	gvec = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind] * gvec[3*ind]).abs().detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Direct TFO applied to filters (Refer Molchanov et al. (ICLR, 2017)) 
def cal_importance_tfo(net, trainloader, num_stop=5000, T=1):
	gvec = model_grads(net) if(trainloader == None) else cal_grad(net, trainloader, num_stop=num_stop, T=T)
	gvec = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind] * gvec[3*ind]).sum(dim=1).abs().detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# Gradient calculator for Fisher importance
def cal_grad_fisher(net, trainloader, num_stop=5000, T=1):
	net.eval()
	num_data = 0  # count the number of datum points in the dataloader
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for inputs, targets in trainloader:
		if(num_data >= num_stop):
			break
		net.zero_grad()
		tmp_num_data = inputs.size(0)
		outputs = net(inputs.to(device)) / T
		loss = criterion(outputs, targets.to(device))
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update (gradient is squared for fisher importance)
		gbase = [ gbase1 + g1.pow(2).detach().clone() * float(tmp_num_data) for gbase1, g1 in zip(gbase, gradsH) ]
		num_data += float(tmp_num_data)

	gbase = [gbase1 / num_data for gbase1 in gbase]

	return gbase

# Fisher importance (Squared TFO; Refer Theis et al. (ECCV, 2018)) 
def cal_importance_fisher(net, trainloader, num_stop=5000, T=1):
	gvec_squared = model_grads(net) if(trainloader == None) else cal_grad_fisher(net, trainloader, num_stop=num_stop, T=T)
	gvec_squared = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec_squared]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind].pow(2) * gvec_squared[3*ind]).sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# NVIDIA's Fisher importance implementation (Squared TFO applied to BN; Refer Molchanov et al. (CVPR, 2019)) 
def cal_importance_nvidia_fisher(net, trainloader, num_stop=5000, T=1):
	gvec_squared = model_grads(net) if(trainloader == None) else cal_grad_fisher(net, trainloader, num_stop=num_stop, T=T)
	gvec_squared = [gvec1.reshape(gvec1.shape[0], -1) for gvec1 in gvec_squared]
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [(l_params[3*ind+1].pow(2) * gvec_squared[3*ind+1] + l_params[3*ind+2].pow(2) * gvec_squared[3*ind+2]).sum(dim=1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### Other magnitude-based measures (included for completeness sake; not used in paper) ###
# L1-norm based importance (Refer Li et al. (ICLR, 2017)) 
def cal_importance_l1(net):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [l_params[3*ind].norm(1,1).detach().clone() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

# BN-scale based importance (Refer NetSlim (ICCV, 2017)) 
def cal_importance_bn(net):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	list_imp = [l_params[3*ind+1].abs().detach().clone().squeeze() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### General importance estimation function ###
def cal_importance(net, pruning_type, trainloader, num_stop=5000, num_of_classes=100, T=1, grasp_T=200, params_init=None):
	### Measures used in paper
	# Magnitude-based measure
	if(pruning_type=='mag_based'):
		nlist = cal_importance_mag(net)
	# Loss-preservation based measures
	elif(pruning_type=='loss_based'):
		nlist = cal_importance_loss_based(net, trainloader, num_stop=num_stop, T=T)
	# Loss-preservation measure biased towards small magnitude parameters
	elif(pruning_type=='biased_loss'):
		nlist = cal_importance_biased_loss(net, trainloader, num_stop=num_stop, T=T, params_init=params_init)
	# GraSP: Increase in gradient norm based measure
	elif(pruning_type=='grasp'):
		nlist = cal_importance_grasp(net, trainloader, n_samples=int(num_stop/num_of_classes), n_classes=num_of_classes, T=grasp_T)
	# Gradient-norm preservation biased measure
	elif(pruning_type=='grad_preserve'):
		nlist = cal_importance_grad_preserve(net, trainloader, n_samples=int(num_stop/num_of_classes), n_classes=num_of_classes, T=grasp_T)
	### Extra loss-based measures (Not used in the paper)
	elif(pruning_type=='biased_loss_tracked'):
		nlist = cal_importance_biased_loss_tracked(net, trainloader, num_stop=num_stop, T=T)
	elif(pruning_type=='tfo'):
		nlist = cal_importance_tfo(net, trainloader, num_stop=num_stop, T=T)
	elif(pruning_type=='fisher'):
		nlist = cal_importance_fisher(net, trainloader, num_stop=num_stop, T=T)
	elif(pruning_type=='nvidia'):
		nlist = cal_importance_nvidia_fisher(net, trainloader, num_stop=num_stop, T=T)
	### Extra magnitude-based measures (Not used in the paper)
	elif(pruning_type=='l1'):
		nlist = cal_importance_l1(net)
	elif(pruning_type=='bn'):
		nlist = cal_importance_bn(net)
	return nlist
