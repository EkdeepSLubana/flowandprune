import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from models import *

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.deterministic = True
cudnn.benchmark = False

######### Constrain pruning to 95% at max #########
def constrain_ratios(imp_order, osize):
    lids = np.unique(imp_order[:,0])
    constraints = (0.05 * osize).astype(int) + 1
    for lid_ind, lid in enumerate(lids):
        inds = np.where(imp_order[:,0] == lid)
        imp_order[inds[0][-constraints[lid_ind]:], 2] = 1e7
    return imp_order

######### Pruned network constructor #########
# VGG 
class VGG_p(nn.Module):
    def __init__(self, cfg, num_classes=100):
        super(VGG_p, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-2], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(x)]
                in_channels = x
        return nn.Sequential(*layers)

# MobileNet
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet_p(nn.Module):
    def __init__(self, in_first, cfg, num_classes=100):
        super(MobileNet_p, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, in_first, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_first)
        self.layers = self._make_layers(in_planes=in_first)
        self.linear = nn.Linear(cfg[-1], num_classes)
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet
class MapLayer(torch.nn.Module):
    def __init__(self, out_planes):
        super(MapLayer, self).__init__()
        self.map = torch.ones(1, out_planes, 1, 1).to(device)    
    def forward(self, x):
        return self.map * x

class BasicBlock_p(nn.Module):
    expansion = 1
    def __init__(self, lnum, in_planes, mid_planes, out_planes, stride=1, n_shortcuts=0):
        super(BasicBlock_p, self).__init__()
        if stride != 1:
            self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_planes)
            self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.map = MapLayer(out_planes)   
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, n_shortcuts, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_shortcuts),
                MapLayer(n_shortcuts)
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_planes)
            self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.map = MapLayer(out_planes)
            self.shortcut = nn.Sequential()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.map(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet
class ResNet_p(nn.Module):
    def __init__(self, cfg, num_blocks, num_classes=100):
        super(ResNet_p, self).__init__()
        self.in_planes = cfg['base']
        self.conv1 = nn.Conv2d(3, cfg['base'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg['base'])
        self.layer1 = self._make_layer(BasicBlock_p, cfg['l1'], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock_p, cfg['l2'], num_blocks[1], stride=2, n_shortcuts=cfg['l2s'])
        self.layer3 = self._make_layer(BasicBlock_p, cfg['l3'], num_blocks[2], stride=2, n_shortcuts=cfg['l3s'])
        self.layer4 = self._make_layer(BasicBlock_p, cfg['l4'], num_blocks[3], stride=2, n_shortcuts=cfg['l4s'])
        self.linear = nn.Linear(cfg['l4'][-1] * BasicBlock_p.expansion, num_classes)
    def _make_layer(self, block, l_cfg, num_blocks, stride, n_shortcuts=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for lnum, stride in enumerate(strides):
            layers.append(block(lnum, self.in_planes, l_cfg[2*lnum], l_cfg[2*lnum+1], stride, n_shortcuts))
            self.in_planes = l_cfg[2*lnum+1]
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResPruned(layer_cfgs, num_blocks=[3,4,6,3], num_classes=100):
    return ResNet_p(layer_cfgs, num_blocks=num_blocks, num_classes=num_classes)

# ResNet-56
class ResNet_cifar_p(nn.Module):
    def __init__(self, cfg, num_blocks, num_classes=100):
        super(ResNet_cifar_p, self).__init__()
        self.in_planes = cfg['base']
        self.conv1 = nn.Conv2d(3, cfg['base'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg['base'])
        self.layer1 = self._make_layer(BasicBlock_p, cfg['l1'], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock_p, cfg['l2'], num_blocks[1], stride=2, n_shortcuts=cfg['l2s'])
        self.layer3 = self._make_layer(BasicBlock_p, cfg['l3'], num_blocks[2], stride=2, n_shortcuts=cfg['l3s'])
        self.linear = nn.Linear(cfg['l3'][-1] * BasicBlock_p.expansion, num_classes)
    def _make_layer(self, block, l_cfg, num_blocks, stride, n_shortcuts=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for lnum, stride in enumerate(strides):
            layers.append(block(lnum, self.in_planes, l_cfg[2*lnum], l_cfg[2*lnum+1], stride, n_shortcuts))
            self.in_planes = l_cfg[2*lnum+1]
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResPruned_cifar(layer_cfgs, num_blocks=[9,9,9], num_classes=100):
    return ResNet_cifar_p(layer_cfgs, num_blocks=num_blocks, num_classes=num_classes)

######### Sort importance and differentiate into layers #########
# VGG
def vgg_order_and_ratios(imp_order, prune_ratio):
    imp_sort = np.argsort(imp_order[:,2])
    temp_order = imp_order[imp_sort]
    n_prune = int(prune_ratio * imp_order.shape[0])
    prune_list = temp_order[0:n_prune]
    imp_order_layer = {}
    ratios = []
    for l_index in [2, 5, 9, 12, 16, 19, 23, 26, 30, 33]:
        nlist = temp_order[(temp_order[:,0] == l_index), 1].astype(int)
        imp_order_layer.update({l_index: nlist})
        nlist = np.sort(prune_list[(prune_list[:,0] == l_index), 1].astype(int))
        ratios.append(nlist.shape[0])
    return imp_order_layer, np.array(ratios)

# MobileNet
def mobile_order_and_ratios(imp_order, prune_ratio):
    imp_sort = np.argsort(imp_order[:,2])
    temp_order = imp_order[imp_sort]
    n_prune = int(prune_ratio * imp_order.shape[0])
    prune_list = temp_order[0:n_prune]
    imp_order_layer = {}
    ratios = []
    for l_index in range(27):
        nlist = temp_order[(temp_order[:,0] == l_index), 1].astype(int)
        imp_order_layer.update({l_index: nlist})
        nlist = np.sort(prune_list[(prune_list[:,0] == l_index), 1].astype(int))
        ratios.append(nlist.shape[0])
    return imp_order_layer, np.array(ratios)

def res_order_and_ratios(imp_order, prune_ratio, arch_type=34):
    imp_sort = np.argsort(imp_order[:,2])
    temp_order = imp_order[imp_sort]
    n_prune = int(prune_ratio * imp_order.shape[0])
    prune_list = temp_order[0:n_prune]
    imp_order_layer = {}
    ratios = []
    if(arch_type==34):
        for l_index in range(36):
            nlist = temp_order[(temp_order[:,0] == l_index), 1].astype(int)
            imp_order_layer.update({l_index: nlist})
            nlist = np.sort(prune_list[(prune_list[:,0] == l_index), 1].astype(int))
            ratios.append(nlist.shape[0])
    if(arch_type==18):
        for l_index in range(20):
            nlist = temp_order[(temp_order[:,0] == l_index), 1].astype(int)
            imp_order_layer.update({l_index: nlist})
            nlist = np.sort(prune_list[(prune_list[:,0] == l_index), 1].astype(int))
            ratios.append(nlist.shape[0])
    return imp_order_layer, np.array(ratios)

def res_order_and_ratios_cifar(imp_order, prune_ratio):
    imp_sort = np.argsort(imp_order[:,2])
    temp_order = imp_order[imp_sort]
    n_prune = int(prune_ratio * imp_order.shape[0])
    prune_list = temp_order[0:n_prune]
    imp_order_layer = {}
    ratios = []
    for l_index in range(57):
        nlist = temp_order[(temp_order[:,0] == l_index), 1].astype(int)
        imp_order_layer.update({l_index: nlist})
        nlist = np.sort(prune_list[(prune_list[:,0] == l_index), 1].astype(int))
        ratios.append(nlist.shape[0])
    return imp_order_layer, np.array(ratios)

######### Calculate network size #########
def cal_size(net, name):
    if(name == 'vgg'):
        return vgg_size(net)
    elif(name == 'mobilenet'):
        return mobile_size(net)
    elif(name == 'resnet-34'):
        return res_size(net)
    elif(name == 'resnet-18'):
        return res_size(net)
    elif(name == 'resnet-56'):
        return res_size_cifar(net)

# VGG
def vgg_size(net):
    orig_size = []
    for i in [2, 5, 9, 12, 16, 19, 23, 26, 30, 33]:
        orig_size.append(net.module.features[i].bias.shape[0])
    orig_size = np.array(orig_size)
    return orig_size

# MobileNet
def mobile_size(net):
    orig_size = [net.module.bn1.bias.shape[0]]
    for l_index in range(13):
        orig_size.append(net.module.layers[l_index].bn1.bias.shape[0])
        orig_size.append(net.module.layers[l_index].bn2.bias.shape[0])
    orig_size = np.array(orig_size) 
    return orig_size

def res_size(net):
    orig_size = []
    for l_index in net.module.modules():
        if(isinstance(l_index, nn.BatchNorm2d)):
            orig_size.append(l_index.bias.shape[0])
    orig_size = np.array(orig_size)
    return orig_size

def res_size_cifar(net):
    orig_size = []
    for l_index in net.module.modules():
        if(isinstance(l_index, nn.BatchNorm2d)):
            orig_size.append(l_index.bias.shape[0])
    orig_size = np.array(orig_size)
    return orig_size

######### Layerwise configuration of pruned network #########
# VGG
def cfg_vgg(prune_ratio, orig_size):
    cfg_list = []
    for i in range(4):
        cfg_list.append(orig_size[2*i] - prune_ratio[2*i])
        cfg_list.append(orig_size[2*i+1] - prune_ratio[2*i+1])
        cfg_list.append('M')
    cfg_list.append(orig_size[8] - prune_ratio[8])
    cfg_list.append(orig_size[9] - prune_ratio[9])
    cfg_list.append('M')
    return cfg_list

# MobileNet
def cfg_mobile(prune_ratio, orig_size):
    cfg_list = []
    for i in range(0,27,2):
        cfg_list.append(int(orig_size[i] - prune_ratio[i])) 
    for i in [2, 4, 6, 12]:
        cfg_list[i] = (cfg_list[i], 2)
    return cfg_list

# ResNet
def cfg_res(prune_ratio, orig_size, num_blocks=[3,4,6,3]):
    cfg_list = {}
    cfg_list.update({'base': orig_size[0] - prune_ratio[0]})
    b_id = 0    
    ### First layer has only identity shortcuts ###
    l_list = []
    l_id = 1
    for b_id in range(1,2*num_blocks[0]+1):
        if(b_id % 2 == 1):
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
        else:
            l_list.append(orig_size[b_id])
    cfg_list.update({'l'+str(l_id): l_list.copy()})
    for l_id in range(2,len(num_blocks)+1):
        ### First block has a learned shortcut ###
        l_list = []
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        l_list.append(orig_size[b_id])
        b_id += 1
        cfg_list.update({'l'+str(l_id)+'s': orig_size[b_id]})    
        ### Rest blocks have identity shortcuts ###
        for block_id in range(num_blocks[l_id-1]-1):
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
            b_id += 1
            l_list.append(orig_size[b_id])  
        cfg_list.update({'l'+str(l_id): l_list.copy()})
    return cfg_list

def cfg_res_zero(prune_ratio, orig_size, num_blocks=[3,4,6,3]):
    cfg_list = {}
    cfg_list.update({'base': orig_size[0] - prune_ratio[0]})
    b_id = 0    
    ### First layer has only identity shortcuts ###
    l_list = []
    l_id = 1
    for b_id in range(1,2*num_blocks[0]+1):
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
    cfg_list.update({'l'+str(l_id): l_list.copy()})   
    for l_id in range(2,len(num_blocks)+1):
        ### First block has a learned shortcut ###
        l_list = []
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        cfg_list.update({'l'+str(l_id)+'s': orig_size[b_id] - prune_ratio[b_id]})    
        ### Rest blocks have identity shortcuts ###
        for block_id in range(num_blocks[l_id-1]-1):
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])  
        cfg_list.update({'l'+str(l_id): l_list.copy()})
    return cfg_list

# ResNet
def cfg_res_cifar(prune_ratio, orig_size, num_blocks=[9,9,9]):
    cfg_list = {}
    cfg_list.update({'base': orig_size[0] - prune_ratio[0]})
    b_id = 0    
    ### First layer has only identity shortcuts ###
    l_list = []
    l_id = 1
    for b_id in range(1,2*num_blocks[0]+1):
        if(b_id % 2 == 1):
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
        else:
            l_list.append(orig_size[b_id])
    cfg_list.update({'l'+str(l_id): l_list.copy()})
    for l_id in range(2,len(num_blocks)+1):
        ### First block has a learned shortcut ###
        l_list = []
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        l_list.append(orig_size[b_id])
        b_id += 1
        cfg_list.update({'l'+str(l_id)+'s': orig_size[b_id]})    
        ### Rest blocks have identity shortcuts ###
        for block_id in range(num_blocks[l_id-1]-1):
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
            b_id += 1
            l_list.append(orig_size[b_id])  
        cfg_list.update({'l'+str(l_id): l_list.copy()})
    return cfg_list

def cfg_res_zero_cifar(prune_ratio, orig_size, num_blocks=[9,9,9]):
    cfg_list = {}
    cfg_list.update({'base': orig_size[0] - prune_ratio[0]})
    b_id = 0    
    ### First layer has only identity shortcuts ###
    l_list = []
    l_id = 1
    for b_id in range(1,2*num_blocks[0]+1):
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
    cfg_list.update({'l'+str(l_id): l_list.copy()})   
    for l_id in range(2,len(num_blocks)+1):
        ### First block has a learned shortcut ###
        l_list = []
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        l_list.append(orig_size[b_id] - prune_ratio[b_id])
        b_id += 1
        cfg_list.update({'l'+str(l_id)+'s': orig_size[b_id] - prune_ratio[b_id]})    
        ### Rest blocks have identity shortcuts ###
        for block_id in range(num_blocks[l_id-1]-1):
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])
            b_id += 1
            l_list.append(orig_size[b_id] - prune_ratio[b_id])  
        cfg_list.update({'l'+str(l_id): l_list.copy()})
    return cfg_list

######### Pruning engine #########
# VGG
def vgg_pruner(net, imp_order, prune_ratio, orig_size, init=False, num_classes=100):
    cfg = cfg_vgg(prune_ratio, orig_size)    
    net_pruned = torch.nn.DataParallel(VGG_p(cfg=cfg, num_classes=num_classes).to(device))
    bn = [2, 5, 9, 12, 16, 19, 23, 26, 30, 33]  
    for l in range(len(bn)):
        if(l == 0):
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[bn[l]][n_c:])
            net_pruned.module.features[bn[l]-2].weight.data = net.module.features[bn[l]-2].weight[order_c].data.detach().clone()
            net_pruned.module.features[bn[l]].weight.data = net.module.features[bn[l]].weight[order_c].data.detach().clone()
            net_pruned.module.features[bn[l]].bias.data = net.module.features[bn[l]].bias[order_c].data.detach().clone()
            net_pruned.module.features[bn[l]].running_var.data = net.module.features[bn[l]].running_var[order_c].detach().clone()
            net_pruned.module.features[bn[l]].running_mean.data = net.module.features[bn[l]].running_mean[order_c].detach().clone() 
            continue
        n_p = prune_ratio[l-1]  
        n_c = prune_ratio[l]
        order_p = np.sort(imp_order[bn[l-1]][n_p:])
        order_c = np.sort(imp_order[bn[l]][n_c:])
        net_pruned.module.features[bn[l]-2].weight.data = net.module.features[bn[l]-2].weight[order_c][:,order_p].detach().clone()
        net_pruned.module.features[bn[l]].weight.data = net.module.features[bn[l]].weight[order_c].detach().clone()
        net_pruned.module.features[bn[l]].bias.data = net.module.features[bn[l]].bias[order_c].detach().clone() 
        net_pruned.module.features[bn[l]].running_var.data = net.module.features[bn[l]].running_var[order_c].detach().clone()
        net_pruned.module.features[bn[l]].running_mean.data = net.module.features[bn[l]].running_mean[order_c].detach().clone()  
    n_33 = prune_ratio[-1]
    order_33 = np.sort(imp_order[33][n_33:])
    net_pruned.module.classifier.weight.data = net.module.classifier.weight[:,order_33].detach().clone()
    net_pruned.module.classifier.bias.data = net.module.classifier.bias.detach().clone()    
    return net_pruned, cfg

def vgg_optimizer(base_optimizer, lr=None, net_p=None, imp_order=None, prune_ratio=None, orig_size=None, wd=1e-4, moment=0.9, preserve_momentum=True):
    new_optimizer = optim.SGD(net_p.parameters(), lr=lr, momentum=moment, weight_decay=wd)
    if(preserve_momentum==False):
        return new_optimizer            
    bn = [2, 5, 9, 12, 16, 19, 23, 26, 30, 33]
    for l in range(len(bn)):
        if(l == 0):
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[bn[l]][n_c:])
            for ind in range(3):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer'] = momentum_base[order_c].data.clone()
            continue        
        n_p = prune_ratio[l-1]  
        n_c = prune_ratio[l]
        order_p = np.sort(imp_order[bn[l-1]][n_p:])
        order_c = np.sort(imp_order[bn[l]][n_c:])
        for ind in range(3):
            if(ind==0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*l+ind]]['momentum_buffer'] = momentum_base[order_c].data.clone()
    n_33 = prune_ratio[-1]
    order_33 = np.sort(imp_order[33][n_33:])
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*(l+1)]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][3*(l+1)]]['momentum_buffer'] = momentum_base[:,order_33].data.clone()
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*(l+1)+1]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][3*(l+1)+1]]['momentum_buffer'] = momentum_base.data.clone()
    return new_optimizer

# MobileNet
def mobile_pruner(net, imp_order, prune_ratio, orig_size, num_classes=100):  
    cfg = cfg_mobile(prune_ratio, orig_size)    
    net_pruned = torch.nn.DataParallel(MobileNet_p(in_first=cfg[0], cfg=cfg[1:], num_classes=num_classes))
    n_c = 0
    order_c = []
    order_p = []
    for l in range(0, 27, 2):
        if(l == 0):
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[l][n_c:])
            net_pruned.module.conv1.weight.data = net.module.conv1.weight[order_c].data.detach().clone()            
            net_pruned.module.bn1.weight.data = net.module.bn1.weight[order_c].data.detach().clone()
            net_pruned.module.bn1.bias.data = net.module.bn1.bias[order_c].data.detach().clone()
            net_pruned.module.bn1.running_var.data = net.module.bn1.running_var[order_c].data.detach().clone()
            net_pruned.module.bn1.running_mean.data = net.module.bn1.running_mean[order_c].data.detach().clone()
            continue
        else:
            ind = int((l+1)/2) - 1
            net_pruned.module.layers[ind].conv1.weight.data = net.module.layers[ind].conv1.weight[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn1.weight.data = net.module.layers[ind].bn1.weight[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn1.bias.data = net.module.layers[ind].bn1.bias[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn1.running_var.data = net.module.layers[ind].bn1.running_var[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn1.running_mean.data = net.module.layers[ind].bn1.running_mean[order_c].data.detach().clone()
            order_p = order_c.copy()
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[l][n_c:])
            net_pruned.module.layers[ind].conv2.weight.data = net.module.layers[ind].conv2.weight[order_c][:,order_p].data.detach().clone()
            net_pruned.module.layers[ind].bn2.weight.data = net.module.layers[ind].bn2.weight[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn2.bias.data = net.module.layers[ind].bn2.bias[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn2.running_var.data = net.module.layers[ind].bn2.running_var[order_c].data.detach().clone()
            net_pruned.module.layers[ind].bn2.running_mean.data = net.module.layers[ind].bn2.running_mean[order_c].data.detach().clone()
    n_linear = prune_ratio[-1]
    order_linear = np.sort(imp_order[26][n_linear:])
    net_pruned.module.linear.weight.data = net.module.linear.weight[:,order_linear].detach().clone()
    net_pruned.module.linear.bias.data = net.module.linear.bias.detach().clone()    
    return net_pruned, cfg

# MobileNet
def mobile_optimizer(base_optimizer, lr=None, net_p=None, imp_order=None, prune_ratio=None, orig_size=None, wd=1e-4, moment=0.9, preserve_momentum=True):
    new_optimizer = optim.SGD(net_p.parameters(), lr=lr, momentum=moment, weight_decay=wd)
    if(preserve_momentum==False):
        return new_optimizer
    cfg = cfg_mobile(prune_ratio, orig_size)    
    n_c = 0
    order_c = []
    order_p = []

    for l in range(0, 27, 2):
        ind = int((l+1)/2) - 1
        if(l == 0):
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[l][n_c:])
            for lnum in range(3):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*ind+lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*ind+lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
            continue
        else:
        # Depth-wise layer
            for lnum in range(3):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*(l-1)+lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*(l-1)+lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
            order_p = order_c.copy()
            n_c = prune_ratio[l]
            order_c = np.sort(imp_order[l][n_c:])
        # Point-wise layer
        for lnum in range(3):
            if(lnum==0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*l+lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*l+lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][3*l+lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][3*l+lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
    # Linear layer
    n_linear = prune_ratio[-1]
    order_linear = np.sort(imp_order[26][n_linear:])
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-2]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-2]]['momentum_buffer'] = momentum_base[:,order_linear].data.clone()
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-1]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-1]]['momentum_buffer'] = momentum_base.data.clone()
    return new_optimizer

# ResNet pruner
def res_pruner(net, imp_order, prune_ratio, orig_size, arch_type=34, num_classes=100): 
    if(arch_type==34):
        num_blocks=[3,4,6,3]
    elif(arch_type==18):
        num_blocks=[2,2,2,2]

    cfg = cfg_res(prune_ratio, orig_size, num_blocks=num_blocks)
    cfg_zero = cfg_res_zero(prune_ratio, orig_size, num_blocks=num_blocks)
    net_pruned = torch.nn.DataParallel(ResPruned(cfg, num_blocks=num_blocks, num_classes=num_classes))

    block_id = 0
    order_zeros = {}
    # base
    n_c = orig_size[block_id] - cfg['base']
    order_c = np.sort(imp_order[block_id][n_c:])
    net_pruned.module.conv1.weight.data = net.module.conv1.weight[order_c].data.detach().clone()
    
    net_pruned.module.bn1.weight.data = net.module.bn1.weight[order_c].data.detach().clone()
    net_pruned.module.bn1.bias.data = net.module.bn1.bias[order_c].data.detach().clone()
    net_pruned.module.bn1.running_var.data = net.module.bn1.running_var[order_c].data.detach().clone()
    net_pruned.module.bn1.running_mean.data = net.module.bn1.running_mean[order_c].data.detach().clone()
    order_p = order_c.copy()
    block_id += 1
        
    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        n_c = orig_size[block_id] - cfg['l1'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        net_pruned.module.layer1[block_num].conv1.weight.data = net.module.layer1[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()

        net_pruned.module.layer1[block_num].bn1.weight.data = net.module.layer1[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.bias.data = net.module.layer1[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.running_var.data = net.module.layer1[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.running_mean.data = net.module.layer1[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l1'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        ### Conv
        net_pruned.module.layer1[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer1[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer1[block_num].conv2.weight.data[order_c] = net.module.layer1[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer1[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer1[block_num].bn2.weight.data[order_c] = net.module.layer1[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer1[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer1[block_num].bn2.bias.data[order_c] = net.module.layer1[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer1[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer1[block_num].bn2.running_var.data[order_c] = net.module.layer1[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer1[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer1[block_num].bn2.running_mean.data[order_c] = net.module.layer1[block_num].bn2.running_mean[order_c].data.detach().clone()

        ### These weights are permanently set to zero ###
        net_pruned.module.layer1[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer1[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l1_'+str(block_num): order_c_zero})
        
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        n_c = orig_size[block_id] - cfg['l2'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        ### Conv
        net_pruned.module.layer2[block_num].conv1.weight.data = net.module.layer2[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()
        ### BN
        net_pruned.module.layer2[block_num].bn1.weight.data = net.module.layer2[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.bias.data = net.module.layer2[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.running_var.data = net.module.layer2[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.running_mean.data = net.module.layer2[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l2'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are recalled ###
        ### Conv
        net_pruned.module.layer2[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer2[block_num].conv2.weight.data[order_c] = net.module.layer2[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer2[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer2[block_num].bn2.weight.data[order_c] = net.module.layer2[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer2[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer2[block_num].bn2.bias.data[order_c] = net.module.layer2[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer2[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer2[block_num].bn2.running_var.data[order_c] = net.module.layer2[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer2[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer2[block_num].bn2.running_mean.data[order_c] = net.module.layer2[block_num].bn2.running_mean[order_c].data.detach().clone()
        
        ### These weights are permanently set to zero ###
        net_pruned.module.layer2[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer2[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l2_'+str(block_num): order_c_zero})
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l2s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            ### These weights are recalled ###
            ### Conv
            net_pruned.module.layer2[block_num].shortcut[0].weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[0].weight.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[0].weight.data[order_c] = net.module.layer2[block_num].shortcut[0].weight[order_c][:,order_p0].data.detach().clone()
            ### BN2 weight
            net_pruned.module.layer2[block_num].shortcut[1].weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].weight.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].weight.data[order_c] = net.module.layer2[block_num].shortcut[1].weight[order_c].data.detach().clone()
            ### BN2 bias
            net_pruned.module.layer2[block_num].shortcut[1].bias.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].bias.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].bias.data[order_c] = net.module.layer2[block_num].shortcut[1].bias[order_c].data.detach().clone()
            ### BN2 running_var
            net_pruned.module.layer2[block_num].shortcut[1].running_var.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].running_var.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].running_var.data[order_c] = net.module.layer2[block_num].shortcut[1].running_var[order_c].data.detach().clone()
            ### BN2 running_mean
            net_pruned.module.layer2[block_num].shortcut[1].running_mean.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].running_mean.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].running_mean.data[order_c] = net.module.layer2[block_num].shortcut[1].running_mean[order_c].data.detach().clone()
            
            ### These weights are permanently set to zero ###
            net_pruned.module.layer2[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer2[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            order_zeros.update({'l2s': order_c_zero})
            block_id += 1
            
    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        n_c = orig_size[block_id] - cfg['l3'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        ### Conv
        net_pruned.module.layer3[block_num].conv1.weight.data = net.module.layer3[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()
        ### BN
        net_pruned.module.layer3[block_num].bn1.weight.data = net.module.layer3[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.bias.data = net.module.layer3[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.running_var.data = net.module.layer3[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.running_mean.data = net.module.layer3[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l3'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])
        ### These weights are recalled ###
        ### Conv
        net_pruned.module.layer3[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer3[block_num].conv2.weight.data[order_c] = net.module.layer3[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer3[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer3[block_num].bn2.weight.data[order_c] = net.module.layer3[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer3[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer3[block_num].bn2.bias.data[order_c] = net.module.layer3[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer3[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer3[block_num].bn2.running_var.data[order_c] = net.module.layer3[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer3[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer3[block_num].bn2.running_mean.data[order_c] = net.module.layer3[block_num].bn2.running_mean[order_c].data.detach().clone()

        ### These weights are permanently set to zero ###
        net_pruned.module.layer3[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer3[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l3_'+str(block_num): order_c_zero})
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l3s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            ### These weights are recalled ###
            ### Conv
            net_pruned.module.layer3[block_num].shortcut[0].weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[0].weight.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[0].weight.data[order_c] = net.module.layer3[block_num].shortcut[0].weight[order_c][:,order_p0].data.detach().clone()
            ### BN2 weight
            net_pruned.module.layer3[block_num].shortcut[1].weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].weight.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].weight.data[order_c] = net.module.layer3[block_num].shortcut[1].weight[order_c].data.detach().clone()
            ### BN2 bias
            net_pruned.module.layer3[block_num].shortcut[1].bias.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].bias.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].bias.data[order_c] = net.module.layer3[block_num].shortcut[1].bias[order_c].data.detach().clone()
            ### BN2 running_var
            net_pruned.module.layer3[block_num].shortcut[1].running_var.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].running_var.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].running_var.data[order_c] = net.module.layer3[block_num].shortcut[1].running_var[order_c].data.detach().clone()
            ### BN2 running_mean
            net_pruned.module.layer3[block_num].shortcut[1].running_mean.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].running_mean.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].running_mean.data[order_c] = net.module.layer3[block_num].shortcut[1].running_mean[order_c].data.detach().clone()

            ### These weights are permanently set to zero ###
            net_pruned.module.layer3[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer3[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            order_zeros.update({'l3s': order_c_zero})
            block_id += 1
            
    ### l4
    for block_num in range(int(len(cfg['l4']) / 2)):
        n_c = orig_size[block_id] - cfg['l4'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        ### Conv
        net_pruned.module.layer4[block_num].conv1.weight.data = net.module.layer4[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()
        ### BN
        net_pruned.module.layer4[block_num].bn1.weight.data = net.module.layer4[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer4[block_num].bn1.bias.data = net.module.layer4[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer4[block_num].bn1.running_var.data = net.module.layer4[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer4[block_num].bn1.running_mean.data = net.module.layer4[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l4'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])
        
        ### These weights are recalled ###
        ### Conv
        net_pruned.module.layer4[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer4[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer4[block_num].conv2.weight.data[order_c] = net.module.layer4[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer4[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer4[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer4[block_num].bn2.weight.data[order_c] = net.module.layer4[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer4[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer4[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer4[block_num].bn2.bias.data[order_c] = net.module.layer4[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer4[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer4[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer4[block_num].bn2.running_var.data[order_c] = net.module.layer4[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer4[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer4[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer4[block_num].bn2.running_mean.data[order_c] = net.module.layer4[block_num].bn2.running_mean[order_c].data.detach().clone()
        
        ### These weights are permanently set to zero ###
        net_pruned.module.layer4[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer4[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l4_'+str(block_num): order_c_zero})
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l4s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            ### These weights are recalled ###
            ### Conv
            net_pruned.module.layer4[block_num].shortcut[0].weight.data = torch.zeros_like(net_pruned.module.layer4[block_num].shortcut[0].weight.data).to(device)
            net_pruned.module.layer4[block_num].shortcut[0].weight.data[order_c] = net.module.layer4[block_num].shortcut[0].weight[order_c][:,order_p0].data.detach().clone()
            ### BN2 weight
            net_pruned.module.layer4[block_num].shortcut[1].weight.data = torch.zeros_like(net_pruned.module.layer4[block_num].shortcut[1].weight.data).to(device)
            net_pruned.module.layer4[block_num].shortcut[1].weight.data[order_c] = net.module.layer4[block_num].shortcut[1].weight[order_c].data.detach().clone()
            ### BN2 bias
            net_pruned.module.layer4[block_num].shortcut[1].bias.data = torch.zeros_like(net_pruned.module.layer4[block_num].shortcut[1].bias.data).to(device)
            net_pruned.module.layer4[block_num].shortcut[1].bias.data[order_c] = net.module.layer4[block_num].shortcut[1].bias[order_c].data.detach().clone()
            ### BN2 running_var
            net_pruned.module.layer4[block_num].shortcut[1].running_var.data = torch.zeros_like(net_pruned.module.layer4[block_num].shortcut[1].running_var.data).to(device)
            net_pruned.module.layer4[block_num].shortcut[1].running_var.data[order_c] = net.module.layer4[block_num].shortcut[1].running_var[order_c].data.detach().clone()
            ### BN2 running_mean
            net_pruned.module.layer4[block_num].shortcut[1].running_mean.data = torch.zeros_like(net_pruned.module.layer4[block_num].shortcut[1].running_mean.data).to(device)
            net_pruned.module.layer4[block_num].shortcut[1].running_mean.data[order_c] = net.module.layer4[block_num].shortcut[1].running_mean[order_c].data.detach().clone()
            ### These weights are permanently set to zero ###
            net_pruned.module.layer4[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer4[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            order_zeros.update({'l4s': order_c_zero})
            block_id += 1
            

    ### Linear block
    net_pruned.module.linear.weight.data = net.module.linear.weight.data.detach().clone()
    net_pruned.module.linear.bias.data = net.module.linear.bias.data.detach().clone()
    
    return net_pruned, order_zeros

# Map to zero block for ResNet pruning
def skip_or_prune(net_p, order_zeros, prune_ratio, orig_size, arch_type=34):
    if(arch_type==34):
        num_blocks=[3,4,6,3]
    elif(arch_type==18):
        num_blocks=[2,2,2,2]

    cfg = cfg_res(prune_ratio, orig_size, num_blocks=num_blocks)

    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l1_'+str(block_num)]
        net_p.module.layer1[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer1[block_num].map.map[0,order_c_zero])).to(device)

    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l2_'+str(block_num)]
        net_p.module.layer2[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer2[block_num].map.map[0,order_c_zero])).to(device)
        ### Shortcut
        if(block_num == 0):   
            ### These weights are permanently set to zero ###
            order_c_zero = order_zeros['l2s']
            net_p.module.layer2[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_p.module.layer2[block_num].shortcut[2].map[0,order_c_zero])).to(device)  

    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l3_'+str(block_num)]
        net_p.module.layer3[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer3[block_num].map.map[0,order_c_zero])).to(device)
        ### Shortcut
        if(block_num == 0):   
            ### These weights are permanently set to zero ###
            order_c_zero = order_zeros['l3s']
            net_p.module.layer3[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_p.module.layer3[block_num].shortcut[2].map[0,order_c_zero])).to(device)  

    ### l4
    for block_num in range(int(len(cfg['l4']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l4_'+str(block_num)]
        net_p.module.layer4[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer4[block_num].map.map[0,order_c_zero])).to(device)
        ### Shortcut
        if(block_num == 0):
            ### These weights are permanently set to zero ###
            order_c_zero = order_zeros['l4s']
            net_p.module.layer4[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_p.module.layer4[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            
    return net_p, cfg

# ResNet optimizer
def res_optimizer(base_optimizer, lr=None, net_p=None, imp_order=None, prune_ratio=None, orig_size=None, wd=1e-4, moment=0.9, preserve_momentum=True, arch_type=34):
    new_optimizer = optim.SGD(net_p.parameters(), lr=lr, momentum=moment, weight_decay=wd)
    if(preserve_momentum==False):
        return new_optimizer

    if(arch_type==34):
        num_blocks=[3,4,6,3]
    elif(arch_type==18):
        num_blocks=[2,2,2,2]

    cfg = cfg_res(prune_ratio, orig_size, num_blocks=num_blocks)
    cfg_zero = cfg_res_zero(prune_ratio, orig_size, num_blocks=num_blocks)

    block_id = 0
    lbase = 0

    # base conv layer
    n_c = orig_size[block_id] - cfg['base']
    order_c = np.sort(imp_order[block_id][n_c:])
    for lnum in range(3):
        momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
        new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()

    order_p = order_c.copy()
    block_id += 1
    lbase = lnum + 1

    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        n_c = orig_size[block_id] - cfg['l1'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()

        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l1'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        n_c = orig_size[block_id] - cfg['l2'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l2'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l2s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            for iterate, lnum in enumerate(range(lbase, lbase+3)):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

            block_id += 1
            lbase = lnum + 1
            
    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        n_c = orig_size[block_id] - cfg['l3'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l3'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l3s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            for iterate, lnum in enumerate(range(lbase, lbase+3)):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

            block_id += 1
            lbase = lnum + 1
            
    ### l4
    for block_num in range(int(len(cfg['l4']) / 2)):
        n_c = orig_size[block_id] - cfg['l4'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l4'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l4s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            for iterate, lnum in enumerate(range(lbase, lbase+3)):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

            block_id += 1
            lbase = lnum + 1            

    ### Linear block
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-2]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-2]]['momentum_buffer'] = momentum_base.data.clone()
    lbase = lbase + 1

    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-1]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-1]]['momentum_buffer'] = momentum_base.data.clone()
    lbase = lbase + 1

    return new_optimizer

# ResNet
def res_pruner_cifar(net, imp_order, prune_ratio, orig_size, num_blocks=[9,9,9], num_classes=100):
    cfg = cfg_res_cifar(prune_ratio, orig_size, num_blocks=num_blocks)
    cfg_zero = cfg_res_zero_cifar(prune_ratio, orig_size, num_blocks=num_blocks)
    net_pruned = torch.nn.DataParallel(ResPruned_cifar(cfg, num_blocks=num_blocks, num_classes=num_classes))

    block_id = 0
    order_zeros = {}
    # base
    n_c = orig_size[block_id] - cfg['base']
    order_c = np.sort(imp_order[block_id][n_c:])
    net_pruned.module.conv1.weight.data = net.module.conv1.weight[order_c].data.detach().clone()
    
    net_pruned.module.bn1.weight.data = net.module.bn1.weight[order_c].data.detach().clone()
    net_pruned.module.bn1.bias.data = net.module.bn1.bias[order_c].data.detach().clone()
    net_pruned.module.bn1.running_var.data = net.module.bn1.running_var[order_c].data.detach().clone()
    net_pruned.module.bn1.running_mean.data = net.module.bn1.running_mean[order_c].data.detach().clone()
    order_p = order_c.copy()
    block_id += 1
        
    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        n_c = orig_size[block_id] - cfg['l1'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        net_pruned.module.layer1[block_num].conv1.weight.data = net.module.layer1[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()

        net_pruned.module.layer1[block_num].bn1.weight.data = net.module.layer1[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.bias.data = net.module.layer1[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.running_var.data = net.module.layer1[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer1[block_num].bn1.running_mean.data = net.module.layer1[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l1'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        ### Conv
        net_pruned.module.layer1[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer1[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer1[block_num].conv2.weight.data[order_c] = net.module.layer1[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer1[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer1[block_num].bn2.weight.data[order_c] = net.module.layer1[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer1[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer1[block_num].bn2.bias.data[order_c] = net.module.layer1[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer1[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer1[block_num].bn2.running_var.data[order_c] = net.module.layer1[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer1[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer1[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer1[block_num].bn2.running_mean.data[order_c] = net.module.layer1[block_num].bn2.running_mean[order_c].data.detach().clone()

        ### These weights are permanently set to zero ###
        net_pruned.module.layer1[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer1[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l1_'+str(block_num): order_c_zero})
        
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        n_c = orig_size[block_id] - cfg['l2'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        ### Conv
        net_pruned.module.layer2[block_num].conv1.weight.data = net.module.layer2[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()
        ### BN
        net_pruned.module.layer2[block_num].bn1.weight.data = net.module.layer2[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.bias.data = net.module.layer2[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.running_var.data = net.module.layer2[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer2[block_num].bn1.running_mean.data = net.module.layer2[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l2'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are recalled ###
        ### Conv
        net_pruned.module.layer2[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer2[block_num].conv2.weight.data[order_c] = net.module.layer2[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer2[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer2[block_num].bn2.weight.data[order_c] = net.module.layer2[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer2[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer2[block_num].bn2.bias.data[order_c] = net.module.layer2[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer2[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer2[block_num].bn2.running_var.data[order_c] = net.module.layer2[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer2[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer2[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer2[block_num].bn2.running_mean.data[order_c] = net.module.layer2[block_num].bn2.running_mean[order_c].data.detach().clone()
        
        ### These weights are permanently set to zero ###
        net_pruned.module.layer2[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer2[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l2_'+str(block_num): order_c_zero})
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l2s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            ### These weights are recalled ###
            ### Conv
            net_pruned.module.layer2[block_num].shortcut[0].weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[0].weight.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[0].weight.data[order_c] = net.module.layer2[block_num].shortcut[0].weight[order_c][:,order_p0].data.detach().clone()
            ### BN2 weight
            net_pruned.module.layer2[block_num].shortcut[1].weight.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].weight.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].weight.data[order_c] = net.module.layer2[block_num].shortcut[1].weight[order_c].data.detach().clone()
            ### BN2 bias
            net_pruned.module.layer2[block_num].shortcut[1].bias.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].bias.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].bias.data[order_c] = net.module.layer2[block_num].shortcut[1].bias[order_c].data.detach().clone()
            ### BN2 running_var
            net_pruned.module.layer2[block_num].shortcut[1].running_var.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].running_var.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].running_var.data[order_c] = net.module.layer2[block_num].shortcut[1].running_var[order_c].data.detach().clone()
            ### BN2 running_mean
            net_pruned.module.layer2[block_num].shortcut[1].running_mean.data = torch.zeros_like(net_pruned.module.layer2[block_num].shortcut[1].running_mean.data).to(device)
            net_pruned.module.layer2[block_num].shortcut[1].running_mean.data[order_c] = net.module.layer2[block_num].shortcut[1].running_mean[order_c].data.detach().clone()
            
            ### These weights are permanently set to zero ###
            net_pruned.module.layer2[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer2[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            order_zeros.update({'l2s': order_c_zero})
            block_id += 1
            
    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        n_c = orig_size[block_id] - cfg['l3'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        ### Conv
        net_pruned.module.layer3[block_num].conv1.weight.data = net.module.layer3[block_num].conv1.weight[order_c][:,order_p].data.detach().clone()
        ### BN
        net_pruned.module.layer3[block_num].bn1.weight.data = net.module.layer3[block_num].bn1.weight[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.bias.data = net.module.layer3[block_num].bn1.bias[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.running_var.data = net.module.layer3[block_num].bn1.running_var[order_c].data.detach().clone()
        net_pruned.module.layer3[block_num].bn1.running_mean.data = net.module.layer3[block_num].bn1.running_mean[order_c].data.detach().clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        
        n_c = orig_size[block_id] - cfg_zero['l3'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])
        ### These weights are recalled ###
        ### Conv
        net_pruned.module.layer3[block_num].conv2.weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].conv2.weight.data).to(device)
        net_pruned.module.layer3[block_num].conv2.weight.data[order_c] = net.module.layer3[block_num].conv2.weight[order_c][:,order_p].data.detach().clone()
        ### BN2 weight
        net_pruned.module.layer3[block_num].bn2.weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.weight.data).to(device)
        net_pruned.module.layer3[block_num].bn2.weight.data[order_c] = net.module.layer3[block_num].bn2.weight[order_c].data.detach().clone()
        ### BN2 bias
        net_pruned.module.layer3[block_num].bn2.bias.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.bias.data).to(device)
        net_pruned.module.layer3[block_num].bn2.bias.data[order_c] = net.module.layer3[block_num].bn2.bias[order_c].data.detach().clone()
        ### BN2 running_var
        net_pruned.module.layer3[block_num].bn2.running_var.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.running_var.data).to(device)
        net_pruned.module.layer3[block_num].bn2.running_var.data[order_c] = net.module.layer3[block_num].bn2.running_var[order_c].data.detach().clone()
        ### BN2 running_mean
        net_pruned.module.layer3[block_num].bn2.running_mean.data = torch.zeros_like(net_pruned.module.layer3[block_num].bn2.running_mean.data).to(device)
        net_pruned.module.layer3[block_num].bn2.running_mean.data[order_c] = net.module.layer3[block_num].bn2.running_mean[order_c].data.detach().clone()

        ### These weights are permanently set to zero ###
        net_pruned.module.layer3[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer3[block_num].map.map[0,order_c_zero])).to(device)
        order_zeros.update({'l3_'+str(block_num): order_c_zero})
        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l3s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            ### These weights are recalled ###
            ### Conv
            net_pruned.module.layer3[block_num].shortcut[0].weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[0].weight.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[0].weight.data[order_c] = net.module.layer3[block_num].shortcut[0].weight[order_c][:,order_p0].data.detach().clone()
            ### BN2 weight
            net_pruned.module.layer3[block_num].shortcut[1].weight.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].weight.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].weight.data[order_c] = net.module.layer3[block_num].shortcut[1].weight[order_c].data.detach().clone()
            ### BN2 bias
            net_pruned.module.layer3[block_num].shortcut[1].bias.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].bias.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].bias.data[order_c] = net.module.layer3[block_num].shortcut[1].bias[order_c].data.detach().clone()
            ### BN2 running_var
            net_pruned.module.layer3[block_num].shortcut[1].running_var.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].running_var.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].running_var.data[order_c] = net.module.layer3[block_num].shortcut[1].running_var[order_c].data.detach().clone()
            ### BN2 running_mean
            net_pruned.module.layer3[block_num].shortcut[1].running_mean.data = torch.zeros_like(net_pruned.module.layer3[block_num].shortcut[1].running_mean.data).to(device)
            net_pruned.module.layer3[block_num].shortcut[1].running_mean.data[order_c] = net.module.layer3[block_num].shortcut[1].running_mean[order_c].data.detach().clone()

            ### These weights are permanently set to zero ###
            net_pruned.module.layer3[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_pruned.module.layer3[block_num].shortcut[2].map[0,order_c_zero])).to(device)
            order_zeros.update({'l3s': order_c_zero})
            block_id += 1

    ### Linear block
    net_pruned.module.linear.weight.data = net.module.linear.weight.data.detach().clone()
    net_pruned.module.linear.bias.data = net.module.linear.bias.data.detach().clone()
    
    return net_pruned, order_zeros

# Map to zero block for ResNet pruning
def skip_or_prune_cifar(net_p, order_zeros, prune_ratio, orig_size, num_blocks=[9,9,9]):
    cfg = cfg_res_cifar(prune_ratio, orig_size, num_blocks=num_blocks)

    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l1_'+str(block_num)]
        net_p.module.layer1[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer1[block_num].map.map[0,order_c_zero])).to(device)

    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l2_'+str(block_num)]
        net_p.module.layer2[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer2[block_num].map.map[0,order_c_zero])).to(device)
        ### Shortcut
        if(block_num == 0):   
            ### These weights are permanently set to zero ###
            order_c_zero = order_zeros['l2s']
            net_p.module.layer2[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_p.module.layer2[block_num].shortcut[2].map[0,order_c_zero])).to(device)  

    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        ### These weights are permanently set to zero ###
        order_c_zero = order_zeros['l3_'+str(block_num)]
        net_p.module.layer3[block_num].map.map[0,order_c_zero] = torch.zeros_like((net_p.module.layer3[block_num].map.map[0,order_c_zero])).to(device)
        ### Shortcut
        if(block_num == 0):   
            ### These weights are permanently set to zero ###
            order_c_zero = order_zeros['l3s']
            net_p.module.layer3[block_num].shortcut[2].map[0,order_c_zero] = torch.zeros_like((net_p.module.layer3[block_num].shortcut[2].map[0,order_c_zero])).to(device)  

    return net_p, cfg

# ResNet-56
def res_cifar_optimizer(base_optimizer, lr=None, net_p=None, imp_order=None, prune_ratio=None, orig_size=None, wd=1e-4, moment=0.9, preserve_momentum=True, num_blocks=[9,9,9]):
    new_optimizer = optim.SGD(net_p.parameters(), lr=lr, momentum=moment, weight_decay=wd)
    if(preserve_momentum==False):
        return new_optimizer
    cfg = cfg_res_cifar(prune_ratio, orig_size, num_blocks=num_blocks)
    cfg_zero = cfg_res_zero_cifar(prune_ratio, orig_size, num_blocks=num_blocks)

    block_id = 0
    lbase = 0

    # base conv layer
    n_c = orig_size[block_id] - cfg['base']
    order_c = np.sort(imp_order[block_id][n_c:])
    for lnum in range(3):
        momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
        new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()

    order_p = order_c.copy()
    block_id += 1
    lbase = lnum + 1
        
    ### l1
    for block_num in range(int(len(cfg['l1']) / 2)):
        n_c = orig_size[block_id] - cfg['l1'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()

        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l1'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
    ### l2
    for block_num in range(int(len(cfg['l2']) / 2)):
        n_c = orig_size[block_id] - cfg['l2'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l2'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l2s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            for iterate, lnum in enumerate(range(lbase, lbase+3)):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

            block_id += 1
            lbase = lnum + 1
            
    ### l3
    for block_num in range(int(len(cfg['l3']) / 2)):
        n_c = orig_size[block_id] - cfg['l3'][2*block_num]
        order_c = np.sort(imp_order[block_id][n_c:])
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[order_c].data.clone()
        order_p0 = order_p.copy()
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        n_c = orig_size[block_id] - cfg_zero['l3'][2*block_num+1]
        order_c = np.sort(imp_order[block_id][n_c:])
        order_c_zero = np.sort(imp_order[block_id][0:n_c])

        ### These weights are called back ###
        for iterate, lnum in enumerate(range(lbase, lbase+3)):
            if(iterate == 0):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base[:][:,order_p].data.clone()
            else:
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

        order_c = np.sort(imp_order[block_id])
        order_p = order_c.copy()
        block_id += 1
        lbase = lnum + 1
        
        ### Shortcut
        if(block_num == 0):
            n_c = orig_size[block_id] - cfg_zero['l3s']
            order_c = np.sort(imp_order[block_id][n_c:])
            order_c_zero = np.sort(imp_order[block_id][0:n_c])
            
            for iterate, lnum in enumerate(range(lbase, lbase+3)):
                momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer']
                new_optimizer.state[new_optimizer.param_groups[0]['params'][lnum]]['momentum_buffer'] = momentum_base.data.clone()

            block_id += 1
            lbase = lnum + 1

    ### Linear block
    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-2]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-2]]['momentum_buffer'] = momentum_base.data.clone()

    momentum_base = base_optimizer.state[base_optimizer.param_groups[0]['params'][-1]]['momentum_buffer']
    new_optimizer.state[new_optimizer.param_groups[0]['params'][-1]]['momentum_buffer'] = momentum_base.data.clone()
    
    return new_optimizer