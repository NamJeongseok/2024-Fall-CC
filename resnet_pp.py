import os
import sys
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint


def set_config(task_index, port = 15000):
  node_idx_list = []
  cluster_info = os.environ['SLURM_NODELIST']
  base, node_list = cluster_info.split('-', 1)
  _nodes = node_list[1:-1].split(',')
  for nodes in _nodes:
    if "-" in nodes:
      nodes = nodes.split('-')

      for i in range(int(nodes[0]),int(nodes[1])+1):
        if i < 10:
          node_idx_list.append(f"0{i}")
        else:
          node_idx_list.append(f"{i}")
    else:
      node_idx_list.append(nodes)

  torch_config = {
        'cluster': {'worker': [f"{base}-{node_idx}:{port}" for node_idx in node_idx_list]},
        'task': {'type': 'worker', 'index': task_index}
        }
  
  os.environ['TORCH_CONFIG'] = json.dumps(torch_config)
  print(f"# of worker : {len(node_idx_list)}")


# Connect Nodes.
task_index = int(os.environ['SLURM_PROCID'])
num_workers = int(os.environ['SLURM_NPROCS'])
set_config(task_index)


# Load Dataset (CIFAR10)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = datasets.CIFAR10(root='../dataset', train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
testset = datasets.CIFAR10(root='../dataset', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)


################################################################################
# Todo: Implement Model Split Here.
# import modules as you want.





################################################################################
# Now, schedule the pipe.
# ScheduleGPipe will be a baseline for this assignment.





################################################################################
# Finally, run model.