from datasets import load_dataset
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import time
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

# Prepare dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = load_dataset("stanfordnlp/imdb")

train_dataset = dataset["train"]
test_dataset = dataset["test"]
train_tokens = tokenizer(list(dataset['train']['text']), padding = True, truncation=True)
test_tokens = tokenizer(list(dataset['test']['text']), padding = True, truncation=True)

class TokenData(Dataset):
    def __init__(self, train = False):
        if train:
            self.text_data = list(dataset['train']['text'])
            self.tokens = train_tokens
            self.labels = list(dataset['train']['label'])
        else:
            self.text_data = list(dataset['test']['text'])
            self.tokens = test_tokens
            self.labels = list(dataset['test']['label'])

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.tokens.items():
            sample[k] = torch.tensor(v[idx])
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased') # Pre-trained model

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


################################################################################
# Todo: Implement Model Split Here.
# import modules as you want.





################################################################################
# Now, schedule the pipe.
# ScheduleGPipe will be a baseline for this assignment.





################################################################################
# Finally, run model.