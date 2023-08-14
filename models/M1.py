import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from basic_module import BasicModule
from utils.utils import cosine_similarity, emd_inference_qpth
import torch.nn.functional as F
def get_marginal_distribution(n_batch, n_dim, device='cpu'):
    dist1 = torch.ones((n_batch, n_dim), device=device) / n_dim
    return dist1

def extract_class_indices(indices, cls_id):
    mask = torch.eq(indices, cls_id)
    cls_indices = torch.nonzero(mask).squeeze()
    return cls_indices

class M1(nn.Module):
    def __init__(self, n_sp, n_q, seq_len):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet = list(resnet.children())[:-1]
        mid = len(resnet) // 2
        self.resnet1 = nn.Sequential(*resnet[:mid+5])
        self.resnet2 = nn.Sequential(*resnet[mid+5:])

        n_batch = n_sp * n_q
        self.dist = get_marginal_distribution(n_batch, seq_len)


    def optimal_transport(self, cost_matrix):
        device = cost_matrix.device
        if (self.dist.device != device):
            self.dist = self.dist.to(device)
        cost, flow = emd_inference_qpth(cost_matrix, self.dist, self.dist)
        return flow
    def encode(self, x):
        x = torch.utils.checkpoint.checkpoint(self.resnet1, x)
        # x = self.resnet1(x)
        x = self.resnet2(x)
        return x



    def forward(self, sp_set, sp_class, q_set):
        n_sp = sp_set.shape[0]
        n_q = q_set.shape[0]
        seq_len = sp_set.shape[1]
        
        sp_set = sp_set.view(n_sp*seq_len, *sp_set.shape[2:])
        q_set = q_set.view(n_q*seq_len, *q_set.shape[2:])
        
        # TODO: add activation checkpointing
        sp_set =  self.encode(sp_set)
        q_set = self.encode(q_set)
 
        sp_set = sp_set.view(n_sp, seq_len, -1)
        q_set = q_set.view(n_q, seq_len, -1)

        cost_matrix = torch.zeros((n_q * n_sp, seq_len, seq_len), dtype=torch.float32).to(sp_set.device)
        for i in range(n_q):
            for j in range(n_sp):
                cost_matrix[i*n_sp + j] = 1 - cosine_similarity(q_set[i], sp_set[j])
        
        trans_plan = self.optimal_transport(cost_matrix)
        trans_cost = torch.mul(trans_plan, cost_matrix).sum((-2, -1))
        
        unique_classes, _ = torch.sort(torch.unique(sp_class))
        n_classes = unique_classes.shape[0]
        class_distace = torch.zeros((n_q, n_classes), device=sp_set.device)
        
        for i in range(n_q):
            for j, c in enumerate(unique_classes):
                cls_indices = extract_class_indices(sp_class, c)
                cls_indices = i*n_sp + cls_indices
                class_distace[i, j] = torch.mean(trans_cost[cls_indices])
        
        class_distace = -class_distace
        return class_distace
        
if __name__ == '__main__':
    n_way = 5
    n_sp = 25
    n_q = 25
    seq_len = 8
    img_size = 224
    dev = 'cuda:0'

    args = dict(
        n_sp = n_sp,
        n_q = n_q,
        seq_len = seq_len
    )
    model = M1(**args).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    sp_set = torch.rand(n_sp, seq_len, 3, img_size, img_size).to(dev)
    q_set = sp_set
    sp_class = torch.tensor([0, 1, 2, 3, 4]*n_way, dtype=torch.int32).to(dev)
    q_label = torch.tensor([0, 1, 2, 3, 4]*n_way, dtype=torch.int64).to(dev)
    
    import time
    start_time = time.time()
    # cost_matrix = model(sp_set, sp_class, q_set)
    # loss = F.cross_entropy(cost_matrix, q_label)
    # loss.backward()
    # optimizer.step()
    with torch.no_grad():
        cost_matrix = model(sp_set, sp_class, q_set)
        loss = F.cross_entropy(cost_matrix, q_label)

    end_time = time.time()
    print('Time: ', end_time - start_time)
    
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")


