import torch
import torch.nn as nn
import numpy as np
# import ot
# from easydict import EasyDict as edict
from utils.utils import cosine_similarity, emd_inference_qpth

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, cfg):
        super(C3D, self).__init__()
        # code copy from https://github.com/DavideA/c3d-pytorch
        self.cfg = cfg
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 =  nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 =  nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 =  nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        if "checkpoint" in cfg.keys():
            self.load_state_dict(torch.load(cfg["checkpoint"])['state_dict'], strict=False)
        else:
            self.__init_weight()
        
        nseg = cfg['n_seg']
        self.dist1 = torch.ones((1, nseg)) / nseg
        self.dist2 = torch.ones((1, nseg)) / nseg
        
        if cfg['use_positional_cost']:
            self.ot_reg = 7.0
            self.cost_alpha = 0.4
            self.pos_cost_phi = 1.0
            self.positional_cost = np.zeros((nseg, nseg), dtype=np.float32)
            for i in range(nseg):
                for j in range(nseg):
                    self.positional_cost[i, j] = np.exp(-(1/(self.pos_cost_phi**2)) * (1/((i/nseg - j/nseg)**2+1)))
    
    def optimal_transport(self, sem_cost_matrix):
        if self.cfg['use_positional_cost']:
            cost_matrix = sem_cost_matrix + self.cost_alpha * self.positional_cost
        else:
            cost_matrix = sem_cost_matrix
        
        # normalize cost matrix
        # cost_matrix = cost_matrix / np.maximum(np.max(cost_matrix), 1.0)
        # tuning parameter
        numItermax = self.cfg.numItermax if hasattr(self.cfg, 'numItermax') else 2000
        if self.dist1.device != cost_matrix.device:
            self.dist1 = self.dist1.to(cost_matrix.device)
            self.dist2 = self.dist2.to(cost_matrix.device)
        
        cost_matrix = cost_matrix.unsqueeze(0)
        _, trans_plan = emd_inference_qpth(cost_matrix, self.dist1, self.dist2)
        trans_plan = trans_plan.squeeze(0)
                
        return trans_plan

    def encode(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.bn3(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.bn4(self.conv4b(x)))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.bn5(self.conv5b(x)))
        x = self.pool5(x)
        
        x = x.view(-1, 8192)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.dropout(x)

        return x
    
    def forward(self, support_set, query_set):
        # assumption: 
        # each class instance lies consecutively
        _, _, seglen, c, h, w = support_set.shape
        n_way = self.cfg['n_way']
        n_shot = self.cfg['n_shot']
        n_seg = self.cfg['n_seg']
        total_query = query_set.shape[0]
        total_sp = support_set.shape[0]
        
        sp_set = support_set.transpose(2, 3).reshape((-1, c, seglen, h, w))
        sp_set = self.encode(sp_set)
        sp_set = sp_set.reshape((total_sp, n_seg, -1))
        
        q_set = query_set.transpose(2, 3).reshape((-1, c, seglen, h, w))
        q_set = self.encode(q_set)
        q_set = q_set.reshape((total_query, n_seg, -1))
        
        # total_query: total number of query instances Ex: 5 labels each has 5 query instances, therefore total_query = 25
        # total_sp: total number of support instances Ex: 5 labels each has 1 support instances, therefore total_sp = 5
        # n_seg: number of segments of each video
        # sp_set: total_sp x n_seg x feature_dim
        # q_set: total_query x n_seg x feature_dim
        cost_matrix = torch.zeros((total_query, total_sp, n_seg, n_seg), dtype=torch.float32).to(support_set.device)
        for query in range(total_query):
            for shot in range(total_sp):
                cost_matrix[query, shot] = 1 - cosine_similarity(q_set[query], sp_set[shot])
        
        trans_plan = torch.zeros((total_query, total_sp, n_seg, n_seg), dtype=torch.float32).to(support_set.device)
        for query in range(total_query):
            for shot in range(total_sp):
                trans_plan[query, shot] = self.optimal_transport(cost_matrix[query, shot])
        
        # with regulization term
        # entropy = -torch.sum(trans_plan * torch.log(trans_plan + 1e-10), dim=(-2, -1))
        # trans_cost = torch.mul(cost_matrix, trans_plan).sum((-2, -1)) - self.cfg.entropic_reg*entropy
        trans_cost = torch.mul(trans_plan, cost_matrix).sum((-2, -1))

        trans_cost = trans_cost.reshape((total_query, n_way, n_shot)).mean(dim=-1)
        trans_cost = -trans_cost
        
        return trans_cost
                        
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    device = 'cuda:0'
    args = edict(
        use_positional_cost = False,
        n_seg = 4,
        n_way = 5,
        n_shot = 1,
        entropic_reg = 1/7.0
    )
    sp_set = torch.rand(5, 4, 16, 3, 112, 112).to(device)
    q_set = torch.rand(5, 4, 16, 3, 112, 112).to(device)
    net = C3D(args)
    net.to(device)

    # ckpt_path = 'c3d_sports1m-pretrained.pt'
    # ckpt = torch.load(ckpt_path)
    # net.load_state_dict(ckpt['state_dict'])
    output = net(sp_set, q_set)
    print(output.shape)
