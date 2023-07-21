import torch
import torch.nn as nn
import numpy as np
import ot
from easydict import EasyDict as edict

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, cfg):
        super(C3D, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()
        
        nseg = cfg['n_seg']
        self.ot_dist= np.ones((nseg), dtype=np.float64) / nseg
        
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
        
        numItermax = self.cfg.numItermax if hasattr(self.cfg, 'numItermax') else 10000
        trans_plan = ot.sinkhorn(self.ot_dist, self.ot_dist, cost_matrix, self.cfg.entropic_reg, numItermax=numItermax)
        if (np.any(np.isnan(trans_plan))):
            trans_plan = self.ot_dist[:, np.newaxis] * self.ot_dist[np.newaxis, :]
        
        return trans_plan

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
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

        cost_matrix = torch.zeros((total_query, total_sp, n_seg, n_seg), dtype=torch.float32).to(support_set.device)
        for query in range(total_query):
            for shot in range(total_sp):
                cost_matrix[query, shot] = torch.cdist(sp_set[shot], q_set[query], p=2)
        
        cost_matrix_c = cost_matrix.detach().cpu().numpy().astype(np.float64)
        trans_plan = np.zeros((total_query, total_sp, n_seg, n_seg), dtype=np.float64)
        for query in range(total_query):
            for shot in range(total_sp):
                trans_plan[query, shot] = self.optimal_transport(cost_matrix_c[query, shot])

        
        trans_plan = torch.from_numpy(trans_plan).to(cost_matrix.device).float()
        entropy = -torch.sum(trans_plan * torch.log(trans_plan + 1e-10), dim=(-2, -1))
        trans_cost = torch.mul(cost_matrix, trans_plan).sum((-2, -1)) - self.cfg.entropic_reg*entropy
        trans_cost = trans_cost.reshape((total_query, n_way, n_shot)).mean(dim=-1)
        trans_cost = -trans_cost
        
        return trans_cost
                        
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
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
    sp_set = torch.rand(5, 4, 3, 16, 112, 112).to(device)
    q_set = torch.rand(5, 4, 3, 16, 112, 112).to(device)
    net = C3D(args)
    net.to(device)

    # ckpt_path = 'c3d_sports1m-pretrained.pt'
    # ckpt = torch.load(ckpt_path)
    # net.load_state_dict(ckpt['state_dict'])
    output = net(sp_set, q_set)
