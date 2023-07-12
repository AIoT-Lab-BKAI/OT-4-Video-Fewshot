import torch
import torch.nn as nn
import numpy as np
import ot
def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-6), dim=(-2, -1))



class C3DModel(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, nseg):
        super(C3DModel, self).__init__()

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
        
        self.ot_reg = 7.0
        self.cost_alpha = 0.4
        self.pos_cost_phi = 1.0
        self.ot_dist= np.ones((nseg), dtype=np.float32) / nseg
        self.positional_cost = np.zeros((nseg, nseg), dtype=np.float32)
        for i in range(nseg):
            for j in range(nseg):
                self.positional_cost[i, j] = np.exp(-(1/(self.pos_cost_phi**2)) * (1/((i/nseg - j/nseg)**2+1)))
    
    def optimal_transport(self, sem_cost_matrix):
        cost_matrix = sem_cost_matrix + self.cost_alpha * self.positional_cost
        trans_plan = ot.sinkhorn(self.ot_dist, self.ot_dist, cost_matrix, self.ot_reg)
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
        nbatch, nway, kshot, nseg, c, seglen, h, w = support_set.shape
        nbatch, nquery, nseg, c, seglen, h, w = query_set.shape

        sp_set = support_set.reshape((-1, c, seglen, h, w))
        sp_set = self.encode(sp_set)
        sp_set = sp_set.reshape((nbatch, nway, kshot, nseg, -1))

        q_set = query_set.reshape((-1, c, seglen, h, w))
        q_set = self.encode(q_set)
        q_set = q_set.reshape((nbatch, nquery, nseg, -1))


        cost_matrix = torch.zeros((nbatch, nquery, nway, kshot, nseg, nseg), dtype=torch.float32).to(support_set.device)
        for batch in range(nbatch):
            for query in range(nquery):
                for label in range(nway):
                    for shot in range(kshot):
                        cost_matrix[batch, query, label, shot] = torch.cdist(sp_set[batch, label, shot], q_set[batch, query], p=2)
        
        cost_matrix_c = cost_matrix.detach().cpu().numpy()
        trans_plan = np.zeros((nbatch, nquery, nway, kshot, nseg, nseg), dtype=np.float32)
        for batch in range(nbatch):
            for query in range(nquery):
                for label in range(nway):
                    for shot in range(kshot):
                        trans_plan[batch, query, label, shot] = self.optimal_transport(cost_matrix_c[batch, query, label, shot])
        trans_plan = torch.from_numpy(trans_plan).to(cost_matrix.device)
        trans_cost = torch.mul(cost_matrix, trans_plan).sum((-2, -1)) - (1/self.ot_reg)*entropy(trans_plan)
        trans_cost = torch.mean(trans_cost, dim=(-1))
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
    sp_set = torch.rand(1, 5, 1, 4, 3, 16, 112, 112).to(device)
    q_set = torch.rand(1, 5, 4, 3, 16, 112, 112).to(device)
    net = C3DModel(4)
    net.to(device)

    # ckpt_path = 'c3d_sports1m-pretrained.pt'
    # ckpt = torch.load(ckpt_path)
    # net.load_state_dict(ckpt['state_dict'])
    output = net(sp_set, q_set)
    import pdb; pdb.set_trace()