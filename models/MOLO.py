import torch
import torch.nn as nn
import torchvision.models as torch_models
import torch.nn.functional as F

from .components import PositionalEncoder, Transformer_v2, PreNormattention, Attention, FeedForward, DoubleConv2, Up2, OutConv
from einops import rearrange
from utils.utils import get_marginal_distribution, ot_distance
from easydict import EasyDict as edict

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class ResNet50(nn.Module):
    
    def __init__(self):
        super().__init__()
        backbone = torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V2)
        last_layer_idx = -2
        backbone = list(backbone.children())[:last_layer_idx]
        mid_point = len(backbone) // 2 + 4
        # mid_point = len(backbone)
        self.layer1 = nn.Sequential(*backbone[:mid_point])
        self.layer2 = nn.Sequential(*backbone[mid_point:])
    
    def forward(self, x):
        x = torch.utils.checkpoint.checkpoint(self.layer1, x)
        x = self.layer2(x)
        return x

class CNN_BiMHM_MoLo(nn.Module):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_BiMHM_MoLo, self).__init__()
        self.backbone_name = "resnet50"
        self.backbone = ResNet50()
        
        self.args = edict(args)
        
        if hasattr(self.args,"USE_CONTRASTIVE") and self.args.USE_CONTRASTIVE:
            if hasattr(self.args,"TEMP_COFF") and self.args.TEMP_COFF:
                self.scale = self.args.TEMP_COFF
                self.scale_motion = self.args.TEMP_COFF
            else:
                self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale.data.fill_(1.0)

                self.scale_motion = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale_motion.data.fill_(1.0)

        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        if self.backbone_name == "resnet50":
            self.mid_dim = 2048
            self.pre_reduce = nn.Sequential()
        else:
            self.mid_dim = 512
            self.pre_reduce = nn.Sequential()
        if hasattr(self.args,"POSITION_A") and hasattr(self.args,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.POSITION_A, B_scale=self.args.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        if hasattr(self.args,"HEAD") and self.args.HEAD:
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = self.args.HEAD, dim_head_k = self.mid_dim//self.args.HEAD, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = self.args.HEAD, dim_head_k = self.mid_dim//self.args.HEAD, dropout_atte = 0.2)
            
        else:
            
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
        # for frame motion reconstruction    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.factor = 8
        self.motion_reduce = nn.Conv3d(self.mid_dim, self.mid_dim//self.factor, kernel_size=(3,3,3), padding=(1,1,1), groups=1)
        self.motion_conv = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim//self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_up = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim, kernel_size=1, padding=0, groups=1)
        if hasattr(self.args, "USE_CLASSIFICATION") and self.args.USE_CLASSIFICATION:
            if hasattr(self.args, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        bilinear = True
        # factor = 2 if bilinear else 1
        factor = 1
        n_classes = 3
        self.up1 = Up2(self.mid_dim//self.factor, 128 // factor, bilinear, kernel_size=2)
        self.up2 = Up2(128, 32 // factor, bilinear, kernel_size=4)
        self.up3 = Up2(32, 16, bilinear, kernel_size=4)
        self.outc = OutConv(16, n_classes)
        # set_trace()

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        # Resnet
        support_features = self.pre_reduce(self.backbone(support_images)).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.pre_reduce(self.backbone(target_images)).squeeze()   # [200, 2048, 7, 7]

        dim = int(support_features.shape[1])
        
        # motion encoder
        support_features_motion = self.motion_reduce(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)   # [40, 128, 7, 7]
        support_features_motion_conv = self.motion_conv(support_features_motion)   # [40, 128, 7, 7]
        support_features_motion = support_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - support_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        support_features_motion = support_features_motion.reshape(-1, dim//self.factor, 7, 7)
        
        target_features_motion = self.motion_reduce(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)
        target_features_motion_conv = self.motion_conv(target_features_motion)
        target_features_motion = target_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - target_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        target_features_motion = target_features_motion.reshape(-1, dim//self.factor, 7, 7)
        
        # motion decoder
        feature_motion_recons = torch.cat([support_features_motion, target_features_motion], dim=0)
        feature_motion_recons = self.up1(feature_motion_recons)
        feature_motion_recons = self.up2(feature_motion_recons)
        feature_motion_recons = self.up3(feature_motion_recons)
        feature_motion_recons = self.outc(feature_motion_recons)
        
        # Temporal Attention for motions
        support_features_motion = self.relu(self.motion_up(support_features_motion))
        support_features_motion = self.avg_pool(support_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        support_bs = int(support_features_motion.shape[0])
        support_features_motion = torch.cat((self.class_token_motion.expand(support_bs, -1, -1), support_features_motion), dim=1)
        support_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(support_features_motion)))

        target_features_motion = self.relu(self.motion_up(target_features_motion))
        target_features_motion = self.avg_pool(target_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        target_bs = int(target_features_motion.shape[0])
        target_features_motion = torch.cat((self.class_token_motion.expand(target_bs, -1, -1), target_features_motion), dim=1)  
        target_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(target_features_motion)))   # [5, 9, 2048]
        
        # Temporal Attention for frame features
        support_features = self.avg_pool(support_features).squeeze()
        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        support_features = torch.cat((self.class_token.expand(support_bs, -1, -1), support_features), dim=1)
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features)))   # [5, 9, 2048]

        target_features = self.avg_pool(target_features).squeeze()
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = torch.cat((self.class_token.expand(target_bs, -1, -1), target_features), dim=1)
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features)))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args, "USE_CLASSIFICATION") and self.args.USE_CLASSIFICATION:
            if hasattr(self.args, "NUM_CLASS"):
                if hasattr(self.args, "USE_LOCAL") and self.args.USE_LOCAL:
                    class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.NUM_CLASS))
                else:
                    class_logits = self.classification_layer(torch.cat([support_features.mean(1)+support_features_motion.mean(1), target_features.mean(1)+target_features_motion.mean(1)], 0))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        
        unique_labels, _ = torch.unique(support_labels)
        support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_features_motion = [torch.mean(torch.index_select(support_features_motion, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features_motion = torch.stack(support_features_motion)
        
        return support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons
    
    def ot_distance(self, cost):
        uni_dist = get_marginal_distribution(cost.shape[0], cost.shape[1], cost.device)
        costs, flows = ot_distance(cost, uni_dist, uni_dist)
        return costs

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['query_set'] # [200, 3, 224, 224]
        support_images = support_images.view(-1, 3, 224, 224)
        support_labels = support_labels.view(-1)
        target_images = target_images.view(-1, 3, 224, 224)

        support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons = self.get_feats(support_images, target_images, support_labels)
        # motion reconstruction loss
        if "train" in inputs:
            support_images_re = support_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            target_images_re = target_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            input_recons = torch.cat([support_images_re[:,1:,:]-support_images_re[:,:-1,:], target_images_re[:,1:,:]- target_images_re[:,:-1,:]], dim=0).reshape(-1, 3, 224, 224)
            loss_recons = (feature_motion_recons - input_recons) ** 2   # [280, 3, 224, 224]
            loss_recons = loss_recons.mean()  # [N, L], mean loss per patch
        else:
            loss_recons = 0
        

        unique_labels, _ = torch.unique(support_labels)
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features_g = support_features[:,0,:]
        target_features_g = target_features[:,0,:]
        support_features = support_features[:,1:,:]
        target_features = target_features[:,1:,:]
        # calculate contrastive
        # support to query
        class_sim_s2q = cos_sim(support_features, target_features_g)  # [5, 8, 35]
        class_dists_s2q = 1 - class_sim_s2q
        class_dists_s2q = [torch.sum(torch.index_select(class_dists_s2q, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q = torch.stack(class_dists_s2q).squeeze(1)
        if hasattr(self.args,"USE_CONTRASTIVE") and self.args.USE_CONTRASTIVE:
            class_dists_s2q = rearrange(class_dists_s2q * self.scale, 'c q -> q c')

        # query to support 
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s   
        class_dists_q2s = [torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze(2)
        if hasattr(self.args,"USE_CONTRASTIVE") and self.args.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')
        
        support_features_motion_g = support_features_motion[:,0,:]
        target_features_motion_g = target_features_motion[:,0,:]
        support_features_motion = support_features_motion[:,1:,:]
        target_features_motion = target_features_motion[:,1:,:]
        # calculate contrastive
        # support to query
        class_sim_s2q_motion = cos_sim(support_features_motion, target_features_motion_g)  # [5, 8, 35]
        class_dists_s2q_motion = 1 - class_sim_s2q_motion
        class_dists_s2q_motion = [torch.sum(torch.index_select(class_dists_s2q_motion, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q_motion = torch.stack(class_dists_s2q_motion).squeeze(1)
        if hasattr(self.args,"USE_CONTRASTIVE") and self.args.USE_CONTRASTIVE:
            class_dists_s2q_motion = rearrange(class_dists_s2q_motion * self.scale_motion, 'c q -> q c')

        # query to support 
        class_sim_q2s_motion = cos_sim(target_features_motion, support_features_motion_g)  # [35, 8, 5]
        class_dists_q2s_motion = 1 - class_sim_q2s_motion   
        class_dists_q2s_motion = [torch.sum(torch.index_select(class_dists_q2s_motion, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s_motion = torch.stack(class_dists_q2s_motion).squeeze(2)
        if hasattr(self.args,"USE_CONTRASTIVE") and self.args.USE_CONTRASTIVE:
            class_dists_q2s_motion = rearrange(class_dists_q2s_motion * self.scale_motion, 'c q -> q c')
        
        # calculate class distance for each query
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]
        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> (tb sb) ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]
        cum_dists = self.ot_distance(dists)
        cum_dists = rearrange(cum_dists, '(tb sb) -> tb sb', tb = n_queries, sb = n_support)
        class_dists = cum_dists

        # calculate class distance for each query motion
        support_features_motion = rearrange(support_features_motion, 'b s d -> (b s) d')  # [200, 2048]
        target_features_motion = rearrange(target_features_motion, 'b s d -> (b s) d')    # [200, 2048]
        frame_sim_motion = cos_sim(target_features_motion, support_features_motion)    # [200, 200]
        frame_dists_motion = 1 - frame_sim_motion   
        
        dists_motion = rearrange(frame_dists_motion, '(tb ts) (sb ss) -> (tb sb) ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]
        cum_dists_motion = self.ot_distance(dists_motion)
        cum_dists_motion = rearrange(cum_dists_motion, '(tb sb) -> tb sb', tb = n_queries, sb = n_support)
        class_dists_motion = cum_dists_motion

        
        if hasattr(self.args, "LOGIT_BALANCE_COFF") and self.args.LOGIT_BALANCE_COFF:
            class_dists = class_dists + self.args.LOGIT_BALANCE_COFF*class_dists_motion
        else:
            class_dists = class_dists + 0.3*class_dists_motion
    
        return_dict = {'logits': - class_dists,  "logits_s2q": -class_dists_s2q, "logits_q2s": -class_dists_q2s, "logits_s2q_motion": -class_dists_s2q_motion, "logits_q2s_motion": -class_dists_q2s_motion, "loss_recons": loss_recons,}
        inputs.update(return_dict)
        return inputs

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

if __name__ == '__main__':
    import yaml
    with open("test.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    from easydict import EasyDict as edict
    cfg = edict(cfg)
    
    
    n_way = 5
    n_shot = 5
    n_query = 10
    seq_len = 8
    img_size = 224
    dev = 'cuda:0'
    total_sp = n_way * n_shot
    total_q = n_way * n_query
    
    sp_set = torch.rand((total_sp * seq_len, 3, img_size, img_size)).to(dev)
    q_set = torch.rand((total_q * seq_len, 3, img_size, img_size)).to(dev)
    sp_label = torch.arange(n_way).repeat(n_shot).to(dev)
    q_label = torch.arange(n_way).repeat(n_query).to(dev)

    model = CNN_BiMHM_MoLo(cfg)
    optimizer = torch.optim.Adam(model.parameters())
    model.to(dev)

    import time
    start_time = time.time()
    
    inputs = {"support_set": sp_set, "support_labels": sp_label, "target_set": q_set, "target_labels": q_label}
    model_dict = model(inputs)
    
    loss = F.cross_entropy(model_dict["logits"], q_label.long())
    loss += model_dict["loss_recons"]
    loss.backward()
    optimizer.step()

    end_time = time.time()
    print('Time: ', end_time - start_time)

    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")

    import pdb; pdb.set_trace()
