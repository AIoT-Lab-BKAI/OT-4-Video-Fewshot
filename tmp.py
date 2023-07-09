import torch

ckpt = torch.load('c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth')
new_state = {}
for k, v in ckpt['state_dict'].items():
    k1 = k.replace('backbone.', '')
    k1 = k1.replace('1a', '1')
    k1 = k1.replace('2a', '2')
    k1 = k1.replace('.conv', '')
    if 'cls' not in k1:
        new_state[k1] = v
ckpt['state_dict'] = new_state
torch.save(ckpt, 'c3d_sports1m-pretrained.pt')


