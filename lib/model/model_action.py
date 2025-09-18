import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat
        
class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=4, dropout_ratio=0., version='class', hidden_dim=2048, num_joints=16):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version=='class':
            self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, num_joints=num_joints)
        elif version=='embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        else:
            raise Exception('Version Error.')

        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        
    def forward(self, x):
        '''
            Input: (N, M x T x 17 x 3) 
        '''
        N, M, T, J, C = x.shape
        x = x.reshape(N*M, T, J, C)        
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, M, T, self.feat_J, -1])      # (N, M, T, J, C)
        out = self.head(feat)

        # extract features, for contrastive learning
        N, M, T, J, C = feat.shape  # 获取输入的形状
        feat = feat.permute(0, 1, 3, 4, 2)  # (N, M, T, J, C) -> (N, M, J, C, T)
        # 将时间步（T）和坐标维度（C）分开
        feat = feat.mean(dim=-1)  # 对 C 维度（每个关节点的坐标）求平均，得到 (N, M, J, T)
        feat = feat.reshape(N, M, -1)  # (N, M, J*C)  将关节点特征与时间步数融合
        feat = feat.mean(dim=1)  # 对 M（可能是多个通道或输入分支）维度求平均
        feat = self.fc1(feat)  # 将特征通过全连接层变换为隐层
        feat = F.normalize(feat, dim=-1)  # 对嵌入特征进行归一化，保证每个样本在同一空间内有相同的尺度
        return feat, out