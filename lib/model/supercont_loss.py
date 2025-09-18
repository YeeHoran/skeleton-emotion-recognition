import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised Contrastive Loss
    Args:
        features: torch.Tensor, shape [batch_size, feature_dim]
        labels: torch.Tensor, shape [batch_size], int class labels
        temperature: float, temperature scaling
    Returns:
        loss: torch scalar
    """
    device = features.device
    batch_size = features.shape[0]

    # 归一化特征向量
    features = F.normalize(features, p=2, dim=1)

    # 计算相似度矩阵 (cosine similarity)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # 创建 mask，使同类样本为1，不同类样本为0
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # mask 去掉自己与自己的对比
    mask = mask - torch.eye(batch_size, device=device)

    # 计算对比损失
    exp_sim = torch.exp(similarity_matrix) * (1 - torch.eye(batch_size, device=device))  # 排除自己
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # 只保留同类的对比项
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # Loss = -mean over batch
    loss = -mean_log_prob_pos.mean()
    return loss

