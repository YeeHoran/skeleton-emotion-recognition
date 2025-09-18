import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConProtoLoss(nn.Module):
    """
    多分类严重不平衡场景下的 Supervised Contrastive + Prototype Loss
    - 多数类 (Angry) 使用 NT-Xent 保持均匀
    - 少数类 (Neutral/Happy/Sad) 使用 SupCon + 原型对齐
    - 原型条件吸引：f(x)·p_c <= 0.5 才计算 proto loss
    """
    def __init__(self, num_classes=4, feature_dim=512, temperature_major=0.5, temperature_minor=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature_major = temperature_major  # NT-Xent 温度，多数类
        self.temperature_minor = temperature_minor  # SupCon 温度，少数类

        # 初始化原型向量，单位球面
        proto_vectors = torch.randn(num_classes, feature_dim)
        proto_vectors = F.normalize(proto_vectors, dim=1)
        self.register_parameter('prototypes', nn.Parameter(proto_vectors))

        # class id 映射，默认顺序：Angry=0, Neutral=1, Happy=2, Sad=3
        self.class_map = {0: 'Angry', 1: 'Neutral', 2: 'Happy', 3: 'Sad'}

    def nt_xent_loss(self, features):
        """
        NT-Xent 无监督对比损失
        features: [N, D]
        """
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.t()) / self.temperature_major
        labels = torch.arange(features.size(0), device=features.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def supcon_loss(self, features, labels):
        """
        SupCon 有监督对比损失
        features: [N, D]
        labels: [N]
        """
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.t()) / self.temperature_minor

        # mask: label 相同为 1，否则 0
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask = mask.float()
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(features.size(0), device=features.device))
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = loss.mean()
        return loss

    def proto_loss(self, features, labels):
        """
        Prototype alignment loss
        - 条件：f(x)·p_c <= 0.5
        """
        features = F.normalize(features, dim=1)  # [N, D]
        proto_vectors = F.normalize(self.prototypes, dim=1)  # [num_classes, D]

        # 确保 labels 是 LongTensor，且 shape = [N]
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()

        # 为每个样本取对应类别原型
        proto_for_samples = proto_vectors[labels]  # [N, D]

        sim = (features * proto_for_samples).sum(dim=1)  # [N]

        mask = sim <= 0.5  # 条件吸引
        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        loss = 1 - sim[mask]  # 拉近距离
        return loss.mean()

    def forward(self, features, labels):
        """
        features: [N, D]
        labels: [N] 或 [N,1]，类别索引 0~3
        """
        features = F.normalize(features, dim=1)  # [N, D]

        # 处理 labels 形状，保证为 [N] LongTensor
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()

        loss_total = torch.tensor(0.0, device=features.device)

        # ---- 多数类 Angry ----
        mask_angry = labels == 0
        if mask_angry.sum() > 1:
            loss_angry = self.nt_xent_loss(features[mask_angry])
            loss_total += loss_angry

        # ---- 少数类 Neutral / Happy / Sad ----
        for c in [1, 2, 3]:
            mask_c = labels == c
            if mask_c.sum() > 1:
                feat_c = features[mask_c]
                label_c = labels[mask_c]

                # SupCon 有监督对比损失
                loss_c = self.supcon_loss(feat_c, label_c)

                # Prototype 对齐
                proto_vectors = F.normalize(self.prototypes, dim=1)  # [num_classes, D]
                proto_for_samples = proto_vectors[label_c]  # [N_c, D]

                sim = (feat_c * proto_for_samples).sum(dim=1)  # [N_c]
                mask_proto = sim <= 0.5
                if mask_proto.sum() > 0:
                    loss_proto = 1 - sim[mask_proto]
                    loss_proto = loss_proto.mean()
                else:
                    loss_proto = torch.tensor(0.0, device=features.device)

                loss_total += loss_c + loss_proto

        return loss_total

