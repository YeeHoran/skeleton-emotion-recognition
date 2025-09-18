import torch
import torch.nn as nn
import torch.nn.functional as F

class CACMemoryBank(nn.Module):
    def __init__(self, feat_dim, bank_size=2048, r=0.05, device="cuda"):
        """
        CAC Loss with Memory Bank
        Args:
            feat_dim: 特征维度
            bank_size: memory bank 容量
            r: 邻域比例 (CAC 原始论文推荐 5%)
            device: 设备
        """
        super().__init__()
        self.bank_size = bank_size
        self.feat_dim = feat_dim
        self.r = r
        self.device = device

        # 初始化 memory bank
        self.register_buffer("features", torch.zeros(bank_size, feat_dim, device=device))
        self.register_buffer("labels", torch.zeros(bank_size, dtype=torch.long, device=device))
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def update_bank(self, new_feats, new_labels):
        """更新 memory bank"""
        n = new_feats.size(0)
        if n > self.bank_size:
            new_feats = new_feats[:self.bank_size]
            new_labels = new_labels[:self.bank_size]
            n = self.bank_size

        # 循环写入
        end = min(self.ptr + n, self.bank_size)
        self.features[self.ptr:end] = new_feats[:end - self.ptr]
        self.labels[self.ptr:end] = new_labels[:end - self.ptr]

        if end - self.ptr < n:  # wrap around
            remain = n - (end - self.ptr)
            self.features[0:remain] = new_feats[-remain:]
            self.labels[0:remain] = new_labels[-remain:]

        self.ptr = (self.ptr + n) % self.bank_size
        if self.ptr == 0:
            self.full = True

    def forward(self, inputs, ground_truth):
        """
        CAC Loss with memory bank
        Args:
            inputs: 当前 batch embeddings, shape (N, d)
            ground_truth: 当前 batch labels, shape (N,)
        Returns:
            loss: CAC loss (scalar)
        """
        device = inputs.device
        N = inputs.size(0)

        # 归一化
        norm_inputs = F.normalize(inputs, dim=1)

        # memory bank 有效长度
        bank_len = self.bank_size if self.full else self.ptr
        if bank_len == 0:  # 如果 bank 还空，就不算 loss
            return torch.tensor(0.0, device=device, requires_grad=True)

        bank_feats = F.normalize(self.features[:bank_len], dim=1)
        bank_labels = self.labels[:bank_len]

        # 拼接当前 batch + memory bank
        all_feats = torch.cat([norm_inputs, bank_feats], dim=0)  # (N+M, d)
        all_labels = torch.cat([ground_truth, bank_labels], dim=0)  # (N+M,)

        # 计算距离矩阵
        sim_matrix = torch.matmul(norm_inputs, all_feats.T)  # (N, N+M)
        dist_matrix = 1 - sim_matrix  # cosine distance

        # 排除自身
        dist_matrix[:, :N].fill_diagonal_(float("inf"))

        # 邻居数量
        k = max(1, int(all_feats.size(0) * self.r))

        # 找最近邻
        _, nn_idx = torch.topk(dist_matrix, k, largest=False)  # (N, k)

        # 取邻居标签
        neighbor_labels = all_labels[nn_idx]  # (N, k)
        expanded_labels = ground_truth.unsqueeze(1).expand_as(neighbor_labels)
        same_class = (neighbor_labels == expanded_labels).float()

        # CAC = 平均同类比例
        consistency_per_sample = same_class.mean(dim=1)
        cac = consistency_per_sample.mean()

        # Loss = 1 - CAC
        loss = 1 - cac

        # 更新 memory bank (detach 避免梯度流入)
        self.update_bank(inputs.detach(), ground_truth.detach())

        return loss
