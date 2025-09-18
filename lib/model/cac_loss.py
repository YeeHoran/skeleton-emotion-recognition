import torch
import torch.nn.functional as F


def class_alignment_consistency(inputs, ground_truth, r):
    """
    Batch version of Class Alignment Consistency (CAC).

    Args:
        inputs (torch.Tensor): Embeddings, shape (N, d)
        ground_truth (torch.Tensor): Labels, shape (N,)
        r (float): Ratio of neighbors to consider (default 0.05)

    Returns:
        torch.Tensor: CAC loss (scalar)
    """
    N = inputs.size(0)
    device = inputs.device

    # Normalize embeddings
    norm_inputs = F.normalize(inputs, dim=1)

    # Compute cosine similarity (N, N)
    sim_matrix = torch.matmul(norm_inputs, norm_inputs.T)
    dist_matrix = 1 - sim_matrix  # cosine distance

    # Exclude self by setting diagonal to inf
    dist_matrix.fill_diagonal_(float("inf"))

    # Number of neighbors
    k = max(1, int(N * r))

    # Get top-k nearest neighbors for each sample
    _, nn_idx = torch.topk(dist_matrix, k, largest=False)  # (N, k)

    # Gather neighbor labels
    neighbor_labels = ground_truth[nn_idx]  # (N, k)
    expanded_labels = ground_truth.unsqueeze(1).expand_as(neighbor_labels)  # (N, k)

    # Compare labels: 1 if same, 0 otherwise
    same_class = (neighbor_labels == expanded_labels).float()

    # Per-sample consistency
    consistency_per_sample = same_class.mean(dim=1)  # (N,)

    # CAC = average consistency
    cac = consistency_per_sample.mean()

    # Loss = 1 - CAC
    loss = 1 - cac
    return loss
