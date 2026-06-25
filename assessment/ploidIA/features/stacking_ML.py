import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────────────────────
# Aggregation strategies
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_mean(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean pool over valid frames. Shape: (embed_dim,)"""
    valid = x[mask]
    return valid.mean(dim=0).numpy()


def aggregate_mean_std(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean + std over valid frames. Shape: (2 * embed_dim,)"""
    valid = x[mask]
    mean  = valid.mean(dim=0)
    std   = valid.std(dim=0) if valid.shape[0] > 1 else torch.zeros_like(mean)
    return torch.cat([mean, std]).numpy()


def aggregate_4stat(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean + std + min + max. Shape: (4 * embed_dim,)"""
    valid = x[mask]
    mean  = valid.mean(dim=0)
    std   = valid.std(dim=0) if valid.shape[0] > 1 else torch.zeros_like(mean)
    vmin  = valid.min(dim=0).values
    vmax  = valid.max(dim=0).values
    return torch.cat([mean, std, vmin, vmax]).numpy()


def aggregate_concat_pca(
    X: list[torch.Tensor],
    masks: list[torch.Tensor],
    n_components: int = 128,
    fit: bool = True,
    pca: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """
    Concatenate (flatten) all frames, then reduce with PCA.
    NOTE: all videos must have the same number of valid frames T.
    Call with fit=False and pass the train PCA for val set.
    """
    rows = []
    for x, mask in zip(X, masks):
        valid = x[mask]
        rows.append(valid.reshape(-1).numpy())   # (T*D,)

    mat = np.stack(rows)                         # (N, T*D)
    if fit:
        pca = PCA(n_components=min(n_components, mat.shape[0] - 1, mat.shape[1]))
        mat = pca.fit_transform(mat)
    else:
        mat = pca.transform(mat)
    return mat, pca


def aggregate_weighted_mean(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Weighted mean where each frame's weight is the average cosine similarity
    with its neighbours, giving more importance to frames that are consistent
    with the surrounding context.

    Frame 0 is assigned the weight of the first pair (A[0], B[0]).
    Frame T-1 is assigned the weight of the last pair (A[T-2], B[T-1]).
    All other frames are assigned the mean of the two adjacent pair weights.
    """
    valid = x[mask]
    T = valid.shape[0]
    if T == 0:
        return np.zeros(x.shape[-1])            # FIX: use x.shape[-1] not valid.shape[1]
    if T == 1:
        return valid[0].numpy()                 # FIX: was returning a Tensor, not ndarray

    A = valid[:-1]                              # frames 0..T-2
    B = valid[1:]                               # frames 1..T-1
    pair_sim = 1 - F.cosine_similarity(A, B, dim=1) # (T-1,)  similarity of each consecutive pair

    # Assign a weight to every frame (including frame 0 and frame T-1)
    # FIX: original code dropped frame 0 entirely by only summing over B
    w_first = pair_sim[:1]                      # weight for frame 0  = sim(0,1)
    w_last  = pair_sim[-1:]                     # weight for frame T-1 = sim(T-2,T-1)
    w_mid   = (pair_sim[:-1] + pair_sim[1:]) / 2  # mean of adjacent pair weights
    weights = torch.cat([w_first, w_mid, w_last])  # (T,)

    weighted_sum = (valid * weights[:, None]).sum(dim=0)
    return (weighted_sum / (weights.sum() + 1e-8)).numpy()



def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    ang = pos / torch.pow(10_000, (2 * (dim // 2)) / d_model)
    ang[:, 0::2] = torch.sin(ang[:, 0::2])
    ang[:, 1::2] = torch.cos(ang[:, 1::2])
    return ang                


def aggregate_pos_encoding(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Position-weighted mean: each valid frame is scaled element-wise by its
    sinusoidal positional encoding before averaging.

    FIX 1: was ignoring the mask and processing padding frames.
    FIX 2: pos_enc[:, None] broadcast to (T, 1, D) causing (T, T, D) output;
            pos_enc has shape (T, D) so it multiplies directly with valid (T, D).
    """
    valid = x[mask]                         
    T, D = valid.shape
    if T == 0:
        return np.zeros(D)
    pos_enc = sinusoidal_encoding(T, D)     
    weighted = valid * pos_enc              
    return weighted.mean(dim=0).numpy()