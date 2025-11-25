import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv

class HierAST(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 stopgrad_between_levels=False):
        super().__init__()
        self.tok = GCNConv(in_dim, hid_dim)
        self.stm = GCNConv(in_dim, hid_dim)
        self.blk = GCNConv(in_dim, hid_dim)

        self.n_t = nn.LayerNorm(hid_dim)
        self.n_s = nn.LayerNorm(hid_dim)
        self.n_b = nn.LayerNorm(hid_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.stopgrad_between_levels = stopgrad_between_levels

    def forward(self, x, token_ei, stmt_ei, block_ei):
        xt = self.n_t(F.relu(self.tok(x, token_ei)))
        x1 = xt.detach() if self.stopgrad_between_levels else xt
        xs = self.n_s(F.relu(self.stm(x1, stmt_ei)))
        x2 = xs.detach() if self.stopgrad_between_levels else xs
        xb = self.n_b(F.relu(self.blk(x2, block_ei)))
        h = torch.cat([xt, xs, xb], dim=-1)
        out = self.out_proj(h)
        return out  # [N_ast_total, D]

class GGNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_steps=6, out_dim=None):
        super().__init__()
        self.input = nn.Linear(in_dim, hid_dim) if in_dim != hid_dim else nn.Identity()
        self.ggnn = GatedGraphConv(out_channels=hid_dim, num_layers=n_steps)
        self.norm = nn.LayerNorm(hid_dim)
        self.out = nn.Identity() if (out_dim is None or out_dim == hid_dim) else nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.input(x)
        x = self.ggnn(x, edge_index)
        x = self.norm(F.relu(x))
        return self.out(x)

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, heads=4, out_dim=None):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hid_dim) if in_dim != hid_dim else nn.Identity()
        self.gat1 = GATConv(hid_dim, hid_dim // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hid_dim, hid_dim // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.out = nn.Identity() if (out_dim is None or out_dim == hid_dim) else nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.proj_in(x)
        x = self.norm1(F.elu(self.gat1(x, edge_index)))
        x = self.norm2(F.elu(self.gat2(x, edge_index)))
        return self.out(x)

class AutoMaskGenerator(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, emb):  # [B, L, D]
        norms = torch.norm(emb, p=2, dim=-1)   # [B, L]
        mask = norms < self.eps               # True=pad
        all_pad = mask.all(dim=1)
        mask[all_pad, 0] = False
        return mask

def masked_region_pool(node_feat, region_nodes, node_mask, reduce="mean"):
    B, K, M = region_nodes.shape
    D = node_feat.size(-1)
    safe_idx = torch.where(node_mask, region_nodes, torch.zeros_like(region_nodes))
    flat = safe_idx.view(-1)
    picked = node_feat[flat].view(B, K, M, D)
    picked = picked * node_mask[..., None].to(picked.dtype)
    cnt = node_mask.sum(dim=2, keepdim=True).clamp(min=1)
    if reduce == "mean":
        pooled = picked.sum(dim=2) / cnt
    elif reduce == "max":
        neg_inf = torch.finfo(picked.dtype).min
        masked = picked.masked_fill(~node_mask[..., None], neg_inf)
        pooled = masked.max(dim=2).values
    else:
        raise ValueError("reduce must be mean/max")
    return pooled  # [B, K, D]

def pool_region_lines(encoded_lines, region_line_indices, region_line_mask):
    B, L, D = encoded_lines.shape
    _, K, l = region_line_indices.shape
    safe_idx = torch.where(region_line_mask, region_line_indices, torch.zeros_like(region_line_indices))
    safe_idx = safe_idx.clamp(min=0, max=L-1)
    b_idx = torch.arange(B, device=encoded_lines.device)[:, None, None].expand(B, K, l)
    gathered = encoded_lines[b_idx, safe_idx]  # [B, K, l, D]
    gathered = gathered * region_line_mask[..., None].to(gathered.dtype)
    cnt = region_line_mask.sum(dim=2, keepdim=True).clamp(min=1)
    return gathered.sum(dim=2) / cnt  # [B, K, D]

class VulDetectionModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, heads=4, ggnn_steps=6, attr_dim: int = 0,
                 topk_regions: int = 12, topk_pooling: str = "weighted_mean"):
        """
        `attr_dim`: The dimension (F) of the region attribute feature. If 0 or not provided, `region_attr` is not used.

        `topk_regions`: The K value for selective aggregation (at most K regions per sample).

        `topk_pooling`: 'mean' or 'weighted_mean' (weighted average using region scores).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attr_dim = int(attr_dim)
        self.topk = int(topk_regions)
        assert topk_pooling in ("mean", "weighted_mean")
        self.topk_pooling = topk_pooling

        self.ast_enc  = HierAST(in_dim=input_dim, hid_dim=hidden_dim, out_dim=hidden_dim)
        self.cfg_enc  = GGNNEncoder(in_dim=input_dim, hid_dim=hidden_dim, n_steps=ggnn_steps, out_dim=hidden_dim)
        self.pdg_enc  = GATEncoder(in_dim=input_dim, hid_dim=hidden_dim, heads=heads, out_dim=hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=4*hidden_dim,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.line_masker = AutoMaskGenerator()
        self.sem_enc = nn.TransformerEncoder(enc_layer, num_layers=3)

        if self.attr_dim > 0:
            self.attr_enc = nn.Sequential(
                nn.Linear(self.attr_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            fuse_in = hidden_dim * 5  # ast/cfg/pdg/sem/attr
        else:
            self.attr_enc = None
            fuse_in = hidden_dim * 4  # ast/cfg/pdg/sem

        self.region_fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.region_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        """
        返回：
        - region_logits: [B, K]

        - region_mask: [B, K] bool

        - region_feat: [B, K, D]

        - cls_logits: [B] (Graph-level classification logits)

        - topk_indices: [B, K_sel] The index of the selected region for each sample (K_sel = min(K, number of valid regions)). If less than K, padded with -1.

        - topk_scores: [B, K_sel] The score of the selected region (after sigmoid), 0 for regions less than K.
        """
        # ===== 1) AST → Region Pooling =====
        ast_h = self.ast_enc(data.ast_x, data.token_edge_index, data.stmt_edge_index, data.block_edge_index)
        ast_r = masked_region_pool(ast_h, data.region_ast_nodes, data.region_ast_node_mask)

        # ===== 2) CFG → Region Pooling =====
        cfg_h = self.cfg_enc(data.cfg_x, data.cfg_edge_index)
        cfg_r = masked_region_pool(cfg_h, data.region_cfg_nodes, data.region_cfg_node_mask)

        # ===== 3) PDG → Region Pooling =====
        pdg_h = self.pdg_enc(data.pdg_x, data.pdg_edge_index)
        pdg_r = masked_region_pool(pdg_h, data.region_pdg_nodes, data.region_pdg_node_mask)

        # ===== 4) Line semantics → Region line number pooling =====
        lines = data.global_code_embeddings
        pad_mask = self.line_masker(lines)
        sem = self.sem_enc(lines, src_key_padding_mask=pad_mask)
        sem_r = pool_region_lines(sem, data.region_line_numbers, data.region_line_mask)

        # ===== 5) Regional Attributes =====
        feats_to_cat = [ast_r, cfg_r, pdg_r, sem_r]
        if self.attr_enc is not None and hasattr(data, "region_attr") and data.region_attr is not None and data.region_attr.numel() > 0:
            attr_proj = self.attr_enc(data.region_attr)  # [B,K,D]
            region_mask = self._get_region_mask(data)    # [B,K]
            attr_proj = attr_proj * region_mask[..., None].to(attr_proj.dtype)
            feats_to_cat.append(attr_proj)

        # ===== 6) Regional Integration =====
        region_feat = torch.cat(feats_to_cat, dim=-1)     # [B,K,4D] or [B,K,5D]
        region_feat = self.region_fuse(region_feat)       # [B,K,D]

        # ===== 7) Regional Scoring =====
        region_logits = self.region_scorer(region_feat).squeeze(-1)  # [B,K]
        region_mask   = self._get_region_mask(data)                  # [B,K] bool
        region_probs  = torch.sigmoid(region_logits)                 # [B,K] in [0,1]

        # ===== 8) Aggregation =====
        graph_feat, topk_idx, topk_scores = self._topk_pool(region_feat, region_probs, region_mask,
                                                            k=self.topk, mode=self.topk_pooling)  # [B,D], [B,K_sel], [B,K_sel]

        # ===== 9) Image-level classification =====
        cls_logits = self.graph_classifier(graph_feat).squeeze(-1)   # [B]

        return {
            "region_logits": region_logits,   # [B,K]
            "region_mask":   region_mask,     # [B,K]
            "region_feat":   region_feat,     # [B,K,D]
            "cls_logits":    cls_logits,      # [B]
            "topk_indices":  topk_idx,        # [B, K_sel]
            "topk_scores":   topk_scores,     # [B, K_sel]
        }

    def _get_region_mask(self, data):
        for name in ["region_ast_region_mask", "region_cfg_region_mask", "region_pdg_region_mask", "region_attr_mask"]:
            if hasattr(data, name) and getattr(data, name) is not None:
                return getattr(data, name).bool()
        raise ValueError("No region_*_region_mask found in batch data.")

    @staticmethod
    def _topk_pool(region_feat: torch.Tensor,
                   region_scores: torch.Tensor,
                   region_mask: torch.Tensor,
                   k: int,
                   mode: str = "mean"):
        """
        region_feat: [B,K,D]

        region_scores: [B,K] (already [0,1] scores)

        region_mask: [B,K] True=Valid

        k: Maximum number of regions selected

        mode: 'mean' or 'weighted_mean'

        Returns:

        graph_feat: [B,D]

        topk_idx: [B,K] (or K_sel), positions less than K are set to -1

        topk_vals: [B,K] (or K_sel), positions less than K are set to 0
        """
        B, K, D = region_feat.shape
        k = int(max(1, min(k, K)))

        # Set the scores of invalid regions to -inf to ensure they are not selected.
        # Note: region_scores ∈ [0,1], here it is replaced with -1e9 as an invalidation flag.
        scores = region_scores.masked_fill(~region_mask, -1e9)  # [B,K]

        # Select top-k (default is all selected)
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)   # [B,k], [B,k]

        sel_valid = topk_vals > -1e8                            # [B,k] bool

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, k, D)    # [B,k,D]
        sel_feat = torch.gather(region_feat, 1, gather_idx)     # [B,k,D]

        if mode == "weighted_mean":
            w = torch.clamp(topk_vals, min=0.0)
            w = w * sel_valid.float()                           # [B,k]
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
            graph_feat = (sel_feat * w.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]
        else:
            cnt = sel_valid.sum(dim=1, keepdim=True).clamp(min=1)        # [B,1]
            graph_feat = (sel_feat * sel_valid.unsqueeze(-1).float()).sum(dim=1) / cnt

        pad_idx = torch.where(sel_valid, topk_idx, torch.full_like(topk_idx, -1))
        pad_vals = torch.where(sel_valid, torch.clamp(topk_vals, min=0.0), torch.zeros_like(topk_vals))

        return graph_feat, pad_idx, pad_vals
