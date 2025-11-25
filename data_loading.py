import logging
import os
import torch
import random
import pickle
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch_geometric.data import Dataset, Batch
from typing import Optional, List, Dict
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import json

class GlobalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 32,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            max_regions: int = 5,
            emb_dim: int = 128,
            seed: int = 42,
            cwe_mode: bool = False,
            max_global_lines: int = 4096,
            drop_long_global: bool = True,
            save_lists_dir: Optional[str] = None,
            cross_data_path: Optional[str] = None,
            cross_test_ratio: float = 1.0
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_regions = max_regions
        self.emb_dim = emb_dim
        self.seed = seed
        self.cwe_mode    = cwe_mode

        self.max_global_lines = max_global_lines
        self.drop_long_global = drop_long_global

        self.global_train = None
        self.global_val = None
        self.global_test = None
        self.contrast_train = None
        self.contrast_val = None

        self.save_lists_dir = save_lists_dir

        self.train_file_list: List[str] = []
        self.val_file_list: List[str] = []
        self.test_file_list: List[str] = []

        self.cross_data_path = cross_data_path
        self.cross_test_ratio = cross_test_ratio

    def get_train_filenames(self) -> List[str]:
        return list(self.train_file_list)

    def get_val_filenames(self) -> List[str]:
        return list(self.val_file_list)

    def get_test_filenames(self) -> List[str]:
        return list(self.test_file_list)

    def prepare_data(self):
        if self.cwe_mode:
            if not os.path.isdir(self.data_path):
                raise FileNotFoundError(f"In CWE mode, specify a directory containing multiple .pkl files instead of a single file: {self.data_path}")
        else:
            if not os.path.isfile(self.data_path):
                raise FileNotFoundError(f"Data file does not exist: {self.data_path}")

        if self.cross_data_path is not None:
            if not os.path.isfile(self.cross_data_path):
                raise FileNotFoundError(f"Cross-dataset test data file does not exist: {self.cross_data_path}")

    def setup(self, stage: str = None):
        """
        Case A : Provide `cross_data_path`

        - Split train/val from `data_path`

        - Randomly sample `cross_test_ratio` from `cross_data_path` as test

        Case B: Do not provide `cross_data_path` (Keep original logic)

        - Randomly split train/val/test from a single dataset using an 8:1:1 ratio or a custom ratio

        Case C: `cwe_mode=True` (Multiple pkl directories)

        - Keep your previous CWE splitting logic; specifying `cross_data_path` will result in an error (to avoid semantic confusion)
        """
        # CWE mode:
        if self.cwe_mode:
            if self.cross_data_path is not None:
                raise ValueError("The current implementation does not support the simultaneous use of cwe_mode and cross_data_path. Please enable only one of them.")
            train_list, val_list, test_list = [], [], []
            g = torch.Generator().manual_seed(self.seed)

            for fname in sorted(os.listdir(self.data_path)):
                rank_zero_info(f"Loading {fname}")
                if not fname.endswith(".pkl"):
                    continue
                full_path = os.path.join(self.data_path, fname)
                with open(full_path, "rb") as f:
                    data_i = pickle.load(f)

                n = len(data_i)
                n_train = int(self.train_ratio * n)
                n_val   = int(self.val_ratio   * n)
                n_test  = n - n_train - n_val

                subt = random_split(data_i, [n_train, n_val, n_test], generator=g)
                train_list.extend([subt[0].dataset[i] for i in subt[0].indices])
                val_list.extend([subt[1].dataset[i] for i in subt[1].indices])
                test_list.extend([subt[2].dataset[i] for i in subt[2].indices])

            self.global_train = train_list
            self.global_val   = val_list
            self.global_test  = test_list

            rank_zero_info(f"[CWE] train={len(self.global_train)}  val={len(self.global_val)}  test={len(self.global_test)}")
            return

        # Non-CWE mode:
        # 1) First load data_path for train/val
        with open(self.data_path, "rb") as f:
            dbd_full = pickle.load(f)

        # 2) The source of the test is determined by whether cross_data_path is provided.
        if self.cross_data_path is not None:
            # —— Situation A: data_path -> train/val; cross_data_path -> test ——
            total = len(dbd_full)
            n_train = int(self.train_ratio * total)
            n_val   = total - n_train
            g = torch.Generator().manual_seed(self.seed)
            dbd_train, dbd_val = random_split(dbd_full, [n_train, n_val], generator=g)
            self.global_train, self.global_val = dbd_train, dbd_val

            with open(self.cross_data_path, "rb") as f:
                reveal_full = pickle.load(f)
            n_reveal = len(reveal_full)
            n_test = max(1, int(round(self.cross_test_ratio * n_reveal)))

            rng = random.Random(self.seed)
            idxs = list(range(n_reveal))
            rng.shuffle(idxs)
            pick = idxs[:n_test]
            self.global_test = [reveal_full[i] for i in pick]

        else:
            # —— Case B: Tripartite division of a single dataset
            self._safe_data_split(dbd_full)
            rank_zero_info(f"[Single] train={len(self.global_train)}  val={len(self.global_val)}  test={len(self.global_test)}")

    def _safe_data_split(self, full_dataset):
        """Perform hierarchical data partitioning"""
        # Global partition (8:1:1)
        total = len(full_dataset)
        train_size = int(self.train_ratio * total)
        val_size = int(self.val_ratio * total)
        test_size = total - train_size - val_size

        # Fixed random seed guarantees reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        self.global_train, self.global_val, self.global_test = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

    def _extract_filenames(self, subset) -> List[str]:
        """
        Extract the file_name field of the sample from the subset.
        The subset can be a list or torch.utils.data.Subset.
        """
        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            items = [subset.dataset[i] for i in subset.indices]
        else:
            items = list(subset) if subset is not None else []

        names = []
        for s in items:
            fn = getattr(s, "file_name", None)
            if fn is not None:
                names.append(str(fn))
        return names

    def _save_split_lists(self, out_dir: str):
        """
        Save the filename list of the three groups train/val/test as .txt and .json to prepare for the subsequent perturbation test set.
        """
        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "train": self.train_file_list,
            "val": self.val_file_list,
            "test": self.test_file_list,
        }

        for split, lst in payload.items():
            with open(os.path.join(out_dir, f"{split}_filenames.txt"), "w", encoding="utf-8") as f:
                for fn in lst:
                    f.write(fn + "\n")

        with open(os.path.join(out_dir, "split_filenames.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rank_zero_info(f"Saved filename lists to: {out_dir}")

    def _create_weighted_sampler(self, subset):
        labels = [1 if p.y == 1 else 0 for p in subset]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = [class_weights[label] for label in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(
            self.global_train,
            batch_size=self.batch_size,
            sampler=self._create_weighted_sampler(self.global_train),
            shuffle=False,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.global_val,
            batch_size=self.batch_size,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.global_test,
            batch_size=self.batch_size,
            collate_fn=self._classification_collate,
            num_workers=8
        )

    def _classification_collate(self, batch):
        try:
            import psutil
            mem = psutil.virtual_memory()
            # print(f"Memory usage: {mem.percent}% | Available: {mem.available / 1024 ** 3:.1f}GB")

            return self._build_cls_batch(batch)
        except Exception as e:
            print(f"Collate error: {str(e)}")
            torch.save(batch, "error_batch.pt")
            raise

    def _build_cls_batch(self, samples):
        max_regions = int(self.max_regions)
        d = int(self.emb_dim)
        L_max = int(getattr(self, "max_global_lines", 4096))

        B = len(samples)

        # ====== Statistical Padding Scale ======

        # Maximum Length of Regional Nodes (AST/CFG/PDG)
        max_ast_region_nodes = 0
        max_cfg_region_nodes = 0
        max_pdg_region_nodes = 0
        max_region_lines = 0
        max_attr_dim = 0

        for s in samples:
            for t in getattr(s, "region_ast_nodes_list", []):
                max_ast_region_nodes = max(max_ast_region_nodes, int(t.size(0)))
            for t in getattr(s, "region_cfg_nodes_list", []):
                max_cfg_region_nodes = max(max_cfg_region_nodes, int(t.size(0)))
            for t in getattr(s, "region_pdg_nodes_list", []):
                max_pdg_region_nodes = max(max_pdg_region_nodes, int(t.size(0)))
            for t in getattr(s, "region_line_numbers_lists", []):
                max_region_lines = max(max_region_lines, int(t.size(0)))
            if hasattr(s, "region_attr") and s.region_attr is not None and s.region_attr.ndim == 2:
                max_attr_dim = max(max_attr_dim, int(s.region_attr.size(1)))

        ast_x_list = []
        token_eis, stmt_eis, block_eis = [], [], []
        ast_batch_list = []

        cfg_x_list, cfg_eis, cfg_batch_list = [], [], []
        pdg_x_list, pdg_eis, pdg_batch_list = [], [], []

        ast_region_lists = []
        cfg_region_lists = []
        pdg_region_lists = []
        region_line_lists = []
        region_score_lists = []
        file_names = []
        ys = []

        ast_node_offset = 0
        cfg_node_offset = 0
        pdg_node_offset = 0

        for i, s in enumerate(samples):
            ast_x_list.append(s.ast_x)
            N_ast = int(s.ast_x.size(0))

            if s.token_edge_index.numel() > 0:
                token_eis.append(s.token_edge_index + ast_node_offset)
            if s.stmt_edge_index.numel() > 0:
                stmt_eis.append(s.stmt_edge_index + ast_node_offset)
            if s.block_edge_index.numel() > 0:
                block_eis.append(s.block_edge_index + ast_node_offset)

            ast_batch_list.append(torch.full((N_ast,), i, dtype=torch.long))
            ast_node_offset += N_ast

            if hasattr(s, "cfg_x") and s.cfg_x is not None:
                x = s.cfg_x
                ei = s.cfg_edge_index
                cfg_x_list.append(x)
                cfg_eis.append(ei + cfg_node_offset if ei.numel() > 0 else ei)
                cfg_batch_list.append(torch.full((x.size(0),), i, dtype=torch.long))
                cfg_node_offset += int(x.size(0))
            else:
                cfg_x_list.append(torch.empty((0, d)))
                cfg_eis.append(torch.empty((2, 0), dtype=torch.long))
                cfg_batch_list.append(torch.empty((0,), dtype=torch.long))

            if hasattr(s, "pdg_x") and s.pdg_x is not None:
                x = s.pdg_x
                ei = s.pdg_edge_index
                pdg_x_list.append(x)
                pdg_eis.append(ei + pdg_node_offset if ei.numel() > 0 else ei)
                pdg_batch_list.append(torch.full((x.size(0),), i, dtype=torch.long))
                pdg_node_offset += int(x.size(0))
            else:
                pdg_x_list.append(torch.empty((0, d)))
                pdg_eis.append(torch.empty((2, 0), dtype=torch.long))
                pdg_batch_list.append(torch.empty((0,), dtype=torch.long))

            ast_region_lists.append(getattr(s, "region_ast_nodes_list", []))
            cfg_region_lists.append(getattr(s, "region_cfg_nodes_list", []))
            pdg_region_lists.append(getattr(s, "region_pdg_nodes_list", []))
            region_line_lists.append(getattr(s, "region_line_numbers_lists", []))

            region_score_lists.append(getattr(s, "region_score_gt", torch.zeros(0)))
            ys.append(s.y)
            file_names.append(s.file_name)

        ast_x = torch.cat(ast_x_list, dim=0) if len(ast_x_list) > 0 else torch.empty((0, d))
        token_edge_index = (torch.cat(token_eis, dim=1)
                            if len(token_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        stmt_edge_index = (torch.cat(stmt_eis, dim=1)
                           if len(stmt_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        block_edge_index = (torch.cat(block_eis, dim=1)
                            if len(block_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        ast_batch = torch.cat(ast_batch_list, dim=0) if len(ast_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        cfg_x = torch.cat(cfg_x_list, dim=0) if len(cfg_x_list) > 0 else torch.empty((0, d))
        cfg_edge_index = (torch.cat(cfg_eis, dim=1)
                          if len(cfg_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        cfg_batch = torch.cat(cfg_batch_list, dim=0) if len(cfg_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        pdg_x = torch.cat(pdg_x_list, dim=0) if len(pdg_x_list) > 0 else torch.empty((0, d))
        pdg_edge_index = (torch.cat(pdg_eis, dim=1)
                          if len(pdg_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        pdg_batch = torch.cat(pdg_batch_list, dim=0) if len(pdg_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        region_score = torch.zeros((B, max_regions), dtype=torch.float32)
        for i, s in enumerate(region_score_lists):
            L = min(int(s.size(0)), max_regions)
            if L > 0:
                region_score[i, :L] = s[:L]

        ast_offsets = []
        c = 0
        for s in samples:
            ast_offsets.append(c)
            c += int(s.ast_x.size(0))

        cfg_offsets, pdg_offsets = [], []
        c = 0
        for s in samples:
            cfg_offsets.append(c)
            c += int(s.cfg_x.size(0)) if hasattr(s, "cfg_x") and s.cfg_x is not None else 0
        c = 0
        for s in samples:
            pdg_offsets.append(c)
            c += int(s.pdg_x.size(0)) if hasattr(s, "pdg_x") and s.pdg_x is not None else 0

        ast_region_nodes, ast_region_mask, ast_node_mask = self._pad_2d_indices(
            [ast_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_ast_region_nodes if max_ast_region_nodes > 0 else 0,
            offset=0
        )
        cfg_region_nodes, cfg_region_mask, cfg_node_mask = self._pad_2d_indices(
            [cfg_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_cfg_region_nodes if max_cfg_region_nodes > 0 else 0,
            offset=0
        )
        pdg_region_nodes, pdg_region_mask, pdg_node_mask = self._pad_2d_indices(
            [pdg_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_pdg_region_nodes if max_pdg_region_nodes > 0 else 0,
            offset=0
        )

        for i in range(B):
            if ast_region_nodes.size(2) > 0 and ast_region_mask[i].any():
                ast_region_nodes[i, ast_region_mask[i]] += ast_offsets[i]
            if cfg_region_nodes.size(2) > 0 and cfg_region_mask[i].any():
                cfg_region_nodes[i, cfg_region_mask[i]] += cfg_offsets[i]
            if pdg_region_nodes.size(2) > 0 and pdg_region_mask[i].any():
                pdg_region_nodes[i, pdg_region_mask[i]] += pdg_offsets[i]

        F = int(max_attr_dim)
        if F > 0:
            region_attr = torch.zeros((B, max_regions, F), dtype=torch.float32)
        else:
            region_attr = torch.empty((B, max_regions, 0), dtype=torch.float32)

        region_attr_mask = ast_region_mask.clone() if ast_region_mask.numel() > 0 else torch.empty((B, max_regions), dtype=torch.bool)

        for i, s in enumerate(samples):
            if not hasattr(s, "region_attr") or s.region_attr is None or s.region_attr.numel() == 0:
                continue
            attr_i = s.region_attr
            Ri = int(attr_i.size(0))
            Fi = int(attr_i.size(1)) if attr_i.ndim == 2 else 0
            L = min(Ri, max_regions)
            C = min(F, Fi)
            if L > 0 and C > 0:
                region_attr[i, :L, :C] = attr_i[:L, :C]

        region_line_numbers, region_line_region_mask, region_line_mask = self._pad_2d_lines(
            [region_line_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_lines=max_region_lines if max_region_lines > 0 else 0
        )

        global_embeddings, global_mask = self._pad_global_embeddings(
            [s.global_code_embedding for s in samples],
            pad_lines=L_max,
            d=d
        )

        y = torch.cat(ys, dim=0) if len(ys) > 0 else torch.empty((0,), dtype=torch.long)

        batch = Batch(
            ast_x=ast_x,
            num_nodes=ast_x.size(0),
            token_edge_index=token_edge_index,
            stmt_edge_index=stmt_edge_index,
            block_edge_index=block_edge_index,
            ast_batch=ast_batch,

            y=y,
            file_name=file_names,

            global_code_embeddings=global_embeddings,  # (B, L_max, d)
            global_emb_mask=global_mask,  # (B, L_max)

            region_ast_nodes=ast_region_nodes,  # (B, K, MaxAstNodes)
            region_ast_region_mask=ast_region_mask,  # (B, K)
            region_ast_node_mask=ast_node_mask,  # (B, K, MaxAstNodes)

            cfg_x=cfg_x,
            cfg_edge_index=cfg_edge_index,
            cfg_batch=cfg_batch,
            region_cfg_nodes=cfg_region_nodes,  # (B, K, MaxCfgNodes)
            region_cfg_region_mask=cfg_region_mask,  # (B, K)
            region_cfg_node_mask=cfg_node_mask,  # (B, K, MaxCfgNodes)

            pdg_x=pdg_x,
            pdg_edge_index=pdg_edge_index,
            pdg_batch=pdg_batch,
            region_pdg_nodes=pdg_region_nodes,  # (B, K, MaxPdgNodes)
            region_pdg_region_mask=pdg_region_mask,  # (B, K)
            region_pdg_node_mask=pdg_node_mask,  # (B, K, MaxPdgNodes)

            region_line_numbers=region_line_numbers,  # (B, K, MaxLines)
            region_line_mask=region_line_mask,  # (B, K, MaxLines)
            region_score=region_score,  # (B, K)

            region_attr=region_attr,  # (B, K, F)
            region_attr_mask=region_attr_mask,  # (B, K)

        )
        batch.num_graphs = batch.y.size(0)
        return batch

    def _pad_2d_indices(self, lists_per_sample, max_regions, pad_nodes, offset=0):
        """
        Packs "List[List[Tensor_varlen]]" into (B, max_regions, pad_nodes) and returns a mask:

        - idx_tensor: LongTensor[B, max_regions, pad_nodes] (fill empty spaces with 0)

        - region_mask: BoolTensor[B, max_regions] (whether the region exists)

        - node_mask: BoolTensor[B, max_regions, pad_nodes] (whether the position is a valid node)

        lists_per_sample: Length B, each element is a List[Tensor(ni,)] or a List[list[int]]

        offset: +offset for all indices (used for node offset after cross-sample concatenation)
        """
        B = len(lists_per_sample)
        idx_tensor = torch.zeros((B, max_regions, pad_nodes), dtype=torch.long)
        region_mask = torch.zeros((B, max_regions), dtype=torch.bool)
        node_mask = torch.zeros((B, max_regions, pad_nodes), dtype=torch.bool)

        for i, region_list in enumerate(lists_per_sample):
            region_list = region_list[:max_regions]
            for r, ids in enumerate(region_list):
                if isinstance(ids, torch.Tensor):
                    ids = ids.detach().cpu().tolist()
                region_mask[i, r] = True
                length = min(len(ids), pad_nodes)
                if length > 0:
                    idx_tensor[i, r, :length] = torch.tensor(ids[:length], dtype=torch.long) + offset
                    node_mask[i, r, :length] = True
        return idx_tensor, region_mask, node_mask

    def _pad_2d_lines(self, lists_per_sample, max_regions, pad_lines):
        """
        Pack “List[List[Tensor_varlen_lines]]” into (B, max_regions, pad_lines) + mask for use in region_line_numbers_lists.
        """
        B = len(lists_per_sample)
        line_tensor = torch.zeros((B, max_regions, pad_lines), dtype=torch.long)
        region_mask = torch.zeros((B, max_regions), dtype=torch.bool)
        line_mask = torch.zeros((B, max_regions, pad_lines), dtype=torch.bool)

        for i, region_list in enumerate(lists_per_sample):
            region_list = region_list[:max_regions]
            for r, lines in enumerate(region_list):
                if isinstance(lines, torch.Tensor):
                    lines = lines.detach().cpu().tolist()
                region_mask[i, r] = True
                length = min(len(lines), pad_lines)
                if length > 0:
                    line_tensor[i, r, :length] = torch.tensor(lines[:length], dtype=torch.long)
                    line_mask[i, r, :length] = True
        return line_tensor, region_mask, line_mask

    def _pad_global_embeddings(self, emb_list, pad_lines, d):
        """
        emb_list: List[Tensor(num_lines, d)]
        return：
          - global_embeddings: FloatTensor[B, pad_lines, d]
          - global_mask:       BoolTensor[B, pad_lines]
        """
        B = len(emb_list)
        out = torch.zeros((B, pad_lines, d), dtype=emb_list[0].dtype if B > 0 else torch.float32)
        mask = torch.zeros((B, pad_lines), dtype=torch.bool)
        for i, e in enumerate(emb_list):
            if e is None or e.numel() == 0:
                continue
            L = min(int(e.size(0)), pad_lines)
            out[i, :L] = e[:L]
            mask[i, :L] = True
        return out, mask