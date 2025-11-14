import os
import gensim
from nltk.tokenize import RegexpTokenizer
import logging
import torch
import json
from torch_geometric.data import Data
import traceback
import pickle
import re
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from collections import Counter
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
from ordered_set import OrderedSet
import math
import argparse
from collections import defaultdict, deque
from torch_geometric.utils import subgraph as pyg_subgraph
import statistics, math, time

import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self,
                 c_dir: str,
                 js_dir: str,
                 vul_lines_path: str,
                 w2v_path: str,
                 max_regions: int,
                 seed: int = 42):
        c_regexp = (
            r'\w+|->|\+\+|--|<=|>=|==|!=|'
            r'<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|'
            r'&=|<<=|>>=|^=|\|=|::|'
            r'[!@#$%^&*()_+\-=\[\]{};\':"\|,.<>/?]'
        )
        self.tokenizer = RegexpTokenizer(c_regexp)

        self.original_c = c_dir
        self.original_js = js_dir
        self.vul_lines = vul_lines_path
        self.w2v_path = w2v_path

        if not os.path.exists(self.w2v_path):
            raise FileNotFoundError(f"Word2Vec file not found: {self.w2v_path}")
        self.word_vectors   = gensim.models.KeyedVectors.load(self.w2v_path, mmap='r')
        self.embedding_size = self.word_vectors.vector_size

        self.rng = random.Random(seed)
        self.max_regions = max_regions

    def run(self):
        print(f"Preprocessing dataset at:\n  C dir:   {self.original_c}\n  JSON dir:{self.original_js}\n  Vul lines:{self.vul_lines}\n  W2V:      {self.w2v_path}")
        data = self.build_classify_dataset()
        out = f"{self.dataset_name}-{self.max_regions}-Score.pkl"
        self.save_processed_data(data, out)
        print(f"Done! Saved to {out}")

    def build_classify_dataset(self):
        path_config = {
            "original_c": self.original_c,
            "original_js": self.original_js,
            "vul_lines": self.vul_lines
        }

        with open(path_config["vul_lines"], 'r') as f:
            vul_lines_data = json.load(f)

        dataset = []
        error_log = []
        c_files = [
            f for f in os.listdir(path_config["original_c"])
            if f.endswith(".c")
        ]

        pbar = tqdm(c_files, desc="Processing C-JSON pairs")

        for c_file in pbar:
            try:
                sample_id = c_file.split(".")[0]
                js_file = f"{sample_id}.json"
                c_path = os.path.join(path_config["original_c"], c_file)
                js_path = os.path.join(path_config["original_js"], js_file)

                if not os.path.exists(js_path):
                    raise FileNotFoundError(f"Missing JSON file: {js_file}")

                vul_lines = vul_lines_data.get(c_file, [])
                data = self.prepare_torch_graph(
                    js_path=js_path,
                    c_path=c_path,
                    vul_lines=vul_lines
                )

                dataset.append(data)

            except Exception as e:
                Traceback = traceback.format_exc()
                print(f"Traceback:{Traceback}")
                error_msg = (
                    f"[{c_file}] processing error\n"
                    f"  C path: {c_path}\n"
                    f"  JSON path: {js_path}\n"
                    f"  Error: {repr(e)}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                error_log.append(error_msg)
                continue

        self._save_error_log(error_log)

        return dataset

    def _save_error_log(self, error_log: List[str]):
        if error_log:
            with open(f"{self.dataset_name}-{self.max_regions}_error.log", "w") as f:
                f.write("\n".join(error_log))
            logger.warning(f"Encountered {len(error_log)} errors, see log file for details")

    def prepare_torch_graph(self, js_path: str, c_path: str, vul_lines) -> Data:
        with open(js_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        file_name = os.path.basename(js_path)

        # ---------------------------- Generate AST data ----------------------------
        # Read and filter AST
        ast_nodes = json_data['ast_nodes']
        ast_edges = json_data['ast_edges']
        ast_nodes, ast_edges = self.filter_nodes(ast_nodes, ast_edges)

        ast_nodes_dict = {node['id']: node for node in ast_nodes}
        # Build the adjacency list and node dictionary
        adjacency_map = defaultdict(list)
        for edge in ast_edges:
            parent, child = edge[0], edge[1]
            adjacency_map[parent].append(child)

        high_risk_ids = self.detect_high_risk_nodes(ast_nodes)
        regions = self.process_ast(ast_nodes, ast_edges, high_risk_ids, adjacency_map)
        max_depth = 2
        prev_length = -1
        while True:
            refined = self.refine_regions(regions, max_depth=max_depth)
            current_length = len(refined)
            if current_length == prev_length:
                break

            max_depth += 1
            prev_length = current_length

        node_index_lists = list(refined.values())

        region_count = len(node_index_lists)
        if region_count == 0:
            raise Exception(f"Region partitioning result of 0 was detected (AST that Joern could not successfully extract) in file {file_name}\n")

        sorted_node_index_lists = [sorted(sublist) for sublist in node_index_lists]
        if len(sorted_node_index_lists) > 1:
            last_list = sorted_node_index_lists[-1]
            sublists = self.split_continuous_sublists(last_list)
            sorted_node_index_lists = sorted_node_index_lists[:-1] + sublists

        filtered_lists = [sublist for sublist in sorted_node_index_lists if len(sublist) > 1]
        filtered_lists = self.deduplicate_nested_list(filtered_lists)
        region_line_numbers = self.line_number_extract(ast_nodes_dict, filtered_lists)

        merged_regions = self.merge_regions(filtered_lists, ast_nodes_dict, adjacency_map, region_line_numbers)

        vul_line_node_ids = self.get_vul_line_nodes(ast_nodes_dict, vul_lines)
        combined_risk_ids = set(high_risk_ids) | set(vul_line_node_ids)

        for region in merged_regions:
            region_lines = self.line_number_extract(ast_nodes_dict, [region['nodes']])[0]
            region['lines'] = region_lines
            region['high_risk_count'] = sum(1 for nid in region['nodes']
                                            if nid in combined_risk_ids)
            region['score'] = self.region_quality_evaluation(region, vul_lines)

        final_regions = merged_regions

        final_nodes_index_lists = [region['nodes'] for region in final_regions]
        normalized_node_index_lists = self.normalize_lists(ast_nodes, final_nodes_index_lists)
        normalized_node_index_lists_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in
                                               normalized_node_index_lists]

        final_score_lists = [region['score'] for region in final_regions]
        region_score_gt_tensor = torch.tensor(final_score_lists, dtype=torch.float32)
        region_score_gt_tensor = torch.nan_to_num(region_score_gt_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        region_score_gt_tensor = region_score_gt_tensor.clamp_(0.0, 1.0)

        region_prior_vals = [
            self.compute_region_prior(region, ast_nodes_dict, adjacency_map, high_risk_ids)
            for region in final_regions
        ]
        region_prior_tensor = torch.tensor(region_prior_vals, dtype=torch.float32)
        region_prior_tensor = torch.nan_to_num(region_prior_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        region_prior_tensor = region_prior_tensor.clamp_(0.0, 1.0)

        final_region_line_numbers = self.line_number_extract(ast_nodes_dict, final_nodes_index_lists)
        region_line_numbers_lists_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in
                                             final_region_line_numbers]

        region_attr_list_all = []
        region_attr_names = None
        for region in final_regions:
            feats, names = self.build_region_attrs(region, ast_nodes_dict, adjacency_map, combined_risk_ids)
            region_attr_list_all.append(feats)
            if region_attr_names is None:
                region_attr_names = names
        region_attr_tensor_all = torch.tensor(region_attr_list_all, dtype=torch.float32)  # [R_total, F]

        global_code_embeddings = self.global_code_embedding(c_path)  # (n_lines, d)

        token_edges, stmt_edges, block_edges = self.build_hierarchical_edges(ast_nodes, ast_edges)

        normalized_token_edges = self.normalize_graph(ast_nodes, token_edges)
        normalized_stmt_edges = self.normalize_graph(ast_nodes, stmt_edges)
        normalized_block_edges = self.normalize_graph(ast_nodes, block_edges)
        normalized_ast_edges = self.normalize_graph(ast_nodes, ast_edges)

        ast_x = self.generate_node_embeddings(ast_nodes, "AST")
        ast_edge_index = torch.tensor(normalized_ast_edges, dtype=torch.int64).t().contiguous()
        token_edge_index = torch.tensor(normalized_token_edges, dtype=torch.int64).t().contiguous()
        stmt_edge_index = torch.tensor(normalized_stmt_edges, dtype=torch.int64).t().contiguous()
        block_edge_index = torch.tensor(normalized_block_edges, dtype=torch.int64).t().contiguous()

        region_nodes_list = []

        subgraphs = self.extract_region_feature(normalized_node_index_lists_tensors, ast_x, ast_edge_index)

        for region in subgraphs:
            region_nodes_list.append(region['nodes'])

        cfg_edges = json_data.get('cfg_edges', [])
        pdg_edges = json_data.get('cdg_edges', []) + json_data.get('ddg_edges', [])
        normalized_cfg_edges = self.normalize_graph(ast_nodes, cfg_edges)
        normalized_pdg_edges = self.normalize_graph(ast_nodes, pdg_edges)

        cfg_x, cfg_edge_index, cfg_id_map = self._build_subgraph_x_and_edge_index(
            edges_base_on_ast=normalized_cfg_edges,
            ast_nodes=ast_nodes,
            ast_x=ast_x,
            graph_type="CFG",
            use_regen_embedding=False
        )

        pdg_x, pdg_edge_index, pdg_id_map = self._build_subgraph_x_and_edge_index(
            edges_base_on_ast=normalized_pdg_edges,
            ast_nodes=ast_nodes,
            ast_x=ast_x,
            graph_type="PDG",
            use_regen_embedding=False
        )

        K = getattr(self, "max_regions", None)

        R_total = len(region_nodes_list)
        K_eff = K if (K is not None and K > 0) else R_total
        K_eff = min(K_eff, R_total)

        selected_idx = list(range(R_total))

        if K_eff is not None and K_eff > 0:
            R = len(region_nodes_list)
            if R > K_eff:
                prior_np = region_prior_tensor.detach().cpu().numpy().tolist()
                sorted_idx = sorted(range(R), key=lambda i: prior_np[i], reverse=True)
                selected_idx = sorted_idx[:K]

                def _select_list_by_idx(lst):
                    return [lst[i] for i in selected_idx]

                region_nodes_list = _select_list_by_idx(region_nodes_list)
                region_line_numbers_lists_tensors = _select_list_by_idx(region_line_numbers_lists_tensors)

                idx_tensor = torch.tensor(selected_idx, dtype=torch.long)
                region_score_gt_tensor = region_score_gt_tensor.index_select(0, idx_tensor)

                region_attr_tensor = region_attr_tensor_all.index_select(0, idx_tensor)
            else:
                region_attr_tensor = region_attr_tensor_all
        else:
            region_attr_tensor = region_attr_tensor_all

        region_cfg_nodes_list = self._map_regions_to_subgraph_indices(region_nodes_list, cfg_id_map)
        region_pdg_nodes_list = self._map_regions_to_subgraph_indices(region_nodes_list, pdg_id_map)

        data = Data(
            ast_x=ast_x,
            token_edge_index=token_edge_index,
            stmt_edge_index=stmt_edge_index,
            block_edge_index=block_edge_index,

            global_code_embedding=global_code_embeddings,

            region_line_numbers_lists=region_line_numbers_lists_tensors,

            region_ast_nodes_list=region_nodes_list,
            region_cfg_nodes_list=region_cfg_nodes_list,
            region_pdg_nodes_list=region_pdg_nodes_list,

            cfg_x=cfg_x,
            cfg_edge_index=cfg_edge_index,

            pdg_x=pdg_x,
            pdg_edge_index=pdg_edge_index,

            region_score_gt=region_score_gt_tensor,

            region_attr=region_attr_tensor,  # [num_regions, F]

            y=torch.tensor([self.get_target(file_name)], dtype=torch.long),

            file_name=file_name[:-5]
        )

        print("==== Data Details ====")
        print(f"{data=}")
        print("==== ==== ==== ==== ====\n")
        return data

    def _map_regions_to_subgraph_indices(self, region_nodes_list, id_map):
        """
        Maps the list of AST nodes for a region to the local index space of a subgraph (CFG/PDG).

        Parameters

        - `region_nodes_list`: List[Tensor|list], elements are the original AST ids (int)

        - `id_map`: Dict[old_ast_id -> local_subgraph_id] (from the return of `_build_subgraph_x_and_edge_index`)

        Returns

        - `List[Tensor[long]]`: A list of local node indices for each region in the subgraph (may be an empty Tensor)
        """
        out = []
        for nodes in region_nodes_list:
            if isinstance(nodes, torch.Tensor):
                nodes = nodes.detach().cpu().tolist()
            mapped = [id_map[n] for n in nodes if n in id_map]
            mapped = self._unique_preserve_order(mapped)
            out.append(torch.tensor(mapped, dtype=torch.long))
        return out

    def _build_subgraph_x_and_edge_index(
            self,
            edges_base_on_ast,
            ast_nodes,
            ast_x,
            graph_type: str,
            use_regen_embedding=True,
    ):
        if not edges_base_on_ast:
            empty_x = torch.empty((0, ast_x.size(1)), dtype=ast_x.dtype, device=ast_x.device)
            empty_e = torch.empty((2, 0), dtype=torch.int64, device=ast_x.device)
            return empty_x, empty_e, {}

        used_old_ids = []
        for u, v in edges_base_on_ast:
            used_old_ids.append(int(u))
            used_old_ids.append(int(v))
        used_old_ids = sorted(set(used_old_ids))

        old2new = {old_id: new_id for new_id, old_id in enumerate(used_old_ids)}

        if use_regen_embedding:
            sub_nodes = [ast_nodes[old_id] for old_id in used_old_ids]
            sub_x = self.generate_node_embeddings(sub_nodes, graph_type)
        else:
            idx = torch.tensor(used_old_ids, dtype=torch.long, device=ast_x.device)
            sub_x = ast_x.index_select(0, idx)

        remapped = [(old2new[int(u)], old2new[int(v)]) for u, v in edges_base_on_ast]
        sub_edge_idx = torch.tensor(remapped, dtype=torch.int64, device=ast_x.device).t().contiguous()

        id_map = old2new
        return sub_x, sub_edge_idx, id_map

    def compute_region_prior(self, region, ast_nodes_dict, adjacency_map, high_risk_ids):
        nodes = region['nodes']  # List[int]
        lines = region.get('lines', [])
        n = max(1, len(nodes))
        node_set = set(nodes)

        high_risk_cnt = sum(1 for nid in nodes if nid in high_risk_ids)
        risk_ratio = high_risk_cnt / n  # 0~1

        out_deg_sum = 0
        for nid in nodes:
            out_deg_sum += len(adjacency_map.get(nid, []))
        avg_branch = out_deg_sum / n
        avg_branch = min(avg_branch / 3.0, 1.0)

        internal_edges = 0
        for u in nodes:
            for v in adjacency_map.get(u, []):
                if v in node_set:
                    internal_edges += 1
        density = internal_edges / n
        density = min(density / 4.0, 1.0)

        size_score = min(len(nodes) / 30.0, 1.0)
        if lines:
            ln_span = max(lines) - min(lines) + 1
            span_score = min(ln_span / 50.0, 1.0)
        else:
            span_score = 0.0

        prior = (
                0.50 * risk_ratio +
                0.20 * avg_branch +
                0.15 * density +
                0.10 * size_score +
                0.05 * span_score
        )
        prior = max(0.0, min(1.0, prior))

        region['_prior_components'] = {
            'risk_ratio': float(risk_ratio),
            'avg_branch': float(avg_branch),
            'density': float(density),
            'size_score': float(size_score),
            'span_score': float(span_score),
            'prior': float(prior),
        }
        return prior

    def extract_region_feature(self, region_nodes_list, node_features, edge_index):
        subgraphs = []

        for region_nodes in region_nodes_list:
            region_set = set(region_nodes.tolist())

            sub_node_features = node_features[region_nodes]

            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(region_nodes.tolist())}

            src, dst = edge_index
            mask = torch.tensor([(s.item() in region_set and d.item() in region_set)
                                 for s, d in zip(src, dst)], dtype=torch.bool)

            sub_edge_index = edge_index[:, mask]

            sub_edge_index = torch.stack([
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[0]]),
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[1]])
            ])

            subgraphs.append({
                'nodes': region_nodes,
                'features': sub_node_features,
                'edges': sub_edge_index
            })

        return subgraphs

    def global_code_embedding(self, c_path: str) -> torch.Tensor:
        with open(c_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = [line.rstrip('\n') for line in f]

        line_embeddings = []
        for line in all_lines:
            try:
                cleaned_line = line.strip()
                if not cleaned_line:
                    emb = np.zeros(self.word_vectors.vector_size)
                else:
                    cleaned_line = line.replace('\t', ' ').replace('\n', ' ')
                    tokens = self.tokenizer.tokenize(cleaned_line)
                    vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
                    emb = np.mean(vectors, axis=0) if vectors else np.zeros(self.word_vectors.vector_size)
                if emb.shape != (self.word_vectors.vector_size,):
                    emb = np.zeros(self.word_vectors.vector_size)

            except Exception as e:
                print(f"Error in line {len(line_embeddings) + 1}: {str(e)}")
                emb = np.zeros(self.word_vectors.vector_size)

            line_embeddings.append(emb)

        embeddings_tensor = torch.from_numpy(np.array(line_embeddings, dtype=np.float32))   # (n_lines, d)
        assert embeddings_tensor.ndim == 2, f"The dimension f is incorrect; it should be (n, d). The actual result is {embeddings_tensor.shape}."
        return embeddings_tensor

    def region_quality_evaluation(self, region, vul_lines):
        line_coverage = len(set(region['lines']) & set(vul_lines)) / (len(vul_lines) + 1e-8)

        size_score = np.log1p(len(region['nodes'])) / 5.0

        risk_density = region['high_risk_count'] / (len(region['nodes']) + 1e-8)

        score = 0.6 * line_coverage + 0.3 * risk_density + 0.1 * size_score

        return score

    def get_vul_line_nodes(self, ast_nodes_dict, vul_lines):
        vul_lines = set(vul_lines)
        return [
            nid for nid, node in ast_nodes_dict.items()
            if node.get('lineNumber', -1) != -1
               and int(node['lineNumber']) in vul_lines
        ]

    def calculate_region_score(self, node_id_list, nodes_dict, adjacency_map):
        node_scores = [self.calculate_risk_score(nodes_dict[nid], adjacency_map, nodes_dict)
                       for nid in node_id_list]

        cross_file_risk = any(
            any(kw in nodes_dict[nid]['code'].lower()
                for kw in ['extern', 'import', 'dll'])
            for nid in node_id_list
        )

        base_score = max(node_scores)
        weight = 1.0 + 0.15 * len([s for s in node_scores if s > 0.7])
        if cross_file_risk:
            weight *= 1.3

        raw_score = base_score * weight

        return raw_score

    def merge_regions(self, region_list, nodes_dict, adjacency_map, region_line_numbers, overlap_threshold=0.6,
                      score_diff=0.3):
        regions = [{
            'nodes': region,
            'score': self.calculate_region_score(region, nodes_dict, adjacency_map),
            'lines': region_line_numbers[i]
        } for i, region in enumerate(region_list)]

        merged = []
        while regions:
            current = regions.pop(0)
            candidates = []

            for target in regions:
                overlap = len(set(current['lines']) & set(target['lines'])) / \
                          len(set(current['lines']) | set(target['lines']))
                score_gap = abs(current['score'] - target['score'])

                if overlap >= overlap_threshold and score_gap <= score_diff:
                    candidates.append(target)

            if candidates:
                merged_region = {
                    'nodes': current['nodes'] + [n for c in candidates for n in c['nodes']],
                    'score': max([current['score']] + [c['score'] for c in candidates]),
                    'lines': sorted(set(current['lines'] + [l for c in candidates for l in c['lines']]))
                }
                regions = [r for r in regions if r not in candidates]
                regions.insert(0, merged_region)
            else:
                merged.append(current)

        return merged

    def deduplicate_nested_list(self, nested_list):
        seen = {}
        for sublist in nested_list:
            key = tuple(sublist)
            if key not in seen:
                seen[key] = sublist
        return list(seen.values())

    def split_continuous_sublists(self, lst):
        if not lst:
            return []
        sublists = []
        if len(lst) > 2:
            start_id = 1
        else:
            start_id = 0
        current = [lst[start_id]]
        for num in lst[(start_id + 1):]:
            if num == current[-1] + 1:
                current.append(num)
            else:
                sublists.append(current)
                current = [num]
        sublists.append(current)
        return sublists

    def detect_high_risk_nodes(self, ast_nodes):
        HIGH_RISK_TYPES = {'CALL', 'CONTROL_STRUCTURE', 'JUMP_TARGET'}
        VUL_PATTERNS = {
            'memcpy', 'strcpy', 'memmove', 'sprintf', 'strcat', 'strncpy',
            'vsprintf', 'gets', 'scanf', 'strncat',

            'malloc', 'free', 'realloc', 'calloc', 'strdup', 'fopen',
            'fclose', 'fdopen', 'popen',

            'strncmp', 'strlen', 'atoi', 'atol', 'atof', 'getenv',
            'system', 'sscanf', 'access',

            'if', 'switch', 'for', 'while', 'goto', 'sizeof'
        }
        high_risk_nodes = []
        for node in ast_nodes:
            if node.get('_label') in HIGH_RISK_TYPES:
                code = node.get('code', '')
                matched_patterns = [pattern for pattern in VUL_PATTERNS if pattern in code]
                if matched_patterns:
                    high_risk_nodes.append(node['id'])
        return high_risk_nodes

    def calculate_risk_score(self, node, adj_map, nodes):
        static_weights = {
            'memcpy': 1.0, 'strcpy': 1.0, 'memmove': 0.9, 'sprintf': 0.9, 'strcat': 0.95,
            'strncpy': 0.85, 'vsprintf': 0.9, 'gets': 1.0, 'scanf': 0.8, 'strncat': 0.8,

            'malloc': 0.8, 'free': 0.8, 'realloc': 0.7, 'calloc': 0.75,
            'strdup': 0.7, 'fopen': 0.6, 'fclose': 0.6, 'fdopen': 0.6, 'popen': 0.65,

            'strncmp': 0.6, 'strlen': 0.6, 'atoi': 0.5, 'atol': 0.5, 'atof': 0.5,
            'getenv': 0.4, 'system': 0.8, 'sscanf': 0.5, 'access': 0.4,

            'if': 0.4, 'switch': 0.4, 'for': 0.4, 'while': 0.4, 'goto': 0.4, 'sizeof': 0.4
        }
        VUL_PATTERNS = {
            'memcpy', 'strcpy', 'memmove', 'sprintf', 'strcat', 'strncpy',
            'vsprintf', 'gets', 'scanf', 'strncat',

            'malloc', 'free', 'realloc', 'calloc', 'strdup', 'fopen',
            'fclose', 'fdopen', 'popen',

            'strncmp', 'strlen', 'atoi', 'atol', 'atof', 'getenv',
            'system', 'sscanf', 'access',

            'if', 'switch', 'for', 'while', 'goto', 'sizeof'
        }
        code = node.get('code', '')
        matched_patterns = [pattern for pattern in VUL_PATTERNS if pattern in code]
        if matched_patterns:
            for pattern in matched_patterns:
                base_score = static_weights.get(pattern, 0.4)
        else:
            base_score = 0.4

        if len(adj_map.get(node['id'], [])) > 3:
            base_score *= 1.3

        if any('input' in nodes[c_id]['code']
               for c_id in adj_map.get(node['id'], [])):
            base_score *= 1.3

        return min(base_score, 1.0)

    def filter_common_patterns(self, nodes, threshold=0.05):
        code_counter = defaultdict(int)
        for node in nodes:
            code_counter[node['code']] += 1
        total = len(nodes)
        return [
            code for code, count in code_counter.items()
            if count / total > threshold
        ]

    def normalized_edges(self, nodes, edges):
        normalized_edges = []
        for edge in edges:
            normalized_edge = self.normalize_graph(nodes, edge)
            normalized_edges.append(normalized_edge)

        return normalized_edges

    def line_number_extract(self, ast_nodes_dict, node_index_lists):
        region_line_numbers = []

        for region in node_index_lists:
            line_set = set()
            for node_id in region:
                node = ast_nodes_dict.get(node_id)
                if node:
                    line_number = node.get('lineNumber', 0)
                    line_set.update([line_number])
            sorted_lines = sorted(line_set)
            region_line_numbers.append(sorted_lines)
        return region_line_numbers

    def filter_nodes(self, ast_nodes, ast_edges):
        node_ids = {node['id'] for node in ast_nodes}

        root_node_id = min(node_ids)

        nodes_to_remove = set()
        nodes_to_remove.add(root_node_id)

        first_child_id = None
        second_child_id = None

        for edge in ast_edges:
            if edge[0] == root_node_id:
                if first_child_id is None:
                    first_child_id = edge[1]
                    nodes_to_remove.add(first_child_id)
                else:
                    second_child_id = edge[1]
                    nodes_to_remove.add(second_child_id)
                    break

        filtered_nodes = [node for node in ast_nodes if node['id'] not in nodes_to_remove]

        filtered_edges = []
        for edge in ast_edges:
            if edge[0] not in nodes_to_remove and edge[1] not in nodes_to_remove:
                filtered_edges.append(edge)

        return filtered_nodes, filtered_edges

    def normalize_lists(self, nodes, node_index_lists):
        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}

        normalized_lists = [[old_to_new_id_dict[node_id] for node_id in sublist] for sublist in node_index_lists]

        return normalized_lists

    def refine_regions(self, original_regions, max_depth):
        parent_child_map = defaultdict(list)
        for level in sorted(original_regions.keys()):
            if level + 1 not in original_regions: continue
            for parent_id in original_regions[level]:
                parent_nodes = set(original_regions[level][parent_id])
                for child_id in original_regions[level + 1]:
                    if child_id in parent_nodes:
                        parent_child_map[(level, parent_id)].append(child_id)

        def split_region(current_level, current_id, current_nodes, processed_depth):
            if processed_depth >= max_depth:
                return {current_id: current_nodes}

            children = parent_child_map.get((current_level, current_id), [])
            if not children:
                return {current_id: current_nodes}

            split_result = {}
            remaining = set(current_nodes)

            for child_id in children:
                child_nodes = set(original_regions[current_level + 1][child_id])
                split_result[child_id] = list(child_nodes)
                remaining -= child_nodes

            if remaining:
                split_result[current_id] = list(remaining)

            final_result = {}
            for sub_id in list(split_result.keys()):
                if sub_id == current_id:
                    final_result.update({sub_id: split_result[sub_id]})
                else:
                    final_result.update(
                        split_region(
                            current_level + 1,
                            sub_id,
                            split_result[sub_id],
                            processed_depth + 1
                        )
                    )
            return final_result

        final_regions = {}
        for base_level in sorted(original_regions.keys()):
            if base_level > 1: break

            for region_id, region_nodes in original_regions[base_level].items():
                final_regions.update(
                    split_region(
                        current_level=base_level,
                        current_id=region_id,
                        current_nodes=region_nodes,
                        processed_depth=1
                    )
                )

        return final_regions

    def generate_node_embeddings(self, nodes, graph_type):
        node_embedding_dict = {}
        for n in nodes:
            if 'code' in n:
                n_code = n['code']
            else:
                n_code = ""
            try:
                n_code = n_code.replace('\\t', ' ')
                if not n_code:
                    code_embedding = np.zeros(self.word_vectors.vector_size)
                else:
                    code_embedding = np.nanmean([self.word_vectors[x] for x in self.tokenizer.tokenize(n_code)], axis=0)    # 代替普通均值/方差计算，自动忽略NaN值
            except KeyError as e:
                raise RuntimeError(f"Embedding error {e} in {graph_type} graph")

            node_embedding_dict[n['id']] = code_embedding

        node_embeddings = np.array([node_embedding_dict[node['id']] for node in nodes], dtype=np.float32)
        return torch.tensor(node_embeddings, dtype=torch.float)

    def normalize_graph(self, nodes, edges):
        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}

        normalized_edges = [[old_to_new_id_dict[edge[0]], old_to_new_id_dict[edge[1]]] for edge in edges]

        return normalized_edges

    def _remap_rare_labels_to_unknown(self, ast_nodes):
        for n in ast_nodes:
            lab = n.get('_label', '')
            if lab in ('TYPE_DECL', 'MEMBER'):
                n['_label'] = 'UNKNOWN'

    def _node_level(self, label: str) -> str:
        if label in ('METHOD', 'BLOCK', 'CONTROL_STRUCTURE'):
            return 'block'
        if label in ('CALL', 'RETURN'):
            return 'statement'
        return 'token'

    def build_hierarchical_edges(self, ast_nodes, ast_edges):
        """

        Input:

        - ast_nodes: List containing { 'id': int, '_label': str, ... }

        - ast_edges: List containing elements of the form [parent_id, child_id]

        Output:

        - token_edges: List[[u,v]]

        - stmt_edges: List[[u,v]]

        - block_edges: List[[u,v]]

        Rules:

        - Token level: At least one end is a token

        - Statement level: Neither end contains a token, and at least one end is a statement (the other end is either a statement or a block)

        - Block level: Both ends are blocks

        """
        self._remap_rare_labels_to_unknown(ast_nodes)

        id2level = {}
        for n in ast_nodes:
            lab = n.get('_label', '')
            id2level[n['id']] = self._node_level(lab)

        token_e_set = set()
        stmt_e_set = set()
        block_e_set = set()

        for e in ast_edges:
            u, v = int(e[0]), int(e[1])
            lu = id2level.get(u, 'token')
            lv = id2level.get(v, 'token')

            if lu == 'block' and lv == 'block':
                block_e_set.add((u, v))
                continue

            if (lu != 'token' and lv != 'token') and ('statement' in (lu, lv)):
                stmt_e_set.add((u, v))
                continue

            if 'token' in (lu, lv):
                token_e_set.add((u, v))
                continue

            token_e_set.add((u, v))

        token_edges = [[u, v] for (u, v) in token_e_set]
        stmt_edges = [[u, v] for (u, v) in stmt_e_set]
        block_edges = [[u, v] for (u, v) in block_e_set]

        return token_edges, stmt_edges, block_edges

    def get_target(self, file_name: str) -> int:
        """Get the target given a file name."""
        target_str = file_name.split("_")[1]
        non_num = re.compile(r'[^\d]')
        target = int(non_num.sub('', target_str))
        return target

    def save_processed_data(self, data, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_processed_data(self, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

class JoernError(Exception):
    pass

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom.*")

if __name__ == "__main__":
    DATASET_CONFIGS = {
        "FFMPeg+Qemu": {
            "c_dir": "./dataset/devign/c/",
            "js_dir": "./dataset/devign/js/",
            "vul_lines": "./dataset/devign/Devign_vulnerable_lines.json",
            "w2v_path": "./dataset/devign/W2V/Devign-128-20.wordvectors",
        },
        "BigVul": {
            "c_dir": "./dataset/fan/c/",
            "js_dir": "./dataset/fan/js/",
            "vul_lines": "./dataset/fan/Fan_vulnerable_lines.json",
            "w2v_path": "./dataset/fan/W2V/Fan-128-20.wordvectors",
        },
        "DiverseVul": {
            "c_dir": "./dataset/DiverseVul/c/",
            "js_dir": "./dataset/DiverseVul/js/",
            "vul_lines": "./dataset/DiverseVul/DiverseVul_vulnerable_lines.json",
            "w2v_path": "./dataset/DiverseVul/W2V/DiverseVul-128-20.wordvectors",
        },
        "Reveal": {
            "c_dir": "./dataset/Reveal/c/",
            "js_dir": "./dataset/Reveal/js/",
            "vul_lines": None,
            "w2v_path": "./dataset/Reveal/W2V/Reveal-128-20.wordvectors",
        },
        "Location": {
            "c_dir": "./Location/c/",
            "js_dir": "./Location/js/",
            "vul_lines": None,
            "w2v_path": "./Location/Location-128-20.wordvectors",
        },
    }

    parser = argparse.ArgumentParser(description="Preprocess base or perturbed datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_regions", type=int, default=12)
    parser.add_argument("--perturbed_root", type=str, default=None,
                        help="Root of perturbed sets, e.g. './perturbed'")
    parser.add_argument("--modes", type=str, default="api,token",
                        help="Comma-separated modes, default: api,token")
    parser.add_argument("--only_subdirs", type=str, default=None,
                        help="Optional: comma-separated subdir names (e.g., api_s020,token_s060)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]

    # Scenario A: No batch processing, running the baseline set
    if not args.perturbed_root:
        builder = DatasetBuilder(
            c_dir=cfg["c_dir"],
            js_dir=cfg["js_dir"],
            vul_lines_path=cfg["vul_lines"],
            w2v_path=cfg["w2v_path"],
            max_regions=args.max_regions,
            seed=args.seed,
        )
        builder.dataset_name = args.dataset
        if args.dry_run:
            print(f"[DRY] Would run base preprocessing: dataset_name={builder.dataset_name}")
        else:
            builder.run()
        raise SystemExit(0)

    # Scenario B: Batch processing of perturbation sets
    perturbed_root = args.perturbed_root.rstrip("/")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    only_map = None
    if args.only_subdirs:
        only_map = set([s.strip() for s in args.only_subdirs.split(",") if s.strip()])

    def list_subdirs(p: str):
        return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]) if os.path.isdir(p) else []

    print(f"[INFO] Batch preprocessing under: {perturbed_root}")
    print(f"[INFO] Modes: {modes}")
    if only_map:
        print(f"[INFO] Only subdirs: {sorted(only_map)}")

    for mode in modes:
        mode_dir_c = os.path.join(perturbed_root, mode)          # e.g. .../perturbed/api
        mode_dir_js = os.path.join(perturbed_root, f"{mode}_js") # e.g. .../perturbed/api_js
        if not os.path.isdir(mode_dir_c):
            print(f"[WARN] Missing mode dir: {mode_dir_c} (skip)")
            continue
        if not os.path.isdir(mode_dir_js):
            print(f"[WARN] Missing js dir: {mode_dir_js} (skip)")
            continue

        subdirs = list_subdirs(mode_dir_c)
        if only_map:
            subdirs = [s for s in subdirs if s in only_map]
        if not subdirs:
            print(f"[WARN] No subdirs for mode={mode}")
            continue

        for sub in subdirs:
            c_dir_sub = os.path.join(mode_dir_c, sub)
            js_dir_sub = os.path.join(mode_dir_js, sub)
            if not os.path.isdir(js_dir_sub):
                print(f"[WARN] Missing js dir for {mode}:{sub} -> {js_dir_sub} (skip)")
                continue

            dataset_name = f"{args.dataset}_{sub}"
            print(f"[RUN] {dataset_name}")
            print(f"      c_dir={c_dir_sub}")
            print(f"      js_dir={js_dir_sub}")

            builder = DatasetBuilder(
                c_dir=c_dir_sub,
                js_dir=js_dir_sub,
                vul_lines_path=cfg["vul_lines"],
                w2v_path=cfg["w2v_path"],
                max_regions=args.max_regions,
                seed=args.seed,
            )
            builder.dataset_name = dataset_name

            if args.dry_run:
                print(f"[DRY] Would run builder.run() for {dataset_name}")
            else:
                builder.run()

    print("[DONE] Batch preprocessing for perturbed sets completed.")

