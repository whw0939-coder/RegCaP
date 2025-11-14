import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric, Accuracy, Precision, Recall, F1Score
from typing import Optional, Tuple, Union

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import VulDetectionModel


class VulDetectionSystem(pl.LightningModule):
    """
    Multi-task training system:

    - Region score regression: Perform MSE (mask-weighted) on effective regions

    - Graph-level classification: Perform BCE (logits) on the graph representation after Top-K aggregation

    - Validation period t-SNE: Visualize by region features (color = predicted/true score)
    """

    def __init__(
        self,
        model_args,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        attr_dim: int = 0,
        # ====== Loss Weights ======
        lambda_reg: float = 1.0,             # Region Regression Loss Weights
        lambda_cls: float = 1.0,             # Graph-level classification loss weights
        pos_weight: Optional[float] = None,  # Graph-level positive class weights (can be set when classes are unbalanced)
        # ====== t-SNE Visualization ======
        tsne_enable: bool = True,
        tsne_max_points: int = 20000,
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: Union[str, float] = "auto",
        tsne_figsize: Tuple[int, int] = (7, 6),
        tsne_n_iter: int = 1000,
        tsne_random_state: int = 42,
        tsne_cmap: str = "coolwarm",
        tsne_dpi: int = 300,
        pred_dump_dir: str = "./pred_dumps",
        pred_tag: Optional[str] = None,
        dump_region_scores: bool = True,
        dump_logits: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VulDetectionModel(
            hidden_dim=model_args.hidden_dim,
            input_dim=model_args.input_dim,
            heads=getattr(model_args, "gat_heads", 4),
            ggnn_steps=getattr(model_args, "ggnn_steps", 6),
            attr_dim=int(attr_dim),
            topk_regions=model_args.topk_regions,
            topk_pooling="weighted_mean",
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion_reg = self._masked_regression_loss
        if pos_weight is not None:
            self.criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.criterion_cls = nn.BCEWithLogitsLoss()

        self.train_mse = MeanMetric(); self.train_mae = MeanMetric()
        self.val_mse   = MeanMetric(); self.val_mae   = MeanMetric()
        self.test_mse  = MeanMetric(); self.test_mae  = MeanMetric()

        self.train_cls_metrics = torch.nn.ModuleDict({
            "acc": Accuracy(task="binary"),
            "pre": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1":  F1Score(task="binary"),
        })
        self.val_cls_metrics = torch.nn.ModuleDict({
            "acc": Accuracy(task="binary"),
            "pre": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1":  F1Score(task="binary"),
        })
        self.test_cls_metrics = torch.nn.ModuleDict({
            "acc": Accuracy(task="binary"),
            "pre": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1":  F1Score(task="binary"),
        })

        self._val_feats = []
        self._val_pred = []
        self._val_true = []

        self._checked_attr_dim = False

        self._test_records = []

    @staticmethod
    def _masked_regression_loss(pred: torch.Tensor,
                                target: torch.Tensor,
                                mask: torch.Tensor):
        """
        pred/target: [B, K], mask: [B, K] (bool or 0/1)
        Returns: (mse_mean, mae_mean)
        """
        w = mask.float()
        denom = w.sum().clamp(min=1.0)
        mse = ((pred - target) ** 2) * w
        mae = (pred - target).abs() * w
        mse_mean = mse.sum() / denom
        mae_mean = mae.sum() / denom
        return mse_mean, mae_mean

    # ====== Training/Validation/Testing Step-by-Step ======
    def _step_impl(self, batch, stage: str):
        """
        batch:

        - batch.region_score: [B, K] in [0,1]

        - batch.y: [B] in {0,1}

        - If using attributes: batch.region_attr: [B, K, F]
        """
        # Validate attr_dim on the first visit (make sure it matches your current implementation)
        if (not self._checked_attr_dim) and hasattr(batch, "region_attr"):
            F_seen = int(batch.region_attr.size(-1)) if batch.region_attr.ndim == 3 else 0
            F_cfg = int(self.hparams.attr_dim)
            if F_seen != F_cfg:
                raise RuntimeError(
                    f"[region_attr] Dimension inconsistency: when constructing the model, attr_dim={F_cfg}, but the current batch's F={F_seen}."
                    "Please pass the correct attr_dim when creating VulDetectionSystem (pass 0 if there is no attribute)."
                )
            self._checked_attr_dim = True

        out = self.model(batch)

        logits_r = out["region_logits"]              # [B, K]
        mask_r   = out["region_mask"].bool()         # [B, K]
        target_r = batch.region_score.float().clamp(0.0, 1.0)  # [B, K]
        pred_r   = torch.sigmoid(logits_r)           # [B, K] in [0,1]

        cls_logits = out["cls_logits"]               # [B]
        y          = batch.y.float()                 # [B]

        mse_mean, mae_mean = self.criterion_reg(pred_r, target_r, mask_r)
        cls_bce = self.criterion_cls(cls_logits, y)

        loss = self.hparams.lambda_reg * mse_mean + self.hparams.lambda_cls * cls_bce

        probs = torch.sigmoid(cls_logits).detach()
        y_int = y.int()

        if stage == "train":
            self.train_mse.update(mse_mean.detach())
            self.train_mae.update(mae_mean.detach())
            for m in self.train_cls_metrics.values():
                m.update(probs, y_int)

            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("train_cls_bce", cls_bce.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("train_region_mse_step", mse_mean.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))

        elif stage == "val":
            self.val_mse.update(mse_mean.detach())
            self.val_mae.update(mae_mean.detach())
            for m in self.val_cls_metrics.values():
                m.update(probs, y_int)

            if self.hparams.tsne_enable:
                feats = out["region_feat"]      # [B, K, D]
                B, K, D = feats.shape
                m = mask_r.view(B * K)
                feats_flat = feats.reshape(B * K, D)[m]           # [N_eff, D]
                pred_flat  = pred_r.reshape(B * K)[m]             # [N_eff]
                true_flat  = target_r.reshape(B * K)[m]           # [N_eff]
                self._val_feats.append(feats_flat.detach().cpu())
                self._val_pred.append(pred_flat.detach().cpu())
                self._val_true.append(true_flat.detach().cpu())

            self.log("val_cls_bce_step", cls_bce.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("val_region_mse_step", mse_mean.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))

        else:  # test
            self.test_mse.update(mse_mean.detach()); self.test_mae.update(mae_mean.detach())
            for m in self.test_cls_metrics.values():
                m.update(probs, y_int)

            # ====== Collect sample-by-sample records (for ΔF1 / IS) ======
            file_names = getattr(batch, "file_name", None)
            if file_names is None:
                file_names = [f"sample_{i}" for i in range(len(y))]
            B = len(file_names)
            preds = (probs >= 0.5).to(torch.int).cpu().tolist()
            y_cpu = y_int.cpu().tolist()
            prob_cpu = probs.cpu().tolist()
            logit_cpu = cls_logits.detach().cpu().tolist() if self.hparams.dump_logits else [None] * B

            if self.hparams.dump_region_scores:
                pr = pred_r.detach().cpu()       # [B,K]
                mk = mask_r.detach().cpu()       # [B,K]
                region_scores = [pr[i][mk[i]].tolist() for i in range(B)]
            else:
                region_scores = [None] * B

            for i in range(B):
                rec = {
                    "file": file_names[i],
                    "y_true": int(y_cpu[i]),
                    "prob": float(prob_cpu[i]),
                    "pred": int(preds[i]),
                }
                if self.hparams.dump_logits:
                    rec["logit"] = float(logit_cpu[i])
                if self.hparams.dump_region_scores:
                    rec["region_scores"] = region_scores[i]
                self._test_records.append(rec)

        return loss

    # ====== PL hooks ======
    def training_step(self, batch, batch_idx):
        return self._step_impl(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step_impl(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step_impl(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_region_mse", self.train_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_region_mae", self.train_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.train_mse.reset(); self.train_mae.reset()
        vals = {f"train_{k}": m.compute() for k, m in self.train_cls_metrics.items()}
        for k, v in vals.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        for m in self.train_cls_metrics.values():
            m.reset()

    def on_validation_epoch_end(self):
        self.log("val_region_mse", self.val_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_region_mae", self.val_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_mse.reset(); self.val_mae.reset()
        vals = {f"val_{k}": m.compute() for k, m in self.val_cls_metrics.items()}
        for k, v in vals.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        for m in self.val_cls_metrics.values():
            m.reset()

        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            self._val_feats.clear(); self._val_pred.clear(); self._val_true.clear()
            return
        if len(self._val_feats) == 0:
            return

        import os
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE

        tb = getattr(self.logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_figure"):
            self._val_feats.clear(); self._val_pred.clear(); self._val_true.clear()
            return

        X = np.concatenate([t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in self._val_feats], axis=0)
        Y_pred = np.concatenate([t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in self._val_pred], axis=0)
        Y_true = np.concatenate([t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in self._val_true], axis=0)
        self._val_feats.clear(); self._val_pred.clear(); self._val_true.clear()

        N = X.shape[0]
        maxN = int(self.hparams.tsne_max_points)
        if N > maxN:
            rng = np.random.RandomState(int(self.hparams.tsne_random_state))
            idx = rng.choice(N, size=maxN, replace=False)
            X, Y_pred, Y_true = X[idx], Y_pred[idx], Y_true[idx]
            N = maxN
        if N < 5:
            return

        perplexity = float(self.hparams.tsne_perplexity)
        perplexity = max(5.0, min(perplexity, (N - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            learning_rate=self.hparams.tsne_learning_rate,
            n_iter=int(self.hparams.tsne_n_iter),
            random_state=int(self.hparams.tsne_random_state),
            verbose=False,
        )
        X2 = tsne.fit_transform(X)

        def _scatter_continuous(xy, values, title):
            fig = plt.figure(figsize=self.hparams.tsne_figsize, dpi=self.hparams.tsne_dpi)
            ax = fig.add_subplot(111)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=self.hparams.tsne_cmap, s=6, alpha=0.9, vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(sc, ax=ax); cbar.set_label("score (0 → 1)", rotation=90)
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            return fig

        ep = int(self.current_epoch)
        fig_pred = _scatter_continuous(X2, Y_pred, f"Val t-SNE by Pred — epoch {ep}")
        fig_true = _scatter_continuous(X2, Y_true, f"Val t-SNE by GT — epoch {ep}")

        tb.add_figure("tsne/regions_by_pred", fig_pred, global_step=ep, close=True)
        tb.add_figure("tsne/regions_by_true", fig_true, global_step=ep, close=True)

        save_root = getattr(self.logger, "log_dir", "./runs")
        png_dir = os.path.join(save_root, "tsne_pngs"); os.makedirs(png_dir, exist_ok=True)
        f_pred = os.path.join(png_dir, f"epoch{ep:03d}_tsne_pred.png")
        f_true = os.path.join(png_dir, f"epoch{ep:03d}_tsne_true.png")
        fig_pred = _scatter_continuous(X2, Y_pred, f"Val t-SNE by Pred — epoch {ep}")
        fig_true = _scatter_continuous(X2, Y_true, f"Val t-SNE by GT — epoch {ep}")
        fig_pred.savefig(f_pred, dpi=150, bbox_inches="tight"); plt.close(fig_pred)
        fig_true.savefig(f_true, dpi=150, bbox_inches="tight"); plt.close(fig_true)

    def on_test_epoch_end(self):
        self.log("test_region_mse", self.test_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_region_mae", self.test_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.test_mse.reset(); self.test_mae.reset()
        vals = {f"test_{k}": m.compute() for k, m in self.test_cls_metrics.items()}
        for k, v in vals.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        for m in self.test_cls_metrics.values():
            m.reset()

        import json, csv, os
        tag = self.hparams.pred_tag or "test"
        dump_dir = self.hparams.pred_dump_dir
        os.makedirs(dump_dir, exist_ok=True)

        pred_jsonl = os.path.join(dump_dir, f"pred_{tag}.jsonl")
        with open(pred_jsonl, "w", encoding="utf-8") as f:
            for rec in self._test_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = {
            "tag": tag,
            "num_samples": len(self._test_records),
            "metrics": {
                "acc": float(vals.get("test_acc", torch.tensor(0.)).item()),
                "pre": float(vals.get("test_pre", torch.tensor(0.)).item()),
                "rec": float(vals.get("test_rec", torch.tensor(0.)).item()),
                "f1":  float(vals.get("test_f1",  torch.tensor(0.)).item()),
                "region_mse": float(self.trainer.callback_metrics.get("test_region_mse", torch.tensor(0.)).item()) if hasattr(self, "trainer") else None,
                "region_mae": float(self.trainer.callback_metrics.get("test_region_mae", torch.tensor(0.)).item()) if hasattr(self, "trainer") else None,
            }
        }
        with open(os.path.join(dump_dir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        pred_csv = os.path.join(dump_dir, f"pred_{tag}.csv")
        with open(pred_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "y_true", "prob", "pred"])
            for r in self._test_records:
                w.writerow([r["file"], r["y_true"], r["prob"], r["pred"]])

        self._test_records.clear()

        print(f"[TEST DUMP] jsonl={pred_jsonl}")
        print(f"[TEST DUMP] csv  ={pred_csv}")
        print(f"[TEST DUMP] summary={os.path.join(dump_dir, f'summary_{tag}.json')}")


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
