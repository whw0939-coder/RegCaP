import argparse
import os
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from training import VulDetectionSystem
from data_loading import GlobalDataModule
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from dataclasses import dataclass, field
from typing import List

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The PyTorch API of nested tensors")
torch.set_float32_matmul_precision('medium')


@dataclass
class ModelArgs:
    input_dim: int = 128
    hidden_dim: int = 128
    contrast_dim: int = 128
    cls_hidden: int = 128
    num_layers: int = 5
    max_epochs: int = 50
    gpus: List[int] = field(default_factory=lambda: [0])
    device: torch.device = field(init=False)
    topk_regions: int = 5

    def __post_init__(self):
        if self.gpus and torch.cuda.is_available():
            primary_gpu = self.gpus[0]
            self.device = torch.device(f'cuda:{primary_gpu}')
        else:
            self.device = torch.device('cpu')


def _make_trainer_for_eval(gpus):
    tb_logger = TensorBoardLogger(save_dir="./runs_eval", name="vul_multi_obj_eval")
    return pl.Trainer(
        accelerator='gpu' if (gpus and torch.cuda.is_available()) else 'cpu',
        devices=gpus if (gpus and torch.cuda.is_available()) else "auto",
        logger=tb_logger,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )


def _default_pred_tag_from_data_path(p: str) -> str:
    base = os.path.basename(p)
    return os.path.splitext(base)[0] if "." in base else base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', help='List of GPU indices to use')
    parser.add_argument('--data_path', type=str, default="devign.pkl")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--cls_model_path', type=str, default=None,
                        help='If set, test all checkpoints under this directory and exit')
    parser.add_argument('--cwe_mode', action='store_true',
                        help='If set, treat data_path as a directory of CWE .pkl files and merge them')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_regions', type=int, default=12)
    parser.add_argument('--topk_regions', type=int, default=12)

    parser.add_argument('--attr_dim', type=int, default=23)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=None)

    # === Predicting relevant parameters for the drop (used in ΔF1/IS calculation) ===
    parser.add_argument('--pred_dump_dir', type=str, default='./pred_dumps',
                        help='Directory to dump per-sample predictions and summaries')
    parser.add_argument('--pred_tag', type=str, default=None,
                        help='Tag used in output filenames; default = basename(data_path) without extension')
    parser.add_argument('--dump_region_scores', action='store_true',
                        help='Also dump per-sample region_scores (can be large)')
    parser.add_argument('--dump_logits', action='store_true',
                        help='Also dump raw logits')

    # ===== Cross-dataset testing =====
    parser.add_argument('--cross_data_path', type=str, default=None,
                        help='Dataset .pkl (if provided, a subset of it will be used as the test set)')
    parser.add_argument('--cross_test_ratio', type=float, default=1.0,
                        help='The sampling ratio for the test set (default 1.0).')

    args = parser.parse_args()
    rank_zero_info(vars(args))

    # ================ Repro ================
    pl.seed_everything(args.seed, workers=True)

    # ================ Data Loading ===============
    data_module = GlobalDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_regions=args.max_regions,
        emb_dim=128,
        seed=args.seed,
        cwe_mode=args.cwe_mode,
        save_lists_dir="./split_lists",
        cross_data_path=args.cross_data_path,
        cross_test_ratio=args.cross_test_ratio,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # train_files = data_module.get_train_filenames()  # Used for statistical analysis of token/API distribution (standard practice)
    # val_files = data_module.get_val_filenames()
    # test_files = data_module.get_test_filenames()    # Used for subsequent intervention and evaluation on the test set.
    # print(len(train_files), len(val_files), len(test_files))

    # ================ Model Configuration ===============
    model_args = ModelArgs(
        hidden_dim=128,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        topk_regions=args.topk_regions,
    )
    lr = 1e-4

    # pred_tag Default value
    base_tag = args.pred_tag or _default_pred_tag_from_data_path(args.data_path)
    os.makedirs(args.pred_dump_dir, exist_ok=True)

    # ================ Evaluation Mode (Batch Testing ckpt) ===============
    if args.cls_model_path:
        ckpt_dir = args.cls_model_path
        all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        if not all_ckpts:
            rank_zero_info(f"⚠️ No .ckpt found under: {ckpt_dir}")
            return

        # Priority: F1 > Acc > Pre > Rec > MSE > last
        def order_key(p):
            name = os.path.basename(p).lower()
            if 'val_f1' in name:            return (0, name)
            if 'val_acc' in name:           return (1, name)
            if 'val_pre' in name:           return (2, name)
            if 'val_rec' in name:           return (3, name)
            if 'val_region_mse' in name:    return (4, name)
            if name == 'last.ckpt':         return (5, name)
            return (6, name)

        all_ckpts.sort(key=order_key)
        eval_trainer = _make_trainer_for_eval(args.gpus)

        rank_zero_info("\n=== Running evaluation on provided checkpoints ===")
        for ckpt in all_ckpts:
            ckpt_base = os.path.splitext(os.path.basename(ckpt))[0]
            tag = f"{base_tag}__{ckpt_base}"
            rank_zero_info(f"\n✅ Testing checkpoint: {ckpt} (tag={tag})")
            try:
                model = VulDetectionSystem.load_from_checkpoint(
                    ckpt,
                    model_args=model_args,
                    lr=lr,
                    weight_decay=0.0,
                    attr_dim=args.attr_dim,
                    lambda_reg=args.lambda_reg,
                    lambda_cls=args.lambda_cls,
                    pos_weight=args.pos_weight,
                    pred_dump_dir=args.pred_dump_dir,
                    pred_tag=tag,
                    dump_region_scores=args.dump_region_scores,
                    dump_logits=args.dump_logits,
                    map_location=model_args.device,
                )
                rank_zero_info("ℹ️ Loaded model via load_from_checkpoint.")
            except Exception as e:
                rank_zero_info(f"⚠️ load_from_checkpoint failed ({e}), fallback to init.")
                model = VulDetectionSystem(
                    model_args=model_args,
                    lr=lr,
                    weight_decay=0.0,
                    attr_dim=args.attr_dim,
                    lambda_reg=args.lambda_reg,
                    lambda_cls=args.lambda_cls,
                    pos_weight=args.pos_weight,
                    pred_dump_dir=args.pred_dump_dir,
                    pred_tag=tag,
                    dump_region_scores=args.dump_region_scores,
                    dump_logits=args.dump_logits,
                )

            eval_trainer.test(
                model=model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path=ckpt
            )
        return

    # ================ Training ===============
    cls_model = VulDetectionSystem(
        model_args=model_args,
        lr=lr,
        weight_decay=0.0,
        attr_dim=args.attr_dim,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        pos_weight=args.pos_weight,
        tsne_enable=True,
        pred_dump_dir=args.pred_dump_dir,
        pred_tag=base_tag,
        dump_region_scores=args.dump_region_scores,
        dump_logits=args.dump_logits,
    )

    # ---- Checkpoints (Multi-objective monitoring: Four graph-level indicators + regression) ----
    best_mse = ModelCheckpoint(
        monitor='val_region_mse',
        filename='best-val_region_mse-epoch{epoch}-val_region_mse={val_region_mse:.6f}',
        mode='min',
        save_top_k=1
    )
    best_f1 = ModelCheckpoint(
        monitor='val_f1',
        filename='best-val_f1-epoch{epoch}-val_f1={val_f1:.4f}',
        mode='max',
        save_top_k=1
    )
    best_acc = ModelCheckpoint(
        monitor='val_acc',
        filename='best-val_acc-epoch{epoch}-val_acc={val_acc:.4f}',
        mode='max',
        save_top_k=1
    )
    best_pre = ModelCheckpoint(
        monitor='val_pre',
        filename='best-val_pre-epoch{epoch}-val_pre={val_pre:.4f}',
        mode='max',
        save_top_k=1
    )
    best_rec = ModelCheckpoint(
        monitor='val_rec',
        filename='best-val_rec-epoch{epoch}-val_rec={val_rec:.4f}',
        mode='max',
        save_top_k=1
    )
    last_checkpoint = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        monitor=None,
        save_on_train_epoch_end=True
    )

    tb_logger = TensorBoardLogger(save_dir="./runs", name="vul_multi_obj")
    cls_trainer = pl.Trainer(
        accelerator='gpu' if (args.gpus and torch.cuda.is_available()) else 'cpu',
        devices=args.gpus if (args.gpus and torch.cuda.is_available()) else "auto",
        max_epochs=args.max_epochs,
        callbacks=[best_mse, best_f1, best_acc, best_pre, best_rec, last_checkpoint],
        logger=tb_logger,
        enable_progress_bar=True
    )

    # Training + Validation
    cls_trainer.fit(
        cls_model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )

    # ================ Final Test (Multiple Best and Last Tests) ===============
    print("\n🔥 Running Final Test")
    rank_zero_info("\n=== Starting testing ===")
    ckpt_paths = {
        "Best F1":  best_f1.best_model_path,
        "Best Acc": best_acc.best_model_path,
        "Best Precision": best_pre.best_model_path,
        "Best Recall": best_rec.best_model_path,
        "Best MSE": best_mse.best_model_path,
        "Last Model": last_checkpoint.last_model_path
    }
    for model_name, path in ckpt_paths.items():
        if not path:
            rank_zero_info(f"✅ Checkpoint for {model_name} not found, skipping test")
            continue
        tag = f"{base_tag}__final_{model_name.replace(' ', '_')}"
        rank_zero_info(f"\n✅ Testing {model_name} model: {path} (tag={tag})")
        cls_model.hparams.pred_tag = tag
        cls_model.hparams.pred_dump_dir = args.pred_dump_dir
        cls_model.hparams.dump_region_scores = args.dump_region_scores
        cls_model.hparams.dump_logits = args.dump_logits

        cls_trainer.test(
            dataloaders=data_module.test_dataloader(),
            ckpt_path=path
        )


if __name__ == "__main__":
    main()
