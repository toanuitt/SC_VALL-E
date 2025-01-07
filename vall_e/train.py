import json
import logging
from collections import defaultdict

import torch
import os
import pickle
from tqdm import tqdm

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt
from .utils import setup_logging, to_device, trainer
from .vall_e import get_model
from vall_e.models import make_model  # Add this import

_logger = logging.getLogger(__name__)


def load_engines():
    # Create DeepSpeed config
    ds_config = {
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "fp16": {
            "enabled": True,
            "initial_scale_power": 16
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config.warmup_min_lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8
            }
        },
        "zero_optimization": {
            "stage": 2
        }
    }

    # Initialize model first
    model = make_model(config)
    
    # Create engine with model and config
    engine = trainer.Engine(
        model=model,
        config=ds_config
    )

    return Engines({"default": engine})


def main():
    setup_logging(cfg.log_dir)
    
    print('make datasets start')
    if os.path.isfile('./vall_e/dataloaders/train_dl.pkl'):  
        with open('./vall_e/dataloaders/train_dl.pkl', 'rb') as f:
            train_dl = pickle.load(f)
        with open('./vall_e/dataloaders/subtrain_dl.pkl', 'rb') as f:
            subtrain_dl = pickle.load(f)
        with open('./vall_e/dataloaders/val_dl.pkl', 'rb') as f:
            val_dl = pickle.load(f)
    else:    
        train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
        
        if not os.path.exists('./vall_e/dataloaders'):
            os.makedirs('./vall_e/dataloaders')
        
        with open('./vall_e/dataloaders/train_dl.pkl', 'wb') as f:
            pickle.dump(train_dl, f)
        print('train loader saved')
        with open('./vall_e/dataloaders/subtrain_dl.pkl', 'wb') as f:
            pickle.dump(subtrain_dl, f)
        print('subtrain loader saved')
        with open('./vall_e/dataloaders/val_dl.pkl', 'wb') as f:
            pickle.dump(val_dl, f)
        print('validation loader saved')
        
    # train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
    print('make datasets done')
    
    def train_feeder(engines, batch, name):
        model = engines["model"]

        if cfg.model.startswith("ar"):
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resp_list=batch["resp"],
            )
        elif cfg.model.startswith("nar"):
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resps_list=batch["resps"],
            )
        else:
            raise NotImplementedError(cfg.model)

        losses = model.gather_attribute("loss")
        
        loss = torch.stack([*losses.values()]).sum()

        stats = {}
        stats |= {k: v.item() for k, v in losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl):
        log_dir = cfg.log_dir / str(engines.global_step) / name

        model = engines["model"]
        log_dir = cfg.log_dir / str(engines.global_step) / name
        stats = defaultdict(list)
        for batch in tqdm(dl):
            batch: dict = to_device(batch, cfg.device)

            if cfg.model.startswith("ar"):
                resp_list = model(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    max_steps=cfg.max_val_ar_steps,
                    sampling_temperature=cfg.sampling_temperature,
                )
                resps_list = [r.unsqueeze(-1) for r in resp_list]
                
            elif cfg.model.startswith("nar"):
                resps_list = model(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resps_list=[r.unsqueeze(-1) for r in batch["resp"]],
                    sampling_temperature=cfg.sampling_temperature,
                )
            else:
                raise NotImplementedError(cfg.model)

            losses = model.gather_attribute("loss")
            batch_stats = {k: v.item() for k, v in losses.items()}
            for k, v in batch_stats.items():
                stats[k].append(v)
                
            for path, ref, hyp in zip(batch["path"], batch["resps"], resps_list):
                relpath = path.relative_to(cfg.data_root)
                hyp_path = (log_dir / "hyp" / relpath).with_suffix(".wav")
                ref_path = (log_dir / "ref" / relpath).with_suffix(".wav")
                hyp_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                qnt.decode_to_file(ref, ref_path)
                if len(hyp) > 0:
                    qnt.decode_to_file(hyp, hyp_path)

        qnt.unload_model()

        stats = {k: sum(v) / len(v) for k, v in stats.items()}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        _logger.info(f"Eval: {stats}.")

        _logger.info(f"{json.dumps(stats)}.")

    def eval_fn(engines):
        run_eval(engines, "subtrain", subtrain_dl)
        run_eval(engines, "val", val_dl)

    print('training start')
    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
