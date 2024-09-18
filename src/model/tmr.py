### TODO New run with their contrastive loss, current triplet loss
### TODO Think about metrics
### TODO Similarity in InfoNCE based on motion --> motion?

from typing import Dict, Optional
from torch import Tensor

import torch
import torch.nn as nn
from .temos import TEMOS
from .losses import InfoNCE_with_filtering
from .losses import DTWLoss
from .losses import TripletLossCosine
# from .losses import DifferentiableDTWLoss
# from .losses import FastDifferentiableDTWLoss
from .metrics import all_contrastive_metrics
import wandb
# from .all_val_metrics import retrieval
from ..config import read_config
from ..data.collate import collate_text_motion
from omegaconf import DictConfig
import numpy as np
from hydra.utils import instantiate

# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix

# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


class TMR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1, "dtw": 0.1, "new_loss": 10.0},
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
        log_wandb: bool = True,
        use_dtw: bool = True,
        use_contrastive: bool = False,
        dtw_loss_type: str = "euclidean",
        dtw_margin: float = 0.1,
        wandb_name: str = "TMR",
        run_dir: str = "tmr_cos_loss_0.15",
        threshold_dtw: int = 200,
    ) -> None:
        
        # Initialize module like TEMOS
        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
        )

        config_dict = {
            "lr": lr,
            "temp": temperature,
            "threshold_selfsim": threshold_selfsim,
            "threshold_selfsim_metrics": threshold_selfsim_metrics,
            "use_dtw": use_dtw,
            "dtw_loss_type": dtw_loss_type,
            "dtw_margin": dtw_margin,
            "vae": vae,
            "epochs": 500,
        }

        self.threshold_dtw = threshold_dtw

        for key in lmd:
            config_dict[f"lmd_{key}"] = lmd[key]

        self.log_wandb = log_wandb

        if self.log_wandb:
            wandb.init(entity="mukundshankar", project="tmr_with_dtw", name=wandb_name, config=config_dict)

        self.use_dtw = use_dtw
        self.use_contrastive = use_contrastive

        # adding the contrastive loss
        if self.use_contrastive:
            self.text_contrastive_loss_fn = InfoNCE_with_filtering(
                temperature=temperature, threshold_selfsim=threshold_selfsim, threshold_dtw=threshold_dtw, mode="text"
            )
            self.motion_contrastive_loss_fn = InfoNCE_with_filtering(
                temperature=temperature, threshold_selfsim=threshold_selfsim, threshold_dtw=threshold_dtw, mode="motion"
            )

        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # self.log("dtw_loss_status", "initializing")
        if self.use_dtw:
            if dtw_loss_type == "euclidean":
                self.dtw_loss_fn = DTWLoss(dtw_margin)
            elif dtw_loss_type == "cosine":
                self.dtw_loss_fn = TripletLossCosine(dtw_margin)
        # self.log("dtw_loss_status", "initialized")

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

        cfg = read_config("/vulcanscratch/mukunds/downloads/TMR/old_outputs/tmr_humanml3d_guoh3dfeats_vanilla_model")

        self.val_datasets = {}
        self.protocols = ["normal", "threshold", "guo", "nsim"]
        for protocol in self.protocols:
            if protocol not in self.val_datasets:
                if protocol in ["normal", "threshold", "guo"]:
                    dataset = instantiate(cfg.data, split="test")
                    self.val_datasets.update(
                        {key: dataset for key in ["normal", "threshold", "guo"]}
                    )
                elif protocol == "nsim":
                    self.val_datasets[protocol] = instantiate(cfg.data, split="nsim_test")

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        # positive_motion_distance = batch["positive_motion_distance"]
        # positive_text_distance = batch["positive_text_distance"]

        negative_motion_distance = batch["negative_motion_distance"]        
        negative_text_distance = batch["negative_text_distance"]

        positive_sample_x_dict = batch["positive_sample_x_dict"]
        pos_mask = positive_sample_x_dict["mask"]

        negative_sample_x_dict = batch["negative_sample_x_dict"]
        neg_mask = negative_sample_x_dict["mask"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # sentence embeddings
        sent_emb = batch["sent_emb"]

        # text -> motion
        t_motions, t_latents, t_dists = self(text_x_dict, mask=mask, return_all=True)

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, mask=mask, return_all=True)

        _, pos_latents, _ = self(positive_sample_x_dict, mask=pos_mask, return_all=True)
        _, neg_latents, _ = self(negative_sample_x_dict, mask=neg_mask, return_all=True)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        # NOTE no contrastive loss
        if self.use_contrastive:
            # losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)
            losses["text_contrastive"] = self.text_contrastive_loss_fn(t_latents, m_latents, batch['keyid'], sent_emb)
            losses["motion_contrastive"] = self.motion_contrastive_loss_fn(t_latents, m_latents, batch['keyid'], sent_emb)
            # losses["contrastive"] = self.contrastive_loss_fn(t_latents, pos_latents, sent_emb)

        if self.use_dtw:
            # make sure shapes of these are the same
            losses["dtw"] = self.dtw_loss_fn(m_latents, pos_latents, neg_latents)
            # losses["dtw"] = self.dtw_loss_fn(m_latents, pos_latents)
        
        # losses["new_loss"] = self.contrastive_loss_fn(m_latents, pos_latents)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents
        
        if self.log_wandb:
            wandb.log(losses)
        
        return losses

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses, t_latents, m_latents = self.compute_loss(batch, return_all=True)

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(batch["sent_emb"])

        to_log={}

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
            to_log[f"val_{loss_name}"] = loss_val

        if self.log_wandb:
            wandb.log(to_log)
        return losses["loss"]

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        sent_emb = torch.cat(self.validation_step_sent_emb)

        # Compute the similarity matrix
        sim_matrix = get_sim_matrix(t_latents, m_latents).cpu().numpy()

        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_emb.cpu().numpy(),
            threshold=self.threshold_selfsim_metrics,
        )

        # to_log ={}

        for loss_name in sorted(contrastive_metrics):
            loss_val = contrastive_metrics[loss_name]
            # to_log[f"val_{loss_name}"] = loss_val
            self.log(
                f"val_{loss_name}_epoch",
                loss_val,
                on_epoch=True,
                on_step=False,
            )

        # if self.log_wandb:
            # wandb.log(to_log)

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()

    def compute_sim_matrix(self, dataset, keyids, batch_size=256):
        device = self.device
        nsplit = int(np.ceil(len(dataset) / batch_size))
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        for data in all_data_splitted:
            batch = collate_text_motion(data, device=device)

            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]

            # Encode both motion and text
            latent_text = self.encode(text_x_dict, sample_mean=True)
            latent_motion = self.encode(motion_x_dict, sample_mean=True)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
        returned = {
            "sim_matrix": sim_matrix.cpu().numpy(),
            "sent_emb": sent_embs.cpu().numpy(),
        }
        return returned

    def calculate_all_metrics(self):
        batch_size = 256
        results = {}
        metrics_to_log = {}
        for protocol in self.protocols:
            dataset = self.val_datasets[protocol]
            if protocol not in results:
                if protocol in ["normal", "threshold"]:
                    res = self.compute_sim_matrix(
                    dataset, dataset.keyids, batch_size=batch_size
                    )
                    results.update({key: res for key in ["normal", "threshold"]})
                elif protocol == "nsim":
                    res = self.compute_sim_matrix(
                        dataset, dataset.keyids, batch_size=batch_size
                    )
                    results[protocol] = res
                elif protocol == "guo":
                    keyids = sorted(dataset.keyids)
                    N = len(keyids)

                    # make batches of 32
                    idx = np.arange(N)
                    np.random.seed(0)
                    np.random.shuffle(idx)
                    idx_batches = [
                        idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                    ]

                    # split into batches of 32
                    # batched_keyids = [ [32], [32], [...]]
                    results["guo"] = [
                        self.compute_sim_matrix(
                            dataset,
                            np.array(keyids)[idx_batch],
                            batch_size=batch_size,
                        )
                        for idx_batch in idx_batches
                    ]
            result = results[protocol]

            if protocol == "guo":
                all_metrics = []
                for x in result:
                    sim_matrix = x["sim_matrix"]
                    metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                    all_metrics.append(metrics)

                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = round(
                        float(np.mean([metrics[key] for metrics in all_metrics])), 2
                    )

                metrics = avg_metrics
                protocol_name = protocol
            else:
                sim_matrix = result["sim_matrix"]

                protocol_name = protocol
                if protocol == "threshold":
                    emb = result["sent_emb"]
                    threshold = 0.95
                    protocol_name = protocol + f"_{threshold}"
                else:
                    emb, threshold = None, None
                metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)

            metrics_to_log[protocol] = metrics
        
        if self.log_wandb:
            wandb.log(metrics_to_log)
        
        return