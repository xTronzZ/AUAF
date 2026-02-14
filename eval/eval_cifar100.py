"""
Dataset: CIFAR-100
Size:
    - Test set: 10,000 images (100 per class).
    - Full: 60k images (100 classes).
    - Image Size: 32x32 pixels (Low resolution).
Purpose: Out-of-Distribution (OOD) / Classification / Low-Resolution
    - Presents a domain shift challenge for CLIP models trained on high-res web images.
    - Role: Tests the model's performance on blurry, low-resolution inputs. It serves as an excellent scenario for Calibration: determining if the model appropriately degrades its confidence when facing ambiguous or unclear images.
"""

import os,argparse
import json

import torch
import open_clip
from tqdm import tqdm

from dataloaders import CIFAR100DataLoader
from metrics import *
from confidence import *
from utils import *

TABLE = {} # method -> direction -> dict(S=..., R2=..., negS_R2=...)

def table_set(method: str, direction: str, S: float, R2: float):
    if method not in TABLE:
        TABLE[method] = {}
    TABLE[method][direction] = {
        "S": float(S),
        "R2": float(R2),
        "negS_R2": float(-S * R2),
    }

def print_calibration_table_stdout(title: str, table: dict):
    # Only t2i (Text Retrieval for Image = Classification) is relevant? 
    # Current codebase uses i2t for Text Retrieval (Image -> Text), and t2i for Image Retrieval (Text -> Image).
    # Classification is Given Image -> Find Text. So i2t.
    
    cols = [
        ("method", 28),
        ("i2t S", 7), ("i2t R2", 7), ("i2t -SR2", 9),
    ]

    def fmt(x):
        return "" if x is None else f"{x:.2f}"

    header = " | ".join(c[0].ljust(c[1]) for c in cols)
    sep = "-+-".join("-" * c[1] for c in cols)

    print("\n" + title)
    print(header)
    print(sep)

    for method, d in table.items():
        i2t = d.get("image2text", {})
        row = [
            method.ljust(28),
            fmt(i2t.get("S")).rjust(7),
            fmt(i2t.get("R2")).rjust(7),
            fmt(i2t.get("negS_R2")).rjust(9),
        ]
        print(" | ".join(row))


def eval_baselines(modelnames, dataloader, device):
    metrics = {}
    for modelname in modelnames:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=modelname+"-quickgelu",
            pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer(modelname)
        model.to(device)
        model.eval()
    
        img_embs, txt_embs, truth = get_embeddings(model, dataloader, preprocess, tokenizer,  device)
        
        scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
        metrics[modelname] = get_metrics(scores, truth, device)
        print('done')
   
    return metrics

def eval_single(modelname, dataloader, device, method='score_top1var'):
    metrics_rejection = {}
    metrics_calibration = {}
    oracle_metrics_rejection = {}
    if os.path.isfile(f'results/embeddings_cifar100_single_{modelname}.pt'):
        print (f'Loading pre-computed embeddings from results/embeddings_cifar100_single_{modelname}.pt')
        all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_cifar100_single_{modelname}.pt', map_location=device)
    else:
        all_img_embs = []
        all_txt_embs = []
        all_scores = []
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=modelname+"-quickgelu",
            pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer(modelname)
    
        model.to(device)
        model.eval()

        img_embs, txt_embs, truth = get_embeddings(model, dataloader, preprocess, tokenizer, device)
        all_img_embs.append(img_embs)
        all_txt_embs.append(txt_embs)
        scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
        all_scores.append(scores)

        torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_cifar100_single_{modelname}.pt')
        print(f'Saved embeddings into results/embeddings_cifar100_single_{modelname}.pt')

    # calculate confidence
    txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

    metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
    metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
    print('----',method)
    for k in metrics_calibration[modelname].keys():
      print(k)
      spearman_corr, r_squared = compute_calibration_metrics([m[2] for m in metrics_calibration[modelname][k]])
      print("  Spearman Rank Correlation (S):", spearman_corr)
      print("  R^2 for linear regression:", r_squared)
      if k.startswith("text_retrieval"):
          direction = "image2text"
      elif k.startswith("image_retrieval"):
          direction = "text2image"
      table_set(method, direction, -spearman_corr, r_squared)

    # get oracle confidence values
    txt_confidence, img_confidence = get_oracle_confidence(all_txt_embs, all_img_embs, truth)
    oracle_metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)

    return metrics_rejection, metrics_calibration, oracle_metrics_rejection

def eval_MCD(modelnames, dataloader, device, n_passes=50, method='score_top1var'):
    metrics_rejection = {}
    metrics_calibration = {}
    for modelname in modelnames:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=modelname+"-quickgelu",
            pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer(modelname)
    
        for resblock in model.visual.transformer.resblocks:
            resblock.attn.dropout = 0.2
        for resblock in model.transformer.resblocks:
            resblock.attn.dropout = 0.2
    
        model.to(device)
    
        if os.path.isfile(f'results/embeddings_cifar100_montecarlo_{modelname}.pt'):
            print (f'Loading pre-computed embeddings from results/embeddings_cifar100_montecarlo_{modelname}.pt')
            # Load to CPU to avoid OOM
            all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_cifar100_montecarlo_{modelname}.pt', map_location='cpu')
            print('Done!')
        else:
            all_img_embs = []
            all_txt_embs = []
            all_scores = []
            for _ in tqdm(range(n_passes)):
                img_embs, txt_embs, truth = get_embeddings(model, dataloader, preprocess, tokenizer,  device)
                all_img_embs.append(img_embs.cpu())
                all_txt_embs.append(txt_embs.cpu())
                scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
                all_scores.append(scores.cpu())
    
            torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_cifar100_montecarlo_{modelname}.pt')
            print(f'Saved embeddings into results/embeddings_cifar100_montecarlo_{modelname}.pt')
   
        # calculate confidence
        # Inputs are on CPU, so get_confidence/get_scores will run on CPU, avoiding GPU OOM
        txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

        metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
        metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
        print('----', method, '(MCD)')
        for k in metrics_calibration[modelname].keys():
          print(k)
          spearman_corr, r_squared = compute_calibration_metrics([m[2] for m in metrics_calibration[modelname][k]])
          print("  Spearman Rank Correlation (S):", spearman_corr)
          print("  R^2 for linear regression:", r_squared)
          if k.startswith("text_retrieval"):
              direction = "image2text"
          elif k.startswith("image_retrieval"):
              direction = "text2image"
          table_set(method+' (MCD)', direction, -spearman_corr, r_squared)

    return metrics_rejection, metrics_calibration

def eval_ensemble(modelnames, datasets, dataloader, device, method='score_top1var'):
    metrics_rejection = {}
    metrics_calibration = {}
    for modelname in modelnames:
        if os.path.isfile(f'results/embeddings_cifar100_ensemble_{modelname}.pt'):
            print (f'Loading pre-computed embeddings from results/embeddings_cifar100_ensemble_{modelname}.pt')
            # Load to CPU to avoid OOM
            all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_cifar100_ensemble_{modelname}.pt', map_location='cpu')
        else:
            all_img_embs = []
            all_txt_embs = []
            all_scores = []
            for dataset in tqdm(datasets):
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name=modelname,
                    pretrained=dataset
                )
                tokenizer = open_clip.get_tokenizer(modelname)
            
                model.to(device)
                model.eval()

                img_embs, txt_embs, truth = get_embeddings(model, dataloader, preprocess, tokenizer, device)
                all_img_embs.append(img_embs.cpu())
                all_txt_embs.append(txt_embs.cpu())
                scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
                all_scores.append(scores.cpu())
    
            torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_cifar100_ensemble_{modelname}.pt')
            print(f'Saved embeddings into results/embeddings_cifar100_ensemble_{modelname}.pt')
    
        # calculate confidence
        # Inputs are on CPU, so get_confidence/get_scores will run on CPU, avoiding GPU OOM
        txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

        metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
        metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
        print('----', method, '(Ensemble)')
        for k in metrics_calibration[modelname].keys():
          print(k)
          spearman_corr, r_squared = compute_calibration_metrics([m[2] for m in metrics_calibration[modelname][k]])
          print("  Spearman Rank Correlation (S):", spearman_corr)
          print("  R^2 for linear regression:", r_squared)
          if k.startswith("text_retrieval"):
              direction = "image2text"
          elif k.startswith("image_retrieval"):
              direction = "text2image"
          table_set(method+' (Ens.)', direction, -spearman_corr, r_squared)

    return metrics_rejection, metrics_calibration


if __name__ == '__main__':
    # Use CIFAR100DataLoader
    # one_image_per_class=True to ensure 1-to-1 matching (Instance Retrieval context)
    # This selects 100 images (one per class) to match the 100 unique class captions.
    dataloader = CIFAR100DataLoader(split="test", batch_size=100, one_image_per_class=True).get_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile('results/baselines_cifar100.json'):
        print ('Loading pre-computed metrics from results/baselines_cifar100.json')
        with open('results/baselines_cifar100.json') as f:
            metrics = json.load(f)
    else:
        print ('No pre-computed metrics found (results/baselines_cifar100.json). Computing...')
        # Fewer baselines? stick to list.
        metrics = eval_baselines(["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"], dataloader, device)
        print ('Saved computed metrics into results/baselines_cifar100.json')
        with open('results/baselines_cifar100.json', 'w') as f:
            json.dump(metrics,f)
    print_metrics(metrics)

    # Add Single Model Evaluation
    for method in ['top1similarity','adversarial','adversarial_lin']:
        metrics_rejection, metrics_calibration, _ = eval_single("ViT-L-14", dataloader, device, method=method)
        with open(f'results/metrics_rejection_cifar100_single_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_cifar100_single_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)

    for method in ['top1similarity','top1consistency','adversarial','adversarial_lin']:
        metrics_rejection, metrics_calibration = eval_MCD(["ViT-L-14"], dataloader, device, method=method)
        if method == 'score_top1avg': print_metrics(metrics_rejection)
        with open(f'results/metrics_rejection_cifar100_montecarlo_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_cifar100_montecarlo_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)

        ensemble_dbs = ['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k', 'datacomp_xl_s13b_b90k', 'commonpool_xl_clip_s13b_b90k', 'commonpool_xl_laion_s13b_b90k', 'commonpool_xl_s13b_b90k', 'metaclip_400m', 'metaclip_fullcc', 'dfn2b', 'dfn2b_s39b']
        metrics_rejection, metrics_calibration = eval_ensemble(["ViT-L-14"], ensemble_dbs, dataloader, device, method=method)
        if method == 'score_top1avg': print_metrics(metrics_rejection)
        with open(f'results/metrics_rejection_cifar100_ensemble_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_cifar100_ensemble_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)

    print_calibration_table_stdout("CIFAR-100", TABLE)
