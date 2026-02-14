"""
Dataset: Flickr30k
Size:
    - Test set: 1,000 images, 5,000 captions.
    - Full: 31k images.
Purpose: Validation / Smaller Benchmark
    - Similar to MSCOCO but smaller, focusing on people-centric activities.
    - Role: Used alongside MSCOCO to demonstrate generalization across different standard datasets. Due to its small size, it allows for quick validation of code and model parameters.
"""

import os,argparse
import json

import torch
import open_clip
from tqdm import tqdm

from dataloaders import Flickr30kDataLoader
from metrics import *
from confidence import *
from utils import *

TABLE = {}  # method -> direction -> dict(S=..., R2=..., negS_R2=...)

def table_set(method: str, direction: str, S: float, R2: float):
    if method not in TABLE:
        TABLE[method] = {}
    TABLE[method][direction] = {
        "S": float(S),
        "R2": float(R2),
        "negS_R2": float(-S * R2),
    }

def print_calibration_table_stdout(title: str, table: dict):
    cols = [
        ("method", 28),
        ("i2t S", 7), ("i2t R2", 7), ("i2t -SR2", 9),
        ("t2i S", 7), ("t2i R2", 7), ("t2i -SR2", 9),
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
        t2i = d.get("text2image", {})
        row = [
            method.ljust(28),
            fmt(i2t.get("S")).rjust(7),
            fmt(i2t.get("R2")).rjust(7),
            fmt(i2t.get("negS_R2")).rjust(9),
            fmt(t2i.get("S")).rjust(7),
            fmt(t2i.get("R2")).rjust(7),
            fmt(t2i.get("negS_R2")).rjust(9),
        ]
        print(" | ".join(row))


def print_calibration_table_latex(title: str, table: dict):
    # Expected directions: "image2text", "text2image"
    methods = list(table.keys())

    print("\n" + title)
    print(r"\begin{tabular}{lccc|ccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c|}{image2text} & \multicolumn{3}{c}{text2image} \\")
    print(r"method & $S$ & $R^2$ & $-SR^2$ & $S$ & $R^2$ & $-SR^2$ \\")
    print(r"\midrule")
    for m in methods:
        i2t = table[m].get("image2text", {})
        t2i = table[m].get("text2image", {})
        def fmt(d, k):
            return "" if k not in d else f"{d[k]:.2f}"
        row = [
            m,
            fmt(i2t, "S"), fmt(i2t, "R2"), fmt(i2t, "negS_R2"),
            fmt(t2i, "S"), fmt(t2i, "R2"), fmt(t2i, "negS_R2"),
        ]
        print(" {} \\\\".format(" & ".join(row)))
    print(r"\bottomrule")
    print(r"\end{tabular}")


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
   
    return metrics

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
    
        if os.path.isfile(f'results/embeddings_flickr_montecarlo_{modelname}.pt'):
            print (f'Loading pre-computed embeddings from results/embeddings_flickr_montecarlo_{modelname}.pt')
            all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_flickr_montecarlo_{modelname}.pt', map_location=device)
            print('Done!')
        else:
            all_img_embs = []
            all_txt_embs = []
            all_scores = []
            for _ in tqdm(range(n_passes)):
                img_embs, txt_embs, truth = get_embeddings(model, dataloader, preprocess, tokenizer,  device)
                all_img_embs.append(img_embs)
                all_txt_embs.append(txt_embs)
                scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
                all_scores.append(scores)
    
            torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_flickr_montecarlo_{modelname}.pt')
            print(f'Saved embeddings into results/embeddings_flickr_montecarlo_{modelname}.pt')
   
        # calculate confidence
        txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

        metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
        metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
        print('-'*40)
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
        print('-'*40)

    return metrics_rejection, metrics_calibration

def eval_ensemble(modelnames, datasets, dataloader, device, method='score_top1var'):
    metrics_rejection = {}
    metrics_calibration = {}
    for modelname in modelnames:
        if os.path.isfile(f'results/embeddings_flickr_ensemble_{modelname}.pt'):
            print (f'Loading pre-computed embeddings from results/embeddings_flickr_ensemble_{modelname}.pt')
            all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_flickr_ensemble_{modelname}.pt', map_location=device)
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
                all_img_embs.append(img_embs)
                all_txt_embs.append(txt_embs)
                scores  = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()
                all_scores.append(scores)
    
            torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_flickr_ensemble_{modelname}.pt')
            print(f'Saved embeddings into results/embeddings_flickr_ensemble_{modelname}.pt')
    
        # calculate confidence
        txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

        metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
        metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
        print('-'*40)
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
        print('-'*40)

    return metrics_rejection, metrics_calibration


def eval_single(modelname, dataloader, device, method='score_top1var'):
    metrics_rejection = {}
    metrics_calibration = {}
    oracle_metrics_rejection = {}
    if os.path.isfile(f'results/embeddings_flickr_single_{modelname}.pt'):
        print (f'Loading pre-computed embeddings from results/embeddings_flickr_single_{modelname}.pt')
        all_txt_embs, all_img_embs, truth, all_scores = safe_torch_load(f'results/embeddings_flickr_single_{modelname}.pt', map_location=device)
    else:
        all_img_embs = []
        all_txt_embs = []
        all_scores = []
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=modelname,
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

        torch.save([all_txt_embs, all_img_embs, truth, all_scores], f'results/embeddings_flickr_single_{modelname}.pt')
        print(f'Saved embeddings into results/embeddings_flickr_single_{modelname}.pt')

    # calculate confidence
    txt_confidence, img_confidence = get_confidence(all_txt_embs, all_img_embs, method)

    metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)
    metrics_calibration[modelname] = get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device)
    print('-'*40)
    print('----', method)
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
    print('-'*40)

    # get oracle confidence values
    txt_confidence, img_confidence = get_oracle_confidence(all_txt_embs, all_img_embs, truth)
    oracle_metrics_rejection[modelname] = get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device)

    return metrics_rejection, metrics_calibration, oracle_metrics_rejection



if __name__ == '__main__':
    dataloader = Flickr30kDataLoader(split="test", batch_size=32).get_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile('results/baselines_flickr.json'):
        print ('Loading pre-computed metrics from results/baselines_flickr.json')
        with open('results/baselines_flickr.json') as f:
            metrics = json.load(f)
    else:
        print ('No pre-computed metrics found (results/baselines_flickr.json). Computing...')
        metrics = eval_baselines(["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"], dataloader, device)
        print ('Saved computed metrics into results/baselines_flickr.json')
        with open('results/baselines_flickr.json', 'w') as f:
            json.dump(metrics,f)
    print_metrics(metrics)

    # Add Single Model Evaluation for all methods
    for method in ['top1similarity','adversarial','adversarial_lin']:
        metrics_rejection, metrics_calibration, oracle_metrics_rejection = eval_single("ViT-L-14", dataloader, device, method=method)
        with open(f'results/metrics_rejection_flickr_single_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_flickr_single_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)
        # Oracle is same for all methods (based on truth), but let's save it once
        if method == 'top1similarity':
             with open('results/metrics_rejection_flickr_single_oracle.json', 'w') as f:
                 json.dump(oracle_metrics_rejection, f)

    for method in ['top1similarity','top1consistency','adversarial','adversarial_lin']:
        metrics_rejection, metrics_calibration = eval_MCD(["ViT-L-14"], dataloader, device, method=method)
        if method == 'score_top1avg': print_metrics(metrics_rejection)
        with open(f'results/metrics_rejection_flickr_montecarlo_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_flickr_montecarlo_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)

        ensemble_dbs = ['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k', 'datacomp_xl_s13b_b90k', 'commonpool_xl_clip_s13b_b90k', 'commonpool_xl_laion_s13b_b90k', 'commonpool_xl_s13b_b90k', 'metaclip_400m', 'metaclip_fullcc', 'dfn2b', 'dfn2b_s39b']
        metrics_rejection, metrics_calibration = eval_ensemble(["ViT-L-14"], ensemble_dbs, dataloader, device, method=method)
        if method == 'score_top1avg': print_metrics(metrics_rejection)
        with open(f'results/metrics_rejection_flickr_ensemble_{method}.json', 'w') as f:
            json.dump(metrics_rejection, f)
        with open(f'results/metrics_calibration_flickr_ensemble_{method}.json', 'w') as f:
            json.dump(metrics_calibration, f)


    #print_calibration_table_latex("Flickr30K", TABLE)
    print_calibration_table_stdout("Flickr30K", TABLE)

