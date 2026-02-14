import os
import json
import torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from dataloaders import HatefulMemesDataLoader
from utils import autocast_ctx

# Configuration
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "hateful_memes")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def construct_prompts_ensemble(texts):
    # Ensemble of templates
    templates_0 = [
        "a harmless meme with text: '{}'",
        "a benign meme with text: '{}'",
        "a funny meme with text: '{}'",
        "a non-offensive meme with text: '{}'"
    ]
    templates_1 = [
        "a hateful meme with text: '{}'",
        "an offensive meme with text: '{}'",
        "a racist meme with text: '{}'",
        "a sexist or mean meme with text: '{}'"
    ]
    
    # We will simply flatten this to: [Batch, N_Templates]
    # But compute_batch_predictions expects list of strings [Batch]
    # To keep code simple, let's just picking a BETTER single prompt for now to avoid dimension mismatch hell
    # Or, we can do manual averaging in compute_batch?
    # Let's start with a much richer single prompt.
    pass

def construct_prompts(texts):
    # Improved Prompts with specific keywords and quotes
    prompts_class0 = [f"a harmless, benign, or funny meme with the text: '{t}'" for t in texts]
    prompts_class1 = [f"a hateful, racist, sexist, or offensive meme with the text: '{t}'" for t in texts]
    return prompts_class0, prompts_class1

def compute_batch_predictions(model, preprocess, tokenizer, batch, device, n_passes=1):
    images = batch["images"]
    texts = batch["texts"]
    labels = batch["labels"].to(device)
    
    # Preprocess images
    image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Preprocess texts
    p0, p1 = construct_prompts(texts)
    text_inputs_0 = tokenizer(p0).to(device)
    text_inputs_1 = tokenizer(p1).to(device)
    
    probs_list = []
    img_embs_list = []
    txt_embs_list = []
    
    autocast = autocast_ctx(device)
    
    model.train() # Enable dropout
    
    with torch.no_grad(), autocast:
        for _ in range(n_passes):
            # Encode
            img_feats = model.encode_image(image_inputs) # [Batch, D]
            txt_feats_0 = model.encode_text(text_inputs_0) # [Batch, D]
            txt_feats_1 = model.encode_text(text_inputs_1) # [Batch, D]
            
            # Normalize
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats_0 = F.normalize(txt_feats_0, dim=-1)
            txt_feats_1 = F.normalize(txt_feats_1, dim=-1)
            
            # Pack Text Features: [Batch, 2, D]
            txt_feats = torch.stack([txt_feats_0, txt_feats_1], dim=1)
            
            # Compute Logits
            logits_0 = (img_feats * txt_feats_0).sum(dim=-1) # [Batch]
            logits_1 = (img_feats * txt_feats_1).sum(dim=-1) # [Batch]
            
            logits = torch.stack([logits_0, logits_1], dim=1)
            logit_scale = model.logit_scale.exp()
            scaled_logits = logits * logit_scale
            probs = F.softmax(scaled_logits, dim=-1) # [Batch, 2]
            
            probs_list.append(probs)
            img_embs_list.append(img_feats)
            txt_embs_list.append(txt_feats)
            
    # Stack over N_passes
    return (torch.stack(probs_list), 
            torch.stack(img_embs_list), 
            torch.stack(txt_embs_list), 
            labels)

def calc_uncertainty_metrics(all_probs, all_img_embs, all_txt_embs):
    total_samples = all_probs.shape[1]
    
    # 1. Similarity (Max Prob)
    mean_probs = all_probs.mean(dim=0) # [Total, 2]
    prob_class1 = mean_probs[:, 1]
    conf_similarity = mean_probs.max(dim=1).values
    
    # 2. Consistency
    hard_preds = all_probs.argmax(dim=-1) # [N, Total]
    votes_1 = (hard_preds == 1).sum(dim=0).float()
    n_passes = all_probs.shape[0]
    votes_0 = n_passes - votes_1
    conf_consistency = torch.max(votes_1, votes_0) / n_passes
    
    # 3. Adversarial (Linear Approx)
    img_mean = all_img_embs.mean(dim=0) # [Total, D]
    txt_mean = all_txt_embs.mean(dim=0) # [Total, 2, D]
    
    conf_adversarial_list = []
    print("Computing Adversarial Confidence...")
    eps = 1e-8
    
    for i in range(total_samples):
        q = img_mean[i].detach().clone().requires_grad_(True)
        gallery = txt_mean[i].detach()
        
        sim = F.cosine_similarity(q.unsqueeze(0), gallery, dim=1)
        sorted_sim, _ = torch.sort(sim, descending=True)
        f = sorted_sim[0] - sorted_sim[1]
        
        grad_f = torch.autograd.grad(f, q, retain_graph=False)[0]
        grad_norm = torch.norm(grad_f) + eps
        
        delta = f / grad_norm
        conf = torch.tanh(delta).item()
        conf_adversarial_list.append(conf)
        
    conf_adversarial = torch.tensor(conf_adversarial_list)
    
    # 4. Joint (Average)
    conf_joint = (conf_similarity + conf_consistency + conf_adversarial) / 3.0
    
    return {
        "prob_class1": prob_class1,
        "conf_similarity": conf_similarity,
        "conf_consistency": conf_consistency,
        "conf_adversarial": conf_adversarial,
        "conf_joint": conf_joint
    }

def eval_rejection(scores, labels, confidence, metric='auroc'):
    sorted_indices = torch.argsort(confidence, descending=True)
    n_samples = len(labels)
    step_size = max(1, n_samples // 20)
    
    results = []
    
    for n_removed in range(0, n_samples - step_size, step_size):
        n_keep = n_samples - n_removed
        keep_indices = sorted_indices[:n_keep]
        
        curr_scores = scores[keep_indices].cpu().numpy()
        curr_labels = labels[keep_indices].cpu().numpy()
        
        try:
            if metric == 'auroc':
                if len(np.unique(curr_labels)) < 2:
                    current_metric = 0.5 
                else:
                    current_metric = roc_auc_score(curr_labels, curr_scores)
            elif metric == 'accuracy':
                preds = (curr_scores > 0.5).astype(int)
                current_metric = accuracy_score(curr_labels, preds)
        except Exception as e:
            current_metric = 0.0
            
        results.append({
            "n_removed": n_removed,
            "fraction_removed": n_removed / n_samples,
            "metric_val": current_metric
        })
        
    return results

def main():
    dataloader = HatefulMemesDataLoader(
        data_root=DATA_ROOT, 
        split="seen/dev_seen", 
        batch_size=32
    ).get_dataloader()
    
    model_name = "ViT-L-14"
    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    dropout_p = 0.1
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        for resblock in model.visual.transformer.resblocks:
            resblock.attn.dropout = dropout_p
    if hasattr(model, 'transformer'):
        for resblock in model.transformer.resblocks:
            resblock.attn.dropout = dropout_p
            
    model.to(DEVICE)
    model.train() 
    
    all_probs_batches = []
    all_img_batches = []
    all_txt_batches = []
    all_labels_batches = []
    
    N_PASSES = 20
    
    print("Running Inference...")
    for batch in tqdm(dataloader):
        p, img, txt, lbl = compute_batch_predictions(model, preprocess, tokenizer, batch, DEVICE, n_passes=N_PASSES)
        all_probs_batches.append(p.cpu())
        all_img_batches.append(img.cpu())
        all_txt_batches.append(txt.cpu())
        all_labels_batches.append(lbl.cpu())
        
    all_probs = torch.cat(all_probs_batches, dim=1) # [N, Total, 2]
    all_img_embs = torch.cat(all_img_batches, dim=1) # [N, Total, D]
    all_txt_embs = torch.cat(all_txt_batches, dim=1) # [N, Total, 2, D]
    all_labels = torch.cat(all_labels_batches, dim=0) # [Total]
    
    print(f"Total samples: {len(all_labels)}")
    
    metrics = calc_uncertainty_metrics(all_probs, all_img_embs, all_txt_embs)
    prob_class1 = metrics["prob_class1"]
    
    baseline_auroc = roc_auc_score(all_labels.numpy(), prob_class1.numpy())
    print(f"Baseline AUROC: {baseline_auroc:.4f}")
    
    curve_sim = eval_rejection(prob_class1, all_labels, metrics["conf_similarity"], metric='auroc')
    curve_con = eval_rejection(prob_class1, all_labels, metrics["conf_consistency"], metric='auroc')
    curve_adv = eval_rejection(prob_class1, all_labels, metrics["conf_adversarial"], metric='auroc')
    curve_joint = eval_rejection(prob_class1, all_labels, metrics["conf_joint"], metric='auroc')
    
    results = {
        "baseline": {"auroc": baseline_auroc},
        "curve_similarity": curve_sim,
        "curve_consistency": curve_con,
        "curve_adversarial": curve_adv,
        "curve_joint": curve_joint
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/hateful_memes_results_joint.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Results saved to results/hateful_memes_results_joint.json")
    
    print("\n--- Rejection Curve (AUROC) ---")
    print(f"{'Rem%':<6} | {'Sim':<8} | {'Con':<8} | {'Adv':<8} | {'Joint':<8}")
    for i in range(0, len(curve_sim), 5):
        r1 = curve_sim[i]
        r2 = curve_con[i]
        r3 = curve_adv[i]
        r4 = curve_joint[i]
        frac = r1['fraction_removed'] * 100
        print(f"{frac:<6.1f} | {r1['metric_val']:.4f}   | {r2['metric_val']:.4f}   | {r3['metric_val']:.4f}   | {r4['metric_val']:.4f}")

if __name__ == "__main__":
    main()
