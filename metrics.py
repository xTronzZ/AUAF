import torch
import torch.nn.functional as F

import numpy as np
from scipy.stats import spearmanr, linregress

from tqdm import tqdm
from contextlib import suppress

from dataloaders import dataloader_with_indices, batchify

# Functions for Retrieval Evaluation (from CLIP_benchmark)

def get_metrics(scores, positive_pairs, device, recall_k_list=[1,5,10], batch_size=32):
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = [(0,(batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item())]
        metrics[f"text_retrieval_recall@{recall_k}"] = [(0,(batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item())]

    return metrics

def get_metrics_with_selective_removal(all_scores, truth, txt_confidence, img_confidence, device, recall_k_list=[1,5,10], batch_size=32):

    all_scores_tensor = torch.stack(all_scores)
    mean_scores = torch.mean(all_scores_tensor, dim=0)

    metrics = {}

    # text-to-image
    for recall_k in recall_k_list:
        num_queries = truth.shape[0]
        confidence_indices = torch.argsort(txt_confidence, descending=False)  # sort by increasing confidence (== decreasing uncertainty)
                                                                              # the lower the confidence the more uncertainty
    
        metrics[f"image_retrieval_recall@{recall_k}"] = []
        
        step_size = max(1, num_queries // 20)
        for n_removed in range(0, num_queries + 1, step_size):
            #create a subset of scores by excluding the most uncertain
            remaining_indices = confidence_indices[n_removed:]
            scores_subset = mean_scores[remaining_indices]
            truth_subset = truth[remaining_indices]
    
            if len(scores_subset) == 0: #if there is nothing left we skip the metrics
                 pass
            else:
                metric = (batchify(recall_at_k, scores_subset, truth_subset, batch_size, device, k=recall_k)>0).float().mean().item()
                metrics[f"image_retrieval_recall@{recall_k}"].append((n_removed, metric))
                #print(f"R@{recall_k} ({n_removed} removed) = {metric}")

    # image-to-text
    truth = truth.T # Transpose
    mean_scores = mean_scores.T
    for recall_k in recall_k_list:
        num_queries = truth.shape[0]
        confidence_indices = torch.argsort(img_confidence, descending=False)  # sort by increasing confidence (== decreasing uncertainty)
                                                                              # the lower the confidence the more uncertainty
    
        metrics[f"text_retrieval_recall@{recall_k}"] = []
    
        step_size = max(1, num_queries // 20)
        for n_removed in range(0, num_queries + 1, step_size):
            #create a subset of scores by excluding the most uncertain
            remaining_indices = confidence_indices[n_removed:]
            scores_subset = mean_scores[remaining_indices]
            truth_subset = truth[remaining_indices]
    
            if len(scores_subset) == 0: #if there is nothing left we skip the metrics
                 pass
            else:
                metric = (batchify(recall_at_k, scores_subset, truth_subset, batch_size, device, k=recall_k)>0).float().mean().item()
                metrics[f"text_retrieval_recall@{recall_k}"].append((n_removed, metric))
                #print(f"R@{recall_k} ({n_removed} removed) = {metric}")

    return metrics

def get_metrics_by_confidence_intervals(all_scores, truth, txt_confidence, img_confidence, device,
                                          recall_k_list=[1],
                                          num_bins=10, batch_size=32):
    """
    Compute retrieval recall metrics by binning queries based on their confidence values.
    
    For each modality (text-to-image and image-to-text), we:
      - Compute the range (min and max) of the confidence values.
      - Divide that range into num_bins equally spaced bins.
      - For each bin, select queries with confidence values in that bin and compute the recall@k metric.
      
    Args:
        all_scores (list[torch.Tensor]): List of score tensors to be averaged.
        truth (torch.Tensor): Ground truth indices (for text-to-image retrieval). 
                              For image-to-text, truth will be transposed.
        txt_confidence (torch.Tensor): Confidence values for text queries.
        img_confidence (torch.Tensor): Confidence values for image queries.
        device (torch.device): Device on which to perform computations.
        recall_k_list (list[int], optional): List of k values for recall@k. Defaults to [1, 5, 10].
        num_bins (int, optional): Number of bins to divide the confidence range into. Defaults to 10.
        batch_size (int, optional): Batch size to use in batchify. Defaults to 32.
        
    Returns:
        dict: A dictionary containing metrics. For example, keys like "image_retrieval_recall@1"
              will map to a list of tuples, where each tuple is of the form:
                  ((lower_bound, upper_bound), avg_confidence_in_bin, metric_value)
    """

    # Fix error in some metrics (adversarial_lin) that produce nans
    # Replace NaN with median of valid values to avoid collapsing all values to 0
    txt_nan_mask = torch.isnan(txt_confidence)
    img_nan_mask = torch.isnan(img_confidence)
    
    if txt_nan_mask.any():
        txt_valid = txt_confidence[~txt_nan_mask]
        txt_replacement = txt_valid.median() if len(txt_valid) > 0 else torch.tensor(0.5)
        txt_confidence = torch.where(txt_nan_mask, txt_replacement, txt_confidence)
        print(f"  Warning: Replaced {txt_nan_mask.sum().item()} NaN values in txt_confidence with {txt_replacement.item():.4f}")
    
    if img_nan_mask.any():
        img_valid = img_confidence[~img_nan_mask]
        img_replacement = img_valid.median() if len(img_valid) > 0 else torch.tensor(0.5)
        img_confidence = torch.where(img_nan_mask, img_replacement, img_confidence)
        print(f"  Warning: Replaced {img_nan_mask.sum().item()} NaN values in img_confidence with {img_replacement.item():.4f}") 

    # Average the scores
    all_scores_tensor = torch.stack(all_scores)
    mean_scores = torch.mean(all_scores_tensor, dim=0)
    metrics = {}

    ### TEXT-TO-IMAGE RETRIEVAL
    # Define bins based on the range of txt_confidence
    txt_conf_min = txt_confidence.min().item()
    txt_conf_max = txt_confidence.max().item()
    txt_bin_edges = torch.linspace(txt_conf_min, txt_conf_max, steps=num_bins + 1)
    
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = []
        # Process each bin interval
        for i in range(num_bins):
            lower_bound = txt_bin_edges[i].item()
            upper_bound = txt_bin_edges[i+1].item()
            # For the last bin, include the upper bound
            if i < num_bins - 1:
                bin_mask = (txt_confidence >= lower_bound) & (txt_confidence < upper_bound)
            else:
                bin_mask = (txt_confidence >= lower_bound) & (txt_confidence <= upper_bound)
            
            bin_indices = torch.where(bin_mask)[0]
            if len(bin_indices) == 0:
                continue  # Skip empty bins
            
            scores_subset = mean_scores[bin_indices]
            truth_subset = truth[bin_indices]
            # Compute recall@k metric using your helper functions.
            # Here, batchify(recall_at_k, ...) is assumed to return a tensor where values >0 indicate a hit.
            metric_val = (batchify(recall_at_k, scores_subset, truth_subset, batch_size, device, k=recall_k) > 0).float().mean().item()
            avg_conf = txt_confidence[bin_indices].mean().item()
            
            metrics[f"image_retrieval_recall@{recall_k}"].append(((lower_bound, upper_bound), avg_conf, metric_val))
    
    ### IMAGE-TO-TEXT RETRIEVAL
    # For image-to-text, we transpose the matrices.
    truth_T = truth.T
    mean_scores_T = mean_scores.T
    
    # Define bins based on the range of img_confidence
    img_conf_min = img_confidence.min().item()
    img_conf_max = img_confidence.max().item()
    img_bin_edges = torch.linspace(img_conf_min, img_conf_max, steps=num_bins + 1)
    
    for recall_k in recall_k_list:
        metrics[f"text_retrieval_recall@{recall_k}"] = []
        for i in range(num_bins):
            lower_bound = img_bin_edges[i].item()
            upper_bound = img_bin_edges[i+1].item()
            if i < num_bins - 1:
                bin_mask = (img_confidence >= lower_bound) & (img_confidence < upper_bound)
            else:
                bin_mask = (img_confidence >= lower_bound) & (img_confidence <= upper_bound)
            
            bin_indices = torch.where(bin_mask)[0]
            if len(bin_indices) == 0:
                continue
            
            scores_subset = mean_scores_T[bin_indices]
            truth_subset = truth_T[bin_indices]
            metric_val = (batchify(recall_at_k, scores_subset, truth_subset, batch_size, device, k=recall_k) > 0).float().mean().item()
            avg_conf = img_confidence[bin_indices].mean().item()
            
            metrics[f"text_retrieval_recall@{recall_k}"].append(((lower_bound, upper_bound), avg_conf, metric_val))
    
    return metrics



def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def compute_calibration_metrics(recall_values, uncertainty_levels=None):
    """
    Computes two calibration metrics between uncertainty levels and Recall@1 values:
    
    1. Spearman rank correlation (S): Measures the monotonic relationship between
       uncertainty levels and Recall@1. For an ideal model (if uncertainty is defined
       in increasing order) this would be -1.
       
    2. R^2 of a linear regression fit: Measures how well the drop in performance (Recall@1)
       follows a linear trend with increasing uncertainty levels.
    
    Args:
        recall_values (list or np.array): The Recall@1 values for the bins. For example,
            you might have 10 values, one per uncertainty bin.
        uncertainty_levels (list or np.array, optional): The uncertainty levels corresponding
            to the bins. If None, defaults to np.arange(len(recall_values)).
            IMPORTANT: To expect a Spearman correlation of -1 for an ideal model,
            define uncertainty_levels such that they increase with uncertainty.
            (For instance, uncertainty_levels = [0, 1, 2, ..., 9].)
    
    Returns:
        spearman_corr (float): The Spearman rank correlation coefficient.
        r_squared (float): The R^2 value from the linear regression fit.
    """
    # Convert recall_values to a NumPy array.
    recall_values = np.array(recall_values)
    
    # Handle empty input
    if len(recall_values) == 0:
        print("  Warning: No calibration data available (empty recall_values)")
        return 0.0, 0.0
    
    # If uncertainty_levels not provided, create default levels.
    if uncertainty_levels is None:
        uncertainty_levels = np.arange(len(recall_values))
    else:
        uncertainty_levels = np.array(uncertainty_levels)
    
    # Need at least 2 points for correlation/regression
    if len(recall_values) < 2:
        print(f"  Warning: Insufficient data points ({len(recall_values)}) for calibration metrics")
        return 0.0, 0.0
    
    # Compute the Spearman rank correlation.
    spearman_corr, spearman_p = spearmanr(uncertainty_levels, recall_values)
    
    # Compute the linear regression between uncertainty_levels (x) and recall_values (y).
    reg_result = linregress(uncertainty_levels, recall_values)
    r_squared = reg_result.rvalue ** 2  # rvalue is the correlation coefficient from the regression
    
    return spearman_corr, r_squared


def print_metrics(metrics):
    modelnames = metrics.keys()
    print("+-----+----------------------------------+----------------------------------+")
    print("|     |          Text Retrieval          |          Image Retrieval         |")
    print("+-----+----------------------------------+----------------------------------+")
    print("|     |       R@1     R@5     R@10       |       R@1     R@5     R@10       |")
    for i,modelname in enumerate(modelnames):
        print("|  %i  |       %.2f    %.2f    %.2f       |       %.2f    %.2f    %.2f       |" % (i, metrics[modelname]['text_retrieval_recall@1'][0][1], 
            metrics[modelname]['text_retrieval_recall@5'][0][1], metrics[modelname]['text_retrieval_recall@10'][0][1], metrics[modelname]['image_retrieval_recall@1'][0][1], metrics[modelname]['image_retrieval_recall@5'][0][1], metrics[modelname]['image_retrieval_recall@10'][0][1]))
    print("+-----+----------------------------------+----------------------------------+")
    for i,modelname in enumerate(modelnames):
        print(f'({i} {modelname})')

