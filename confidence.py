import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.autograd import grad

from collections import Counter

from utils import get_scores

def ranking_confidence_top1similarity(all_txt_embs, all_img_embs):
    # use cosine similarity score between query and top1 item as a confidence measure
    # the larger the similarity score the more confident are the models' predictions

    all_scores_tensor = get_scores(all_txt_embs, all_img_embs) # Shape: (N, 5000, 1000)

    # text-to-image retrieval
    txt_confidence = torch.max(torch.mean(all_scores_tensor, dim=0), dim=1).values.cpu() # average scores of all predictors and get top1 score

    # image-to-text retrieval
    img_confidence = torch.max(torch.mean(all_scores_tensor, dim=0).T, dim=1).values.cpu() # average scores of all predictors and get top1 score

    return txt_confidence, img_confidence


def ranking_confidence_top1consistency(all_txt_embs, all_img_embs):
    """
    Compute ranking confidence scores based on top-1 consistency across Monte Carlo dropout samples.
    
    For text queries, each dropout sample is used to rank the gallery images and the most frequent 
    top-1 image (across samples) is taken as the robust prediction. The confidence is the fraction 
    of dropout samples that vote for that same top-1.
    
    Similarly, for image queries, each dropout sample is used to rank the gallery texts.
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
    
    Returns:
      txt_confidence: a tensor of shape (num_text,) with confidence scores for text queries.
      img_confidence: a tensor of shape (num_img,) with confidence scores for image queries.
    """
    # Stack the Monte Carlo samples:
    txt_mc = torch.stack(all_txt_embs, dim=0)  # (N, num_text, D)
    img_mc = torch.stack(all_img_embs, dim=0)  # (N, num_img, D)
    N = txt_mc.shape[0]
    num_text = txt_mc.shape[1]
    num_img = img_mc.shape[1]
    
    # ---- Text queries ranking images ----
    txt_conf_list = []
    for i in range(num_text):
        top1_indices = []
        for k in range(N):
            # Get the i-th text query sample from the k-th dropout sample:
            q = txt_mc[k, i, :]  # (D,)
            # Use the k-th dropout sample for all gallery images:
            gallery = img_mc[k, :, :]  # (num_img, D)
            sim = F.cosine_similarity(q.unsqueeze(0), gallery, dim=1)  # (num_img,)
            top1_idx = int(torch.argmax(sim).item())
            top1_indices.append(top1_idx)
        # Count which gallery index appears most frequently:
        counter = Counter(top1_indices)
        mode_count = counter.most_common(1)[0][1]
        confidence = mode_count / N
        txt_conf_list.append(confidence)
    txt_confidence = torch.tensor(txt_conf_list)
    
    # ---- Image queries ranking texts ----
    img_conf_list = []
    for j in range(num_img):
        top1_indices = []
        for k in range(N):
            # For image queries, each image's dropout sample is used to rank all texts:
            q = img_mc[k, j, :]  # (D,)
            gallery = txt_mc[k, :, :]  # (num_text, D)
            sim = F.cosine_similarity(q.unsqueeze(0), gallery, dim=1)  # (num_text,)
            top1_idx = int(torch.argmax(sim).item())
            top1_indices.append(top1_idx)
        counter = Counter(top1_indices)
        mode_count = counter.most_common(1)[0][1]
        confidence = mode_count / N
        img_conf_list.append(confidence)
    img_confidence = torch.tensor(img_conf_list)
    
    return txt_confidence, img_confidence

def ranking_confidence_top5consistency(all_txt_embs, all_img_embs, top_k=5):
    """
    Compute ranking confidence scores based on top-k (default top-5) consistency across Monte Carlo dropout samples.
    
    For text queries, each dropout sample is used to rank the gallery images and the top-k candidates are extracted.
    A consensus top-k set is computed from all dropout samples, and the confidence is defined as the average
    Jaccard similarity between each dropout sample's top-k set and the consensus top-k set.
    
    Similarly, for image queries, each dropout sample is used to rank the gallery texts.
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
      top_k: number of top candidates to consider (default is 5)
    
    Returns:
      txt_confidence: a tensor of shape (num_text,) with confidence scores for text queries.
      img_confidence: a tensor of shape (num_img,) with confidence scores for image queries.
    """
    # Helper function: compute Jaccard similarity between two sets
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    # Stack the Monte Carlo samples:
    txt_mc = torch.stack(all_txt_embs, dim=0)  # shape: (N, num_text, D)
    img_mc = torch.stack(all_img_embs, dim=0)  # shape: (N, num_img, D)
    N = txt_mc.shape[0]
    num_text = txt_mc.shape[1]
    num_img = img_mc.shape[1]
    
    # ---- Text queries ranking images ----
    txt_conf_list = []
    for i in range(num_text):
        topk_sets = []
        all_indices = []
        for s in range(N):
            # Get the i-th text query from the s-th dropout sample
            q = txt_mc[s, i, :]  # (D,)
            # Use the corresponding dropout sample for the gallery images:
            gallery = img_mc[s, :, :]  # (num_img, D)
            sim = F.cosine_similarity(q.unsqueeze(0), gallery, dim=1)  # (num_img,)
            # Get the top_k indices
            _, topk_idxs = torch.topk(sim, k=top_k)
            topk_set = set(topk_idxs.tolist())
            topk_sets.append(topk_set)
            all_indices.extend(topk_idxs.tolist())
        # Compute consensus top_k set as the top_k most frequent indices across dropout samples:
        counter = Counter(all_indices)
        consensus_topk = set([idx for idx, count in counter.most_common(top_k)])
        # Compute the average Jaccard similarity over all dropout samples:
        jaccard_scores = [jaccard_similarity(tk_set, consensus_topk) for tk_set in topk_sets]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        txt_conf_list.append(avg_jaccard)
    txt_confidence = torch.tensor(txt_conf_list)
    
    # ---- Image queries ranking texts ----
    img_conf_list = []
    for j in range(num_img):
        topk_sets = []
        all_indices = []
        for s in range(N):
            q = img_mc[s, j, :]  # (D,)
            gallery = txt_mc[s, :, :]  # (num_text, D)
            sim = F.cosine_similarity(q.unsqueeze(0), gallery, dim=1)  # (num_text,)
            _, topk_idxs = torch.topk(sim, k=top_k)
            topk_set = set(topk_idxs.tolist())
            topk_sets.append(topk_set)
            all_indices.extend(topk_idxs.tolist())
        counter = Counter(all_indices)
        consensus_topk = set([idx for idx, count in counter.most_common(top_k)])
        jaccard_scores = [jaccard_similarity(tk_set, consensus_topk) for tk_set in topk_sets]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        img_conf_list.append(avg_jaccard)
    img_confidence = torch.tensor(img_conf_list)
    
    return txt_confidence, img_confidence

def _adv_perturbation_for_query(query, gallery, original_top1, step_size, max_iter):
    """
    Helper function: given a query vector and a gallery (matrix of embeddings),
    perform a PGD attack to find the minimum perturbation (in L2 norm) required
    to change the top-1 candidate.
    """
    q0 = query.clone().detach()
    q_adv = q0.clone().detach().requires_grad_(True)
    #print('------------START----------------')
    for _ in range(max_iter):
        # Compute cosine similarities between q_adv and each gallery item.
        sim = F.cosine_similarity(q_adv.unsqueeze(0), gallery, dim=1)  # (num_gallery,)
        i_top_new = torch.argmax(sim).item()
        if i_top_new != original_top1:
            break  # Top-1 has flipped.
        # To push the ranking, we try to reduce the margin:
        # Let loss = sim[original_top1] - (max_{j != original_top1} sim[j])
        sim_clone = sim.clone()
        sim_clone[original_top1] = -1e4  # exclude original top-1
        runnerup_val = sim_clone.max()
        loss = sim[original_top1] - runnerup_val
        #print(loss)
        loss.backward()
        grad_val = q_adv.grad
        if grad_val is None or grad_val.norm() == 0:
            break
        # Normalize the gradient.
        grad_normalized = grad_val / (grad_val.norm() + 1e-10)
        with torch.no_grad():
            q_adv = q_adv - step_size * grad_normalized
        q_adv = q_adv.detach().requires_grad_(True)
    #print('------------END------------------')
    perturbation_norm = (q_adv - q0).norm()
    return perturbation_norm

def ranking_confidence_adversarial(all_txt_embs, all_img_embs, step_size=0.025, max_iter=50, lambda_scale=0.1):
    """
    Compute confidence scores based on the robustness to adversarial perturbations.
    
    For each query (text or image), we use the mean embedding over dropout samples and 
    compute the minimum L2 perturbation required to flip the top-1 ranking (when comparing to 
    the gallery, computed as the mean over dropout samples of the other modality). The confidence 
    is then defined as a (normalized) measure of the required perturbation (larger perturbation 
    indicates a more robust—and thus more confident—ranking).
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
      step_size: PGD step size.
      max_iter: maximum number of PGD iterations.
      lambda_scale: scaling constant to normalize the perturbation norm (e.g., set so that if the 
                    required norm is greater than lambda_scale, confidence saturates at 1).
    
    Returns:
      txt_confidence: tensor of shape (num_text,) with confidence scores for text queries.
      img_confidence: tensor of shape (num_img,) with confidence scores for image queries.
    """
    # Compute mean embeddings over dropout samples.
    txt_mean = torch.stack(all_txt_embs, dim=0).mean(dim=0)  # (num_text, D)
    img_mean = torch.stack(all_img_embs, dim=0).mean(dim=0)    # (num_img, D)
    num_text = txt_mean.shape[0]
    num_img = img_mean.shape[0]
    
    txt_conf_list = []
    for i in range(num_text):
        q = txt_mean[i]
        # For a text query, gallery = image mean embeddings.
        sim = F.cosine_similarity(q.unsqueeze(0), img_mean, dim=1)  # (num_img,)
        original_top1 = torch.argmax(sim).item()
        perturb_norm = _adv_perturbation_for_query(q, img_mean, original_top1, step_size, max_iter)
        # Define confidence: here, we use a simple linear mapping capped at 1.
        #conf = min(perturb_norm / lambda_scale, 1.0)
        conf = torch.tanh(perturb_norm)
        txt_conf_list.append(conf)
    txt_confidence = torch.tensor(txt_conf_list)
    
    img_conf_list = []
    for j in range(num_img):
        q = img_mean[j]
        # For an image query, gallery = text mean embeddings.
        sim = F.cosine_similarity(q.unsqueeze(0), txt_mean, dim=1)  # (num_text,)
        original_top1 = torch.argmax(sim).item()
        perturb_norm = _adv_perturbation_for_query(q, txt_mean, original_top1, step_size, max_iter)
        #conf = min(perturb_norm / lambda_scale, 1.0)
        conf = torch.tanh(perturb_norm)
        img_conf_list.append(conf)
    img_confidence = torch.tensor(img_conf_list)
    
    return txt_confidence, img_confidence

def ranking_confidence_adversarial_linear(all_txt_embs, all_img_embs, eps=1e-8):
    """
    Compute ranking confidence based on adversarial robustness.
    
    For each query (text or image) we use the mean embedding (over dropout samples)
    and compute the cosine similarities with the gallery mean embeddings. We then define
    a score f = (top1_sim - top2_sim) and approximate the minimum perturbation (via a linearization)
    required to change the top-1 ranking. The higher the required perturbation, the more robust
    (and confident) the ranking.
    
    The confidence is defined as:
         confidence = tanh(delta)
    where delta = f / (||grad f|| + eps)
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
      eps: small constant to avoid division by zero.
    
    Returns:
      txt_confidence: tensor of shape (num_text,) with confidence scores for text queries.
      img_confidence: tensor of shape (num_img,) with confidence scores for image queries.
    """

    # Compute mean embeddings over dropout samples.
    txt_mean = torch.stack(all_txt_embs, dim=0).mean(dim=0)  # (num_text, D)
    img_mean = torch.stack(all_img_embs, dim=0).mean(dim=0)  # (num_img, D)

    num_text = txt_mean.shape[0]
    num_img = img_mean.shape[0]

    # Helper: given query embeddings and gallery embeddings, compute adversarial confidence.
    def compute_adv_conf(query_mean, gallery_mean):
        conf_list = []
        device = query_mean.device  # Get the device from input tensors
        # Pre-compute gallery norms (not strictly needed but may help speed up)
        for i in range(query_mean.shape[0]):
            # Use query i as variable.
            q = query_mean[i].clone().detach().requires_grad_(True)  # (D,)
            # Compute cosine similarities with all gallery items.
            sim = F.cosine_similarity(q.unsqueeze(0), gallery_mean, dim=1)  # (num_gallery,)
            # Get top-1 and top-2 indices.
            sorted_sim, indices = torch.sort(sim, descending=True)
            top1_sim = sorted_sim[0]
            top2_sim = sorted_sim[1] if sorted_sim.numel() > 1 else sorted_sim[0]
            # Define the margin function.
            f = top1_sim - top2_sim
            
            # Handle edge case where f is very small or zero
            if f.item() < eps:
                # Very small margin means low confidence
                conf_list.append(torch.tensor(0.0, device=device))
                continue
                
            # Compute gradient.
            try:
                grad_f = torch.autograd.grad(f, q, retain_graph=False)[0]
                grad_norm = torch.norm(grad_f)
                
                # Handle zero gradient case
                if grad_norm < eps:
                    # Zero gradient with non-zero margin means very high confidence
                    conf_list.append(torch.tensor(1.0, device=device))
                    continue
                    
                delta = f / (grad_norm + eps)
                # Map delta to [0,1] via tanh (for positive delta).
                conf = torch.tanh(delta)
                
                # Check for NaN
                if torch.isnan(conf):
                    conf = torch.tensor(0.0, device=device)
                    
                conf_list.append(conf)
            except RuntimeError as e:
                # Gradient computation failed
                print(f"  Warning: Gradient computation failed for query {i}: {e}")
                conf_list.append(torch.tensor(0.0, device=device))
                
        return torch.stack(conf_list)  # (num_q,)

    # For text queries, use txt_mean as query and img_mean as gallery.
    txt_confidence = compute_adv_conf(txt_mean, img_mean)
    # For image queries, swap roles.
    img_confidence = compute_adv_conf(img_mean, txt_mean)

    return txt_confidence, img_confidence


def _adv_perturbation_for_query_mc(query, gallery, original_top1, step_size, max_iter):
    """
    Given a single query vector and a gallery matrix, perform a PGD attack to find the minimal
    L2 perturbation required to flip the top-1 candidate.
    """
    q0 = query.clone().detach()
    q_adv = q0.clone().detach().requires_grad_(True)
    for _ in range(max_iter):
        # Compute cosine similarities between q_adv and each gallery item.
        sim = F.cosine_similarity(q_adv.unsqueeze(0), gallery, dim=1)  # shape: (num_gallery,)
        # To exclude the original top-1 candidate, set its score to a value below the minimal possible cosine similarity.
        sim_clone = sim.clone()
        sim_clone[original_top1] = -1.1  # since cosine similarity is in [-1, 1]
        runnerup_val = sim_clone.max()
        loss = sim[original_top1] - runnerup_val
        loss.backward()
        grad_val = q_adv.grad
        if grad_val is None or grad_val.norm() == 0:
            break
        # Normalize the gradient.
        grad_normalized = grad_val / (grad_val.norm() + 1e-10)
        with torch.no_grad():
            q_adv = q_adv - step_size * grad_normalized
        q_adv = q_adv.detach().requires_grad_(True)
    perturbation_norm = (q_adv - q0).norm()
    return perturbation_norm

def ranking_confidence_adversarial_mc(all_txt_embs, all_img_embs, step_size=0.05, max_iter=50):
    """
    Compute adversarial perturbation confidence using Monte Carlo dropout samples for the query.
    
    Instead of computing the perturbation on the mean query embedding, we compute it for each dropout
    sample of the query and then aggregate these perturbation norms. The gallery is represented using the 
    mean embedding over its dropout samples.
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
      step_size: PGD step size.
      max_iter: maximum PGD iterations.
      
    Returns:
      txt_confidence: tensor of shape (num_text,) with adversarial confidence for text queries.
      img_confidence: tensor of shape (num_img,) with adversarial confidence for image queries.
    """
    # Compute the gallery mean embeddings.
    img_mean = torch.stack(all_img_embs, dim=0).mean(dim=0)  # (num_img, D)
    txt_mean = torch.stack(all_txt_embs, dim=0).mean(dim=0)    # (num_text, D)
    
    # Process text queries:
    txt_mc = torch.stack(all_txt_embs, dim=0)  # shape: (N, num_text, D)
    N, num_text, D = txt_mc.shape
    txt_conf_list = []
    for q_idx in range(num_text):
        perturbations = []
        for s in range(N):
            # For each dropout sample, get the query embedding.
            q = txt_mc[s, q_idx, :].clone().detach()
            # Compute cosine similarities with the gallery (using gallery mean).
            sim = F.cosine_similarity(q.unsqueeze(0), img_mean, dim=1)  # shape: (num_img,)
            original_top1 = torch.argmax(sim).item()
            # Compute the perturbation required for this dropout sample.
            perturb_norm = _adv_perturbation_for_query_mc(q, img_mean, original_top1, step_size, max_iter)
            perturbations.append(perturb_norm)
        # Aggregate the perturbation norms (e.g., take the average).
        avg_perturb = torch.stack(perturbations).mean()
        # Map the perturbation norm to a confidence value.
        # Here, we use tanh so that a larger required perturbation (more robust) gives a higher confidence.
        conf = torch.tanh(avg_perturb)
        txt_conf_list.append(conf)
    txt_confidence = torch.stack(txt_conf_list)
    
    # Process image queries (using text mean as gallery):
    img_mc = torch.stack(all_img_embs, dim=0)  # shape: (N, num_img, D)
    N_img, num_img, D_img = img_mc.shape
    img_conf_list = []
    for q_idx in range(num_img):
        perturbations = []
        for s in range(N_img):
            q = img_mc[s, q_idx, :].clone().detach()
            sim = F.cosine_similarity(q.unsqueeze(0), txt_mean, dim=1)
            original_top1 = torch.argmax(sim).item()
            perturb_norm = _adv_perturbation_for_query_mc(q, txt_mean, original_top1, step_size, max_iter)
            perturbations.append(perturb_norm)
        avg_perturb = torch.stack(perturbations).mean()
        conf = torch.tanh(avg_perturb)
        img_conf_list.append(conf)
    img_confidence = torch.stack(img_conf_list)
    
    return txt_confidence, img_confidence

def linear_perturbation_for_query(query, gallery, original_top1, epsilon=0.1):
    """
    Compute an approximate adversarial perturbation using a linear approximation.
    
    Instead of using PGD, we compute the gradient of the cosine similarity w.r.t the query embedding
    and use it to approximate the minimal perturbation required to flip the top-1 ranking.
    
    Parameters:
      query: (D,) tensor representing the query embedding.
      gallery: (num_gallery, D) tensor of gallery embeddings.
      original_top1: Index of the original top-1 gallery item.
      epsilon: Step size to simulate the perturbation direction.
      
    Returns:
      perturbation_norm: L2 norm of the estimated perturbation vector.
    """
    # Compute cosine similarities between query and gallery.
    sim = F.cosine_similarity(query.unsqueeze(0), gallery, dim=1)  # shape: (num_gallery,)
    
    # Get the gradient of the similarity with respect to the query embedding.
    # We want to perturb the query in the direction that maximizes the similarity to the current top-2 candidate.
    sim_clone = sim.clone()
    sim_clone[original_top1] = -1.1  # Exclude the original top-1 candidate.
    runnerup_idx = torch.argmax(sim_clone).item()
    
    # Compute the gradient with respect to the query embedding
    sim_grad = torch.autograd.grad(sim[original_top1], query, retain_graph=False)[0]
    
    # Calculate perturbation: perturb query in the direction of the gradient of the runner-up.
    perturbation = epsilon * sim_grad / (sim_grad.norm() + 1e-10)  # Normalize gradient to get a unit direction
    
    perturbation_norm = perturbation.norm()
    return perturbation_norm

def ranking_confidence_adversarial_mc_linear(all_txt_embs, all_img_embs, epsilon=0.1):
    """
    Compute adversarial perturbation confidence using Monte Carlo dropout samples for the query,
    using a linear approximation for the perturbation.
    
    Parameters:
      all_txt_embs: list of N tensors, each with shape (num_text, D)
      all_img_embs: list of N tensors, each with shape (num_img, D)
      epsilon: Perturbation step size for linear approximation.
      
    Returns:
      txt_confidence: tensor of shape (num_text,) with adversarial confidence for text queries.
      img_confidence: tensor of shape (num_img,) with adversarial confidence for image queries.
    """
    # Compute the gallery mean embeddings.
    img_mean = torch.stack(all_img_embs, dim=0).mean(dim=0)  # (num_img, D)
    txt_mean = torch.stack(all_txt_embs, dim=0).mean(dim=0)  # (num_text, D)
    
    # Process text queries:
    txt_mc = torch.stack(all_txt_embs, dim=0)  # shape: (N, num_text, D)
    N, num_text, D = txt_mc.shape
    txt_conf_list = []
    for q_idx in range(num_text):
        perturbations = []
        for s in range(N):
            # For each dropout sample, get the query embedding.
            q = txt_mc[s, q_idx, :].clone().detach().requires_grad_(True)
            # Compute cosine similarities with the gallery (using gallery mean).
            sim = F.cosine_similarity(q.unsqueeze(0), img_mean, dim=1)  # shape: (num_img,)
            original_top1 = torch.argmax(sim).item()
            # Compute the perturbation required for this dropout sample using the linear approximation.
            perturb_norm = linear_perturbation_for_query(q, img_mean, original_top1, epsilon)
            perturbations.append(perturb_norm)
        # Aggregate the perturbation norms (e.g., take the average).
        avg_perturb = torch.stack(perturbations).mean()
        # Map the perturbation norm to a confidence value.
        conf = torch.tanh(avg_perturb)
        txt_conf_list.append(conf)
    txt_confidence = torch.stack(txt_conf_list)
    
    # Process image queries (using text mean as gallery):
    img_mc = torch.stack(all_img_embs, dim=0)  # shape: (N, num_img, D)
    N_img, num_img, D_img = img_mc.shape
    img_conf_list = []
    for q_idx in range(num_img):
        perturbations = []
        for s in range(N_img):
            q = img_mc[s, q_idx, :].clone().detach().requires_grad_(True)
            sim = F.cosine_similarity(q.unsqueeze(0), txt_mean, dim=1)
            original_top1 = torch.argmax(sim).item()
            perturb_norm = linear_perturbation_for_query(q, txt_mean, original_top1, epsilon)
            perturbations.append(perturb_norm)
        avg_perturb = torch.stack(perturbations).mean()
        conf = torch.tanh(avg_perturb)
        img_conf_list.append(conf)
    img_confidence = torch.stack(img_conf_list)
    
    return txt_confidence, img_confidence

def ranking_confidence_joint_bidirectional(all_txt_embs, all_img_embs, base_method='adversarial_lin'):
    """
    Compute a joint bidirectional confidence score by fusing independent text-to-image and 
    image-to-text confidences. This implements a "Cycle Consistency" idea where a pair 
    is considered confident only if both directions are confident.
    
    Formula: Harmonic Mean = 2 * (C_t2i * C_i2t) / (C_t2i + C_i2t)
    
    Returns:
      joint_conf: tensor of joint confidence scores. Returned twice to match the signature 
                  of other functions (txt_conf, img_conf).
    """
    # Verify base method exists
    if base_method not in confidence_methods:
        raise ValueError(f"Base method {base_method} not found in confidence_methods")
        
    # Get independent confidences
    # Note: confidence_methods is a global dict defined below. 
    # Python resolves this at runtime, so it is fine.
    func = confidence_methods[base_method]
    txt_conf, img_conf = func(all_txt_embs, all_img_embs)
    
    # Check if we can fuse them (must be same length, i.e., 1-to-1 pairs)
    if txt_conf.shape != img_conf.shape:
        print(f"Warning: Bidirectional fusion requires matched shapes. "
              f"Got txt {txt_conf.shape} vs img {img_conf.shape}. "
              f"Returning independent confidences.")
        return txt_conf, img_conf
        
    # Harmonic Mean Fusion
    eps = 1e-6
    joint_conf = 2 * (txt_conf * img_conf) / (txt_conf + img_conf + eps)
    
    return joint_conf, joint_conf

confidence_methods = {
        'top1similarity': ranking_confidence_top1similarity,
	'top1consistency': ranking_confidence_top1consistency,
	'top5consistency': ranking_confidence_top5consistency,
	'adversarial': ranking_confidence_adversarial,
        'adversarial_lin': ranking_confidence_adversarial_linear,
        'adversarial_mc': ranking_confidence_adversarial_mc, # TOO SLOW!
        'adversarial_mc_lin': ranking_confidence_adversarial_mc_linear, 
        'joint_bidirectional': ranking_confidence_joint_bidirectional,
        }

def get_confidence(all_txt_embs, all_img_embs, method=''):
    f = confidence_methods[method]
    return f(all_txt_embs, all_img_embs)

def get_oracle_confidence(txt_embs, img_embs, truth):
    if isinstance(txt_embs, list):
        txt_embs = txt_embs[0]
        img_embs = img_embs[0]
    scores = F.normalize(txt_embs, dim=-1) @ F.normalize(img_embs, dim=-1).t()

    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    sorted_truth = torch.gather(truth, dim=1, index=sorted_indices)
    first_relevant_txt = (sorted_truth == 1).int().argmax(dim=1)
    txt_confidence = 1./first_relevant_txt

    sorted_indices = torch.argsort(scores.T, dim=-1, descending=True)
    sorted_truth = torch.gather(truth.T, dim=1, index=sorted_indices)
    first_relevant_img = (sorted_truth == 1).int().argmax(dim=1)
    img_confidence = 1./first_relevant_img
    return txt_confidence, img_confidence
