import torch
import torch.nn.functional as F

from tqdm import tqdm
from contextlib import suppress

from dataloaders import dataloader_with_indices, batchify


def safe_torch_load(path, map_location=None):
    """torch.load wrapper compatible across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def autocast_ctx(device: str, enabled: bool = True):
    """Return an autocast context manager compatible across torch versions."""
    if not enabled:
        return suppress()

    # torch>=2: torch.amp.autocast
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            # preferred signature
            return torch.amp.autocast(device_type=device)
        except TypeError:
            # older signature
            return torch.amp.autocast(device)

    # torch<2: torch.cuda.amp.autocast (cuda only)
    if device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast()

    return suppress()


def get_scores(all_txt_embs, all_img_embs):
    try:
        all_txt_embs_t = torch.stack(all_txt_embs)  # Shape: (N, 5000, 768)
        all_img_embs_t = torch.stack(all_img_embs)  # Shape: (N, 1000, 768)
        # Normalize in batch
        all_txt_embs_t = F.normalize(all_txt_embs_t, dim=-1)  # Shape: (N, 5000, 768)
        all_img_embs_t = F.normalize(all_img_embs_t, dim=-1)  # Shape: (N, 1000, 768)
        # Transpose all_img_embs to (N, 768, 1000)
        all_img_embs_t = all_img_embs_t.transpose(1, 2)
        # Use batched matrix multiplication
        all_scores_tensor = torch.bmm(all_txt_embs_t, all_img_embs_t)  # Shape: (N, 5000, 1000)
        return all_scores_tensor
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU OOM in get_scores, switching to CPU for large matrix computation...")
            torch.cuda.empty_cache()
            # Move inputs to CPU
            all_txt_embs_cpu = [t.cpu() for t in all_txt_embs]
            all_img_embs_cpu = [t.cpu() for t in all_img_embs]
            
            all_txt_embs_t = torch.stack(all_txt_embs_cpu)
            all_img_embs_t = torch.stack(all_img_embs_cpu)
            
            all_txt_embs_t = F.normalize(all_txt_embs_t, dim=-1)
            all_img_embs_t = F.normalize(all_img_embs_t, dim=-1)
            
            # Compute on CPU
            all_scores_tensor = torch.bmm(all_txt_embs_t, all_img_embs_t.transpose(1, 2))
            return all_scores_tensor
        else:
            raise e

def get_embeddings(model, dataloader, preprocess, tokenizer,  device, amp=True):
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = autocast_ctx
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = torch.stack([preprocess(item.convert("RGB")) for item in batch_images])
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast(device, enabled=amp):

            batch_images_emb = model.encode_image(batch_images)
            batch_texts_emb = model.encode_text(batch_texts_tok)

        batch_images_emb_list.append(batch_images_emb)
        batch_texts_emb_list.append(batch_texts_emb)
        texts_image_index.extend(batch_texts_image_index)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros((texts_emb.shape[0], images_emb.shape[0]), dtype=bool)
    positive_pairs[torch.arange(len(positive_pairs)), texts_image_index] = True

    return images_emb, texts_emb, positive_pairs.to(device)

