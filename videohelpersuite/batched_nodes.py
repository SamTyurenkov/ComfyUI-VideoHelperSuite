import torch
from nodes import VAEEncode
from comfy.utils import ProgressBar


class VAEDecodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    def decode(self, vae, samples, per_batch):
        total = samples["samples"].shape[0]
        pbar = ProgressBar(total)
        first = vae.decode(samples["samples"][0:min(per_batch, total)])
        out = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=first.device)
        end = first.shape[0]
        out[0:end] = first
        del first
        pbar.update(end)
        for start_idx in range(end, total, per_batch):
            batch = vae.decode(samples["samples"][start_idx:start_idx+per_batch])
            end = start_idx + batch.shape[0]
            out[start_idx:end] = batch
            del batch
            pbar.update(per_batch)
        return (out, )


class VAEEncodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", ), "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    def encode(self, vae, pixels, per_batch):
        total = pixels.shape[0]
        pbar = ProgressBar(total)
        first_pixels = pixels[0:min(per_batch, total)]
        try:
            first_pixels = vae.vae_encode_crop_pixels(first_pixels)
        except:
            first_pixels = VAEEncode.vae_encode_crop_pixels(first_pixels)
        first = vae.encode(first_pixels[:,:,:,:3])
        out = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=first.device)
        end = first.shape[0]
        out[0:end] = first
        del first, first_pixels
        pbar.update(end)
        for start_idx in range(end, total, per_batch):
            sub_pixels = pixels[start_idx:start_idx+per_batch]
            try:
                sub_pixels = vae.vae_encode_crop_pixels(sub_pixels)
            except:
                sub_pixels = VAEEncode.vae_encode_crop_pixels(sub_pixels)
            batch = vae.encode(sub_pixels[:,:,:,:3])
            end = start_idx + batch.shape[0]
            out[start_idx:end] = batch
            del batch, sub_pixels
            pbar.update(per_batch)
        return ({"samples": out}, )
