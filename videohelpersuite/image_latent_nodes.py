from torch import Tensor
import torch

import comfy.utils

from .utils import BIGMIN, BIGMAX, select_indexes_from_str, convert_str_to_indexes, select_indexes, \
        effective_batch_size, write_tensor_chunks


class MergeStrategies:
    MATCH_A = "match A"
    MATCH_B = "match B"
    MATCH_SMALLER = "match smaller"
    MATCH_LARGER = "match larger"

    list_all = [MATCH_A, MATCH_B, MATCH_SMALLER, MATCH_LARGER]


class ScaleMethods:
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    AREA = "area"
    BICUBIC = "bicubic"
    BISLERP = "bislerp"

    list_all = [NEAREST_EXACT, BILINEAR, AREA, BICUBIC, BISLERP]


class CropMethods:
    DISABLED = "disabled"
    CENTER = "center"

    list_all = [DISABLED, CENTER]


def resolve_merge_template(height_a, width_a, height_b, width_b, merge_strategy):
    a_size = width_a * height_a
    b_size = width_b * height_b
    use_a_as_template = True
    if merge_strategy == MergeStrategies.MATCH_A:
        pass
    elif merge_strategy == MergeStrategies.MATCH_B:
        use_a_as_template = False
    elif merge_strategy in (MergeStrategies.MATCH_SMALLER, MergeStrategies.MATCH_LARGER):
        if a_size <= b_size:
            use_a_as_template = merge_strategy == MergeStrategies.MATCH_SMALLER
        else:
            use_a_as_template = merge_strategy == MergeStrategies.MATCH_LARGER
    if use_a_as_template:
        return height_a, width_a, False, True
    return height_b, width_b, True, False


def merge_image_batches(images_a, images_b, merge_strategy, scale_method, crop, per_batch=0):
    total = images_a.shape[0] + images_b.shape[0]
    needs_scale = images_a.shape[1] != images_b.shape[1] or images_a.shape[2] != images_b.shape[2]
    target_h, target_w, scale_a, scale_b = resolve_merge_template(
        images_a.shape[1], images_a.shape[2], images_b.shape[1], images_b.shape[2], merge_strategy
    )
    batch_size = effective_batch_size(total, images_a.shape[1:], per_batch, images_a.element_size())

    def scale_chunk(chunk, do_scale):
        if not do_scale:
            return chunk
        chunk = chunk.movedim(-1, 1)
        chunk = comfy.utils.common_upscale(chunk, target_w, target_h, scale_method, crop)
        return chunk.movedim(1, -1)

    if not needs_scale and batch_size >= total:
        return torch.cat((images_a, images_b), dim=0)

    if needs_scale:
        first = scale_chunk(images_a[:1], scale_a)
    else:
        first = images_a[:1]
    out = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=first.device)
    del first
    offset = write_tensor_chunks(out, 0, images_a, batch_size, lambda c: scale_chunk(c, scale_a))
    write_tensor_chunks(out, offset, images_b, batch_size, lambda c: scale_chunk(c, scale_b))
    return out


def merge_latent_batches(samples_a, samples_b, merge_strategy, scale_method, crop, per_batch=0):
    total = samples_a.shape[0] + samples_b.shape[0]
    needs_scale = samples_a.shape[2] != samples_b.shape[2] or samples_a.shape[3] != samples_b.shape[3]
    target_h, target_w, scale_a, scale_b = resolve_merge_template(
        samples_a.shape[2], samples_a.shape[3], samples_b.shape[2], samples_b.shape[3], merge_strategy
    )
    batch_size = effective_batch_size(total, samples_a.shape[1:], per_batch, samples_a.element_size())

    def scale_chunk(chunk, do_scale):
        if not do_scale:
            return chunk
        return comfy.utils.common_upscale(chunk, target_w, target_h, scale_method, crop)

    if not needs_scale and batch_size >= total:
        return torch.cat((samples_a, samples_b), dim=0)

    if needs_scale:
        first = scale_chunk(samples_a[:1], scale_a)
    else:
        first = samples_a[:1]
    out = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=first.device)
    del first
    offset = write_tensor_chunks(out, 0, samples_a, batch_size, lambda c: scale_chunk(c, scale_a))
    write_tensor_chunks(out, offset, samples_b, batch_size, lambda c: scale_chunk(c, scale_b))
    return out


def merge_mask_batches(mask_a, mask_b, merge_strategy, scale_method, crop, per_batch=0):
    total = mask_a.shape[0] + mask_b.shape[0]
    needs_scale = mask_a.shape[1] != mask_b.shape[1] or mask_a.shape[2] != mask_b.shape[2]
    target_h, target_w, scale_a, scale_b = resolve_merge_template(
        mask_a.shape[1], mask_a.shape[2], mask_b.shape[1], mask_b.shape[2], merge_strategy
    )
    batch_size = effective_batch_size(total, mask_a.shape[1:], per_batch, mask_a.element_size())

    def scale_chunk(chunk, do_scale):
        if not do_scale:
            return chunk
        chunk = torch.unsqueeze(chunk, 1)
        chunk = comfy.utils.common_upscale(chunk, target_w, target_h, scale_method, crop)
        return torch.squeeze(chunk, 1)

    if not needs_scale and batch_size >= total:
        return torch.cat((mask_a, mask_b), dim=0)

    if needs_scale:
        first = scale_chunk(mask_a[:1], scale_a)
    else:
        first = mask_a[:1]
    out = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=first.device)
    del first
    offset = write_tensor_chunks(out, 0, mask_a, batch_size, lambda c: scale_chunk(c, scale_a))
    write_tensor_chunks(out, offset, mask_b, batch_size, lambda c: scale_chunk(c, scale_b))
    return out


class SplitLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT", "LATENT", "INT")
    RETURN_NAMES = ("LATENT_A", "A_count", "LATENT_B", "B_count")
    FUNCTION = "split_latents"

    def split_latents(self, latents: dict[str, Tensor], split_index: int):
        latents_len = len(latents["samples"])
        group_a = latents.copy()
        group_b = latents.copy()
        for key, val in latents.items():
            if type(val) == Tensor and len(val) == latents_len:
                group_a[key] = latents[key][:split_index]
                group_b[key] = latents[key][split_index:]
        return (group_a, group_a["samples"].size(0), group_b, group_b["samples"].size(0))


class SplitImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("IMAGE_A", "A_count", "IMAGE_B", "B_count")
    FUNCTION = "split_images"

    def split_images(self, images: Tensor, split_index: int):
        group_a = images[:split_index]
        group_b = images[split_index:]
        return (group_a, group_a.size(0), group_b, group_b.size(0))


class SplitMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT", "MASK", "INT")
    RETURN_NAMES = ("MASK_A", "A_count", "MASK_B", "B_count")
    FUNCTION = "split_masks"

    def split_masks(self, mask: Tensor, split_index: int):
        group_a = mask[:split_index]
        group_b = mask[split_index:]
        return (group_a, group_a.size(0), group_b, group_b.size(0))


class MergeLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents_A": ("LATENT",),
                "latents_B": ("LATENT",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
                "per_batch": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "merge"

    def merge(self, latents_A: dict, latents_B: dict, merge_strategy: str, scale_method: str, crop: str, per_batch=0):
        latents_A = latents_A.copy()["samples"]
        latents_B = latents_B.copy()["samples"]

        merged = {"samples": merge_latent_batches(
            latents_A, latents_B, merge_strategy, scale_method, crop, per_batch
        )}
        return (merged, len(merged["samples"]),)


class MergeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_A": ("IMAGE",),
                "images_B": ("IMAGE",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
                "per_batch": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "merge"

    def merge(self, images_A: Tensor, images_B: Tensor, merge_strategy: str, scale_method: str, crop: str, per_batch=0):
        all_images = merge_image_batches(
            images_A, images_B, merge_strategy, scale_method, crop, per_batch
        )
        return (all_images, all_images.size(0),)


class MergeMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_A": ("MASK",),
                "mask_B": ("MASK",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
                "per_batch": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "merge"

    def merge(self, mask_A: Tensor, mask_B: Tensor, merge_strategy: str, scale_method: str, crop: str, per_batch=0):
        all_masks = merge_mask_batches(
            mask_A, mask_B, merge_strategy, scale_method, crop, per_batch
        )
        return (all_masks, all_masks.size(0),)


class SelectEveryNthLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_latents": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "select_latents"

    def select_latents(self, latents: dict[str, Tensor], select_every_nth: int, skip_first_latents: int):
        latents = latents.copy()
        latents_len = len(latents["samples"])
        for key, val in latents.items():
            if type(val) == Tensor and len(val) == latents_len:
                latents[key] = val[skip_first_latents::select_every_nth]
        return (latents, latents["samples"].size(0))
    

class SelectEveryNthImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                    
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "select_images"

    def select_images(self, images: Tensor, select_every_nth: int, skip_first_images: int):
        sub_images = images[skip_first_images::select_every_nth]
        return (sub_images, sub_images.size(0))
    

class SelectEveryNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_masks": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "select_masks"

    def select_masks(self, mask: Tensor, select_every_nth: int, skip_first_masks: int):
        sub_mask = mask[skip_first_masks::select_every_nth]
        return (sub_mask, sub_mask.size(0))


class GetLatentCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, latents: dict):
        return (latents["samples"].size(0),)


class GetImageCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, images: Tensor):
        return (images.size(0),)
    

class GetMaskCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, mask: Tensor):
        return (mask.size(0),)


class RepeatLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, latents: dict[str, Tensor], multiply_by: int):
        latents = latents.copy()
        latents_len = len(latents["samples"])
        for key, val in latents.items():
            if type(val) == Tensor and len(val) == latents_len:
                total = latents_len * multiply_by
                out = torch.empty((total, *val.shape[1:]), dtype=val.dtype, device=val.device)
                for n in range(multiply_by):
                    out[n * latents_len:(n + 1) * latents_len] = val
                latents[key] = out
        return (latents, latents["samples"].size(0),)


class RepeatImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, images: Tensor, multiply_by: int):
        total = images.shape[0] * multiply_by
        out = torch.empty((total, *images.shape[1:]), dtype=images.dtype, device=images.device)
        for n in range(multiply_by):
            out[n * images.shape[0]:(n + 1) * images.shape[0]] = images
        return (out, out.size(0),)


class RepeatMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, mask: Tensor, multiply_by: int):
        total = mask.shape[0] * multiply_by
        out = torch.empty((total, *mask.shape[1:]), dtype=mask.dtype, device=mask.device)
        for n in range(multiply_by):
            out[n * mask.shape[0]:(n + 1) * mask.shape[0]] = mask
        return (out, out.size(0),)


select_description = """Use comma-separated indexes to select items in the given order.
Supports negative indexes, python-style ranges (end index excluded),
as well as range step.

Acceptable entries (assuming 16 items provided, so idxs 0 to 15 exist):
0         -> Returns [0]
-1        -> Returns [15]
0, 1, 13  -> Returns [0, 1, 13]
0:5, 13   -> Returns [0, 1, 2, 3, 4, 13]
0:-1      -> Returns [0, 1, 2, ..., 13, 14]
0:5:-1    -> Returns [4, 3, 2, 1, 0]
0:5:2     -> Returns [0, 2, 4]
::-1     -> Returns [15, 14, 13, ..., 2, 1, 0]
"""
class SelectLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latent": ("LATENT",),
                    "indexes": ("STRING", {"default": "0"}),
                    "err_if_missing": ("BOOLEAN", {"default": True}),
                    "err_if_empty": ("BOOLEAN", {"default": True}),
                },
            }
    
    DESCRIPTION = select_description
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select"

    def select(self, latent: dict[str, Tensor], indexes: str, err_if_missing: bool, err_if_empty: bool):
        # latents are a dict and may contain different stuff (like noise_mask), so need to account for it all
        latent = latent.copy()
        latents_len = len(latent["samples"])
        real_idxs = convert_str_to_indexes(indexes, latents_len, allow_missing=not err_if_missing)
        if err_if_empty and len(real_idxs) == 0:
            raise Exception(f"Nothing was selected based on indexes found in '{indexes}'.")
        for key, val in latent.items():
            if type(val) == Tensor and len(val) == latents_len:
                latent[key] = select_indexes(val, real_idxs)
        return (latent,)


class SelectImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "indexes": ("STRING", {"default": "0"}),
                    "err_if_missing": ("BOOLEAN", {"default": True}),
                    "err_if_empty": ("BOOLEAN", {"default": True}),
                },
            }
    
    DESCRIPTION = select_description
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"

    def select(self, image: Tensor, indexes: str, err_if_missing: bool, err_if_empty: bool):
        to_return = select_indexes_from_str(input_obj=image, indexes=indexes,
                                        err_if_missing=err_if_missing, err_if_empty=err_if_empty)
        to_return_type = type(to_return)
        return (to_return,)


class SelectMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "indexes": ("STRING", {"default": "0"}),
                    "err_if_missing": ("BOOLEAN", {"default": True}),
                    "err_if_empty": ("BOOLEAN", {"default": True}),
                },
            }
    
    DESCRIPTION = select_description
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "select"

    def select(self, mask: Tensor, indexes: str, err_if_missing: bool, err_if_empty: bool):
        return (select_indexes_from_str(input_obj=mask, indexes=indexes,
                                        err_if_missing=err_if_missing, err_if_empty=err_if_empty),)
