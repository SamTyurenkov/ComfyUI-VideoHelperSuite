import json
import os
import re
import shutil
import subprocess
import threading
import uuid

import numpy as np
import torch
from torch import Tensor

import folder_paths
from comfy.utils import ProgressBar

from .logger import logger
from .utils import BIGMAX, BIGMIN, ENCODE_ARGS, ffmpeg_path, floatOrInt, hash_path, strip_path, validate_path, embed_comfy_video_metadata

DISK_TYPE = "VHS_DISK_MEDIA"
PREFERRED_ROOT = "/root/autodl-tmp"
DISK_SUBDIR = "vhs_disk"

# Same contract as RAM VHS: IMAGE/PNG is full-range RGB; disk h264 is TV BT.709.
VF_RGB_FULL_TO_YUV_TV = "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv"
VF_YUV_TV_TO_RGB_FULL = "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=full"
SCALE_YUV_TV = "in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=tv"
RGB_INPUT_COLOR = [
    "-color_range", "pc", "-colorspace", "rgb",
    "-color_primaries", "bt709", "-color_trc", "bt709",
]
YUV_OUTPUT_COLOR = [
    "-color_range", "tv", "-colorspace", "bt709",
    "-color_primaries", "bt709", "-color_trc", "bt709",
]

_nvenc_available = None


def _ffmpeg():
    if ffmpeg_path is None:
        raise ProcessLookupError(
            "ffmpeg is required for disk nodes and could not be found."
        )
    return ffmpeg_path


def _ffprobe():
    ffmpeg = _ffmpeg()
    name = "ffprobe.exe" if ffmpeg.lower().endswith(".exe") else "ffprobe"
    sibling = os.path.join(os.path.dirname(ffmpeg), name)
    if os.path.isfile(sibling):
        return sibling
    found = shutil.which("ffprobe")
    if found:
        return found
    raise ProcessLookupError("ffprobe is required for disk nodes and could not be found.")


def default_disk_root():
    env = os.environ.get("VHS_DISK_ROOT")
    if env:
        return env
    if os.path.isdir(PREFERRED_ROOT) and os.access(PREFERRED_ROOT, os.W_OK):
        return os.path.join(PREFERRED_ROOT, DISK_SUBDIR)
    return os.path.join(folder_paths.get_temp_directory(), DISK_SUBDIR)


def _allowed_write_roots():
    roots = []
    for path in (
        PREFERRED_ROOT,
        os.environ.get("VHS_DISK_ROOT"),
        folder_paths.get_temp_directory(),
    ):
        if not path:
            continue
        try:
            if path == PREFERRED_ROOT and not os.path.isdir(path):
                continue
            os.makedirs(path, exist_ok=True)
            roots.append(os.path.realpath(path))
        except OSError:
            continue
    return roots


def _assert_under(path, roots):
    real = os.path.realpath(path)
    for root in roots:
        try:
            if os.path.commonpath([root, real]) == root:
                return real
        except ValueError:
            continue
    raise Exception(f"Path is outside allowed disk roots: {path}")


def resolve_disk_root(base_dir):
    roots = _allowed_write_roots()
    last_error = None
    for base in (strip_path(base_dir) or "", default_disk_root()):
        if not base:
            continue
        probe = os.path.abspath(base)
        allowed = False
        for root in roots:
            try:
                if os.path.commonpath([root, probe]) == root:
                    allowed = True
                    break
            except ValueError:
                continue
        if not allowed:
            continue
        try:
            os.makedirs(base, exist_ok=True)
            return _assert_under(base, roots)
        except Exception as e:
            last_error = e
    raise Exception(f"Cannot create disk root: {last_error}")


def has_nvenc():
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    try:
        res = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, check=True,
        )
        _nvenc_available = b"h264_nvenc" in res.stdout
    except Exception:
        _nvenc_available = False
    return _nvenc_available


def pick_encoder(encoder):
    if encoder == "auto":
        encoder = "h264_nvenc" if has_nvenc() else "libx264"
    if encoder == "h264_nvenc" and not has_nvenc():
        logger.warn("h264_nvenc not available, falling back to libx264")
        encoder = "libx264"
    return encoder


def _truthy(value, default=True):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() not in ("false", "0", "")
    return bool(value)


def encoder_args(encoder, **kwargs):
    encoder = pick_encoder(encoder)
    if "bitrate" in kwargs and kwargs.get("crf") is None:
        bitrate = int(kwargs.get("bitrate", 50))
        suffix = "M" if _truthy(kwargs.get("megabit", True)) else "K"
        br = f"{bitrate}{suffix}"
        if encoder == "h264_nvenc":
            args = [
                "-c:v", "h264_nvenc", "-preset", "p4",
                "-profile:v", "high", "-level", "5.2",
                "-rc", "cbr", "-b:v", br, "-pix_fmt", "yuv420p",
            ]
        else:
            args = [
                "-c:v", "libx264", "-preset", "veryfast",
                "-b:v", br, "-maxrate", br, "-bufsize", f"{bitrate * 2}{suffix}",
                "-pix_fmt", "yuv420p",
            ]
    else:
        crf = int(kwargs.get("crf", kwargs.get("quality", 18)))
        if encoder == "h264_nvenc":
            args = [
                "-c:v", "h264_nvenc", "-preset", "p4",
                "-profile:v", "high", "-level", "5.2",
                "-rc", "constqp", "-qp", str(crf), "-cq", str(crf),
                "-pix_fmt", "yuv420p",
            ]
        else:
            args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    return args + YUV_OUTPUT_COLOR


def rgb_input_args(width, height, fps):
    return [
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        *RGB_INPUT_COLOR,
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    ]


def rgb_to_yuv_vf(width, height):
    enc_w, enc_h = even_size(width, height)
    parts = []
    if (enc_w, enc_h) != (width, height):
        parts.append(f"pad={enc_w}:{enc_h}:(ow-iw)/2:(oh-ih)/2")
    parts.append(VF_RGB_FULL_TO_YUV_TV)
    return ["-vf", ",".join(parts)], (enc_w, enc_h)


def yuv_to_rgb_vf(*parts):
    filters = [part for part in parts if part] + [VF_YUV_TV_TO_RGB_FULL]
    return ["-vf", ",".join(filters)]


def yuv_keep_filter(scale_wh, fps):
    if scale_wh:
        width, height = scale_wh
        scale = f"scale=w={width}:h={height}:{SCALE_YUV_TV}"
    else:
        scale = f"scale={SCALE_YUV_TV}"
    return f"{scale},fps={fps}"


NVENC_RATE_WIDGETS = [
    ["bitrate", "INT", {"default": 50, "min": 1, "max": 400, "step": 1}],
    ["megabit", "BOOLEAN", {"default": True}],
]
X264_RATE_WIDGETS = [
    ["crf", "INT", {"default": 18, "min": 0, "max": 51, "step": 1}],
]


def encoder_widgets():
    return {
        "encoder": (["auto", "h264_nvenc", "libx264"], {
            "formats": {
                "auto": NVENC_RATE_WIDGETS,
                "h264_nvenc": NVENC_RATE_WIDGETS,
                "libx264": X264_RATE_WIDGETS,
            },
        }),
    }


def run_ffmpeg(args, stdin=None):
    try:
        res = subprocess.run(args, input=stdin, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(*ENCODE_ARGS) if e.stderr else str(e)
        raise Exception("ffmpeg failed:\n" + err)
    if res.stderr:
        text = res.stderr.decode(*ENCODE_ARGS)
        if text.strip():
            logger.warn(text)
    return res


def probe_video(path):
    args = [
        _ffprobe(), "-v", "error", "-show_entries",
        "stream=width,height,nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate:format=duration",
        "-select_streams", "v:0", "-of", "json", path,
    ]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
        data = json.loads(res.stdout.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(*ENCODE_ARGS) if e.stderr else str(e)
        raise Exception("ffprobe failed:\n" + err)
    streams = data.get("streams") or [{}]
    stream = streams[0]
    fmt = data.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 24.0
    count = 0
    for key in ("nb_frames", "nb_read_frames"):
        raw = stream.get(key)
        if raw and raw not in ("N/A", "0"):
            try:
                count = int(raw)
                break
            except ValueError:
                pass
    duration = float(fmt.get("duration") or 0)
    if count <= 0 and duration > 0:
        count = max(1, int(round(duration * fps)))
    if count <= 0:
        count = _count_frames(path)
    return {
        "path": os.path.realpath(path),
        "kind": "video",
        "fps": float(fps),
        "count": int(count),
        "width": width,
        "height": height,
        "duration": duration if duration > 0 else (count / fps if fps else 0),
    }


def _parse_rate(value):
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _count_frames(path):
    args = [
        _ffprobe(), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path,
    ]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
        raw = res.stdout.decode(*ENCODE_ARGS).strip().split(",")[0]
        return int(raw)
    except Exception:
        return 0


def media_from_path(path, fps_hint=0):
    media = probe_video(path)
    if fps_hint > 0:
        media["fps"] = float(fps_hint)
        media["duration"] = media["count"] / media["fps"] if media["fps"] else media["duration"]
    return media


def require_media(media):
    if not isinstance(media, dict) or not media.get("path"):
        raise Exception("Invalid VHS_DISK_MEDIA")
    path = os.path.realpath(media["path"])
    if not os.path.isfile(path):
        raise Exception(f"Disk media file is missing: {path}")
    out = dict(media)
    out["path"] = path
    return out


def even_size(width, height):
    return width - (width % 2), height - (height % 2)


def iter_rgb_bytes(images: Tensor, pbar=None):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    total = images.shape[0]
    for i in range(total):
        frame = images[i]
        if frame.device.type != "cpu":
            frame = frame.cpu()
        arr = np.clip(frame.numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        yield arr.tobytes()
        if pbar is not None:
            pbar.update(1)


def encode_images_to_file(images, out_path, fps, encoder, pbar=None, **enc_kwargs):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    height, width = int(images.shape[1]), int(images.shape[2])
    vf, (enc_w, enc_h) = rgb_to_yuv_vf(width, height)
    args = [
        _ffmpeg(), "-y", "-v", "error",
        *rgb_input_args(width, height, fps),
    ] + vf + encoder_args(encoder, **enc_kwargs) + ["-an", out_path]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err_chunks = []
    drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()))
    drain.start()
    try:
        for chunk in iter_rgb_bytes(images, pbar):
            proc.stdin.write(chunk)
        proc.stdin.close()
        drain.join()
        rc = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        drain.join()
        raise
    stderr = b"".join(err_chunks)
    if rc != 0:
        raise Exception("ffmpeg encode failed:\n" + stderr.decode(*ENCODE_ARGS))
    media = media_from_path(out_path, fps_hint=fps)
    media["count"] = int(images.shape[0])
    media["width"], media["height"] = enc_w, enc_h
    if media["fps"]:
        media["duration"] = media["count"] / media["fps"]
    return media


def concat_media(media_a, media_b, out_path, encoder, merge_strategy="match A", **enc_kwargs):
    a = require_media(media_a)
    b = require_media(media_b)
    same = (
        a["width"] == b["width"] and a["height"] == b["height"]
        and abs(a["fps"] - b["fps"]) < 0.01
    )
    if same:
        list_path = out_path + ".concat.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            f.write(_concat_entry(a["path"]))
            f.write(_concat_entry(b["path"]))
        try:
            run_ffmpeg([
                _ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", "-an", out_path,
            ])
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)
        return media_from_path(out_path, fps_hint=a["fps"])

    if merge_strategy == "match B":
        target_w, target_h, fps = b["width"], b["height"], b["fps"]
        scale_a, scale_b = True, False
    else:
        target_w, target_h, fps = a["width"], a["height"], a["fps"]
        scale_a, scale_b = False, True
        if merge_strategy == "match smaller":
            if b["width"] * b["height"] < a["width"] * a["height"]:
                target_w, target_h, fps = b["width"], b["height"], b["fps"]
                scale_a, scale_b = True, False
        elif merge_strategy == "match larger":
            if b["width"] * b["height"] > a["width"] * a["height"]:
                target_w, target_h, fps = b["width"], b["height"], b["fps"]
                scale_a, scale_b = True, False
    target_w, target_h = even_size(target_w, target_h)
    fa = yuv_keep_filter((target_w, target_h) if scale_a else None, fps)
    fb = yuv_keep_filter((target_w, target_h) if scale_b else None, fps)
    filter_complex = f"[0:v]{fa}[v0];[1:v]{fb}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
    run_ffmpeg([
        _ffmpeg(), "-y", "-v", "error",
        "-i", a["path"], "-i", b["path"],
        "-filter_complex", filter_complex, "-map", "[v]",
    ] + encoder_args(encoder, **enc_kwargs) + ["-an", out_path])
    return media_from_path(out_path, fps_hint=fps)


def _concat_entry(path):
    path = os.path.abspath(path).replace("\\", "/")
    path = path.replace("'", r"'\''")
    return f"file '{path}'\n"


def _ffmpeg_raw(args):
    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(*ENCODE_ARGS) if e.stderr else str(e)
        raise Exception("ffmpeg decode failed:\n" + err)
    return res.stdout


def new_clip_path(base_dir, unique_id, prefix="clip"):
    root = resolve_disk_root(base_dir)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(unique_id or uuid.uuid4().hex[:8]))
    return os.path.join(root, f"{prefix}_{safe_id}.mp4")


def load_frame_window(media, start, count, pbar=None):
    media = require_media(media)
    total = int(media["count"])
    if total <= 0:
        raise Exception("Disk media has no frames")
    if start < 0:
        start = max(0, total + start)
    if count <= 0:
        count = total - start
        logger.warn(f"Load Disk Frames count=0 loads the rest of the clip ({count} frames)")
    count = min(count, total - start)
    if count <= 0:
        raise Exception("Requested frame window is empty")
    fps = float(media["fps"]) or 24.0
    width, height = int(media["width"]), int(media["height"])
    if width <= 0 or height <= 0:
        raise Exception("Disk media has invalid size")
    frame_bytes = height * width * 3
    end = start + count - 1
    raw = _ffmpeg_raw([
        _ffmpeg(), "-v", "error", "-i", media["path"],
        *yuv_to_rgb_vf(f"select=between(n\\,{start}\\,{end})"),
        "-vsync", "0", "-frames:v", str(count),
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    got = len(raw) // frame_bytes
    # One-frame -sseof at 1/fps seeks past EOF on NVENC clips; only use it as
    # a last-N fallback with a couple of extra frames of padding.
    if got <= 0:
        extra = 8
        leftover = max((count + extra) / fps, 0.25)
        raw = _ffmpeg_raw([
            _ffmpeg(), "-v", "error",
            "-sseof", f"-{leftover:.6f}", "-i", media["path"],
            *yuv_to_rgb_vf(),
            "-vsync", "0", "-frames:v", str(count + extra),
            "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ])
        got = len(raw) // frame_bytes
    if got > count:
        raw = raw[-(count * frame_bytes):]
        got = count
    if got != count:
        logger.warn(f"Requested {count} disk frames, ffmpeg returned {got}")
    if got <= 0:
        raise Exception("ffmpeg returned no RGB frames")
    arr = np.frombuffer(raw, dtype=np.uint8, count=got * frame_bytes)
    frames = arr.reshape((got, height, width, 3)).copy()
    if pbar is not None:
        pbar.update(got)
    return torch.from_numpy(frames).to(dtype=torch.float32) / 255.0


def mux_audio(video_path, audio, out_path, fps, frame_count):
    channels = audio["waveform"].size(1)
    min_dur = frame_count / fps + 1
    args = [
        _ffmpeg(), "-y", "-v", "error", "-i", video_path,
        "-ar", str(audio["sample_rate"]), "-ac", str(channels),
        "-f", "f32le", "-i", "-",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-af", f"apad=whole_dur={min_dur}", "-shortest", out_path,
    ]
    audio_data = audio["waveform"].squeeze(0).transpose(0, 1).numpy().tobytes()
    run_ffmpeg(args, stdin=audio_data)


class ImagesToDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (floatOrInt, {"default": 24, "min": 1, "step": 1}),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "save"

    def save(self, images, frame_rate, base_dir, encoder, unique_id=None, **kwargs):
        out_path = new_clip_path(base_dir, unique_id, "images")
        pbar = ProgressBar(images.shape[0] if images.ndim == 4 else 1)
        media = encode_images_to_file(
            images, out_path, frame_rate, encoder, pbar, **kwargs,
        )
        return (media, media["count"])


class PathToDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": ["mp4", "webm", "mkv", "mov", "avi"]}),
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "wrap"

    def wrap(self, video):
        path = strip_path(video)
        if path is None or validate_path(path) != True:
            raise Exception("video is not a valid path: " + str(video))
        media = media_from_path(os.path.realpath(path))
        return (media, media["count"])

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(strip_path(video))

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True, allow_url=False)


class MergeDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk_A": (DISK_TYPE,),
                "disk_B": (DISK_TYPE,),
                "merge_strategy": (["match A", "match B", "match smaller", "match larger"],),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "merge"

    def merge(self, disk_A, disk_B, merge_strategy, base_dir, encoder, unique_id=None, **kwargs):
        out_path = new_clip_path(base_dir, unique_id, "merge")
        media = concat_media(
            disk_A, disk_B, out_path, encoder, merge_strategy, **kwargs,
        )
        return (media, media["count"])


class AppendImagesToDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "images": ("IMAGE",),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "append"

    def append(self, disk, images, base_dir, encoder, unique_id=None, **kwargs):
        media = require_media(disk)
        tail_path = new_clip_path(base_dir, unique_id, "append_tail")
        out_path = new_clip_path(base_dir, unique_id, "append")
        pbar = ProgressBar(images.shape[0] if images.ndim == 4 else 1)
        tail = encode_images_to_file(
            images, tail_path, media["fps"], encoder, pbar, **kwargs,
        )
        result = concat_media(
            media, tail, out_path, encoder, "match A", **kwargs,
        )
        try:
            os.remove(tail_path)
        except OSError:
            pass
        return (result, result["count"])


class LoadDiskFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "start": ("INT", {"default": 0, "min": -BIGMAX, "max": BIGMAX, "step": 1}),
                "count": ("INT", {"default": 1, "min": 0, "max": BIGMAX, "step": 1}),
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "count")
    FUNCTION = "load"

    def load(self, disk, start, count):
        pbar = ProgressBar(max(count, 1))
        images = load_frame_window(disk, start, count, pbar)
        return (images, images.shape[0])


class DiskInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"disk": (DISK_TYPE,)}}

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("path", "fps", "count", "width", "height", "duration")
    FUNCTION = "info"

    def info(self, disk):
        media = require_media(disk)
        return (
            media["path"],
            float(media["fps"]),
            int(media["count"]),
            int(media["width"]),
            int(media["height"]),
            float(media.get("duration") or 0),
        )


class DiskCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "filename_prefix": ("STRING", {"default": "DiskCombine"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "combine"

    def combine(self, disk, filename_prefix="DiskCombine", save_output=True, audio=None, prompt=None, extra_pnginfo=None):
        media = require_media(disk)
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        counter = 1
        matcher = re.compile(rf"{re.escape(filename)}_(\d+)\D*\..+", re.IGNORECASE)
        if os.path.isdir(full_output_folder):
            for existing_file in os.listdir(full_output_folder):
                match = matcher.fullmatch(existing_file)
                if match:
                    counter = max(counter, int(match.group(1)) + 1)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.mp4")
        run_ffmpeg([_ffmpeg(), "-y", "-v", "error", "-i", media["path"], "-c", "copy", file_path])
        output_files = [file_path]
        waveform = None
        if audio is not None:
            try:
                waveform = audio["waveform"]
            except Exception:
                waveform = None
        if waveform is not None:
            audio_path = os.path.join(full_output_folder, f"{filename}_{counter:05}-audio.mp4")
            mux_audio(file_path, audio, audio_path, media["fps"], media["count"])
            output_files.append(audio_path)
            preview = audio_path
        else:
            preview = file_path
        embed_comfy_video_metadata(preview, prompt, extra_pnginfo)
        gif_preview = {
            "filename": os.path.basename(preview),
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": "video/mp4",
            "frame_rate": media["fps"],
            "fullpath": preview,
        }
        return {
            "ui": {
                "images": [gif_preview],
                "gifs": [gif_preview],
                "animated": (True,),
            },
            "result": ((save_output, output_files),),
        }


KNOWN_INTERP_CKPTS = [
    "rife_v4.25_heavy.safetensors",
    "film_net_fp16.safetensors",
]


def _interp_ckpt_names():
    try:
        if hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(
                "frame_interpolation",
                os.path.join(folder_paths.models_dir, "frame_interpolation"),
            )
    except Exception:
        pass
    names = []
    try:
        names = list(folder_paths.get_filename_list("frame_interpolation"))
    except Exception:
        names = []
    seen = set()
    ordered = []
    for name in KNOWN_INTERP_CKPTS + names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


_interp_patchers = {}


def _load_interp_patcher(model_name):
    import comfy.model_patcher
    import comfy.utils as comfy_utils
    from comfy import model_management
    from comfy_extras.frame_interpolation_models.ifnet import IFNet, detect_rife_config

    getter = getattr(folder_paths, "get_full_path_or_raise", None)
    if getter is not None:
        model_path = getter("frame_interpolation", model_name)
    else:
        model_path = folder_paths.get_full_path("frame_interpolation", model_name)
        if not model_path:
            raise Exception(f"Frame interpolation model not found: {model_name}")
    cached = _interp_patchers.get(model_path)
    if cached is not None:
        return cached

    sd = comfy_utils.load_torch_file(model_path, safe_load=True)
    is_film = (
        "extract.extract_sublevels.convs.0.0.conv.weight" in sd
        or "film" in model_name.lower()
    )
    if is_film:
        from comfy_extras.frame_interpolation_models.film_net import FILMNet
        model = FILMNet()
        model.load_state_dict(sd)
    else:
        sd = comfy_utils.state_dict_prefix_replace(sd, {"module.": "", "flownet.": ""})
        key_map = {}
        for k in sd:
            for i in range(5):
                if k.startswith(f"block{i}."):
                    key_map[k] = f"blocks.{i}.{k[len(f'block{i}.'):]}"
        if key_map:
            sd = {key_map.get(k, k): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith(("teacher.", "caltime."))}
        try:
            head_ch, channels = detect_rife_config(sd)
        except (KeyError, ValueError) as e:
            raise Exception(f"Unrecognized frame interpolation model: {model_name}") from e
        model = IFNet(head_ch=head_ch, channels=channels)
        model.load_state_dict(sd)

    dtype = torch.float16 if model_management.should_use_fp16(model_management.get_torch_device()) else torch.float32
    model.eval().to(dtype)
    Patcher = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    patcher = Patcher(
        model,
        load_device=model_management.get_torch_device(),
        offload_device=model_management.unet_offload_device(),
    )
    _interp_patchers[model_path] = patcher
    return patcher


def _hwc_to_rgb_bytes(frame):
    if frame.device.type != "cpu":
        frame = frame.cpu()
    arr = np.clip(frame.numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.tobytes()


class _FfmpegFrameWriter:
    def __init__(self, out_path, width, height, fps, encoder, **enc_kwargs):
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.written = 0
        self.out_path = out_path
        vf, (enc_w, enc_h) = rgb_to_yuv_vf(self.width, self.height)
        self.enc_w, self.enc_h = enc_w, enc_h
        args = [
            _ffmpeg(), "-y", "-v", "error",
            *rgb_input_args(self.width, self.height, self.fps),
        ] + vf + encoder_args(encoder, **enc_kwargs) + ["-an", out_path]
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.err_chunks = []
        self.drain = threading.Thread(target=lambda: self.err_chunks.append(self.proc.stderr.read()))
        self.drain.start()
        rc = self.proc.poll()
        if rc is not None:
            raise Exception(self._encoder_died(rc))

    def _encoder_died(self, rc=None):
        try:
            self.drain.join(timeout=2)
        except Exception:
            pass
        if rc is None:
            rc = self.proc.poll()
        err = b"".join(self.err_chunks).decode(*ENCODE_ARGS).strip()
        return f"ffmpeg encoder died (exit {rc}):\n{err or '(no stderr)'}"

    def write_bytes(self, chunk):
        if self.proc.poll() is not None:
            raise Exception(self._encoder_died())
        try:
            self.proc.stdin.write(chunk)
        except BrokenPipeError as e:
            raise Exception(self._encoder_died()) from e
        self.written += 1

    def write_hwc(self, frame):
        self.write_bytes(_hwc_to_rgb_bytes(frame))

    def write_bchw(self, frame, height, width):
        frame = frame[:, :, :height, :width].float().clamp(0.0, 1.0)
        self.write_hwc(frame[0].movedim(0, -1).contiguous())

    def close(self):
        try:
            self.proc.stdin.close()
            self.drain.join()
            rc = self.proc.wait()
        except Exception:
            self.proc.kill()
            self.proc.wait()
            self.drain.join()
            raise
        stderr = b"".join(self.err_chunks)
        if rc != 0:
            raise Exception("ffmpeg encode failed:\n" + stderr.decode(*ENCODE_ARGS))
        media = media_from_path(self.out_path, fps_hint=self.fps)
        media["count"] = int(self.written)
        media["width"], media["height"] = self.enc_w, self.enc_h
        if media["fps"]:
            media["duration"] = media["count"] / media["fps"]
        return media


def _iter_disk_hwc(media, every=1):
    media = require_media(media)
    width, height = int(media["width"]), int(media["height"])
    frame_bytes = height * width * 3
    if frame_bytes <= 0:
        raise Exception("Disk media has invalid size")
    every = max(int(every), 1)
    select = f"select=not(mod(n\\,{every}))" if every > 1 else None
    args = [
        _ffmpeg(), "-v", "error", "-i", media["path"],
        "-vsync", "0", "-an",
        *yuv_to_rgb_vf(select),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
    err_chunks = []
    drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()))
    drain.start()
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8, count=frame_bytes).reshape((height, width, 3)).copy()
            yield torch.from_numpy(arr).to(dtype=torch.float32) / 255.0
        rc = proc.wait()
        drain.join()
        if rc != 0:
            err = b"".join(err_chunks).decode(*ENCODE_ARGS)
            if err.strip():
                raise Exception("ffmpeg decode failed:\n" + err)
    except Exception:
        proc.kill()
        proc.wait()
        drain.join()
        raise
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


def interpolate_disk_clip(media, ckpt_name, multiplier, out_path, encoder, keep_duration=True, pbar=None, **enc_kwargs):
    from comfy import model_management
    from comfy.ldm.common_dit import pad_to_patch_size

    media = require_media(media)
    total = int(media["count"])
    width, height = int(media["width"]), int(media["height"])
    fps = float(media["fps"]) or 24.0
    if total < 2 or multiplier < 2:
        run_ffmpeg([_ffmpeg(), "-y", "-v", "error", "-i", media["path"], "-c", "copy", "-an", out_path])
        copied = media_from_path(out_path, fps_hint=fps)
        copied["count"] = total
        return copied

    patcher = _load_interp_patcher(ckpt_name)
    device = patcher.load_device
    dtype = patcher.model_dtype()
    inference_model = patcher.model
    activation_mem = 0
    mem_fn = getattr(inference_model, "memory_used_forward", None)
    if callable(mem_fn):
        try:
            activation_mem = mem_fn((2, height, width, 3), dtype)
        except Exception:
            activation_mem = 0
    model_management.load_models_gpu([patcher], memory_required=activation_mem)
    inference_model = patcher.model
    align = getattr(inference_model, "pad_align", 1)
    if align <= 1 and hasattr(inference_model, "pyramid_levels"):
        align = 2 ** max(int(inference_model.pyramid_levels) - 1, 1)
    num_interp = multiplier - 1
    t_values = [t / multiplier for t in range(1, multiplier)]
    fps_out = fps * multiplier if keep_duration else fps
    writer = _FfmpegFrameWriter(out_path, width, height, fps_out, encoder, **enc_kwargs)
    total_steps = (total - 1) * num_interp
    if pbar is None:
        pbar = ProgressBar(max(total_steps, 1))

    def prepare(frame_hwc):
        frame = frame_hwc.unsqueeze(0).movedim(-1, 1).to(dtype=dtype, device=device)
        if align > 1:
            frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
        return frame

    extract = getattr(inference_model, "extract_features", None)
    multi_fn = getattr(inference_model, "forward_multi_timestep", None)
    feat_cache = {}
    prev_gpu = None
    frames_in = 0
    try:
        for frame_hwc in _iter_disk_hwc(media):
            frames_in += 1
            src_bytes = _hwc_to_rgb_bytes(frame_hwc)
            img1 = prepare(frame_hwc)
            if prev_gpu is None:
                writer.write_bytes(src_bytes)
                prev_gpu = img1
                if extract is not None:
                    feat_cache["next"] = extract(img1)
                continue
            img0 = prev_gpu
            if extract is not None:
                feat_cache["img0"] = feat_cache.pop("next") if "next" in feat_cache else extract(img0)
                feat_cache["img1"] = extract(img1)
                feat_cache["next"] = feat_cache["img1"]
            used_multi = False
            if multi_fn is not None:
                try:
                    mids = multi_fn(img0, img1, t_values, cache=feat_cache)
                    for k in range(mids.shape[0]):
                        writer.write_bchw(mids[k:k + 1], height, width)
                        pbar.update(1)
                    used_multi = True
                except model_management.OOM_EXCEPTION:
                    model_management.soft_empty_cache()
                    multi_fn = None
            if not used_multi:
                sample = img0
                ts = torch.tensor(t_values, device=device, dtype=dtype).reshape(num_interp, 1, 1, 1)
                ts = ts.expand(-1, 1, sample.shape[2], sample.shape[3])
                for j in range(num_interp):
                    kwargs = {"timestep": ts[j:j + 1]}
                    try:
                        mid = inference_model(img0, img1, cache=feat_cache, **kwargs)
                    except TypeError:
                        mid = inference_model(img0, img1, timestep=ts[j:j + 1])
                    writer.write_bchw(mid, height, width)
                    pbar.update(1)
            writer.write_bytes(src_bytes)
            prev_gpu = img1
        media_out = writer.close()
    except Exception:
        try:
            writer.proc.kill()
            writer.proc.wait()
        except Exception:
            pass
        raise
    if frames_in != total:
        logger.warn(f"Disk interpolate decoded {frames_in} frames, probe said {total}")
    return media_out


class DiskInterpolate:
    @classmethod
    def INPUT_TYPES(s):
        ckpts = _interp_ckpt_names()
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "ckpt_name": (ckpts,),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16, "step": 1}),
                "keep_duration": ("BOOLEAN", {"default": True}),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "interpolate"

    def interpolate(self, disk, ckpt_name, multiplier, keep_duration, base_dir, encoder, unique_id=None, **kwargs):
        media = require_media(disk)
        out_path = new_clip_path(base_dir, unique_id, "interp")
        total = max(int(media["count"]) - 1, 1) * (int(multiplier) - 1)
        pbar = ProgressBar(max(total, 1))
        result = interpolate_disk_clip(
            media, ckpt_name, int(multiplier), out_path, encoder,
            keep_duration=keep_duration, pbar=pbar, **kwargs,
        )
        return (result, result["count"])


COLOR_MATCH_METHODS = ["reinhard_lab", "reinhard_rgb", "mkl_rgb"]
_STATS_MAX_SIDE = 128


class _ChannelMoments:
    def __init__(self):
        self.n = 0
        self.sum = None
        self.outer = None

    def add_hwc(self, hwc):
        pix = hwc.reshape(-1, 3).detach().float().cpu()
        if pix.numel() == 0:
            return
        if self.sum is None:
            self.sum = torch.zeros(3, dtype=torch.float64)
            self.outer = torch.zeros(3, 3, dtype=torch.float64)
        pix64 = pix.to(dtype=torch.float64)
        self.n += pix64.shape[0]
        self.sum += pix64.sum(0)
        self.outer += pix64.T @ pix64

    def mean(self):
        if self.n <= 0:
            raise Exception("No pixels gathered for color stats")
        return (self.sum / self.n).float()

    def cov(self):
        mean = self.sum / self.n
        cov = self.outer / self.n - torch.outer(mean, mean)
        return cov.float()

    def std(self):
        return self.cov().diag().clamp_min(1e-8).sqrt()


def _downsample_hwc(hwc, max_side=_STATS_MAX_SIDE):
    h, w = int(hwc.shape[0]), int(hwc.shape[1])
    m = max(h, w)
    if m <= max_side:
        return hwc.float()
    scale = max_side / m
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    x = hwc.float().movedim(-1, 0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="area")
    return x[0].movedim(0, -1).contiguous()


def _rgb_to_lab_hwc(hwc, device=None):
    import kornia
    x = hwc.float()
    if device is not None:
        x = x.to(device=device)
    nchw = x.unsqueeze(0).movedim(-1, 1).contiguous()
    lab = kornia.color.rgb_to_lab(nchw)
    return lab[0].movedim(0, -1).contiguous()


def _lab_to_rgb_nchw(lab_nchw):
    import kornia
    return kornia.color.lab_to_rgb(lab_nchw).clamp(0, 1)


def _mkl_matrix(cov_src, cov_ref):
    def _sqrtm(c):
        w, v = torch.linalg.eigh(c)
        w = w.clamp_min(1e-8)
        return v @ torch.diag(w.sqrt()) @ v.T

    def _invsqrtm(c):
        w, v = torch.linalg.eigh(c)
        w = w.clamp_min(1e-8)
        return v @ torch.diag(w.rsqrt()) @ v.T

    eye = torch.eye(3, dtype=torch.float32, device=cov_src.device)
    cs = cov_src.float() + eye * 1e-6
    ct = cov_ref.float() + eye * 1e-6
    cs_h = _sqrtm(cs)
    return _invsqrtm(cs) @ _sqrtm(cs_h @ ct @ cs_h) @ _invsqrtm(cs)


def _prep_stats_frame(hwc, method):
    small = _downsample_hwc(hwc)
    if method == "reinhard_lab":
        return _rgb_to_lab_hwc(small, device="cpu")
    return small.cpu()


def _collect_clip_moments(media, source_stats, sample_every, method):
    moments = _ChannelMoments()
    if source_stats == "first":
        frame = next(_iter_disk_hwc(media), None)
        if frame is None:
            raise Exception("Source clip has no frames")
        moments.add_hwc(_prep_stats_frame(frame, method))
        return moments
    every = 1 if source_stats == "full" else max(int(sample_every), 1)
    n = 0
    for frame in _iter_disk_hwc(media, every=every):
        moments.add_hwc(_prep_stats_frame(frame, method))
        n += 1
    if n <= 0:
        raise Exception("Source clip has no frames")
    return moments


def _collect_image_moments(image, method):
    moments = _ChannelMoments()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    for i in range(image.shape[0]):
        moments.add_hwc(_prep_stats_frame(image[i], method))
    return moments


def _apply_locked_grade(frame, method, src_mean, src_std, ref_mean, ref_std, matrix, strength, chroma_only, device):
    src = frame.unsqueeze(0).to(device=device, dtype=torch.float32).movedim(-1, 1).contiguous()
    s = float(strength)
    if method == "reinhard_lab":
        import kornia
        lab = kornia.color.rgb_to_lab(src)
        mean = src_mean.to(device=device).view(1, 3, 1, 1)
        scale = (ref_std / src_std.clamp_min(1e-6)).to(device=device).view(1, 3, 1, 1)
        shift = ref_mean.to(device=device).view(1, 3, 1, 1)
        graded = (lab - mean) * scale + shift
        if chroma_only:
            graded = torch.cat([lab[:, :1], graded[:, 1:]], dim=1)
        corrected = _lab_to_rgb_nchw(graded)
    elif method == "reinhard_rgb":
        mean = src_mean.to(device=device).view(1, 3, 1, 1)
        scale = (ref_std / src_std.clamp_min(1e-6)).to(device=device).view(1, 3, 1, 1)
        shift = ref_mean.to(device=device).view(1, 3, 1, 1)
        corrected = ((src - mean) * scale + shift).clamp(0, 1)
    else:
        pix = src.movedim(1, -1).reshape(-1, 3)
        A = matrix.to(device=device, dtype=torch.float32)
        graded = (pix - src_mean.to(device=device)) @ A.T + ref_mean.to(device=device)
        corrected = graded.view(1, src.shape[2], src.shape[3], 3).movedim(-1, 1).clamp(0, 1)
        if chroma_only:
            import kornia
            src_lab = kornia.color.rgb_to_lab(src)
            out_lab = kornia.color.rgb_to_lab(corrected)
            corrected = _lab_to_rgb_nchw(torch.cat([src_lab[:, :1], out_lab[:, 1:]], dim=1))
    mixed = (1.0 - s) * src + s * corrected
    return mixed[0].movedim(0, -1).contiguous().cpu().clamp_(0, 1)


class SplitDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT", DISK_TYPE, "INT")
    RETURN_NAMES = ("disk_A", "A_count", "disk_B", "B_count")
    FUNCTION = "split"

    def split(self, disk, split_index, base_dir, encoder, unique_id=None, **kwargs):
        media = require_media(disk)
        total = int(media["count"])
        idx = int(split_index)
        if idx < 0:
            idx = total + idx
        idx = max(0, min(idx, total))
        if idx <= 0 or idx >= total:
            raise Exception("split_index must leave frames on both sides")
        path_a = new_clip_path(base_dir, unique_id, "split_a")
        path_b = new_clip_path(base_dir, unique_id, "split_b")
        writer_a = _FfmpegFrameWriter(path_a, media["width"], media["height"], media["fps"], encoder, **kwargs)
        writer_b = _FfmpegFrameWriter(path_b, media["width"], media["height"], media["fps"], encoder, **kwargs)
        pbar = ProgressBar(total)
        i = 0
        try:
            for frame in _iter_disk_hwc(media):
                if i < idx:
                    writer_a.write_hwc(frame)
                else:
                    writer_b.write_hwc(frame)
                i += 1
                pbar.update(1)
            a = writer_a.close()
            b = writer_b.close()
        except Exception:
            try:
                writer_a.proc.kill()
                writer_b.proc.kill()
            except Exception:
                pass
            raise
        return (a, a["count"], b, b["count"])


class ReverseDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "reverse"

    def reverse(self, disk, base_dir, encoder, unique_id=None, **kwargs):
        media = require_media(disk)
        width, height = int(media["width"]), int(media["height"])
        fps = float(media["fps"]) or 24.0
        frame_bytes = height * width * 3
        out_path = new_clip_path(base_dir, unique_id, "reverse")
        raw_path = out_path + ".raw"
        args = [
            _ffmpeg(), "-v", "error", "-i", media["path"],
            *yuv_to_rgb_vf(),
            "-vsync", "0", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
        err_chunks = []
        drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()))
        drain.start()
        n = 0
        try:
            with open(raw_path, "wb") as raw:
                while True:
                    chunk = proc.stdout.read(frame_bytes)
                    if not chunk or len(chunk) < frame_bytes:
                        break
                    raw.write(chunk)
                    n += 1
            rc = proc.wait()
            drain.join()
            if rc != 0:
                err = b"".join(err_chunks).decode(*ENCODE_ARGS)
                if err.strip():
                    raise Exception("ffmpeg decode failed:\n" + err)
            if n <= 0:
                raise Exception("ffmpeg returned no RGB frames")
            writer = _FfmpegFrameWriter(out_path, width, height, fps, encoder, **kwargs)
            pbar = ProgressBar(n)
            try:
                with open(raw_path, "rb") as raw:
                    for i in range(n - 1, -1, -1):
                        raw.seek(i * frame_bytes)
                        writer.write_bytes(raw.read(frame_bytes))
                        pbar.update(1)
                return (writer.close(), n)
            except Exception:
                try:
                    writer.proc.kill()
                    writer.proc.wait()
                except Exception:
                    pass
                raise
        finally:
            try:
                os.remove(raw_path)
            except OSError:
                pass


class DiskColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "method": (COLOR_MATCH_METHODS, {"default": "reinhard_lab"}),
                "source_stats": (["sampled", "first", "full"], {"default": "sampled"}),
                "sample_every": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "chroma_only": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "optional": {
                "image_ref": ("IMAGE",),
                "disk_ref": (DISK_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "match"

    def match(self, disk, method, source_stats, sample_every, chroma_only, strength, base_dir, encoder,
              unique_id=None, image_ref=None, disk_ref=None, **kwargs):
        media = require_media(disk)
        if image_ref is None and disk_ref is None:
            raise Exception("Provide image_ref or disk_ref — the look to match, not a per-frame pair")
        if float(strength) == 0:
            out_path = new_clip_path(base_dir, unique_id, "cm")
            run_ffmpeg([_ffmpeg(), "-y", "-v", "error", "-i", media["path"], "-c", "copy", "-an", out_path])
            copied = media_from_path(out_path, fps_hint=media["fps"])
            copied["count"] = media["count"]
            return (copied, copied["count"])

        src_m = _collect_clip_moments(media, source_stats, sample_every, method)
        if disk_ref is not None:
            ref_media = require_media(disk_ref)
            ref_sample = "full" if source_stats == "full" else "sampled"
            ref_m = _collect_clip_moments(ref_media, ref_sample, sample_every, method)
        else:
            ref_m = _collect_image_moments(image_ref, method)

        src_mean, src_std = src_m.mean(), src_m.std()
        ref_mean, ref_std = ref_m.mean(), ref_m.std()
        matrix = _mkl_matrix(src_m.cov(), ref_m.cov()) if method == "mkl_rgb" else None

        from comfy import model_management
        device = model_management.get_torch_device()

        out_path = new_clip_path(base_dir, unique_id, "cm")
        writer = _FfmpegFrameWriter(out_path, media["width"], media["height"], media["fps"], encoder, **kwargs)
        pbar = ProgressBar(max(int(media["count"]), 1))
        try:
            for frame in _iter_disk_hwc(media):
                out = _apply_locked_grade(
                    frame, method, src_mean, src_std, ref_mean, ref_std, matrix,
                    float(strength), bool(chroma_only), device,
                )
                writer.write_hwc(out)
                pbar.update(1)
            result = writer.close()
        except Exception:
            try:
                writer.proc.kill()
                writer.proc.wait()
            except Exception:
                pass
            raise
        return (result, result["count"])


class DiskRTXUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"], {"default": "ULTRA"}),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                **encoder_widgets(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "upscale"

    def upscale(self, disk, scale, width, height, quality, base_dir, encoder, unique_id=None, **kwargs):
        try:
            import nvvfx
        except ImportError as e:
            raise Exception("nvidia-vfx is required for Disk RTX Upscale") from e
        media = require_media(disk)
        src_w, src_h = int(media["width"]), int(media["height"])
        if int(width) > 0 and int(height) > 0:
            out_w, out_h = int(width), int(height)
        else:
            out_w = int(src_w * float(scale))
            out_h = int(src_h * float(scale))
        out_w = max(8, round(out_w / 8) * 8)
        out_h = max(8, round(out_h / 8) * 8)
        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)
        out_path = new_clip_path(base_dir, unique_id, "vsr")
        writer = _FfmpegFrameWriter(out_path, out_w, out_h, media["fps"], encoder, **kwargs)
        pbar = ProgressBar(max(int(media["count"]), 1))
        try:
            with nvvfx.VideoSuperRes(selected) as sr:
                sr.output_width = out_w
                sr.output_height = out_h
                sr.load()
                for frame in _iter_disk_hwc(media):
                    inp = frame.movedim(-1, 0).float().contiguous().cuda()
                    # nvvfx capsule aliases C++ memory — clone before the next run()
                    up = torch.from_dlpack(sr.run(inp).image).clone()
                    writer.write_hwc(up.movedim(0, -1).float().clamp(0, 1))
                    del inp, up
                    pbar.update(1)
            result = writer.close()
        except Exception:
            if writer is not None:
                try:
                    writer.proc.kill()
                    writer.proc.wait()
                except Exception:
                    pass
            raise
        return (result, result["count"])
