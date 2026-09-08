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
from .utils import BIGMAX, ENCODE_ARGS, ffmpeg_path, floatOrInt, hash_path, strip_path, validate_path

DISK_TYPE = "VHS_DISK_MEDIA"
PREFERRED_ROOT = "/root/autodl-tmp"
DISK_SUBDIR = "vhs_disk"

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


def encoder_args(encoder, quality):
    encoder = pick_encoder(encoder)
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(quality), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality), "-pix_fmt", "yuv420p"]


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


def encode_images_to_file(images, out_path, fps, encoder, quality, pbar=None):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    height, width = int(images.shape[1]), int(images.shape[2])
    enc_w, enc_h = even_size(width, height)
    vf = []
    if (enc_w, enc_h) != (width, height):
        vf = ["-vf", f"pad={enc_w}:{enc_h}:(ow-iw)/2:(oh-ih)/2"]
    args = [
        _ffmpeg(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    ] + vf + encoder_args(encoder, quality) + ["-an", out_path]
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


def concat_media(media_a, media_b, out_path, encoder, quality, merge_strategy="match A"):
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
    fa = f"scale={target_w}:{target_h},fps={fps}" if scale_a else f"fps={fps}"
    fb = f"scale={target_w}:{target_h},fps={fps}" if scale_b else f"fps={fps}"
    filter_complex = f"[0:v]{fa}[v0];[1:v]{fb}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
    run_ffmpeg([
        _ffmpeg(), "-y", "-v", "error",
        "-i", a["path"], "-i", b["path"],
        "-filter_complex", filter_complex, "-map", "[v]",
    ] + encoder_args(encoder, quality) + ["-an", out_path])
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
        "-vf", f"select=between(n\\,{start}\\,{end})",
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
                "encoder": (["auto", "h264_nvenc", "libx264"],),
                "quality": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "save"

    def save(self, images, frame_rate, base_dir, encoder, quality, unique_id=None):
        out_path = new_clip_path(base_dir, unique_id, "images")
        pbar = ProgressBar(images.shape[0] if images.ndim == 4 else 1)
        media = encode_images_to_file(images, out_path, frame_rate, encoder, quality, pbar)
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
                "encoder": (["auto", "h264_nvenc", "libx264"],),
                "quality": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "merge"

    def merge(self, disk_A, disk_B, merge_strategy, base_dir, encoder, quality, unique_id=None):
        out_path = new_clip_path(base_dir, unique_id, "merge")
        media = concat_media(disk_A, disk_B, out_path, encoder, quality, merge_strategy)
        return (media, media["count"])


class AppendImagesToDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "images": ("IMAGE",),
                "base_dir": ("STRING", {"default": os.path.join(PREFERRED_ROOT, DISK_SUBDIR)}),
                "encoder": (["auto", "h264_nvenc", "libx264"],),
                "quality": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = (DISK_TYPE, "INT")
    RETURN_NAMES = ("disk", "count")
    FUNCTION = "append"

    def append(self, disk, images, base_dir, encoder, quality, unique_id=None):
        media = require_media(disk)
        tail_path = new_clip_path(base_dir, unique_id, "append_tail")
        out_path = new_clip_path(base_dir, unique_id, "append")
        pbar = ProgressBar(images.shape[0] if images.ndim == 4 else 1)
        tail = encode_images_to_file(images, tail_path, media["fps"], encoder, quality, pbar)
        result = concat_media(media, tail, out_path, encoder, quality, "match A")
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
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/disk"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "combine"

    def combine(self, disk, filename_prefix="DiskCombine", save_output=True, audio=None):
        media = require_media(disk)
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, _, _, _ = folder_paths.get_save_image_path(
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
        return {
            "ui": {
                "gifs": [{
                    "filename": os.path.basename(preview),
                    "subfolder": "",
                    "type": "output" if save_output else "temp",
                    "format": "video/mp4",
                    "frame_rate": media["fps"],
                    "fullpath": preview,
                }]
            },
            "result": ((save_output, output_files),),
        }
