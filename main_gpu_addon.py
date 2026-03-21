# -*- coding: utf-8 -*-
"""
GameDiffusion – GUI preview separate from floating capture window
- Floating capture window: thick red border + draggable handle.
  The 512×512 interior is a real hole via SetWindowRgn (never green).
- Preview is rendered ONLY in the GUI (right panel). No overlay window.
- Capture region tracks the floating window LIVE (no restart needed).
- Works with dxcam (pref) and falls back to MSS (MSS may capture overlays).

Patched to run WITHOUT Torch installed:
- No top-level torch import (lazy import only).
- Start button refuses to launch worker until Torch is available.
"""

import importlib.util, os, sys
# ==== _internal GPU bootstrap (flat) ====
from pathlib import Path
APP_ROOT = (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent)
INTERNAL_DIR = APP_ROOT / "_internal"
try:
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ========================================

import tempfile, time
import queue
import random
import threading
import pathlib
import array

import subprocess
from collections import deque
from multiprocessing import get_context, Queue
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional, Deque, Any
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import PIL.Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
import tkinter as tk
import venv
import re
RUNTIME_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Local", "ScreenDiffusion", "runtime")
PY_ENV = os.path.join(RUNTIME_DIR, "venv")
_PIP_DL_PERCENT_RE = re.compile(r"Downloading\s+[^\n]*\s(\d{1,3})%")
_PIP_LEGACY_BAR_RE = re.compile(r"\[\s*(\d{1,3})%\]")
import shutil
GPU_DIAGNOSTIC = True

APP_DIR   = pathlib.Path(os.getenv("LOCALAPPDATA", os.getcwd())) / "ScreenDiffusion"
RUNTIME   = APP_DIR / "runtime"
PY_EXE    = RUNTIME / "Scripts" / "python.exe"
PIP_EXE   = RUNTIME / "Scripts" / "pip.exe"
BOOT_FLAG = "SD_RUNTIME_BOOTSTRAPPED"   

import ctypes
import ctypes.wintypes as wint
_user32 = ctypes.windll.user32
_gdi32  = ctypes.windll.gdi32

CreateRectRgn = _gdi32.CreateRectRgn
CombineRgn    = _gdi32.CombineRgn
DeleteObject  = _gdi32.DeleteObject
SetWindowRgn  = _user32.SetWindowRgn
RGN_OR        = 2

CUSTOM_COLORS = {
    "success": "#10B981",
    "error": "#EF4444", 
    "surface": "#374151"
}

SPOUT_INSTALL_HINT = (
    "Spout dependencies missing. Install with: "
    ".\\.venv\\Scripts\\python.exe -m pip install SpoutGL==0.1.1 PyOpenGL==3.1.10"
)

GWL_EXSTYLE       = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TOOLWINDOW  = 0x00000080
GetWindowLongW    = _user32.GetWindowLongW
SetWindowLongW    = _user32.SetWindowLongW
SetLayeredWindowAttributes = _user32.SetLayeredWindowAttributes
LWA_ALPHA         = 0x00000002


def _check_spout_dependencies() -> tuple[bool, str]:
    try:
        import SpoutGL  # noqa: F401
        from OpenGL import GL  # noqa: F401
        return True, "Spout ready"
    except Exception as e:
        return False, f"{SPOUT_INSTALL_HINT} | Details: {e}"


def _resize_fit_pad_rgba(frame: np.ndarray, target_w: int, target_h: int):
    """Fit entire frame into target while preserving aspect ratio and padding."""
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if src_w <= 0 or src_h <= 0:
        blank = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        mask = np.zeros((target_h, target_w), dtype=np.float32)
        return blank, mask, 1.0, 1.0

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    if (new_w, new_h) != (src_w, src_h):
        try:
            import cv2
            interp = cv2.INTER_AREA if (new_w < src_w or new_h < src_h) else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        except Exception:
            resized = np.array(
                PIL.Image.fromarray(frame, "RGBA").resize((new_w, new_h), resample=PIL.Image.BILINEAR)
            )
    else:
        resized = frame

    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    mask[off_y:off_y + new_h, off_x:off_x + new_w] = 1.0

    # Compensation factors for motion vectors when sender size differs.
    flow_scale_x = float(new_w) / float(src_w)
    flow_scale_y = float(new_h) / float(src_h)
    return out, mask, flow_scale_x, flow_scale_y


def _packet_from_rgba(
    rgba: np.ndarray,
    target_w: int,
    target_h: int,
    torch_mod,
    motion_rgba: Optional[np.ndarray] = None,
):
    rgba_fit, mask2d, flow_sx, flow_sy = _resize_fit_pad_rgba(rgba, target_w, target_h)
    color = rgba_fit[..., :3].astype(np.float32) / 255.0
    color_t = torch_mod.from_numpy(color).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)

    motion_t = None
    if motion_rgba is not None:
        motion_fit, _, _, _ = _resize_fit_pad_rgba(motion_rgba, target_w, target_h)
        motion = motion_fit[..., :3].astype(np.float32) / 255.0
        motion_t = torch_mod.from_numpy(motion).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)

    mask_t = torch_mod.from_numpy(mask2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return {
        "color": color_t,
        "motion": motion_t,
        "mask": mask_t,
        "flow_scale_x": float(flow_sx),
        "flow_scale_y": float(flow_sy),
    }


def _decode_motion_flow(
    motion_rgb,
    mask,
    motion_scale: float,
    invert_y: bool,
    use_b_magnitude: bool,
    b_scale: float,
    flow_scale_x: float,
    flow_scale_y: float,
):
    # motion_rgb is [N,3,H,W] in [0,1]
    import torch

    rg = motion_rgb[:, 0:2, :, :]
    flow = (rg * 2.0 - 1.0) * float(motion_scale)
    if invert_y:
        flow[:, 1:2, :, :] *= -1.0
    flow[:, 0:1, :, :] *= float(flow_scale_x)
    flow[:, 1:2, :, :] *= float(flow_scale_y)

    if use_b_magnitude:
        b = motion_rgb[:, 2:3, :, :].clamp(0.0, 1.0)
        mag = b * float(b_scale)
        flow = flow * mag

    # Guard against malformed input and keep warping numerically stable.
    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    # Reduce tiny global bias drift (common with imperfect neutral encoding around 0.5).
    if mask is not None:
        valid = mask.clamp(0.0, 1.0)
        denom = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean_flow = (flow * valid).sum(dim=(2, 3), keepdim=True) / denom
        flow = flow - mean_flow * (mean_flow.abs() < 0.75).to(flow.dtype)
    else:
        mean_flow = flow.mean(dim=(2, 3), keepdim=True)
        flow = flow - mean_flow * (mean_flow.abs() < 0.75).to(flow.dtype)

    # Soft deadzone removes sub-pixel jitter that accumulates into feedback artifacts.
    deadzone_px = 0.10
    mag = torch.sqrt((flow[:, 0:1, :, :] ** 2) + (flow[:, 1:2, :, :] ** 2))
    shrink = (mag - deadzone_px).clamp(min=0.0) / (mag + 1e-6)
    flow = flow * shrink

    # Hard clamp max displacement per frame to avoid catastrophic warp blow-ups.
    max_flow_px = 6.0
    mag = torch.sqrt((flow[:, 0:1, :, :] ** 2) + (flow[:, 1:2, :, :] ** 2))
    clamp_scale = (max_flow_px / (mag + 1e-6)).clamp(max=1.0)
    flow = flow * clamp_scale

    if mask is not None:
        flow = flow * mask
    return flow


def _warp_prev_with_flow(prev_tensor, flow_pixels):
    # prev_tensor: [N,3,H,W], flow_pixels: [N,2,H,W] in pixel units
    import torch
    n, _, h, w = prev_tensor.shape
    device = prev_tensor.device
    dtype = prev_tensor.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    base_x = xs.unsqueeze(0).expand(n, -1, -1)
    base_y = ys.unsqueeze(0).expand(n, -1, -1)

    # Backward warp: sample previous frame at (x - flow) to align with current frame.
    sample_x = base_x - flow_pixels[:, 0, :, :]
    sample_y = base_y - flow_pixels[:, 1, :, :]
    if w > 1:
        norm_x = (sample_x / (w - 1)) * 2.0 - 1.0
    else:
        norm_x = sample_x * 0.0
    if h > 1:
        norm_y = (sample_y / (h - 1)) * 2.0 - 1.0
    else:
        norm_y = sample_y * 0.0

    grid = torch.stack((norm_x, norm_y), dim=-1)
    return torch.nn.functional.grid_sample(
        prev_tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


if getattr(sys, "frozen", False):
    tensorrt_dirs = [INTERNAL_DIR / "tensorrt", INTERNAL_DIR / "tensorrt_libs"]
    for trt_dir in tensorrt_dirs:
        if trt_dir.exists():
            try:
                os.add_dll_directory(str(trt_dir))
                os.environ["PATH"] = str(trt_dir) + os.pathsep + os.environ.get("PATH", "")
            except:
                pass




def _prime_dll_search():
    """Ensure Windows can find DLLs when importing from _internal."""
    import os, sys
    from pathlib import Path
    if os.name != "nt":
        return
    
    print("[DLL Search] Priming DLL search paths...")
    
    # List of packages that have DLL directories
    dll_packages = [
        "torch",
        "tensorrt",
        #"tensorrt_libs",
        #"onnxruntime",
        "xformers",
    ]
    
    # Add package lib directories
    for package in dll_packages:
        pkg_dir = INTERNAL_DIR / package
        if pkg_dir.exists():
            # Try lib subdirectory
            lib_dir = pkg_dir / "lib"
            if lib_dir.exists():
                print(f"[DLL Search] Found {package} lib: {lib_dir}")
                try:
                    os.add_dll_directory(str(lib_dir))
                    print(f"[DLL Search] ✅ Added {package} lib to DLL directories")
                except Exception as e:
                    print(f"[DLL Search] ⚠ Error adding {package} lib: {e}")
                
                # Also add to PATH
                current_path = os.environ.get("PATH", "")
                lib_str = str(lib_dir)
                if lib_str not in current_path:
                    os.environ["PATH"] = lib_str + os.pathsep + current_path
            
            # Also check package root directory for DLLs
            if any(pkg_dir.glob("*.dll")):
                print(f"[DLL Search] Found DLLs in {package} root: {pkg_dir}")
                try:
                    os.add_dll_directory(str(pkg_dir))
                    print(f"[DLL Search] ✅ Added {package} root to DLL directories")
                except Exception as e:
                    print(f"[DLL Search] ⚠ Error adding {package} root: {e}")
                
                current_path = os.environ.get("PATH", "")
                pkg_str = str(pkg_dir)
                if pkg_str not in current_path:
                    os.environ["PATH"] = pkg_str + os.pathsep + current_path
        else:
            print(f"[DLL Search] ⚠ {package} directory not found at: {pkg_dir}")
    
    # Also check for other common DLL locations
    possible_dll_dirs = [
        INTERNAL_DIR / "bin",
        INTERNAL_DIR / "Library" / "bin",
    ]
    
    for dll_dir in possible_dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
                print(f"[DLL Search] Added: {dll_dir}")
            except Exception:
                pass
    
    print("[DLL Search] DLL search path setup complete")

def try_import_dependency(name: str):
    """Return torch module if present, else None (never raises at import-time)."""
    try:
        import importlib
        return importlib.import_module("torch")
    except Exception:
        return None

class PreloadedDependencies:
    _torch = None
    _streamdiffusion = None
    _stream_wrapper = None
    _cached_models = {}
    _numpy = None
    _pil = None
    
    @classmethod
    def get_torch(cls):
        if cls._torch is None:
            try:
                _prime_dll_search()  # Set up DLL paths first
                cls._torch = try_import_dependency("torch")
            except Exception as e:
                print(f"[Preload] Could not load torch: {e}")
                cls._torch = None
        return cls._torch
    
    @classmethod
    def get_streamdiffusion(cls):
        if cls._streamdiffusion is None:
            cls._streamdiffusion = try_import_dependency("streamdiffusion")
        return cls._streamdiffusion
    
    @classmethod
    def get_stream_wrapper(cls):
        if cls._stream_wrapper is None:
            cls._stream_wrapper = _load_stream_wrapper()
        return cls._stream_wrapper
    
    @classmethod
    def get_numpy(cls):
        if cls._numpy is None:
            import numpy as np
            cls._numpy = np
        return cls._numpy
    
    @classmethod
    def get_pil(cls):
        if cls._pil is None:
            from PIL import Image, ImageDraw
            cls._pil = Image
            cls._pil_draw = ImageDraw
        return cls._pil
    
    @classmethod
    def preload_all(cls):
        """Preload all heavy dependencies in background"""
        import threading
        
        def background_preload():
            cls.get_torch()  # Still try torch if available
            cls.get_streamdiffusion()  # Try streamdiffusion
            cls.get_numpy()
            cls.get_pil()
        
        thread = threading.Thread(target=background_preload, daemon=True)
        thread.start()
        return thread

if not getattr(sys, "frozen", False):
    # Only preload in dev mode
    torch = PreloadedDependencies.get_torch()
else:
    # In frozen mode, torch will be loaded on-demand after DLL setup
    torch = None

def _patch_torch_imports():
    """Patch PyTorch imports to handle missing standard library modules in frozen environments."""
    import sys
    import types
    
    if 'unittest.mock' not in sys.modules:
        try:
            import unittest.mock as mock
            sys.modules['unittest.mock'] = mock
        except ImportError:
            class MinimalMock:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return self
                def __getattr__(self, name):
                    return self
            
            mock_module = types.ModuleType('unittest.mock')
            mock_module.MagicMock = MinimalMock
            mock_module.Mock = MinimalMock
            mock_module.patch = MinimalMock
            mock_module.PropertyMock = MinimalMock
            mock_module.ANY = object()
            
            sys.modules['unittest.mock'] = mock_module
def resource_path(name: str) -> str:
    """Return absolute path to a bundled resource (works in dev and PyInstaller)."""
    if getattr(sys, "frozen", False):  # running as EXE
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))  # _internal for onedir
        exe_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, name),                # e.g., _internal/logo.png
            os.path.join(exe_dir, name),            # next to the EXE
            os.path.join(base, "_internal", name),  # some builds place data here
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return os.path.join(exe_dir, name)
    # dev run
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

def _patch_missing_modules():
    """Patch missing standard library modules in frozen environments."""
    import sys
    import types
    
    if 'modulefinder' not in sys.modules:
        try:
            import modulefinder
            sys.modules['modulefinder'] = modulefinder
        except ImportError:
            modulefinder_module = types.ModuleType('modulefinder')
            modulefinder_module.ModuleFinder = type('ModuleFinder', (), {})
            sys.modules['modulefinder'] = modulefinder_module

_patch_missing_modules()
_patch_torch_imports()
_prime_dll_search()

def check_xformers_availability():
    """Check if xformers is available and working - more tolerant version"""
    try:
        import xformers
        import xformers.ops
        
        # Test if xformers is actually working with a simple test
        try:
            # Simple test that doesn't require specific hardware
            if hasattr(xformers.ops, 'memory_efficient_attention'):
                return True, "xformers is available"
            else:
                return False, "xformers installed but memory_efficient_attention not available"
        except Exception as e:
            # xformers is installed but might have compatibility issues
            return False, f"xformers installed but has compatibility issues: {e}"
            
    except ImportError:
        return False, "xformers not installed"
    except Exception as e:
        return False, f"xformers error: {e}"


def verify_streamdiffusion_dependencies():
    """Verify StreamDiffusion dependencies - more lenient version"""
    dependencies = [
        "streamdiffusion",
        "PIL",  # Pillow
        "numpy",
    ]
    
    # These are important but we'll be more lenient
    optional_deps = [
        "torch",
        "torchvision", 
        "torchaudio",
        "diffusers",
        "transformers",
        "cv2",  # opencv-python
        "xformers",
    ]
    
    missing = []
    warnings = []
    
    # Check required dependencies
    for dep in dependencies:
        try:
            if dep == "PIL":
                import PIL.Image
                PIL.Image.new('RGB', (10, 10))
            elif dep == "cv2":
                import cv2
                cv2.__version__
            else:
                module = __import__(dep)
        except ImportError as e:
            missing.append(f"{dep}: {e}")
        except Exception as e:
            warnings.append(f"{dep}: loaded but has issues - {e}")
    
    # Check optional dependencies (don't add to missing)
    for dep in optional_deps:
        try:
            if dep == "PIL" or dep == "cv2":
                continue  # Already checked
            elif dep == "xformers":
                xformers_available, xformers_msg = check_xformers_availability()
                if not xformers_available:
                    warnings.append(f"xformers: {xformers_msg} (optional)")
            else:
                module = __import__(dep)
        except ImportError:
            warnings.append(f"{dep}: not installed (optional)")
        except Exception as e:
            warnings.append(f"{dep}: loaded but has issues - {e}")
    
    return missing, warnings


def maybe_bootstrap_gpu(on_progress=None, progress_callback=None):
    """Install ONLY PyTorch 2.1.0+cu118 and TorchVision 0.16.0+cu118. StreamDiffusion is bundled in EXE."""
    
    if on_progress:
        on_progress("Checking for PyTorch 2.1.0+cu118...")
    
    # Check if correct PyTorch version with CUDA is already available
    torch_available = False
    try:
        import torch
        import torchvision
        
        # Check for exact version
        torch_version = torch.__version__
        if torch_version.startswith("2.1.0") and "+cu118" in torch_version:
            if torch.cuda.is_available():
                if on_progress:
                    on_progress(f"✅ PyTorch {torch_version} with CUDA already available!")
                torch_available = True
                return True
            else:
                if on_progress:
                    on_progress(f"⚠ PyTorch {torch_version} found but CUDA not available")
        else:
            if on_progress:
                on_progress(f"⚠ Found PyTorch {torch_version}, but need 2.1.0+cu118. Reinstalling...")
    except ImportError:
        if on_progress:
            on_progress("PyTorch not found, starting installation...")
    except Exception as e:
        if on_progress:
            on_progress(f"⚠ PyTorch check error: {e}")
    
    # Install specific PyTorch version if not available
    if not torch_available:
        if on_progress:
            on_progress("📦 Installing PyTorch 2.1.0+cu118 and TorchVision 0.16.0+cu118...")
        
        try:
            py_cmd = _python_cmd_for_pip()
            
            # Install specific versions of PyTorch and TorchVision with CUDA 11.8
            torch_cmd = [
                *py_cmd, "-m", "pip", "install",
                "--target", str(INTERNAL_DIR),
                "--upgrade",  # ← Added this!
                "--no-deps",
                "-vv",
                "torch==2.1.0+cu118",
                "torchvision==0.16.0+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
            
            if on_progress:
                on_progress("⏬ Downloading PyTorch 2.1.0 and TorchVision 0.16.0 packages...")
            
            for line in _gpu_run(torch_cmd, on_progress=on_progress, no_window=not GPU_DIAGNOSTIC, progress_callback=progress_callback):
                pass
            
            if on_progress:
                on_progress("✅ PyTorch 2.1.0+cu118 installation complete!")
            
            # Verify installation
            if on_progress:
                on_progress("🔍 Verifying PyTorch installation...")
            
            try:
                # Force reimport
                import sys
                if 'torch' in sys.modules:
                    del sys.modules['torch']
                if 'torchvision' in sys.modules:
                    del sys.modules['torchvision']
                
                import torch
                import torchvision
                
                torch_version = torch.__version__
                torchvision_version = torchvision.__version__
                cuda_available = torch.cuda.is_available()
                
                if on_progress:
                    on_progress(f"Installed: PyTorch {torch_version}, TorchVision {torchvision_version}")
                
                if cuda_available:
                    if on_progress:
                        on_progress(f"✅ Verification successful! CUDA is available")
                    return True
                else:
                    if on_progress:
                        on_progress("⚠ PyTorch installed but CUDA not available (may need GPU drivers)")
                    return True  # Still return True, might be driver issue
                    
            except Exception as e:
                if on_progress:
                    on_progress(f"❌ Verification failed: {e}")
                return False
            
        except Exception as e:
            if on_progress:
                on_progress(f"❌ PyTorch installation failed: {e}")
            import traceback
            if on_progress:
                on_progress(f"Traceback: {traceback.format_exc()}")
            return False
    
    return torch_available


def _startup_probe(log_fn=None, logfile_name="ScreenDiffusion_probe.txt"):
    import sys, os
    from pathlib import Path

    log_path = Path(os.environ.get("LOCALAPPDATA", ".")) / logfile_name

    def p(msg: str):
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    p("=== STARTUP PROBE ===")
    try:
        p(f"frozen? {getattr(sys, 'frozen', False)}")
        p(f"exe: {sys.executable}")
        p(f"argv0: {sys.argv[:2]}")
        p(f"PYTHONPATH: {os.environ.get('PYTHONPATH','')}")
        p(f"PATH[0:3]: {os.environ.get('PATH','').split(os.pathsep)[:3]}")
        p(f"sys.path[0:5]: {sys.path[:5]}")

        sp1 = RUNTIME / "Lib" / "site-packages"
        p(f"RUNTIME: {RUNTIME}  exists={RUNTIME.exists()}")
        p(f"site-packages: {sp1}  exists={sp1.exists()}  on_sys_path={str(sp1) in sys.path}")

        from pathlib import Path as _P
        if getattr(sys, "frozen", False):
            sp2 = _P(sys.executable).parent / "_internal" / "Lib" / "site-packages"
            p(f"_internal site-packages: {sp2}  exists={sp2.exists()}  on_sys_path={str(sp2) in sys.path}")

        try:
            import importlib
            t = importlib.import_module("torch")
            p(f"import torch (exe): OK v{t.__version__}")
        except Exception as e:
            p(f"import torch (exe): FAIL -> {e!r}")

        try:
            for line in _gpu_run([str(PY_EXE), "-c", "import torch,sys;print('VENV_TORCH',torch.__version__)"]):
                p(f"[venv] {line}")
        except Exception as e:
            p(f"[venv] import torch: FAIL -> {e!r}")
    finally:
        p("=== END PROBE ===")

try:
    import os
    if os.environ.get(BOOT_FLAG) == "1":
        os.environ.pop(BOOT_FLAG, None)
except Exception:
    pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

GD_DETERMINISTIC = bool(int(os.getenv('GD_DETERMINISTIC', '0')))

LOCAL_MODEL_PATH = r""
PREVIEW_GAIN = 1.15

def enforce_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def verify_local_model_path_dir(path: str):
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Local model directory not found: {path}")

def _frame_to_rgb(frame: np.ndarray, force_swap_rb: Optional[bool] = False) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if force_swap_rb:
        frame = frame[:, :, ::-1]
    return frame

SHOW = {
    "model_path": True,
    "prompt": True,
    "negative_prompt": True,
    "seed": True,
    "frame_buffer_size": False,
    "acceleration": True,
    "use_denoising_batch": False,
    "cfg_type": True,
    "guidance_scale": True,
    "delta": True,
    "similar_image_filter": True,
    "offline": False,
}

# =========================
# Windows interop
# =========================
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

MONITOR_DEFAULTTONEAREST = 2

def _monitor_rc_from_point(x: int, y: int):
    hmon = _user32.MonitorFromPoint(POINT(x, y), MONITOR_DEFAULTTONEAREST)
    mi = MONITORINFO(); mi.cbSize = ctypes.sizeof(MONITORINFO)
    _user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    r = mi.rcMonitor
    return r.left, r.top, r.right, r.bottom

# =========================
# Floating capture window
# =========================

class FloatingCaptureWindow:
    def __init__(self, master: tk.Tk, inner_size=512, border_px=8, handle_h=28):
        self.master = master
        self.inner_size = int(inner_size)
        self.border_px = int(border_px)
        self.handle_h = int(handle_h)

        total_w = self.inner_size + self.border_px * 2
        total_h = self.handle_h + self.inner_size + self.border_px

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)

        try:
            x0 = int(master.winfo_rootx()) + 50
            y0 = int(master.winfo_rooty()) + 50
        except Exception:
            x0, y0 = 200, 200
        self.win.geometry(f"{total_w}x{total_h}+{x0}+{y0}")

        self.canvas = tk.Canvas(
            self.win, width=total_w, height=total_h,
            highlightthickness=0, bd=0, relief="flat"
        )
        self.canvas.pack(fill="both", expand=True)

        self.border_color = "#ef4444"
        self.handle_color = "#7f1d1d"

        self.canvas.create_rectangle(
            0, 0, total_w, self.handle_h,
            fill=self.handle_color, outline=self.handle_color
        )
        self.canvas.create_text(
            10, self.handle_h // 2, anchor="w",
            text="Capture 512×512", fill="#ffffff", font=("Segoe UI", 9, "bold")
        )
        
        self.canvas.create_rectangle(
            0, self.handle_h, self.border_px, self.handle_h + self.inner_size + self.border_px,
            fill=self.border_color, outline=self.border_color
        )
        self.canvas.create_rectangle(
            self.border_px + self.inner_size, self.handle_h,
            total_w, self.handle_h + self.inner_size + self.border_px,
            fill=self.border_color, outline=self.border_color
        )
        self.canvas.create_rectangle(
            0, self.handle_h + self.inner_size, total_w,
            self.handle_h + self.inner_size + self.border_px,
            fill=self.border_color, outline=self.border_color
        )

        self._apply_window_region(total_w, total_h)

        self._drag_start = None
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)

        try:
            hwnd = int(self.win.winfo_id())
            ex = GetWindowLongW(hwnd, GWL_EXSTYLE)
            # NO WS_EX_TRANSPARENT - SetWindowRgn handles the hole
            SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TOOLWINDOW)
            SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
        except Exception as e:
            print(f"[capwin] style setup failed: {e}")

    def _apply_window_region(self, total_w: int, total_h: int):
        try:
            region = CreateRectRgn(0, 0, total_w, self.handle_h)

            left = CreateRectRgn(0, self.handle_h, self.border_px,
                                 self.handle_h + self.inner_size + self.border_px)
            CombineRgn(region, region, left, RGN_OR); DeleteObject(left)

            right = CreateRectRgn(self.border_px + self.inner_size, self.handle_h,
                                  total_w, self.handle_h + self.inner_size + self.border_px)
            CombineRgn(region, region, right, RGN_OR); DeleteObject(right)

            bottom = CreateRectRgn(0, self.handle_h + self.inner_size, total_w,
                                   self.handle_h + self.inner_size + self.border_px)
            CombineRgn(region, region, bottom, RGN_OR); DeleteObject(bottom)

            SetWindowRgn(self.win.winfo_id(), region, True)
        except Exception as e:
            print(f"[capwin] region failed: {e}")

    def _on_mouse_down(self, evt):
        if 0 <= evt.y <= self.handle_h:
            self._drag_start = (evt.x_root, evt.y_root)
            self._start_geom = (int(self.win.winfo_x()), int(self.win.winfo_y()))

    def _on_mouse_drag(self, evt):
        if not self._drag_start: return
        dx = evt.x_root - self._drag_start[0]
        dy = evt.y_root - self._drag_start[1]
        x, y = self._start_geom
        self.win.geometry(f"+{x+dx}+{y+dy}")
        try:
            if hasattr(self.master, "_on_capture_window_moved"):
                self.master._on_capture_window_moved()
        except Exception:
            pass

    def destroy(self):
        try: self.win.destroy()
        except Exception: pass

    def inner_rect_screen(self):
        self.win.update_idletasks()
        x = int(self.win.winfo_rootx()) + self.border_px
        y = int(self.win.winfo_rooty()) + self.handle_h
        return {"left": x, "top": y, "width": self.inner_size, "height": self.inner_size}

# =========================
# Capture loops
# =========================
def _screen_capture_loop_dx(stop_evt: threading.Event,
                            height: int, width: int,
                            region_ref: Dict[str, Dict[str, int]],
                            max_buffer: int,
                            inputs_list: List[Any]):
    """Optimized DXGI screen-capture loop with pinned CPU staging + load shedding."""
    torch = PreloadedDependencies.get_torch()
    if torch is None:
        return

    def _append_shed(q, item, maxlen):
        """Append with shedding: drop oldest if the consumer is behind."""
        try:
            if hasattr(q, "maxlen") and q.maxlen is not None:
                if len(q) >= q.maxlen:
                    q.popleft()
                q.append(item)
            else:
                while len(q) >= maxlen:
                    q.pop(0)
                q.append(item)
        except Exception:
            pass

    try:
        import dxcam
        camera = dxcam.create(output_idx=0, output_color="RGB")
        camera.start(target_fps=30, video_mode=True)
    except Exception as e:
        print(f"dxcam failed, falling back to MSS: {e}")
        _screen_capture_loop_mss(stop_evt, height, width, region_ref, max_buffer, inputs_list)
        return

    device = torch.device("cuda")
    H_t, W_t = int(height), int(width)
    C = 3
    dtype_cpu = torch.uint8

    staging_cpu = None
    need_resize = None
    screen_mask = None

    frame_count = 0
    last_log_time = time.time()
    frame_api_mode = "auto"  # auto -> region_kw | grab_region | full_frame

    try:
        while not stop_evt.is_set():
            rect = region_ref["rect"]
            L, T = int(rect["left"]), int(rect["top"])
            W, H = int(rect["width"]), int(rect["height"])
            R, B = L + W, T + H

            frame = None
            if frame_api_mode in ("auto", "region_kw"):
                try:
                    frame = camera.get_latest_frame(region=(L, T, R, B))
                    frame_api_mode = "region_kw"
                except TypeError:
                    frame_api_mode = "grab_region"
                except Exception:
                    frame_api_mode = "grab_region"

            if frame is None and frame_api_mode in ("auto", "grab_region"):
                try:
                    frame = camera.grab(region=(L, T, R, B))
                    frame_api_mode = "grab_region"
                except Exception:
                    frame_api_mode = "full_frame"

            if frame is None:
                frame = camera.get_latest_frame()
                if frame is not None:
                    # Some dxcam builds return full output frame and do not accept region input.
                    h_full, w_full = int(frame.shape[0]), int(frame.shape[1])
                    x0 = max(0, L)
                    y0 = max(0, T)
                    x1 = min(w_full, R)
                    y1 = min(h_full, B)
                    if (x1 > x0) and (y1 > y0):
                        frame = frame[y0:y1, x0:x1]
            if frame is None:
                time.sleep(0.001)
                continue

            if frame.shape[-1] == 4:
                frame = frame[..., :3]

            if staging_cpu is None:
                H_c, W_c = int(frame.shape[0]), int(frame.shape[1])
                staging_cpu = torch.empty((H_c, W_c, C), dtype=dtype_cpu, pin_memory=True)
                need_resize = (H_c != H_t) or (W_c != W_t)

            staging_cpu.numpy()[...] = frame

            img = staging_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            img = img.permute(2, 0, 1).unsqueeze(0)
            img.mul_(1.0 / 255.0)

            if need_resize:
                img = torch.nn.functional.interpolate(
                    img, size=(H_t, W_t), mode="bilinear", align_corners=False
                )

            if screen_mask is None:
                screen_mask = torch.ones((1, 1, H_t, W_t), device=img.device, dtype=img.dtype)
            packet = {
                "color": img,
                "motion": None,
                "mask": screen_mask,
                "flow_scale_x": 1.0,
                "flow_scale_y": 1.0,
            }
            _append_shed(inputs_list, packet, max_buffer)

            frame_count += 1
            now = time.time()
            if now - last_log_time > 5.0:
                fps = frame_count / (now - last_log_time)
                try:
                    buf_len = len(inputs_list)
                except Exception:
                    buf_len = -1
                print(f"Capture FPS: {fps:.1f} | Buffer: {buf_len} | Resize:{bool(need_resize)}")
                frame_count = 0
                last_log_time = now

    except Exception as e:
        print(f"Capture loop error: {e}")
        try:
            camera.stop()
        except Exception:
            pass
        _screen_capture_loop_mss(stop_evt, height, width, region_ref, max_buffer, inputs_list)
    finally:
        try:
            camera.stop()
        except Exception:
            pass

def _screen_capture_loop_mss(stop_evt: threading.Event,
                             height: int, width: int,
                             region_ref: Dict[str, Dict[str, int]],
                             max_buffer: int,
                             inputs_list: List[Any]):
    torch = PreloadedDependencies.get_torch()
    if torch is None:
        return
    try:
        import mss
    except Exception:
        return

    with mss.mss() as sct:
        while not stop_evt.is_set():
            rect = region_ref["rect"]
            monitor = {k: int(rect[k]) for k in ("left","top","width","height")}
            raw = sct.grab(monitor)

            bgra  = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
            frame = bgra[:, :, :3][:, :, ::-1]

            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = np.array(PIL.Image.fromarray(frame, "RGB").resize((width, height), resample=PIL.Image.BICUBIC))

            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)\
                         .permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
            packet = {
                "color": tensor,
                "motion": None,
                "mask": torch.ones((1, 1, height, width), dtype=tensor.dtype),
                "flow_scale_x": 1.0,
                "flow_scale_y": 1.0,
            }
            inputs_list.append(packet)
            if isinstance(inputs_list, list) and len(inputs_list) > max_buffer:
                del inputs_list[:-max_buffer]


def _spout_capture_loop(
    stop_evt: threading.Event,
    height: int,
    width: int,
    max_buffer: int,
    inputs_list: List[Any],
    capture_cfg_ref: Dict[str, Any],
    status_cb=None,
):
    torch = PreloadedDependencies.get_torch()
    if torch is None:
        return

    try:
        import SpoutGL
        from OpenGL import GL
    except Exception as e:
        if status_cb:
            status_cb(f"Spout unavailable: {e}")
        return

    def _append_shed(q, item, maxlen):
        try:
            if hasattr(q, "maxlen") and q.maxlen is not None:
                if len(q) >= q.maxlen:
                    q.popleft()
                q.append(item)
            else:
                while len(q) >= maxlen:
                    q.pop(0)
                q.append(item)
        except Exception:
            pass

    color_buffer = None
    motion_buffer = None
    color_w = color_h = 0
    motion_w = motion_h = 0
    last_motion_ok = False
    last_log = 0.0

    with SpoutGL.SpoutReceiver() as color_rx, SpoutGL.SpoutReceiver() as motion_rx:
        bound_color = None
        bound_motion = None
        while not stop_evt.is_set():
            cfg = dict(capture_cfg_ref.get("cfg", {}))
            color_name = str(cfg.get("spout_color_sender", "UE_Color"))
            motion_name = str(cfg.get("spout_motion_sender", "UE_Motion"))

            if color_name != bound_color:
                color_rx.setReceiverName(color_name)
                color_buffer = None
                bound_color = color_name
                if status_cb:
                    status_cb(f"Spout color sender: {color_name}")
            if motion_name != bound_motion:
                motion_rx.setReceiverName(motion_name)
                motion_buffer = None
                bound_motion = motion_name
                if status_cb:
                    status_cb(f"Spout motion sender: {motion_name}")

            try:
                color_rx.waitFrameSync(color_name, 0)
            except Exception:
                pass

            if color_buffer is None or color_rx.isUpdated():
                color_w = max(1, int(color_rx.getSenderWidth() or width))
                color_h = max(1, int(color_rx.getSenderHeight() or height))
                color_buffer = array.array("B", [0] * (color_w * color_h * 4))

            got_color = False
            try:
                got_color = bool(color_rx.receiveImage(color_buffer, GL.GL_RGBA, False, 0))
            except Exception:
                got_color = False
            if not got_color:
                time.sleep(0.002)
                continue

            color_rgba = np.asarray(color_buffer, dtype=np.uint8).reshape((color_h, color_w, 4)).copy()
            color_rx.setFrameSync(color_name)

            motion_rgba = None
            try:
                motion_rx.waitFrameSync(motion_name, 0)
            except Exception:
                pass
            try:
                if motion_buffer is None or motion_rx.isUpdated():
                    motion_w = max(1, int(motion_rx.getSenderWidth() or width))
                    motion_h = max(1, int(motion_rx.getSenderHeight() or height))
                    motion_buffer = array.array("B", [0] * (motion_w * motion_h * 4))
                got_motion = bool(motion_rx.receiveImage(motion_buffer, GL.GL_RGBA, False, 0))
                if got_motion:
                    motion_rgba = np.asarray(motion_buffer, dtype=np.uint8).reshape((motion_h, motion_w, 4)).copy()
                    motion_rx.setFrameSync(motion_name)
                    last_motion_ok = True
                else:
                    last_motion_ok = False
            except Exception:
                last_motion_ok = False

            packet = _packet_from_rgba(
                color_rgba,
                width,
                height,
                torch,
                motion_rgba=motion_rgba,
            )
            _append_shed(inputs_list, packet, max_buffer)

            now = time.time()
            if status_cb and (now - last_log > 5.0):
                mm = "ok" if last_motion_ok else "missing->no warp"
                status_cb(f"Spout running | color={color_name} | motion={motion_name} ({mm})")
                last_log = now

# =========================
# Worker
# =========================
def _load_stream_wrapper():
    # Locate wrapper.py, whether frozen or running from source
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    wrapper_path = os.path.join(base_dir, "wrapper.py")

    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wrapper"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "StreamDiffusionWrapper")


def image_generation_process(
    out_queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    status_queue: Queue,
    control_queue: Queue,
    debug_queue: Queue,
    t_index_list: List[int],
    model_path_dir: str,
    controlnet_paths: List[str],
    controlnet_scales: List[float],
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor_receiver: Connection,
    capture_config: Optional[Dict[str, Any]] = None,
    offline: bool = True,
) -> None:
    
    import sys
    import os
    from pathlib import Path
    
    # Determine APP_ROOT and INTERNAL_DIR for worker process
    if getattr(sys, "frozen", False):
        APP_ROOT_WORKER = Path(sys.executable).parent
    else:
        APP_ROOT_WORKER = Path(__file__).resolve().parent
    
    INTERNAL_DIR_WORKER = APP_ROOT_WORKER / "_internal"
    
    # Add _internal to sys.path FIRST
    internal_str = str(INTERNAL_DIR_WORKER)
    if internal_str not in sys.path:
        sys.path.insert(0, internal_str)
        print(f"[Worker] Added to sys.path: {internal_str}")
    
    # Set up DLL search paths for ALL packages with DLLs
    dll_packages = ["torch", "xformers"]
    
    for package in dll_packages:
        pkg_dir = INTERNAL_DIR_WORKER / package
        if pkg_dir.exists():
            # Try lib subdirectory
            lib_dir = pkg_dir / "lib"
            if lib_dir.exists():
                try:
                    os.add_dll_directory(str(lib_dir))
                    print(f"[Worker] Added DLL dir: {lib_dir}")
                except:
                    pass
                
                current_path = os.environ.get("PATH", "")
                if str(lib_dir) not in current_path:
                    os.environ["PATH"] = str(lib_dir) + os.pathsep + current_path
            
            # Also add package root if it has DLLs
            if any(pkg_dir.glob("*.dll")):
                try:
                    os.add_dll_directory(str(pkg_dir))
                    print(f"[Worker] Added DLL dir: {pkg_dir}")
                except:
                    pass
                
                current_path = os.environ.get("PATH", "")
                if str(pkg_dir) not in current_path:
                    os.environ["PATH"] = str(pkg_dir) + os.pathsep + current_path
    
    # Add common bin directories
    for bin_dir in [INTERNAL_DIR_WORKER / "bin", INTERNAL_DIR_WORKER / "Library" / "bin"]:
        if bin_dir.exists():
            try:
                os.add_dll_directory(str(bin_dir))
            except:
                pass
    
    print("[Worker] DLL search paths configured")


    _prime_dll_search()
    
    import_torch = PreloadedDependencies.get_torch()
    
    def _status(msg: str):
        try: 
            status_queue.put_nowait(msg)
        except Exception: 
            pass

    cap_cfg = dict(capture_config or {})
    capture_source = str(cap_cfg.get("capture_source", "screen")).lower()
    spout_color_sender = str(cap_cfg.get("spout_color_sender", "UE_Color"))
    spout_motion_sender = str(cap_cfg.get("spout_motion_sender", "UE_Motion"))
    spout_output_enabled = bool(cap_cfg.get("spout_output_enabled", False))
    spout_output_sender = str(cap_cfg.get("spout_output_sender", "SD_Output"))
    spout_output_fps_cap = float(cap_cfg.get("spout_output_fps_cap", 30.0))
    motion_warp_enabled = bool(cap_cfg.get("motion_warp_enabled", False))
    motion_scale = float(cap_cfg.get("motion_scale", 1.0))
    warp_strength = float(cap_cfg.get("warp_strength", 0.35))
    motion_invert_y = bool(cap_cfg.get("motion_invert_y", True))
    motion_use_b_magnitude = bool(cap_cfg.get("motion_use_b_magnitude", False))
    motion_b_scale = float(cap_cfg.get("motion_b_scale", 1.0)
                           if cap_cfg.get("motion_b_scale", 1.0) is not None else 1.0)

    try:
        _status("Worker process: initializing...")
    
        if import_torch is None:
            _status("ERROR: Torch not available in worker process")
           
            _prime_dll_search()
            import_torch = PreloadedDependencies.get_torch()
            if import_torch is None:
                _status("CRITICAL: Torch not found after path extension")
                return

        _status(f"Torch loaded: {import_torch.__version__}")
        _status(f"CUDA available: {import_torch.cuda.is_available()}")
    
        StreamDiffusionWrapper = _load_stream_wrapper()
        _status("StreamDiffusionWrapper imported")

        _status("Setting up pipeline...")
        if offline: 
            enforce_offline_mode()
            _status("Offline mode enabled")
        
        verify_local_model_path_dir(model_path_dir)
        _status(f"Model path verified: {model_path_dir}")

        # Initialize Stream Diffusion - wrapper handles all optimizations internally
        _status("Initializing StreamDiffusion...")
        model_root = os.path.dirname(os.path.normpath(model_path_dir))
        model_name = os.path.basename(os.path.normpath(model_path_dir))
        trt_engine_dir = os.path.join(model_root, f"{model_name}-trt")
        if acceleration == "tensorrt":
            os.makedirs(trt_engine_dir, exist_ok=True)
            _status(f"TensorRT engines dir: {trt_engine_dir}")

        stream = StreamDiffusionWrapper(
            model_id_or_path=model_path_dir,
            t_index_list=list(t_index_list),
            frame_buffer_size=frame_buffer_size,
            width=width,
            height=height,
            warmup=2,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            enable_similar_image_filter=enable_similar_image_filter,
            similar_image_filter_threshold=similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
            mode="img2img",
            use_denoising_batch=use_denoising_batch,
            cfg_type=cfg_type,
            seed=seed,
            lora_dict=lora_dict,
            engine_dir=trt_engine_dir,
        )
        _status("StreamDiffusion model prepared")

        # Prepare the pipeline
        _status("Preparing pipeline with prompts...")
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            delta=delta,
        )
        _status("Pipeline prepared successfully")

        region_ref = {"rect": {"left": 0, "top": 0, "width": width, "height": height}}
        if capture_source == "screen":
            _status("Waiting for capture region...")
            first_rect = monitor_receiver.recv()
            region_ref = {"rect": dict(first_rect)}
            _status(f"Capture region received: {first_rect}")
        else:
            _status("Spout mode selected (no capture window region needed)")

        # Setup capture
        inputs = deque(maxlen=frame_buffer_size * 2)
        cap_stop = threading.Event()
        capture_cfg_ref = {
            "cfg": {
                "spout_color_sender": spout_color_sender,
                "spout_motion_sender": spout_motion_sender,
            }
        }

        if capture_source == "spout":
            _status("Starting Spout capture thread...")
            cap_thr = threading.Thread(
                target=_spout_capture_loop,
                args=(
                    cap_stop,
                    height,
                    width,
                    frame_buffer_size,
                    inputs,
                    capture_cfg_ref,
                    _status,
                ),
                daemon=True,
            )
            cap_thr.start()
            _status("Spout capture started")
        else:
            _status("Starting screen capture thread...")
            cap_thr = threading.Thread(
                target=_screen_capture_loop_dx,
                args=(cap_stop, height, width, region_ref, frame_buffer_size, inputs),
                daemon=True,
            )
            cap_thr.start()
            _status("Screen capture started")

        spout_sender = None
        spout_gl = None
        spout_sender_name = spout_output_sender
        spout_out_last_ts = 0.0
        if spout_output_enabled:
            try:
                import SpoutGL
                from OpenGL import GL
                spout_gl = GL
                spout_sender = SpoutGL.SpoutSender()
                spout_sender.setSenderName(spout_sender_name)
                _status(f"Spout output enabled: {spout_sender_name}")
            except Exception as e:
                spout_sender = None
                _status(f"Spout output unavailable: {e}")
    
        current_t_index_list = list(t_index_list)
        frame_count = 0
        prev_output_tensor = None
        _status("Entering main processing loop...")

        # Main processing loop
        while close_queue.empty():
            # Handle control messages from GUI
            try:
                while True:
                    msg = control_queue.get_nowait()
                    if not isinstance(msg, dict): 
                        continue
                    
                    mtype = msg.get("type")

                    if mtype == "set_region" and capture_source == "screen":
                        r = msg.get("region")
                        if isinstance(r, dict) and all(k in r for k in ("left","top","width","height")):
                            region_ref["rect"] = {k:int(r[k]) for k in ("left","top","width","height")}

                    elif mtype == "set_t_at":
                        i = int(msg.get("index", -1))
                        val = int(msg.get("value", 30))
                        if 0 <= i < len(current_t_index_list):
                            current_t_index_list[i] = max(2, min(49, val))
                            stream.set_t_index_list(current_t_index_list)
                            _status(f"t[{i}] → {current_t_index_list[i]}")

                    elif mtype == "set_t_index_list":
                        new_t = msg.get("t_index_list")
                        if isinstance(new_t, (list, tuple)) and len(new_t) > 0:
                            current_t_index_list = [int(max(2, min(49, x))) for x in new_t]
                            stream.set_t_index_list(current_t_index_list)
                            _status(f"t_index_list → {current_t_index_list}")

                    elif mtype == "set_prompt":
                        new_prompt = str(msg.get("prompt", prompt))
                        prompt = new_prompt
                        try:
                            # Fix: stream.update_prompt, not stream.stream.update_prompt
                            stream.update_prompt(prompt, negative_prompt=negative_prompt)
                            _status("Prompt updated live")
                        except Exception:
                            # Fallback: prepare, but preserve t_index_list
                            stream.prepare(
                                prompt=prompt, 
                                negative_prompt=negative_prompt,
                                num_inference_steps=50, 
                                guidance_scale=guidance_scale, 
                                delta=delta
                            )
                            # CRITICAL: Restore the t_index_list after prepare
                            stream.set_t_index_list(current_t_index_list)
                            _status("Pipeline re-prepared with new prompt (t_list restored)")

                    elif mtype == "set_negative_prompt":
                        new_neg_prompt = str(msg.get("negative_prompt", negative_prompt))
                        negative_prompt = new_neg_prompt
                        try:
                            # Fix: stream.update_prompt, not stream.stream.update_prompt
                            stream.update_prompt(prompt, negative_prompt=negative_prompt)
                            _status("Negative prompt updated live")
                        except Exception:
                            # Fallback: prepare, but preserve t_index_list
                            stream.prepare(
                                prompt=prompt, 
                                negative_prompt=negative_prompt,
                                num_inference_steps=50, 
                                guidance_scale=guidance_scale, 
                                delta=delta
                            )
                            # CRITICAL: Restore the t_index_list after prepare
                            stream.set_t_index_list(current_t_index_list)
                            _status("Pipeline re-prepared with new negative prompt (t_list restored)")

                    elif mtype == "set_advanced":
                        cfg = msg.get("cfg")
                        if isinstance(cfg, dict):
                            prev_motion_warp_enabled = bool(motion_warp_enabled)
                            new_capture_source = str(cfg.get("capture_source", capture_source)).lower()
                            if new_capture_source != capture_source:
                                _status("Capture source change requires restart")

                            spout_color_sender = str(cfg.get("spout_color_sender", spout_color_sender))
                            spout_motion_sender = str(cfg.get("spout_motion_sender", spout_motion_sender))
                            capture_cfg_ref["cfg"]["spout_color_sender"] = spout_color_sender
                            capture_cfg_ref["cfg"]["spout_motion_sender"] = spout_motion_sender

                            spout_output_enabled = bool(cfg.get("spout_output_enabled", spout_output_enabled))
                            spout_output_sender = str(cfg.get("spout_output_sender", spout_output_sender))
                            try:
                                spout_output_fps_cap = max(1.0, float(cfg.get("spout_output_fps_cap", spout_output_fps_cap)))
                            except Exception:
                                pass

                            motion_warp_enabled = bool(cfg.get("motion_warp_enabled", motion_warp_enabled))
                            if motion_warp_enabled != prev_motion_warp_enabled:
                                prev_output_tensor = None
                                _status("Motion warp state changed; history reset")
                            try:
                                motion_scale = float(cfg.get("motion_scale", motion_scale))
                                warp_strength = max(0.0, min(1.0, float(cfg.get("warp_strength", warp_strength))))
                                motion_b_scale = float(cfg.get("motion_b_scale", motion_b_scale))
                            except Exception:
                                pass
                            motion_invert_y = bool(cfg.get("motion_invert_y", motion_invert_y))
                            motion_use_b_magnitude = bool(cfg.get("motion_use_b_magnitude", motion_use_b_magnitude))

                            if spout_output_enabled and spout_sender is None:
                                try:
                                    import SpoutGL
                                    from OpenGL import GL
                                    spout_gl = GL
                                    spout_sender = SpoutGL.SpoutSender()
                                    spout_sender_name = spout_output_sender
                                    spout_sender.setSenderName(spout_sender_name)
                                    _status(f"Spout output enabled: {spout_sender_name}")
                                except Exception as e:
                                    _status(f"Spout output unavailable: {e}")
                            elif (not spout_output_enabled) and spout_sender is not None:
                                try:
                                    spout_sender.releaseSender()
                                except Exception:
                                    pass
                                spout_sender = None
                                _status("Spout output disabled")

                            if spout_sender is not None and spout_output_sender != spout_sender_name:
                                try:
                                    spout_sender.releaseSender()
                                except Exception:
                                    pass
                                try:
                                    import SpoutGL
                                    spout_sender = SpoutGL.SpoutSender()
                                    spout_sender_name = spout_output_sender
                                    spout_sender.setSenderName(spout_sender_name)
                                    _status(f"Spout output sender -> {spout_sender_name}")
                                except Exception as e:
                                    spout_sender = None
                                    _status(f"Spout sender update failed: {e}")
                            
            except Exception:
                pass

            # Wait until we have enough frames
            if len(inputs) < frame_buffer_size:
                time.sleep(0.004)
                continue

            try:
                t0 = time.time()
                packet = inputs[-1]
                if not isinstance(packet, dict):
                    packet = {
                        "color": packet,
                        "motion": None,
                        "mask": None,
                        "flow_scale_x": 1.0,
                        "flow_scale_y": 1.0,
                    }

                # Prepare batch tensor
                if frame_buffer_size == 1:
                    batch = packet.get("color")
                else:
                    sampled = []
                    for i in range(frame_buffer_size):
                        idx = max(0, len(inputs) - frame_buffer_size + i)
                        p = inputs[idx]
                        sampled.append(p.get("color") if isinstance(p, dict) else p)
                    batch = import_torch.cat(sampled)

                # Clean up old frames if using list
                if isinstance(inputs, list) and len(inputs) > frame_buffer_size * 2:
                    del inputs[:-frame_buffer_size]

                # Ensure tensor is on correct device/dtype
                if isinstance(batch, import_torch.Tensor):
                    batch = batch.to(device=stream.device, dtype=stream.dtype)

                # Motion-vector warp guidance (single-frame path)
                if (
                    motion_warp_enabled
                    and frame_buffer_size == 1
                    and prev_output_tensor is not None
                    and isinstance(packet.get("motion"), import_torch.Tensor)
                ):
                    try:
                        prev_mean = float(prev_output_tensor.mean().item())
                    except Exception:
                        prev_mean = 0.0
                    if (not np.isfinite(prev_mean)) or prev_mean < 0.01:
                        prev_output_tensor = None
                    if prev_output_tensor is not None:
                        motion_t = packet["motion"].to(device=stream.device, dtype=stream.dtype)
                        mask_t = packet.get("mask")
                        if isinstance(mask_t, import_torch.Tensor):
                            mask_t = mask_t.to(device=stream.device, dtype=stream.dtype)
                        flow = _decode_motion_flow(
                            motion_rgb=motion_t,
                            mask=mask_t,
                            motion_scale=motion_scale,
                            invert_y=motion_invert_y,
                            use_b_magnitude=motion_use_b_magnitude,
                            b_scale=motion_b_scale,
                            flow_scale_x=float(packet.get("flow_scale_x", 1.0)),
                            flow_scale_y=float(packet.get("flow_scale_y", 1.0)),
                        )
                        warped_prev = _warp_prev_with_flow(prev_output_tensor, flow).clamp(0.0, 1.0)
                        s = max(0.0, min(1.0, float(warp_strength)))
                        # Back off warp blending automatically when flow is unusually large.
                        flow_mag = None
                        try:
                            flow_mag = import_torch.sqrt((flow[:, 0:1, :, :] ** 2) + (flow[:, 1:2, :, :] ** 2))
                            flow_med = float(flow_mag.median().item())
                            if np.isfinite(flow_med):
                                if flow_med > 20.0:
                                    s = 0.0
                                elif flow_med > 4.0:
                                    s *= max(0.2, min(1.0, 4.0 / (flow_med + 1e-6)))
                        except Exception:
                            pass
                        # Only blend warped history where motion magnitude is meaningful.
                        if s > 0.0 and isinstance(flow_mag, import_torch.Tensor):
                            motion_floor_px = 0.35
                            motion_span_px = 1.25
                            blend_map = ((flow_mag - motion_floor_px) / motion_span_px).clamp(0.0, 1.0)
                            if isinstance(mask_t, import_torch.Tensor):
                                blend_map = blend_map * mask_t.clamp(0.0, 1.0)
                            try:
                                bm_mean = float(blend_map.mean().item())
                                if np.isfinite(bm_mean) and bm_mean > 0.65:
                                    s *= max(0.15, min(1.0, 0.65 / (bm_mean + 1e-6)))
                            except Exception:
                                pass
                            # Never let motion-warp dominate the full frame.
                            s = min(s, 0.35)
                            blend = (blend_map * s).clamp(0.0, 0.35)
                            # Safety: if blending would affect almost everything, back off hard.
                            try:
                                cover = float((blend > 0.01).float().mean().item())
                                if np.isfinite(cover) and cover > 0.80:
                                    blend = blend * 0.20
                            except Exception:
                                pass
                            candidate = (batch * (1.0 - blend) + warped_prev * blend).clamp(0.0, 1.0)
                            try:
                                in_mean = float(batch.mean().item())
                                cand_mean = float(candidate.mean().item())
                                if np.isfinite(in_mean) and np.isfinite(cand_mean) and in_mean > 0.02 and cand_mean > (in_mean * 0.25):
                                    batch = candidate
                            except Exception:
                                batch = candidate

                # Run through stream diffusion - wrapper handles everything
                res = stream.img2img(batch)

                # Result is already PIL images from wrapper
                images = []
                if isinstance(res, Image.Image):
                    images = [res]
                elif isinstance(res, list):
                    images = res

                # Send results to GUI
                for im in images:
                    try: 
                        out_queue.put(im, block=False)
                        frame_count += 1
                    except queue.Full: 
                        pass

                    # Optional Spout output publishing (rate-capped)
                    if spout_sender is not None and spout_output_enabled:
                        now = time.time()
                        interval = 1.0 / max(1.0, float(spout_output_fps_cap))
                        if now - spout_out_last_ts >= interval:
                            try:
                                rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
                                h_o, w_o = rgba.shape[:2]
                                spout_sender.sendImage(
                                    np.ascontiguousarray(rgba),
                                    w_o,
                                    h_o,
                                    spout_gl.GL_RGBA,
                                    False,
                                    0,
                                )
                                spout_sender.setFrameSync(spout_sender_name)
                                spout_out_last_ts = now
                            except Exception as e:
                                _status(f"Spout output send error: {e}")

                # Keep latest output tensor for next motion-warp step
                if images:
                    try:
                        out_np = np.array(images[-1].convert("RGB"), dtype=np.float32) / 255.0
                        prev_output_tensor = import_torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0)
                        prev_output_tensor = prev_output_tensor.to(device=stream.device, dtype=stream.dtype)
                    except Exception:
                        prev_output_tensor = None

                # Calculate and send FPS
                elapsed = time.time() - t0
                fps = (1.0/elapsed) if elapsed > 0 else 0.0
                while not fps_queue.empty():
                    try: fps_queue.get_nowait()
                    except Exception: break
                fps_queue.put(int(round(fps)))

                if frame_count % 30 == 0:
                    _status(f"Processed {frame_count} frames, FPS: {fps:.1f}")

            except Exception as e:
                _status(f"Processing error: {e}")
                time.sleep(0.01)

        # Cleanup
        _status("Stopping capture...")
        cap_stop.set()
        cap_thr.join(timeout=2.0)
        if spout_sender is not None:
            try:
                spout_sender.releaseSender()
            except Exception:
                pass
        _status("Worker process finished")

    except KeyboardInterrupt:
        _status("Worker interrupted by keyboard")
    except Exception as e:
        _status(f"Worker fatal error: {e}")
        import traceback
        _status(f"Traceback: {traceback.format_exc()}")

# =========================
# GPU Bootstrap Functions
# =========================
def _python_cmd_for_pip():
    import sys, shutil
    if getattr(sys, "frozen", False):
        v = f"-{sys.version_info.major}.{sys.version_info.minor}"
        if shutil.which("py"):
            return ["py", v]
        p = shutil.which("python")
        if p:
            return [p]
        raise RuntimeError("No external Python found in PATH. Install Python and try again.")
    return [sys.executable]

def _gpu_run(cmd, env=None, on_progress=None, cwd=None, shell=False, no_window=True, progress_callback=None):
    """Run a command, stream lines to UI + log; raise on non-zero exit."""
    import os, subprocess, tempfile, time, re, sys
    from pathlib import Path

    log_path = Path(tempfile.gettempdir()) / "screen_diffusion_pip_debug.log"

    def tee(line: str):
        if callable(on_progress):
            try: on_progress(line)
            except Exception: pass
        
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    env2 = dict(os.environ)
    if env: env2.update(env)
    env2["PYTHONUNBUFFERED"] = "1"
    env2["PYTHONIOENCODING"] = "utf-8"
    env2["PIP_PROGRESS_BAR"] = "on"
    env2["PIP_NO_COLOR"] = "1"
    env2["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    creationflags = 0
    startupinfo = None
    if os.name == "nt" and no_window:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        try:
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        except Exception:
            pass

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,  # Unbuffered
        env=env2,
        cwd=cwd,
        shell=shell,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )

    try:
        assert proc.stdout is not None
        buf = bytearray()
        last_yield_time = time.time()

        while True:
            # Read available bytes
            chunk = proc.stdout.read(64)  # Read in small chunks for responsiveness
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            
            buf.extend(chunk)
            
            # Try to decode and process
            try:
                text = buf.decode("utf-8", errors="replace")
                
                # Split on newlines
                lines = text.split('\n')
                
                # Keep the last incomplete line in buffer
                if not text.endswith('\n'):
                    incomplete = lines[-1]
                    buf = bytearray(incomplete.encode("utf-8"))
                    lines = lines[:-1]
                else:
                    buf = bytearray()
                
                # Yield complete lines
                for line in lines:
                    # Handle carriage returns - take the last part after \r
                    if '\r' in line:
                        line = line.split('\r')[-1]
                    
                    line = line.strip()
                    if line:
                        tee(line)
                        yield line
                
                # Also yield progress updates periodically even if line isn't complete
                now = time.time()
                if buf and (now - last_yield_time) > 0.5:  # Every 500ms
                    partial = buf.decode("utf-8", errors="replace")
                    if '\r' in partial:
                        partial = partial.split('\r')[-1]
                    partial = partial.strip()
                    if partial and len(partial) > 20:  # Only if substantial
                        tee(partial)
                        yield partial
                        last_yield_time = now
                        
            except Exception as e:
                # If decode fails, skip this chunk
                pass

        # Flush remaining buffer
        if buf:
            try:
                line = buf.decode("utf-8", errors="replace").strip()
                if line:
                    tee(line)
                    yield line
            except Exception:
                pass

    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")

def _parse_pip_progress(line: str, progress_callback):
    """Parse pip output to extract download/building progress and update progress bar."""
    import re
    
    line = line.strip()
    
    # Download progress patterns
    download_patterns = [
        r"Downloading\s+[^\n]*\s+(\d{1,3})%",
        r"[\|#]\s*(\d{1,3})%",
        r"(\d{1,3})%\s*[\|#]",
    ]
    
    # Installation/building patterns  
    install_patterns = [
        r"Building wheel for.*\((\d{1,3})%\)",
        r"Installing.*\((\d{1,3})%\)",
    ]
    
    # Check for download progress
    for pattern in download_patterns:
        match = re.search(pattern, line)
        if match:
            percent = int(match.group(1))
            progress_callback(percent / 100.0, f"Downloading: {percent}%")
            return
    
    # Check for installation progress
    for pattern in install_patterns:
        match = re.search(pattern, line)
        if match:
            percent = int(match.group(1))
            progress_callback(percent / 100.0, f"Installing: {percent}%")
            return
    
    # Check for completion indicators
    if "Successfully installed" in line:
        progress_callback(1.0, "Installation complete!")
    elif "Building wheel" in line and "complete" in line.lower():
        progress_callback(1.0, "Build complete!")
    elif "Downloaded" in line and "MB" in line:
        progress_callback(1.0, "Download complete!")

def _find_wheel_in_cache(package_name: str, version: str, cuda_version: str = "cu118"):
    """Find a wheel file in pip cache directories."""
    import os
    from pathlib import Path
    import re
    
    cache_locations = []
    
    # Windows pip cache (most common)
    try:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            win_cache = Path(local_app_data) / "pip" / "cache"
            if win_cache.exists():
                cache_locations.append(win_cache)
    except Exception:
        pass
    
    # User pip cache (Linux/Mac style on Windows)
    try:
        user_cache = Path.home() / ".cache" / "pip"
        if user_cache.exists():
            cache_locations.append(user_cache)
    except Exception:
        pass
    
    # AppData Roaming cache
    try:
        roaming = os.environ.get("APPDATA")
        if roaming:
            roaming_cache = Path(roaming) / "pip" / "cache"
            if roaming_cache.exists():
                cache_locations.append(roaming_cache)
    except Exception:
        pass
    
    # Our custom cache
    custom_cache = INTERNAL_DIR / ".pip-cache"
    if custom_cache.exists():
        cache_locations.append(custom_cache)
    
    # Temp pip cache
    try:
        temp_cache = Path(os.environ.get("TEMP", "/tmp")) / "pip-cache"
        if temp_cache.exists():
            cache_locations.append(temp_cache)
    except Exception:
        pass
    
    # Build regex patterns for various naming conventions
    version_escaped = re.escape(version)
    cuda_escaped = re.escape(cuda_version)
    
    patterns = [
        rf"^{package_name}-{version_escaped}[\+%2B]{cuda_escaped}-cp\d+-cp\d+-.*\.whl$",
        rf"^{package_name}-{version_escaped}\.post\d+[\+%2B]{cuda_escaped}-cp\d+-cp\d+-.*\.whl$",
        rf"^{package_name}-{version_escaped}[\+%2B]{cuda_escaped}\.whl$",
        rf"^{package_name}-{version_escaped}.*{cuda_escaped}.*\.whl$",
    ]
    
    best_match = None
    best_pattern_idx = len(patterns)
    
    for cache_dir in cache_locations:
        try:
            for root, dirs, files in os.walk(cache_dir):
                for filename in files:
                    if not filename.endswith('.whl'):
                        continue
                    
                    for idx, pattern in enumerate(patterns):
                        if re.match(pattern, filename, re.IGNORECASE):
                            wheel_path = Path(root) / filename
                            
                            if not (wheel_path.is_file() and os.access(wheel_path, os.R_OK)):
                                continue
                            
                            if idx < best_pattern_idx:
                                best_match = wheel_path
                                best_pattern_idx = idx
                            
                            if idx == 0:
                                return wheel_path
                            
                            break
                            
        except Exception:
            continue
    
    return best_match



def _find_streamdiffusion_in_cache(package_name: str):
    """Find any streamdiffusion wheel file in pip cache directories."""
    import os
    from pathlib import Path
    import re
    
    cache_locations = []
    
    # Windows pip cache (most common)
    try:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            win_cache = Path(local_app_data) / "pip" / "cache"
            if win_cache.exists():
                cache_locations.append(win_cache)
    except Exception:
        pass
    
    # User pip cache (Linux/Mac style on Windows)
    try:
        user_cache = Path.home() / ".cache" / "pip"
        if user_cache.exists():
            cache_locations.append(user_cache)
    except Exception:
        pass
    
    # AppData Roaming cache
    try:
        roaming = os.environ.get("APPDATA")
        if roaming:
            roaming_cache = Path(roaming) / "pip" / "cache"
            if roaming_cache.exists():
                cache_locations.append(roaming_cache)
    except Exception:
        pass
    
    # Our custom cache
    custom_cache = INTERNAL_DIR / ".pip-cache"
    if custom_cache.exists():
        cache_locations.append(custom_cache)
    
    # Temp pip cache
    try:
        temp_cache = Path(os.environ.get("TEMP", "/tmp")) / "pip-cache"
        if temp_cache.exists():
            cache_locations.append(temp_cache)
    except Exception:
        pass
    
    # Build regex pattern for streamdiffusion
    pattern = rf"^{package_name}-.*\.whl$"
    
    for cache_dir in cache_locations:
        try:
            for root, dirs, files in os.walk(cache_dir):
                for filename in files:
                    if not filename.endswith('.whl'):
                        continue
                    
                    if re.match(pattern, filename, re.IGNORECASE):
                        wheel_path = Path(root) / filename
                        
                        if wheel_path.is_file() and os.access(wheel_path, os.R_OK):
                            return wheel_path
                            
        except Exception:
            continue
    
    return None

def install_streamdiffusion(on_progress=None, skip_torch=True, progress_callback=None):
    """Install StreamDiffusion ONLY, without PyTorch dependencies. Returns success status."""
    
    py_cmd = _python_cmd_for_pip()
    streamdiffusion_package = "streamdiffusion[tensorrt]"
    
    if on_progress:
        on_progress("Installing StreamDiffusion (excluding PyTorch dependencies)...")
    
    try:
        # Install StreamDiffusion with --no-deps to avoid installing torch
        cmd = [
            *py_cmd, "-m", "pip", "install",
            "--target", str(INTERNAL_DIR),
            "--no-deps",  # This prevents installing dependencies
            streamdiffusion_package,
        ]
        
        for line in _gpu_run(cmd, on_progress=on_progress, no_window=not GPU_DIAGNOSTIC, progress_callback=progress_callback):
            pass
        
        if on_progress:
            on_progress("✓ StreamDiffusion installed (without dependencies)!")
            
    except Exception as e:
        if on_progress:
            on_progress(f"✗ StreamDiffusion installation failed: {e}")
        return False
    
    # Verify installation
    if on_progress:
        on_progress("Verifying StreamDiffusion installation...")
    
    try:
        # Test if we can import streamdiffusion
        test_cmd = [
            *py_cmd, "-c", 
            "import sys; sys.path.insert(0, r'{}'); import streamdiffusion; print('StreamDiffusion OK')".format(str(INTERNAL_DIR))
        ]
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        if on_progress:
            on_progress("✅ StreamDiffusion verified!")
        return True
    except Exception as e:
        if on_progress:
            on_progress(f"⚠ StreamDiffusion import failed: {e}")
        return False

def _gpu_relaunch_into_runtime():
    """Relaunch the application."""
    import sys, os
    os.environ[BOOT_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
class StreamGUI(ctk.CTk):
    def __init__(self):
        import multiprocessing as mp
        if mp.current_process().name != 'MainProcess':
            print(f"Preventing GUI creation in worker process: {mp.current_process().name}")
            return

        super().__init__()
        
        self.title("Screen Diffusion")
        self.geometry("1180x840")
        self.minsize(1060, 760)
        
        # Set icon early in initialization
        self._set_window_icon()
        
        self.collapse_var = ctk.BooleanVar(value=False)
        self.collapse_btn = None
        self.left_panel = None
        self.prompts_row = None
        self.status_bar = None
        self.header_frame = None
        self._cache_mon_running = False
        self.preview_dim = 512

        self.proc_ctx = None
        self.proc_worker = None
        self.out_q = self.fps_q = self.status_q = self.control_q = self.debug_q = self.close_q = None
        self.monitor_sender = self.monitor_receiver = None
        self.running = False
        self.profiling_running = False
        self._profiling_gpu_proc = None
        self._profiling_proc_proc = None
        self._profiling_gpu_log = None
        self._profiling_proc_log = None
        self._profiling_gpu_fh = None
        self._profiling_proc_fh = None

        self.model_var = ctk.StringVar(value=LOCAL_MODEL_PATH)
        self.prompt_var = ctk.StringVar(value="flip book animation, black and white rough sketch, rough drawing")
        self.neg_prompt_var = ctk.StringVar(value="low quality, bad quality, blurry, low resolution")
        self.seed_var = ctk.StringVar(value="1")
        self.width_var = ctk.IntVar(value=512)
        self.height_var = ctk.IntVar(value=512)
        self.buffer_var = ctk.StringVar(value="1")
        self.accel_var = ctk.StringVar(value="xformers")
        self.denoise_batch_var = ctk.BooleanVar(value=True)
        self.cfg_type_var = ctk.StringVar(value="none")
        self.guidance_var = ctk.StringVar(value="0.0")
        self.delta_var = ctk.StringVar(value="0.0")
        self.sim_filter_var = ctk.BooleanVar(value=False)
        self.sim_thresh_var = ctk.StringVar(value="0.99")
        self.sim_maxskip_var = ctk.StringVar(value="10.0")
        self.offline_var = ctk.BooleanVar(value=True)
        self.capture_source_var = ctk.StringVar(value="screen")
        self.spout_color_sender_var = ctk.StringVar(value="UE_Color")
        self.spout_motion_sender_var = ctk.StringVar(value="UE_Motion")
        self.spout_output_enabled_var = ctk.BooleanVar(value=False)
        self.spout_output_sender_var = ctk.StringVar(value="SD_Output")
        self.spout_output_fps_var = ctk.StringVar(value="30")
        self.motion_warp_enabled_var = ctk.BooleanVar(value=False)
        self.motion_scale_var = ctk.StringVar(value="1.0")
        self.warp_strength_var = ctk.StringVar(value="0.35")
        self.motion_invert_y_var = ctk.BooleanVar(value=True)
        self.motion_use_b_magnitude_var = ctk.BooleanVar(value=False)
        self.motion_b_scale_var = ctk.StringVar(value="1.0")
        self.advanced_expanded_var = ctk.BooleanVar(value=False)
        self.spout_ready, self.spout_status = _check_spout_dependencies()
        
        self._debounce_prompt = self._debounce_neg = self._debounce_region = self._debounce_advanced = None

        self.t_index_list: List[int] = [30]
        self._lockables: List[ctk.CTkBaseClass] = []
        self._step_sliders: List[ctk.CTkSlider] = []

        self.capwin: Optional[FloatingCaptureWindow] = None
        self._ctk_img: Optional[ctk.CTkImage] = None
        self.bind("<h>", lambda e: self._toggle_capture_window())
        self.bind("<H>", lambda e: self._toggle_capture_window())
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            img = Image.open(logo_path)
            self.logo_img = ctk.CTkImage(
                light_image=img, dark_image=img, size=(28, 28)
            )
        except Exception:
            self.logo_img = None
            
        self._build_ui()
        self._setup_input_validation()
        self.after(500, self._check_and_hide_gpu_frame)
        self.bind("<Configure>", self._on_window_configure)
        self._poll_queues()
        self.protocol("WM_DELETE_WINDOW", self.do_quit)

    def _toggle_capture_window(self):
        """Toggle visibility of capture window while keeping capture active"""
        if not self.running or self.capwin is None:
            return
    
        try:
            if self.capwin.win.state() == "normal":
                # Hide the window
                self.capwin.win.withdraw()
                self.hide_capture_btn.configure(text="👁 Show (H)")
                self.status_var.set("Capture window hidden (press H to show)")
            else:
                # Show the window
                self.capwin.win.deiconify()
                self.hide_capture_btn.configure(text="👁 Hide (H)")
                self.status_var.set("Capture window visible")
                # Update capture region in case window was moved
                self._send_region_update()
        except Exception as e:
            print(f"Error toggling capture window: {e}")

    def _set_window_icon(self):
        """Set window icon"""
        try:
            icon_paths = [
                resource_path("icon2.ico"),
                os.path.join(os.path.dirname(__file__), "icon2.ico"),
            ]
            
            for icon_path in icon_paths:
                if os.path.exists(icon_path):
                    self.iconbitmap(icon_path)
                    print(f"Window icon set from: {icon_path}")
                    return True
        except Exception as e:
            print(f"Failed to set window icon: {e}")
        return False

    def _current_advanced_cfg(self) -> Dict[str, Any]:
        def _f(var, default):
            try:
                return float(var.get().strip())
            except Exception:
                return float(default)

        return {
            "capture_source": str(self.capture_source_var.get()).lower(),
            "spout_color_sender": self.spout_color_sender_var.get().strip() or "UE_Color",
            "spout_motion_sender": self.spout_motion_sender_var.get().strip() or "UE_Motion",
            "spout_output_enabled": bool(self.spout_output_enabled_var.get()),
            "spout_output_sender": self.spout_output_sender_var.get().strip() or "SD_Output",
            "spout_output_fps_cap": max(1.0, _f(self.spout_output_fps_var, 30.0)),
            "motion_warp_enabled": bool(self.motion_warp_enabled_var.get()),
            "motion_scale": _f(self.motion_scale_var, 1.0),
            "warp_strength": max(0.0, min(1.0, _f(self.warp_strength_var, 0.35))),
            "motion_invert_y": bool(self.motion_invert_y_var.get()),
            "motion_use_b_magnitude": bool(self.motion_use_b_magnitude_var.get()),
            "motion_b_scale": _f(self.motion_b_scale_var, 1.0),
        }

    def _toggle_advanced_panel(self):
        self.advanced_expanded_var.set(not bool(self.advanced_expanded_var.get()))
        self._refresh_advanced_visibility()

    def _refresh_advanced_visibility(self):
        if not hasattr(self, "advanced_body") or not hasattr(self, "advanced_toggle_btn"):
            return
        expanded = bool(self.advanced_expanded_var.get())
        self.advanced_toggle_btn.configure(text=("Advanced ▾" if expanded else "Advanced ▸"))
        if expanded:
            self.advanced_body.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        else:
            self.advanced_body.grid_forget()
        self._refresh_spout_ui()

    def _refresh_spout_ui(self):
        has_spout_controls = hasattr(self, "_w_capture_source")
        if not has_spout_controls:
            return
        use_spout = (self.capture_source_var.get().lower() == "spout")

        spout_widgets = [
            getattr(self, "_w_spout_color_entry", None),
            getattr(self, "_w_spout_motion_entry", None),
            getattr(self, "_w_spout_output_switch", None),
            getattr(self, "_w_spout_output_sender_entry", None),
            getattr(self, "_w_spout_out_fps_entry", None),
            getattr(self, "_w_motion_warp_switch", None),
            getattr(self, "_w_motion_scale_entry", None),
            getattr(self, "_w_warp_strength_entry", None),
            getattr(self, "_w_motion_invert_y_switch", None),
            getattr(self, "_w_motion_use_b_switch", None),
            getattr(self, "_w_motion_b_scale_entry", None),
        ]

        if (not self.spout_ready) and use_spout:
            self.status_var.set(self.spout_status)

        # Keep advanced Spout controls editable; we validate readiness on Start.
        # This avoids "locking" the section when variable traces refresh widget states.
        for w in spout_widgets:
            if w is None:
                continue
            try:
                w.configure(state="normal")
            except Exception:
                pass
        # B-scale only editable when B modulation is enabled.
        if getattr(self, "_w_motion_b_scale_entry", None) is not None:
            try:
                b_ok = bool(self.motion_use_b_magnitude_var.get())
                self._w_motion_b_scale_entry.configure(state=("normal" if b_ok else "disabled"))
            except Exception:
                pass

    def _on_advanced_control_changed(self, *_args):
        self._refresh_spout_ui()
        if self.running and getattr(self, "control_q", None):
            if self._debounce_advanced is not None:
                try:
                    self.after_cancel(self._debounce_advanced)
                except Exception:
                    pass
            self._debounce_advanced = self.after(120, self._push_advanced_runtime)

    def _push_advanced_runtime(self):
        if not getattr(self, "control_q", None):
            return
        try:
            cfg = self._current_advanced_cfg()
            if cfg.get("capture_source") != self.capture_source_var.get().lower():
                cfg["capture_source"] = self.capture_source_var.get().lower()
            self.control_q.put_nowait({"type": "set_advanced", "cfg": cfg})
        except Exception:
            pass

    def _setup_input_validation(self):
        """Set up validation for numeric entry fields"""
        def validate_numeric_input(P):
            if P == "" or P == "-":
                return True
            try:
                float(P)
                return True
            except ValueError:
                return False
    
        vcmd = (self.register(validate_numeric_input), '%P')
    
        numeric_entries = [
            self._w_seed_entry,
            self._w_guidance_entry,
            self._w_delta_entry,
            self._w_sim_thresh,
            self._w_sim_maxskip,
            getattr(self, "_w_width_entry", None),
            getattr(self, "_w_height_entry", None),
            getattr(self, "_w_spout_out_fps_entry", None),
            getattr(self, "_w_motion_scale_entry", None),
            getattr(self, "_w_warp_strength_entry", None),
            getattr(self, "_w_motion_b_scale_entry", None),
        ]
    
        for entry in numeric_entries:
            if entry:
                entry.configure(validate="key", validatecommand=vcmd)

    def _download_sd_turbo(self):
        """Download sd-turbo fp16 model using Hugging Face CLI"""
        try:
            # Ask user where to save the model
            download_dir = filedialog.askdirectory(
                title="Select directory to download sd-turbo model"
            )
            if not download_dir:
                return  # User cancelled
        
            model_path = os.path.join(download_dir, "sd-turbo-fp16")
    
            # Show download dialog
            self._show_download_dialog(model_path)
    
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to start download: {e}")

    def _show_download_dialog(self, model_path):
        """Show a dialog for downloading the model"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Downloading sd-turbo fp16")
        dialog.geometry("500x240")
        dialog.transient(self)
        dialog.grab_set()
    
        # Add download process tracking
        self._download_process = None
        self._download_cancelled = False
        self._download_thread = None

        # Center the dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog, 
            text="Downloading sd-turbo fp16 model...",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(20, 10))

        ctk.CTkLabel(
            dialog, 
            text=f"Destination: {model_path}",
            wraplength=400
        ).pack(pady=(0, 20))

        progress_bar = ctk.CTkProgressBar(dialog, width=400)
        progress_bar.pack(pady=(0, 10))
        progress_bar.set(0)

        status_label = ctk.CTkLabel(dialog, text="Starting download...")
        status_label.pack(pady=(0, 10))

        cancel_btn = ctk.CTkButton(
            dialog, 
            text="Cancel", 
            command=lambda: self._cancel_download(dialog, model_path)
        )
        cancel_btn.pack(pady=(0, 10))

        # Store references
        self._current_download_dialog = dialog
        self._current_download_path = model_path

        # Start download in a separate thread
        self._download_thread = threading.Thread(
            target=self._run_hf_download,
            args=(model_path, progress_bar, status_label, dialog),
            daemon=True
        )
        self._download_thread.start()

    def _cleanup_partial_download(self, model_path):
        """Remove partially downloaded files"""
        try:
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
                print(f"Cleaned up partial download: {model_path}")
        except Exception as e:
            print(f"Error cleaning up partial download: {e}")

    def _cancel_download(self, dialog, model_path):
        """Cancel the download and clean up"""
        print("Cancelling download...")
        self._download_cancelled = True
    
        # Terminate any subprocess
        if hasattr(self, '_download_process') and self._download_process:
            try:
                print("Terminating download process...")
                self._download_process.terminate()
                import time
                time.sleep(1)
                if self._download_process.poll() is None:
                    self._download_process.kill()
                    print("Download process killed")
            except Exception as e:
                print(f"Error terminating process: {e}")
    
        # Kill any remaining Python download threads by interrupting the main thread
        if hasattr(self, '_download_thread') and self._download_thread:
            # We can't directly kill threads, but we can make them exit quickly
            pass
    
        # Clean up partial download
        self._cleanup_partial_download(model_path)
    
        # Close the dialog
        dialog.destroy()
    
        # Clear references
        if hasattr(self, '_current_download_dialog'):
            del self._current_download_dialog
        if hasattr(self, '_current_download_path'):
            del self._current_download_path
    
        print("Download cancelled successfully")

    def _run_hf_download(self, model_path, progress_bar, status_label, dialog):
        """Run the download using only CLI method for better cancellation control"""
        try:
            model_repo = "stabilityai/sd-turbo"
    
            def update_status(text, progress=0):
                if self._download_cancelled:
                    return False
                try:
                    self.after(0, lambda: status_label.configure(text=text))
                    self.after(0, lambda: progress_bar.set(progress))
                    return True
                except Exception:
                    return False
    
            def check_cancelled():
                return self._download_cancelled
    
            # Use CLI method only - more reliable for cancellation
            if not update_status("Checking Hugging Face CLI...", 0.1):
                return
    
            if check_cancelled():
                return
    
            # Check if huggingface-cli is available
            try:
                subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, check=True, timeout=10)
                use_cli = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                use_cli = False
    
            if not use_cli:
                if not update_status("Installing huggingface_hub...", 0.2):
                    return
                if check_cancelled():
                    return
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "huggingface_hub"
                    ], check=True, capture_output=True, text=True, timeout=120)
                    use_cli = True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    if not update_status(f"Installation failed: {e}", 0):
                        return
                    return
    
            if not update_status("Starting download...", 0.3):
                return
        
            if check_cancelled():
                return
    
            # Create model directory
            os.makedirs(model_path, exist_ok=True)
    
            # Use subprocess for download - this gives us actual process control
            print("Starting CLI download with process control...")
        
            # Build the command
            cmd = [
                "huggingface-cli", "download", model_repo,
                "--local-dir", model_path,
                "--local-dir-use-symlinks", "False",
                "--resume-download"
            ]
        
            # Start the process
            self._download_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stdout and stderr
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        
            # Monitor the process output and check for cancellation
            try:
                last_progress = 0.3
                while True:
                    # Check for cancellation
                    if check_cancelled():
                        print("Cancellation detected, terminating process...")
                        self._download_process.terminate()
                        # Wait a bit for termination
                        import time
                        time.sleep(2)
                        if self._download_process.poll() is None:
                            print("Process still running, killing...")
                            self._download_process.kill()
                        break
                
                    # Check if process finished
                    if self._download_process.poll() is not None:
                        break
                
                    # Read output line by line to track progress
                    line = self._download_process.stdout.readline()
                    if line:
                        print(f"Download: {line.strip()}")
                        # Very basic progress parsing - you can enhance this
                        if "%" in line:
                            # Extract percentage if possible
                            import re
                            match = re.search(r'(\d+)%', line)
                            if match:
                                percent = int(match.group(1))
                                progress = 0.3 + (percent / 100) * 0.7
                                last_progress = progress
                                if not update_status(f"Downloading... {percent}%", progress):
                                    break
                        else:
                            # Update status with current line
                            if not update_status(f"Downloading...", last_progress):
                                break
                
                    # Small delay to prevent busy waiting
                    import time
                    time.sleep(0.1)
            
                # Process finished or was terminated
                returncode = self._download_process.wait(timeout=5)
            
                if check_cancelled():
                    print("Download was cancelled successfully")
                    self._cleanup_partial_download(model_path)
                    return
                
                if returncode == 0:
                    if not update_status("Download completed successfully!", 1.0):
                        return
                    self.after(0, lambda: self.model_var.set(model_path))
                    self.after(2000, dialog.destroy)
                    self.after(0, lambda: messagebox.showinfo(
                        "Download Complete", 
                        f"sd-turbo model downloaded successfully to:\n{model_path}"
                    ))
                else:
                    if not check_cancelled():
                        error_msg = f"Download failed with exit code {returncode}"
                        print(error_msg)
                        if not update_status(error_msg, 0):
                            return
                    
            except Exception as e:
                if not check_cancelled():
                    error_msg = f"Download error: {str(e)}"
                    print(error_msg)
                    if not update_status(error_msg, 0):
                        return
                    
            finally:
                # Ensure process is cleaned up
                if self._download_process and self._download_process.poll() is None:
                    try:
                        self._download_process.terminate()
                        self._download_process.wait(timeout=2)
                    except:
                        try:
                            self._download_process.kill()
                        except:
                            pass
                self._download_process = None
                
        except Exception as e:
            if not self._download_cancelled:
                error_msg = f"Download setup failed: {str(e)}"
                print(error_msg)
                try:
                    self.after(0, lambda: status_label.configure(text=error_msg))
                    self.after(0, lambda: progress_bar.set(0))
                except Exception:
                    pass
    def _toggle_collapse(self):
        """Toggle between collapsed and expanded view"""
        if self.collapse_var.get():
            # Expand - show everything
            self.collapse_var.set(False)
            self.collapse_btn.configure(text="📱 Compact")
            
            # Show all sections
            if self.header_frame:
                self.header_frame.grid(row=0, column=0, sticky="nw", padx=(12, 0), pady=(10, 0))
            if self.left_panel:
                self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(6, 6))
            if self.prompts_row:
                self.prompts_row.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
            if self.status_bar:
                self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
            
            # Restore grid weights
            self.grid_rowconfigure(0, weight=0)  # Header
            self.grid_rowconfigure(1, weight=1)  # Main content
            self.grid_rowconfigure(2, weight=0)  # Prompts
            self.grid_rowconfigure(3, weight=0)  # Status
            self.grid_columnconfigure(0, weight=0)  # Left panel
            self.grid_columnconfigure(1, weight=1)  # Right panel
            
            # Restore original geometry
            self.geometry("1180x840")
            self.minsize(1060, 760)
            
        else:
            # Collapse - hide left panel and bottom sections
            self.collapse_var.set(True)
            self.collapse_btn.configure(text="📖 Expand")
            
            # Hide sections
            if self.header_frame:
                self.header_frame.grid_remove()
            if self.left_panel:
                self.left_panel.grid_remove()
            if self.prompts_row:
                self.prompts_row.grid_remove()
            if self.status_bar:
                self.status_bar.grid_remove()
            
            # Adjust grid weights for collapsed view
            self.grid_rowconfigure(0, weight=0)  # No header
            self.grid_rowconfigure(1, weight=1)  # Only main content
            self.grid_rowconfigure(2, weight=0)  # No prompts
            self.grid_rowconfigure(3, weight=0)  # No status
            self.grid_columnconfigure(0, weight=0)  # No left panel
            self.grid_columnconfigure(1, weight=1)  # Only right panel
            
            # Resize to fit only the preview panel
            preview_size = self.preview_dim + 150  # Add padding for buttons
            self.geometry(f"{preview_size + 50}x{preview_size + 100}")
            self.minsize(preview_size, preview_size)

    def _update_collapse_state(self):
        """Update visibility of UI elements based on collapse state"""
        collapsed = self.collapse_var.get()
        
        # Direct references to the main sections we want to hide/show
        # Left panel (parameters)
        left_panel = None
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkScrollableFrame) and child._width == 440:  # Left panel
                left_panel = child
                break
        
        # Prompts row (bottom section with prompt and negative prompt)
        prompts_row = None
        for child in self.winfo_children():
            if hasattr(child, 'winfo_children') and child.winfo_children():
                # Look for frames that contain textboxes for prompts
                for subchild in child.winfo_children():
                    if hasattr(subchild, 'winfo_children') and any('textbox' in str(widget).lower() for widget in subchild.winfo_children()):
                        prompts_row = child
                        break
                if prompts_row:
                    break
        
        # Status bar (bottom frame)
        status_bar = None
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame) and child._height == 36:  # Status bar
                status_bar = child
                break
        
        # Header (top left)
        header = None
        for child in self.winfo_children():
            if child == self.title_label.master:  # The header frame we created
                header = child
                break
        
        # Show/hide elements
        elements_to_toggle = [left_panel, prompts_row, status_bar, header]
        
        for element in elements_to_toggle:
            if element:
                if collapsed:
                    element.grid_remove()
                else:
                    element.grid()
        
        # Also hide the GPU frame if it exists
        if hasattr(self, 'gpu_frame') and self.gpu_frame:
            if collapsed:
                self.gpu_frame.grid_remove()
            else:
                self.gpu_frame.grid()

    def _check_and_hide_gpu_frame(self):
        """Check if torch is available and hide GPU frame if it is"""
        torch_available = False
    
        try:
            import sys
            import os
        
            # Check if torch directory exists
            torch_dir = INTERNAL_DIR / "torch"
            if not torch_dir.exists():
                print(f"[GPU Check] Torch directory not found at: {torch_dir}")
                return
        
            print(f"[GPU Check] Found torch directory: {torch_dir}")
        
            # Set up paths
            internal_str = str(INTERNAL_DIR)
            if internal_str not in sys.path:
                sys.path.insert(0, internal_str)
        
            # Add DLL paths
            torch_lib_dir = torch_dir / "lib"
            if torch_lib_dir.exists():
                try:
                    os.add_dll_directory(str(torch_lib_dir))
                    print(f"[GPU Check] Added torch lib to DLL search")
                except:
                    pass
            
                current_path = os.environ.get("PATH", "")
                torch_lib_str = str(torch_lib_dir)
                if torch_lib_str not in current_path:
                    os.environ["PATH"] = torch_lib_str + os.pathsep + current_path
        
            _prime_dll_search()
        
            # Now try to import torch (should be clean, first import)
            print("[GPU Check] Attempting fresh torch import...")
            import torch
        
            torch_version = torch.__version__
            print(f"[GPU Check] Torch file: {torch.__file__}")
            print(f"[GPU Check] ✅ PyTorch {torch_version} detected")
        
            try:
                cuda_available = torch.cuda.is_available()
                print(f"[GPU Check] CUDA available: {cuda_available}")
            except Exception as e:
                print(f"[GPU Check] CUDA check failed: {e}")
                cuda_available = False
        
            torch_available = True
            
        except ImportError as e:
            print(f"[GPU Check] PyTorch import failed: {e}")
        except Exception as e:
            print(f"[GPU Check] Error: {e}")
            import traceback
            print(f"[GPU Check] Traceback: {traceback.format_exc()}")
    
        if torch_available:
            try:
                if hasattr(self, 'gpu_frame') and self.gpu_frame.winfo_ismapped():
                    self.gpu_frame.grid_remove()
                    print("[GPU Check] ✅ GPU frame hidden")
            
                if hasattr(self, 'status_var'):
                    import torch
                    self.status_var.set(f"GPU runtime ready ✅ (PyTorch {torch.__version__})")
                    
            except Exception as e:
                print(f"[GPU Check] Error hiding frame: {e}")
        else:
            print("[GPU Check] ⚠ PyTorch not available")

    def _build_ui(self):
        # Root grid
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)

        # Header (left column only)
        left_header = ctk.CTkFrame(self, fg_color="transparent")
        left_header.grid(row=0, column=0, sticky="nw", padx=(12, 0), pady=(10, 0))

        left_header.grid_columnconfigure(0, weight=0)
        left_header.grid_columnconfigure(1, weight=0)
        left_header.grid_columnconfigure(2, weight=1)
        left_header.grid_columnconfigure(3, weight=0)

        if getattr(self, "logo_img", None):
            ctk.CTkLabel(left_header, image=self.logo_img, text="")\
                .grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.title_label = ctk.CTkLabel(
            left_header,
            text="Screen Diffusion",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.title_label.grid(row=0, column=1, sticky="w")

        self.version_label = ctk.CTkLabel(
            left_header,
            text="Version 0.1",
            font=ctk.CTkFont(size=14, slant="italic"),
            text_color="gray70",
        )
        self.version_label.grid(row=0, column=3, sticky="e")
        self.header_frame = left_header
        # LEFT (Parameters)
        left = ctk.CTkScrollableFrame(self, width=440, corner_radius=12)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(6, 6))
        left.grid_columnconfigure(0, weight=1)
        row = 0

        if SHOW.get("model_path", True):
            ctk.CTkLabel(left, text="Model (diffusers folder):", anchor="w")\
                .grid(row=row, column=0, sticky="ew", pady=(4, 0))
            mp = ctk.CTkFrame(left); mp.grid(row=row+1, column=0, sticky="ew")
            mp.grid_columnconfigure(0, weight=1)
            self._w_model_entry = ctk.CTkEntry(mp, textvariable=self.model_var)
            self._w_model_entry.grid(row=0, column=0, sticky="ew", padx=(0,6), pady=6)
            self._w_model_browse = ctk.CTkButton(
                mp, text="Browse", command=self._browse_model, width=90
            )
            self._w_model_browse.grid(row=0, column=1, pady=6)

            self._register_lockables(self._w_model_entry, self._w_model_browse)
            row += 2

        
        g2 = ctk.CTkFrame(left); g2.grid(row=row, column=0, sticky="ew", pady=(4,6))
        for i in range(6): g2.grid_columnconfigure(i, weight=1)
        col = 0
        if SHOW.get("seed", True):
            ctk.CTkLabel(g2, text="Seed").grid(row=0, column=col, sticky="w")
            self._w_seed_entry = ctk.CTkEntry(g2, textvariable=self.seed_var, width=80)
            self._w_seed_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_seed_entry)
        if SHOW.get("frame_buffer_size", True):
            ctk.CTkLabel(g2, text="Buffer").grid(row=0, column=col, sticky="w")
            self._w_buffer_entry = ctk.CTkEntry(g2, textvariable=self.buffer_var, width=70)
            self._w_buffer_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_buffer_entry)
        if SHOW.get("acceleration", True):
            ctk.CTkLabel(g2, text="Acceleration").grid(row=0, column=col, sticky="w")
            self._w_accel_combo = ctk.CTkComboBox(
                g2, values=["none","xformers","tensorrt"], variable=self.accel_var, width=120
            )
            self._w_accel_combo.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_accel_combo)
            ctk.CTkLabel(g2, text="Profiling").grid(row=0, column=col, sticky="w")
            self.profile_btn = ctk.CTkButton(
                g2,
                text="Start Profiling",
                command=self._toggle_profiling,
                width=140,
                fg_color=CUSTOM_COLORS["success"],
                hover_color="#059669",
            )
            self.profile_btn.grid(row=1, column=col, sticky="ew"); col += 1
        if SHOW.get("use_denoising_batch", True):
            self._w_denoise_switch = ctk.CTkSwitch(g2, text="Denoising batch", variable=self.denoise_batch_var)
            self._w_denoise_switch.grid(row=1, column=col, sticky="w"); col += 1
            self._register_lockables(self._w_denoise_switch)
        row += 1

        g3 = ctk.CTkFrame(left); g3.grid(row=row, column=0, sticky="ew", pady=(4,6))
        for i in range(6): g3.grid_columnconfigure(i, weight=1)
        col = 0
        if SHOW.get("cfg_type", True):
            ctk.CTkLabel(g3, text="CFG type").grid(row=0, column=col, sticky="w")
            self._w_cfg_combo = ctk.CTkComboBox(
                g3, values=["none","full","self","initialize"], variable=self.cfg_type_var, width=120
            )
            self._w_cfg_combo.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_cfg_combo)
        if SHOW.get("guidance_scale", True):
            ctk.CTkLabel(g3, text="Guidance").grid(row=0, column=col, sticky="w")
            self._w_guidance_entry = ctk.CTkEntry(g3, textvariable=self.guidance_var, width=70)
            self._w_guidance_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_guidance_entry)
        if SHOW.get("delta", True):
            ctk.CTkLabel(g3, text="Delta").grid(row=0, column=col, sticky="w")
            self._w_delta_entry = ctk.CTkEntry(g3, textvariable=self.delta_var, width=70)
            self._w_delta_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_delta_entry)
        if SHOW.get("offline", True):
            self._w_offline_switch = ctk.CTkSwitch(g3, text="Offline", variable=self.offline_var)
            self._w_offline_switch.grid(row=1, column=col, sticky="w"); col += 1
            self._register_lockables(self._w_offline_switch)
        row += 1
        self.left_panel = left
        # Steps
        steps = ctk.CTkFrame(left); steps.grid(row=row, column=0, sticky="ew", pady=(4,6))
        steps.grid_columnconfigure(0, weight=1)

        # Simple header without add/remove buttons
        ctk.CTkLabel(steps, text="Denoising Step", anchor="w").grid(row=0, column=0, sticky="w", pady=(0,4))

        self._steps_holder = ctk.CTkFrame(steps)
        self._steps_holder.grid(row=1, column=0, sticky="ew", pady=(6,0))
        self._build_steps_ui()
        row += 1

        if SHOW.get("similar_image_filter", True):
            sim = ctk.CTkFrame(left); sim.grid(row=row, column=0, sticky="ew", pady=(4,6))
            sim.grid_columnconfigure(0, weight=0); sim.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(sim, text="Similar Image Filter").grid(row=0, column=0, sticky="w", pady=(0,4))
            self._w_sim_switch  = ctk.CTkSwitch(sim, text="Enable", variable=self.sim_filter_var); self._w_sim_switch.grid(row=1, column=0, sticky="w")
            self._w_sim_thresh  = ctk.CTkEntry(sim, textvariable=self.sim_thresh_var, width=80); self._w_sim_thresh.grid(row=2, column=1, sticky="w", padx=(6,0))
            self._w_sim_maxskip = ctk.CTkEntry(sim, textvariable=self.sim_maxskip_var, width=80); self._w_sim_maxskip.grid(row=3, column=1, sticky="w", padx=(6,0))
            self._register_lockables(self._w_sim_switch, self._w_sim_thresh, self._w_sim_maxskip)
            row += 1

        # Advanced (collapsible): Spout + motion warp controls
        adv_wrap = ctk.CTkFrame(left)
        adv_wrap.grid(row=row, column=0, sticky="ew", pady=(4, 6))
        adv_wrap.grid_columnconfigure(0, weight=1)

        self.advanced_toggle_btn = ctk.CTkButton(
            adv_wrap,
            text="Advanced ▸",
            command=self._toggle_advanced_panel,
            height=30,
            fg_color="#4B5563",
            hover_color="#374151",
        )
        self.advanced_toggle_btn.grid(row=0, column=0, sticky="ew")

        self.advanced_body = ctk.CTkFrame(adv_wrap)
        self.advanced_body.grid_columnconfigure(1, weight=1)

        r_adv = 0
        ctk.CTkLabel(self.advanced_body, text="Capture Source").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
        self._w_capture_source = ctk.CTkComboBox(
            self.advanced_body,
            values=["screen", "spout"],
            variable=self.capture_source_var,
            width=140,
        )
        self._w_capture_source.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=(6, 2))
        self._register_lockables(self._w_capture_source)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Model Width").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_width_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.width_var, width=100)
        self._w_width_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_width_entry)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Model Height").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_height_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.height_var, width=100)
        self._w_height_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_height_entry)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Spout Color Sender").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_spout_color_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.spout_color_sender_var)
        self._w_spout_color_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_spout_color_entry)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Spout Motion Sender").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_spout_motion_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.spout_motion_sender_var)
        self._w_spout_motion_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_spout_motion_entry)
        r_adv += 1

        self._w_spout_output_switch = ctk.CTkSwitch(self.advanced_body, text="Enable Spout Output", variable=self.spout_output_enabled_var)
        self._w_spout_output_switch.grid(row=r_adv, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=2)
        self._register_lockables(self._w_spout_output_switch)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Spout Output Sender").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_spout_output_sender_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.spout_output_sender_var)
        self._w_spout_output_sender_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_spout_output_sender_entry)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Spout Output FPS").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_spout_out_fps_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.spout_output_fps_var, width=100)
        self._w_spout_out_fps_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_spout_out_fps_entry)
        r_adv += 1

        self._w_motion_warp_switch = ctk.CTkSwitch(self.advanced_body, text="Enable Motion Warp", variable=self.motion_warp_enabled_var)
        self._w_motion_warp_switch.grid(row=r_adv, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=2)
        self._register_lockables(self._w_motion_warp_switch)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Motion Scale").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_motion_scale_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.motion_scale_var, width=100)
        self._w_motion_scale_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_motion_scale_entry)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="Warp Strength").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=2)
        self._w_warp_strength_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.warp_strength_var, width=100)
        self._w_warp_strength_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=2)
        self._register_lockables(self._w_warp_strength_entry)
        r_adv += 1

        self._w_motion_invert_y_switch = ctk.CTkSwitch(self.advanced_body, text="Invert Motion Y", variable=self.motion_invert_y_var)
        self._w_motion_invert_y_switch.grid(row=r_adv, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=2)
        self._register_lockables(self._w_motion_invert_y_switch)
        r_adv += 1

        self._w_motion_use_b_switch = ctk.CTkSwitch(self.advanced_body, text="Use B as Magnitude", variable=self.motion_use_b_magnitude_var)
        self._w_motion_use_b_switch.grid(row=r_adv, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=2)
        self._register_lockables(self._w_motion_use_b_switch)
        r_adv += 1

        ctk.CTkLabel(self.advanced_body, text="B Magnitude Scale").grid(row=r_adv, column=0, sticky="w", padx=(6, 6), pady=(2, 6))
        self._w_motion_b_scale_entry = ctk.CTkEntry(self.advanced_body, textvariable=self.motion_b_scale_var, width=100)
        self._w_motion_b_scale_entry.grid(row=r_adv, column=1, sticky="ew", padx=(0, 6), pady=(2, 6))
        self._register_lockables(self._w_motion_b_scale_entry)
        r_adv += 1

        self._w_spout_status_lbl = ctk.CTkLabel(
            self.advanced_body,
            text=("Spout ready" if self.spout_ready else self.spout_status),
            text_color=("gray70" if self.spout_ready else "#FCA5A5"),
            anchor="w",
            justify="left",
            wraplength=380,
        )
        self._w_spout_status_lbl.grid(row=r_adv, column=0, columnspan=2, sticky="ew", padx=(6, 6), pady=(0, 6))
        row += 1

        # GPU Add-On
        self.gpu_frame = ctk.CTkFrame(left)
        self.gpu_frame.grid(row=row, column=0, sticky="ew", pady=(6,6))
        self.gpu_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.gpu_frame, text="GPU Add-On").grid(row=0, column=0, sticky="w")
        self.gpu_btn = ctk.CTkButton(self.gpu_frame, text="Enable GPU (download ~ 2.8GB)", command=self._on_enable_gpu)
        self.gpu_btn.grid(row=1, column=0, sticky="ew", pady=(4,2))
        self.gpu_log = ctk.CTkTextbox(self.gpu_frame, height=80)
        self.gpu_log.grid(row=2, column=0, sticky="ew")
        self.gpu_log.configure(state="disabled")
        self.gpu_log.configure(state="normal"); self.gpu_log.insert("end", "Log ready…\n"); self.gpu_log.configure(state="disabled")
        row += 1

        # RIGHT (Preview)
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(6, 6))
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(2, weight=0)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Preview").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))
        self.preview_container = ctk.CTkFrame(right, corner_radius=8, width=self.preview_dim, height=self.preview_dim)
        self.preview_container.grid(row=1, column=0, sticky="n", padx=12, pady=(8, 12))
        self.preview_container.grid_propagate(False)
        self._photo = CTkImage(light_image=Image.new("RGB", (self.preview_dim, self.preview_dim)),size=(self.preview_dim, self.preview_dim))
        
        self.preview_panel = ctk.CTkLabel(self.preview_container, image=self._photo, text="")
        self.preview_panel.place(x=0, y=0)

                # Action buttons
        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=0)
        actions.grid_columnconfigure(2, weight=1)

        button_container = ctk.CTkFrame(actions, fg_color="transparent")
        button_container.grid(row=0, column=1, sticky="ew")

        self.start_btn = ctk.CTkButton(
            button_container, 
            text="🚀 Start Generation",
            command=self._on_start, 
            width=140,
            height=36,
            fg_color=CUSTOM_COLORS["success"],
            hover_color="#059669",
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        self.stop_btn = ctk.CTkButton(
            button_container, 
            text="⏹ Stop", 
            command=self._on_stop, 
            width=100,
            height=36,
            fg_color=CUSTOM_COLORS["error"],
            hover_color="#DC2626",
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))
        self.stop_btn.configure(state="disabled")

        self.hide_capture_btn = ctk.CTkButton(
            button_container, 
            text="👁 Hide (H)", 
            command=self._toggle_capture_window, 
            width=100,
            height=36,
            fg_color="#6B7280",
            hover_color="#4B5563",
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        self.hide_capture_btn.grid(row=0, column=2, padx=(0, 8))
        self.hide_capture_btn.configure(state="disabled")

        # Collapse/Expand button
        self.collapse_btn = ctk.CTkButton(
            button_container, 
            text="📱 Compact", 
            command=self._toggle_collapse, 
            width=100,
            height=36,
            fg_color="#6B7280",
            hover_color="#4B5563",
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        self.collapse_btn.grid(row=0, column=3, padx=(0, 8)) 

        ctk.CTkButton(
            button_container, 
            text="❌ Quit", 
            command=self.do_quit, 
            width=100,
            height=36,
            fg_color=CUSTOM_COLORS["surface"],
            hover_color=CUSTOM_COLORS["error"],
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=4)

        # Prompts row (spans both columns)
        prompts_row = ctk.CTkFrame(self)
        prompts_row.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
        prompts_row.grid_columnconfigure(0, weight=1); prompts_row.grid_columnconfigure(1, weight=1); prompts_row.grid_rowconfigure(0, weight=1)

        prompt_frame = ctk.CTkFrame(prompts_row, corner_radius=10); prompt_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        prompt_frame.grid_rowconfigure(1, weight=1); prompt_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(prompt_frame, text="Prompt", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.prompt_txt = ctk.CTkTextbox(prompt_frame, height=120); self.prompt_txt.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.prompt_txt.delete("1.0", "end"); self.prompt_txt.insert("1.0", self.prompt_var.get()); self.prompt_txt.bind("<KeyRelease>", self._on_prompt_changed)

        neg_frame = ctk.CTkFrame(prompts_row, corner_radius=10); neg_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        neg_frame.grid_rowconfigure(1, weight=1); neg_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(neg_frame, text="Negative Prompt", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.neg_prompt_txt = ctk.CTkTextbox(neg_frame, height=120); self.neg_prompt_txt.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.neg_prompt_txt.delete("1.0", "end"); self.neg_prompt_txt.insert("1.0", self.neg_prompt_var.get()); self.neg_prompt_txt.bind("<KeyRelease>", self._on_neg_prompt_changed)
        self.prompts_row = prompts_row
        # Status bar
        status = ctk.CTkFrame(self, height=36, corner_radius=12)
        status.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        status.grid_columnconfigure(0, weight=0); status.grid_columnconfigure(1, weight=1)
        self.fps_var = ctk.StringVar(value="FPS: --"); self.status_var = ctk.StringVar(value="idle")
        ctk.CTkLabel(status, textvariable=self.fps_var).grid(row=0, column=0, sticky="w", padx=12)
        ctk.CTkLabel(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=12)
        self.status_bar = status
        for var in (
            self.capture_source_var,
            self.spout_color_sender_var,
            self.spout_motion_sender_var,
            self.spout_output_enabled_var,
            self.spout_output_sender_var,
            self.spout_output_fps_var,
            self.motion_warp_enabled_var,
            self.motion_scale_var,
            self.warp_strength_var,
            self.motion_invert_y_var,
            self.motion_use_b_magnitude_var,
            self.motion_b_scale_var,
            self.width_var,
            self.height_var,
        ):
            try:
                var.trace_add("write", self._on_advanced_control_changed)
            except Exception:
                pass
        self._refresh_advanced_visibility()
    def _validate_numeric_parameters(self):
        """Validate all numeric parameters before starting"""
        try:
            seed_str = self.seed_var.get().strip()
            if seed_str == "":
                seed = 1
            else:
                seed = int(seed_str)
                if seed < 0:
                    messagebox.showwarning("Warning", "Seed should be non-negative. Using 0.")
                    seed = 0
            self.seed_var.set(str(seed))

            buffer_str = self.buffer_var.get().strip()
            if buffer_str == "":
                buffer_size = 1
            else:
                buffer_size = int(buffer_str)
                if buffer_size < 1:
                    messagebox.showwarning("Warning", "Buffer size must be at least 1. Using 1.")
                    buffer_size = 1
            self.buffer_var.set(str(buffer_size))

            guidance_str = self.guidance_var.get().strip()
            if guidance_str == "":
                guidance = 0.0
            else:
                guidance = float(guidance_str)
                if guidance < 0:
                    messagebox.showwarning("Warning", "Guidance scale should be non-negative. Using 0.")
                    guidance = 0.0
            self.guidance_var.set(str(guidance))

            delta_str = self.delta_var.get().strip()
            if delta_str == "":
                delta = 0.0
            else:
                delta = float(delta_str)
                if delta < 0:
                    messagebox.showwarning("Warning", "Delta should be non-negative. Using 0.")
                    delta = 0.0
            self.delta_var.set(str(delta))

            sim_thresh_str = self.sim_thresh_var.get().strip()
            if sim_thresh_str == "":
                sim_thresh = 0.99
            else:
                sim_thresh = float(sim_thresh_str)
                if not (0.0 <= sim_thresh <= 1.0):
                    messagebox.showwarning("Warning", "Similarity threshold should be between 0.0 and 1.0. Using 0.99.")
                    sim_thresh = 0.99
            self.sim_thresh_var.set(str(sim_thresh))

            sim_maxskip_str = self.sim_maxskip_var.get().strip()
            if sim_maxskip_str == "":
                sim_maxskip = 10.0
            else:
                sim_maxskip = float(sim_maxskip_str)
                if sim_maxskip < 0:
                    messagebox.showwarning("Warning", "Max skip frames should be non-negative. Using 10.")
                    sim_maxskip = 10.0
            self.sim_maxskip_var.set(str(sim_maxskip))

            w = int(self.width_var.get())
            h = int(self.height_var.get())
            if w < 64:
                w = 64
            if h < 64:
                h = 64
            self.width_var.set(w)
            self.height_var.set(h)

            spout_out_fps = float(self.spout_output_fps_var.get().strip() or "30")
            self.spout_output_fps_var.set(str(max(1.0, spout_out_fps)))

            motion_scale = float(self.motion_scale_var.get().strip() or "1.0")
            self.motion_scale_var.set(str(motion_scale))

            warp_strength = float(self.warp_strength_var.get().strip() or "0.35")
            warp_strength = max(0.0, min(1.0, warp_strength))
            self.warp_strength_var.set(str(warp_strength))

            b_scale = float(self.motion_b_scale_var.get().strip() or "1.0")
            self.motion_b_scale_var.set(str(b_scale))

            return True

        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid numeric value: {e}")
            return False

    def _register_lockables(self, *ws):
        for w in ws:
            if w is not None: self._lockables.append(w)

    def _set_state(self, widgets, state: str):
        for w in widgets:
            try: w.configure(state=state)
            except Exception: pass

    def _apply_running_state(self):
        if self.running:
            self._set_state(self._lockables, "disabled")
            try: self.prompt_txt.configure(state="normal")
            except Exception: pass
            try: self.neg_prompt_txt.configure(state="normal")
            except Exception: pass
            for s in self._step_sliders:
                try: s.configure(state="normal")
                except Exception: pass
            # Keep advanced tuning controls live while running.
            runtime_widgets = [
                getattr(self, "_w_capture_source", None),
                getattr(self, "_w_spout_color_entry", None),
                getattr(self, "_w_spout_motion_entry", None),
                getattr(self, "_w_spout_output_switch", None),
                getattr(self, "_w_spout_output_sender_entry", None),
                getattr(self, "_w_spout_out_fps_entry", None),
                getattr(self, "_w_motion_warp_switch", None),
                getattr(self, "_w_motion_scale_entry", None),
                getattr(self, "_w_warp_strength_entry", None),
                getattr(self, "_w_motion_invert_y_switch", None),
                getattr(self, "_w_motion_use_b_switch", None),
                getattr(self, "_w_motion_b_scale_entry", None),
            ]
            self._set_state(runtime_widgets, "normal")
            self._refresh_spout_ui()
        else:
            self._set_state(self._lockables, "normal")

    def _add_step(self):
        if self.running: return
        self.t_index_list.append(30)
        self._build_steps_ui()

    def _remove_step(self):
        if len(self.t_index_list) > 1:
            self.t_index_list.pop()
            self._build_steps_ui()
            if self.running: 
                try:
                    self.control_q.put_nowait({"type": "set_t_index_list", "t_index_list": list(self.t_index_list)})
                except Exception:
                    pass

    def _build_steps_ui(self):
        for child in self._steps_holder.winfo_children(): child.destroy()
        self._step_sliders = []
        for idx, val in enumerate(self.t_index_list):
            row = idx
            ctk.CTkLabel(self._steps_holder, text=f"Step {idx+1}")\
                .grid(row=row, column=0, sticky="w", padx=(0,6), pady=4)
            disp = ctk.StringVar(value=str(int(val)))
            ctk.CTkLabel(self._steps_holder, textvariable=disp, width=28)\
                .grid(row=row, column=1, sticky="w", padx=(0,6))
            slider = ctk.CTkSlider(
                self._steps_holder, from_=2, to=49, number_of_steps=48,
                command=lambda v, i=idx, d=disp: self._on_step_changed(i, v, d)
            )
            slider.set(int(val))
            slider.grid(row=row, column=2, sticky="ew", padx=(0,6))
            self._step_sliders.append(slider)
            self._steps_holder.grid_columnconfigure(2, weight=1)
        self._apply_running_state()

    def _on_step_changed(self, index: int, v, disp_var):
        try:
            ival = max(2, min(49, int(float(v))))
        except Exception:
            return

        disp_var.set(str(ival))
        if 0 <= index < len(self.t_index_list):
            self.t_index_list[index] = ival

        if self.running and getattr(self, 'control_q', None):
            try:
                self.control_q.put_nowait({"type": "set_t_at", "index": int(index), "value": int(ival)})
            except Exception:
                pass

    def _on_prompt_changed(self, _evt=None):
        if not self.running: return
        if self._debounce_prompt is not None:
            try: self.after_cancel(self._debounce_prompt)
            except Exception: pass
        self._debounce_prompt = self.after(150, self._push_prompt_runtime)

    def _on_neg_prompt_changed(self, _evt=None):
        if not self.running: return
        if self._debounce_neg is not None:
            try: self.after_cancel(self._debounce_neg)
            except Exception: pass
        self._debounce_neg = self.after(150, self._push_neg_prompt_runtime)

    def _push_prompt_runtime(self):
        if not getattr(self, 'control_q', None): return
        try:
            txt = self.prompt_txt.get("1.0", "end").strip()
            self.control_q.put_nowait({"type": "set_prompt", "prompt": txt})
        except Exception: pass

    def _push_neg_prompt_runtime(self):
        if not getattr(self, 'control_q', None): return
        try:
            txt = self.neg_prompt_txt.get("1.0", "end").strip()
            self.control_q.put_nowait({"type": "set_negative_prompt", "negative_prompt": txt})
        except Exception: pass

    def _overlay_screen_rect(self) -> Dict[str, int]:
        if self.capwin is not None:
            return self.capwin.inner_rect_screen()
        self.update_idletasks()
        x = int(self.preview_panel.winfo_rootx()); y = int(self.preview_panel.winfo_rooty())
        return {"left": x, "top": y, "width": self.preview_dim, "height": self.preview_dim}

    def _send_region_update(self):
        if self.capture_source_var.get().lower() != "screen":
            return
        if self.running and getattr(self, "control_q", None):
            try:
                self.control_q.put_nowait({"type": "set_region", "region": self._overlay_screen_rect()})
            except Exception: pass

    def _on_window_configure(self, _evt=None):
        if self.running and self.capwin is not None:
            try:
                if self._debounce_region is not None: 
                    self.after_cancel(self._debounce_region)
            except Exception: 
                pass
            self._debounce_region = self.after(50, self._send_region_update)

    def _append_gpu_log(self, line: str):
        try:
            self.gpu_log.configure(state="normal")
            self.gpu_log.insert("end", line + "\n")
            self.gpu_log.see("end")
        finally:
            self.gpu_log.configure(state="disabled")

    def _gpu_prog_set(self, frac: float):
        try:
            f = max(0.0, min(1.0, float(frac)))
            self.gpu_prog.set(f)
            self.gpu_pct_var.set(f"{int(round(f*100))}%")
            try: self.update_idletasks()
            except Exception: pass
        except Exception:
            pass

    def _on_enable_gpu(self):
        if getattr(self, "_gpu_install_in_progress", False):
            try:
                self.gpu_btn.configure(state="disabled")
            except Exception:
                pass
            return

        try:
            self.gpu_btn.configure(state="disabled", text="Installing PyTorch...")
        except Exception:
            pass

        self._gpu_install_in_progress = True
        self._append_gpu_log("Starting PyTorch installation...")
        self._gpu_prog_set(0.05)

        def progress_callback(progress, status=""):
            self._gpu_prog_set(progress)
            if status:
                self._append_gpu_log(status)

        def worker():
            try:
                def on_progress(line: str):
                    self._append_gpu_log(line)

                self._append_gpu_log("Installing PyTorch with CUDA support...")
                self._gpu_prog_set(0.05)

                _patch_missing_modules()
                _patch_torch_imports()

                # Install only PyTorch
                success = maybe_bootstrap_gpu(on_progress=on_progress, progress_callback=progress_callback)
        
                if success:
                    self._append_gpu_log("✅ PyTorch installation completed successfully!")
                    self._append_gpu_log("🔄 Restarting application in 3 seconds...")
            
                    # Schedule automatic restart
                    self.after(3000, self._perform_automatic_restart)
                else:
                    self._append_gpu_log("❌ PyTorch installation failed")
                    self._gpu_prog_set(0.0)
                    try:
                        self.gpu_btn.configure(state="normal", text="Install PyTorch - RETRY")
                    except Exception:
                        pass

            except Exception as e:
                self._append_gpu_log(f"❌ Installation failed: {e}")
                import traceback
                self._append_gpu_log(f"Traceback: {traceback.format_exc()}")
                self._gpu_prog_set(0.0)
                try:
                    self.gpu_btn.configure(state="normal", text="Install PyTorch - RETRY")
                except Exception:
                    pass
            finally:
                self._gpu_install_in_progress = False

        import threading
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _perform_automatic_restart(self):
        """Perform automatic restart"""
        try:
            self._append_gpu_log("🔄 Restarting application now...")
            self.update_idletasks()
        
            # Ensure _internal is in sys.path before restarting
            import sys
            internal_str = str(INTERNAL_DIR)
            if internal_str not in sys.path:
                sys.path.insert(0, internal_str)
        
            import time
            time.sleep(1)
        
            _gpu_relaunch_into_runtime()
        except Exception as e:
            self._append_gpu_log(f"❌ Automatic restart failed: {e}")
            self._append_gpu_log("Please manually restart the application")
            messagebox.showinfo(
                "Restart Required", 
                "PyTorch installed successfully!\n\nPlease manually restart the application to use GPU features."
            )


    def _browse_model(self):
        d = filedialog.askdirectory(title="Select diffusers model folder")
        if d: self.model_var.set(d)

    def _on_start(self):
        if self.running: return

        if not self._validate_numeric_parameters():
            return

        if self.capture_source_var.get().lower() == "spout" and not self.spout_ready:
            messagebox.showerror("Spout unavailable", self.spout_status)
            return

        try:
            verify_local_model_path_dir(self.model_var.get())
        except Exception as e:
            messagebox.showerror("Paths error", str(e)); return

        ctx = get_context("spawn")
        self.proc_ctx = ctx
        self.out_q = ctx.Queue(maxsize=2); self.fps_q = ctx.Queue(); self.status_q = ctx.Queue()
        self.control_q = ctx.Queue(); self.debug_q = ctx.Queue(); self.close_q = ctx.Queue()
        self.monitor_sender, self.monitor_receiver = ctx.Pipe()

        controlnet_paths: List[str] = []; controlnet_scales: List[float] = []
        capture_cfg = self._current_advanced_cfg()

        self.proc_worker = ctx.Process(
            target=image_generation_process,
            args=(
                self.out_q, self.fps_q, self.close_q, self.status_q, self.control_q, self.debug_q,
                list(map(int, self.t_index_list)),
                self.model_var.get(), controlnet_paths, controlnet_scales, None,
                self.prompt_txt.get("1.0", "end").strip(),
                self.neg_prompt_txt.get("1.0", "end").strip(),
                int(self.buffer_var.get()),
                int(self.width_var.get()), int(self.height_var.get()),
                self.accel_var.get(), bool(self.denoise_batch_var.get()),
                int(self.seed_var.get()), self.cfg_type_var.get(),
                float(self.guidance_var.get()), float(self.delta_var.get()),
                False, bool(self.sim_filter_var.get()),
                float(self.sim_thresh_var.get()), float(self.sim_maxskip_var.get()),
                self.monitor_receiver, capture_cfg, bool(self.offline_var.get()),
            ),
        )
        self.proc_worker.start()

        if self.capture_source_var.get().lower() == "screen":
            self.capwin = FloatingCaptureWindow(self, inner_size=self.preview_dim, border_px=8, handle_h=28)
            try:
                self.monitor_sender.send(self._overlay_screen_rect())
            except Exception as e:
                print(f"[gui] failed to send capture rect: {e}")
            self.hide_capture_btn.configure(state="normal")
        else:
            self.capwin = None
            self.hide_capture_btn.configure(state="disabled")
            try:
                self.monitor_sender.send({"left": 0, "top": 0, "width": int(self.width_var.get()), "height": int(self.height_var.get())})
            except Exception:
                pass

        self.running = True
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self._apply_running_state()

    def _on_stop(self):
        if not self.running: return
        try:
            if self.close_q is not None:
                try: self.close_q.put_nowait(True)
                except Exception: pass
            if self.proc_worker is not None:
                self.proc_worker.join(timeout=5)
                if self.proc_worker.is_alive(): self.proc_worker.terminate()
            self.proc_worker = None
        finally:
            try:
                if self.capwin is not None:
                    # Show window before destroying if it was hidden
                    try:
                        if self.capwin.win.state() != "normal":
                            self.capwin.win.deiconify()
                    except:
                        pass
                    self.capwin.destroy()
            finally:
                self.capwin = None
            self.running = False
            self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
            self._apply_running_state()

    def _on_capture_window_moved(self):
        if self.running: self._send_region_update()

    def _refresh_profile_button(self):
        if not hasattr(self, "profile_btn"):
            return
        if self.profiling_running:
            self.profile_btn.configure(
                text="Stop Profiling",
                fg_color=CUSTOM_COLORS["error"],
                hover_color="#DC2626",
            )
        else:
            self.profile_btn.configure(
                text="Start Profiling",
                fg_color=CUSTOM_COLORS["success"],
                hover_color="#059669",
            )

    def _toggle_profiling(self):
        if self.profiling_running:
            self._stop_profiling()
        else:
            self._start_profiling()

    def _start_profiling(self):
        smi_path = shutil.which("nvidia-smi")
        if not smi_path:
            messagebox.showerror("Profiling Error", "nvidia-smi not found in PATH.")
            return

        try:
            profiling_dir = APP_ROOT / "profiling"
            profiling_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._profiling_gpu_log = str(profiling_dir / f"gpu_metrics_{ts}.csv")
            self._profiling_proc_log = str(profiling_dir / f"gpu_processes_{ts}.csv")

            self._profiling_gpu_fh = open(self._profiling_gpu_log, "w", buffering=1, encoding="utf-8")
            self._profiling_proc_fh = open(self._profiling_proc_log, "w", buffering=1, encoding="utf-8")

            self._profiling_gpu_proc = subprocess.Popen(
                [
                    smi_path,
                    "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv",
                    "-l",
                    "1",
                ],
                stdout=self._profiling_gpu_fh,
                stderr=subprocess.DEVNULL,
            )
            self._profiling_proc_proc = subprocess.Popen(
                [
                    smi_path,
                    "--query-compute-apps=timestamp,pid,process_name,used_gpu_memory",
                    "--format=csv",
                    "-l",
                    "1",
                ],
                stdout=self._profiling_proc_fh,
                stderr=subprocess.DEVNULL,
            )

            self.profiling_running = True
            self._refresh_profile_button()
            self.status_var.set(f"Profiling started: {self._profiling_gpu_log}")
        except Exception as e:
            self._stop_profiling(silent=True)
            messagebox.showerror("Profiling Error", f"Failed to start profiling: {e}")

    def _stop_profiling(self, silent=False):
        for proc in (self._profiling_gpu_proc, self._profiling_proc_proc):
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self._profiling_gpu_proc = None
        self._profiling_proc_proc = None
        for fh_attr in ("_profiling_gpu_fh", "_profiling_proc_fh"):
            fh = getattr(self, fh_attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            setattr(self, fh_attr, None)
        was_running = self.profiling_running
        self.profiling_running = False
        self._refresh_profile_button()
        if (not silent) and was_running:
            self.status_var.set(f"Profiling stopped. Logs saved in {APP_ROOT / 'profiling'}")

    def _poll_queues(self):
        if self.out_q is not None:
            try:
                while True:
                    pil = self.out_q.get_nowait()
                    if isinstance(pil, Image.Image): self._update_preview(pil)
            except Exception: pass
        if self.fps_q is not None:
            try:
                while True:
                    fps = self.fps_q.get_nowait()
                    self.fps_var.set(f"FPS: {int(fps)}")
            except Exception: pass
        if self.status_q is not None:
            try:
                while True:
                    msg = self.status_q.get_nowait()
                    self.status_var.set(msg)
            except Exception: pass
        self.after(33, self._poll_queues)

    def _update_preview(self, pil_img: Image):
        dim = self.preview_dim
    
        canvas = Image.new("RGB", (dim, dim), (30, 30, 30))
    
        img = pil_img.copy()
        img.thumbnail((dim, dim), PIL.Image.BICUBIC)
        x = (dim - img.width) // 2
        y = (dim - img.height) // 2
        canvas.paste(img, (x, y))
    
        self._ctk_img = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(dim, dim))
        self.preview_panel.configure(image=self._ctk_img)

    def do_quit(self):
        if not messagebox.askyesno("Quit", "Are you sure you want to quit and stop all workers?"):
            return
        if self.running: self._on_stop()
        if self.profiling_running: self._stop_profiling(silent=True)
        self.destroy()

# =========================
# Main
# =========================
def preload_critical_components():
    """Preload components but skip torch in frozen mode"""
    if not getattr(sys, "frozen", False):
        # Only preload in dev mode
        _prime_dll_search()
        PreloadedDependencies.preload_all()
    else:
        print("[Preload] Skipping torch preload in frozen mode (will load on-demand)")

if __name__ == "__main__":
    try:
        print("=== APPLICATION STARTING ===")
        import multiprocessing as mp
        mp.freeze_support()
        
        preload_critical_components()
        
        print("Creating GUI...")
        app = StreamGUI()
        print("GUI created successfully, starting mainloop...")
        
        # Additional icon setting as backup
        def set_window_icon_backup():
            """Backup method to set window icon"""
            try:
                icon_paths = [
                    resource_path("icon2.ico"),
                ]
                
                for icon_path in icon_paths:
                    if os.path.exists(icon_path):
                        if icon_path.lower().endswith('.ico'):
                            app.iconbitmap(icon_path)
                            print(f"Backup icon set from: {icon_path}")
                            break
                        else:
                            # Convert image to temporary .ico
                            try:
                                img = Image.open(icon_path)
                                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                                temp_ico = os.path.join(tempfile.gettempdir(), "temp_app_icon.ico")
                                img.save(temp_ico, format='ICO')
                                app.iconbitmap(temp_ico)
                                print(f"Converted and set icon from: {icon_path}")
                                break
                            except Exception:
                                continue
            except Exception as e:
                print(f"Backup icon setting failed: {e}")
        
        # Run backup icon setting after a short delay
        app.after(500, set_window_icon_backup)
        
        app.mainloop()
        print("=== APPLICATION EXITED ===")
        
    except Exception as e:
        print(f"!!! CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
