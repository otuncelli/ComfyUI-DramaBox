"""
ComfyUI-DramaBox: Custom nodes for DramaBox expressive TTS with voice cloning.

Source: https://github.com/resemble-ai/DramaBox
Models: https://huggingface.co/ResembleAI/Dramabox

This node clones the DramaBox repository on first use and downloads the required
model weights (~17 GB total) from HuggingFace into ComfyUI/models/DramaBox/.
"""

import os
import sys
import subprocess
import tempfile
import logging
import urllib.request
import zipfile
import shutil
from pathlib import Path

import torch
import torchaudio
import folder_paths

logger = logging.getLogger(__name__)


def _save_audio(path: str, waveform: "torch.Tensor", sample_rate: int) -> None:
    """Save audio, falling back gracefully when torchcodec is unavailable.

    Recent torchaudio (≥2.6) defaults torchaudio.save() to the torchcodec backend
    which may not be installed (especially on bleeding-edge CUDA builds). This
    wrapper tries soundfile first, then falls back to the scipy/wave writer.
    """
    # Preferred: explicit soundfile backend (no torchcodec dependency)
    try:
        torchaudio.save(path, waveform, sample_rate, backend="soundfile")
        return
    except Exception:
        pass

    # Last resort: scipy wavfile (always available with numpy)
    try:
        import scipy.io.wavfile as _wavfile
        import numpy as _np
        data = waveform.numpy()
        if data.ndim == 2:
            data = data.T  # scipy expects (samples, channels)
        _wavfile.write(path, sample_rate, data.astype(_np.float32))
        return
    except Exception:
        pass

    # Final fallback: let torchaudio pick whatever backend it has
    torchaudio.save(path, waveform, sample_rate)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NODE_DIR = Path(__file__).parent
DRAMABOX_REPO_DIR = NODE_DIR / "DramaBox"
MODELS_DIR = Path(folder_paths.models_dir) / "DramaBox"

# Pin to a known-good DramaBox commit. The upstream ltx2/ libraries vendored
# in the repo are still on the class-based ltx_pipelines API (blocks.py,
# denoisers.py). Pinning protects us from future SDK refactors landing in
# DramaBox HEAD without warning. Bump deliberately after testing.
DRAMABOX_PIN_SHA = "a70a5818e103c1c9fef22409c1e0c707ebf4f8a7"

# ---------------------------------------------------------------------------
# Repository bootstrap
# ---------------------------------------------------------------------------
_repo_paths_added = False


def _add_repo_paths():
    """Insert DramaBox source directories at the front of sys.path AND evict
    any prior cached ``ltx_pipelines`` / ``ltx_core`` imports so our bundled
    copy wins.

    Some users have ``ltx-pipelines`` pip-installed (Lightricks' LTX-2 SDK
    1.0 moved ``PromptEncoder`` / ``GuidedDenoiser`` from ``utils.blocks`` and
    ``utils.denoisers`` into functional helpers, so DramaBox's class-based
    imports fail against it). If another ComfyUI custom node imports
    ``ltx_pipelines`` before us, ``sys.modules`` caches the pip version and
    a later ``sys.path.insert(0, ...)`` is too late — module resolution is
    already short-circuited. Evicting forces re-resolution against the
    bundled tree.
    """
    global _repo_paths_added
    if _repo_paths_added:
        return
    for subdir in ["ltx2", "src"]:
        p = str(DRAMABOX_REPO_DIR / subdir)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    for name in [
        m for m in list(sys.modules)
        if m == "ltx_pipelines" or m.startswith("ltx_pipelines.")
        or m == "ltx_core" or m.startswith("ltx_core.")
    ]:
        del sys.modules[name]

    # Defensive check: the bundled blocks.py must exist. If a fresh clone
    # ever lands without it (upstream refactor we haven't adopted yet), fail
    # loudly here instead of deep inside the inference import chain.
    blocks_py = DRAMABOX_REPO_DIR / "ltx2" / "ltx_pipelines" / "utils" / "blocks.py"
    if DRAMABOX_REPO_DIR.exists() and not blocks_py.exists():
        raise RuntimeError(
            f"[DramaBox] Bundled ltx_pipelines is missing utils/blocks.py at {blocks_py}. "
            f"This usually means the auto-cloned DramaBox tree got out of sync. "
            f"Delete {DRAMABOX_REPO_DIR} and let it re-clone (pinned to "
            f"{DRAMABOX_PIN_SHA[:7]})."
        )

    _repo_paths_added = True


def _checkout_pin() -> bool:
    """Check out the pinned SHA in DRAMABOX_REPO_DIR. Best-effort: returns
    False if the SHA isn't reachable (e.g. shallow clone), so the caller can
    decide whether to fall back to whatever is currently checked out."""
    try:
        # Ensure the pinned commit is fetched. Cheap on top of --depth=1 since
        # the rest of master is shallow-fetched only as needed.
        subprocess.run(
            ["git", "-C", str(DRAMABOX_REPO_DIR), "fetch", "--depth=1",
             "origin", DRAMABOX_PIN_SHA],
            capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "-C", str(DRAMABOX_REPO_DIR), "-c", "advice.detachedHead=false",
             "checkout", DRAMABOX_PIN_SHA],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(f"[DramaBox] Checked out pinned commit {DRAMABOX_PIN_SHA[:7]}.")
            return True
        logger.warning(
            f"[DramaBox] Could not check out pin {DRAMABOX_PIN_SHA[:7]}: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        pass
    return False


def _clone_via_git():
    """Clone with git, then check out the pinned SHA. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "git", "clone", "--depth=1",
                "https://github.com/resemble-ai/DramaBox.git",
                str(DRAMABOX_REPO_DIR),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("[DramaBox] Repository cloned successfully via git.")
            _checkout_pin()  # best-effort; stays on master if unreachable
            return True
        logger.warning(f"[DramaBox] git clone returned {result.returncode}: {result.stderr}")
    except FileNotFoundError:
        logger.warning("[DramaBox] git not found on PATH, falling back to zipball download.")
    return False


def _clone_via_zipball():
    """Download GitHub zipball at the pinned SHA as a fallback when git is unavailable."""
    url = f"https://github.com/resemble-ai/DramaBox/archive/{DRAMABOX_PIN_SHA}.zip"
    tmp_zip = NODE_DIR / "_dramabox_tmp.zip"
    logger.info(f"[DramaBox] Downloading DramaBox zipball from {url} …")
    try:
        urllib.request.urlretrieve(url, str(tmp_zip))
        with zipfile.ZipFile(str(tmp_zip), "r") as zf:
            zf.extractall(str(NODE_DIR))
        extracted = NODE_DIR / f"DramaBox-{DRAMABOX_PIN_SHA}"
        extracted.rename(DRAMABOX_REPO_DIR)
        logger.info(f"[DramaBox] Repository extracted at pin {DRAMABOX_PIN_SHA[:7]}.")
        return True
    except Exception as e:
        logger.error(f"[DramaBox] Zipball download failed: {e}")
        return False
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


def _heal_bitsandbytes_stub() -> None:
    """Patch ComfyUI's broken bitsandbytes stub in-place.

    Symptom: another package (likely an early-loading ComfyUI node) imported
    bnb during startup, bnb's __init__ partially ran — the C++ cextension
    loaded and ``bitsandbytes._ops`` ran torch.library.define for all bnb
    ops, but a downstream import failed. Python rolled the failed import
    back, deleting ``bitsandbytes`` from ``sys.modules``. Some caller then
    inserted a stub ``ModuleType('bitsandbytes')`` placeholder so its own
    ``is_bitsandbytes_available()``-style checks would pass. The C++ ops,
    however, stay registered with PyTorch's dispatcher — they're process-
    global state and survive module eviction.

    Net result: ``bnb.nn`` is missing on the stub, so ``bnb.nn.Linear4bit``
    fails. We can't re-import bnb (re-running ``bitsandbytes._ops`` re-
    registers the same ops → RuntimeError: duplicate registration). The
    fix is to mutate the existing stub in place — repair its
    ``__path__`` / ``__file__`` so submodule imports resolve, pre-cache
    a no-op stub for the modules that would re-register torch ops, then
    import ``bitsandbytes.nn`` and attach it to the stub. Since
    ``transformers.integrations.bitsandbytes`` captured a reference to the
    same stub object at module-load, mutating it propagates the fix
    everywhere with no eviction needed.
    """
    bnb = sys.modules.get("bitsandbytes")
    if bnb is None:
        return

    healthy = hasattr(bnb, "nn") and getattr(bnb, "__file__", None) is not None
    if healthy:
        return

    logger.warning(
        f"[DramaBox] heal v3 — bitsandbytes is a stub "
        f"(file={getattr(bnb, '__file__', None)}, "
        f"version={getattr(bnb, '__version__', '?')}, "
        f"has_nn={hasattr(bnb, 'nn')}). Patching in place…"
    )

    import importlib  # noqa: PLC0415
    import importlib.machinery  # noqa: PLC0415
    import importlib.metadata  # noqa: PLC0415
    import types  # noqa: PLC0415

    # Bypass the sys.modules cache (which holds the stub with no spec/origin).
    # Try PathFinder first, then walk sys.path manually if that returns nothing.
    pkg_dir = None
    spec = importlib.machinery.PathFinder.find_spec("bitsandbytes", sys.path)
    if spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
    else:
        # Manual fallback — find bitsandbytes/__init__.py on sys.path.
        for entry in sys.path:
            candidate = os.path.join(entry, "bitsandbytes", "__init__.py")
            if os.path.isfile(candidate):
                pkg_dir = os.path.dirname(candidate)
                break

    if not pkg_dir:
        logger.error(
            f"[DramaBox] Cannot locate bitsandbytes on sys.path. "
            f"PathFinder returned spec={spec}. Tried {len(sys.path)} sys.path entries."
        )
        return
    logger.info(f"[DramaBox] Found real bitsandbytes at {pkg_dir}")
    init_path = os.path.join(pkg_dir, "__init__.py")

    # Repair the stub so Python's import machinery can find submodules.
    if not hasattr(bnb, "__path__"):
        bnb.__path__ = [pkg_dir]
    if not getattr(bnb, "__file__", None):
        bnb.__file__ = init_path

    # Pre-cache no-op stubs for every module that would re-register torch
    # ops. The C++ ops are already registered globally (from the prior
    # partial load); running these modules again would call
    # torch.library.define on the same names and raise duplicate-
    # registration RuntimeError. ``bitsandbytes._ops`` also exports
    # decorator factories that backend modules import, so we shim those.
    _noop_decorator = lambda *_a, **_kw: (lambda f: f)
    for name, exports in (
        ("bitsandbytes._ops", {"register_kernel": _noop_decorator,
                                "register_fake": _noop_decorator}),
        ("bitsandbytes.backends.cpu.ops", {}),
        ("bitsandbytes.backends.default.ops", {}),
        ("bitsandbytes.backends.cuda.ops", {}),
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            for k, v in exports.items():
                setattr(stub, k, v)
            sys.modules[name] = stub

    try:
        nn_mod = importlib.import_module("bitsandbytes.nn")
        bnb.nn = nn_mod
        try:
            bnb.__version__ = importlib.metadata.version("bitsandbytes")
        except Exception:
            pass
        # transformers.is_bitsandbytes_multi_backend_available() reads this.
        bnb.features = {"multi_backend"}
        logger.info(
            f"[DramaBox] bitsandbytes stub patched: "
            f"version={bnb.__version__}, Linear4bit={bnb.nn.Linear4bit}"
        )
    except Exception as e:
        import traceback  # noqa: PLC0415
        logger.error(
            "[DramaBox] bitsandbytes stub patch failed:\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__))
        )


def _ensure_repo():
    """Ensure the DramaBox GitHub repository is present and on sys.path."""
    if not DRAMABOX_REPO_DIR.exists():
        logger.info("[DramaBox] DramaBox repository not found – downloading…")
        if not _clone_via_git():
            if not _clone_via_zipball():
                raise RuntimeError(
                    "[DramaBox] Could not download the DramaBox repository.\n"
                    "Please manually clone https://github.com/resemble-ai/DramaBox "
                    f"into {DRAMABOX_REPO_DIR}"
                )
    _add_repo_paths()


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

def _download_models():
    """Download all required model weights and return paths dict.

    All models are stored under ``ComfyUI/models/DramaBox/``:
      - dramabox-dit-v1.safetensors (transformer)
      - dramabox-audio-components.safetensors (audio components)
      - gemma-3-12b-it-bnb-4bit/ (Gemma text encoder directory)
    """
    _ensure_repo()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415

    logger.info("[DramaBox] Verifying model weights (will download if missing)…")
    hf_token = os.environ.get("HF_TOKEN")

    # --- DramaBox-unique weights → ComfyUI/models/DramaBox/ ---
    dramabox_repo = "ResembleAI/Dramabox"
    model_files = {
        "transformer": "dramabox-dit-v1.safetensors",
        "audio_components": "dramabox-audio-components.safetensors",
    }

    paths = {}
    for name, filename in model_files.items():
        local_path = MODELS_DIR / filename
        if local_path.exists():
            logger.info(f"[DramaBox] {filename} found locally.")
        else:
            logger.info(f"[DramaBox] Downloading {filename} from {dramabox_repo}…")
            hf_hub_download(
                repo_id=dramabox_repo,
                filename=filename,
                local_dir=str(MODELS_DIR),
                token=hf_token,
            )
            logger.info(f"[DramaBox] {filename} downloaded.")
        paths[name] = str(local_path)

    # --- Gemma text encoder → ComfyUI/models/DramaBox/ ---
    gemma_dir = MODELS_DIR / "gemma-3-12b-it-bnb-4bit"
    if gemma_dir.exists() and any(gemma_dir.iterdir()):
        logger.info("[DramaBox] Gemma encoder found locally.")
    else:
        logger.info("[DramaBox] Downloading Gemma encoder (unsloth/gemma-3-12b-it-bnb-4bit)…")
        snapshot_download(
            repo_id="unsloth/gemma-3-12b-it-bnb-4bit",
            local_dir=str(gemma_dir),
            token=hf_token,
        )
        logger.info("[DramaBox] Gemma encoder downloaded.")
    paths["gemma_root"] = str(gemma_dir)

    return paths


# ---------------------------------------------------------------------------
# TTSServer singleton (lazy, loaded on first generate() call)
# ---------------------------------------------------------------------------
_tts_server = None


def _get_server():
    """Return the cached TTSServer, creating and loading it on first call."""
    global _tts_server
    if _tts_server is not None:
        return _tts_server

    _ensure_repo()
    paths = _download_models()

    _heal_bitsandbytes_stub()

    from inference_server import TTSServer  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[DramaBox] Loading TTSServer on {device} (first run may take several minutes)…")

    _tts_server = TTSServer(
        checkpoint=paths["transformer"],
        full_checkpoint=paths["audio_components"],
        gemma_root=paths["gemma_root"],
        device=device,
        dtype="bf16",
        compile_model=False,   # torch.compile can be unstable in some setups
        bnb_4bit=True,         # uses pre-quantised unsloth Gemma weights
    )
    logger.info("[DramaBox] TTSServer ready.")
    return _tts_server


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class DramaBoxTTS:
    """
    DramaBox expressive TTS with voice cloning.

    Generates rich, dramatic speech from a structured scene prompt.
    An optional voice reference (10+ seconds) clones the speaker timbre.

    Prompt format:
        <speaker description>, "<dialogue>" <action direction> "<more dialogue>"

    Example:
        A woman speaks warmly, "Hello, how are you today?" She laughs,
        "Hahaha, it is so good to see you!"

    Tips:
    - Phonetic sounds go INSIDE quotes: "Hahaha", "Hmm", "Ugh", "Argh"
    - Stage directions go OUTSIDE: She sighs deeply.  He clears his throat.
    - Match the gender/age in your description to the voice reference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            'A woman speaks warmly, "Hello, how are you today?" '
                            'She laughs, "Hahaha, it is so good to see you!"'
                        ),
                        "tooltip": (
                            "Scene prompt. Put dialogue in double quotes, stage "
                            "directions outside them. Phonetic sounds (Hahaha, Hmm) "
                            "go inside quotes; named actions (She sighs.) go outside."
                        ),
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": (
                            "CFG guidance scale. Lower = more natural delivery; "
                            "higher = more text-faithful. DramaBox default: 2.5."
                        ),
                    },
                ),
                "stg_scale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": (
                            "Skip-token guidance scale. DramaBox default: 1.5."
                        ),
                    },
                ),
            },
            "optional": {
                "voice_sample": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional voice reference for timbre cloning. "
                            "10+ seconds of clean speech recommended."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed for reproducible generations.",
                    },
                ),
                "duration_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Multiply the auto-estimated speech duration. "
                            "1.1 adds 10 %% breathing room. Increase for slower delivery."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/DramaBox"
    DESCRIPTION = (
        "DramaBox expressive TTS with voice cloning. "
        "Generates dramatic, expressive speech from a structured scene prompt. "
        "Requires ~24 GB VRAM on first run."
    )

    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        cfg_scale: float,
        stg_scale: float,
        voice_sample=None,
        seed: int = 42,
        duration_multiplier: float = 1.1,
    ):
        if not text or not text.strip():
            raise ValueError("[DramaBox] Text prompt cannot be empty.")

        server = _get_server()

        # ---- Write voice reference to a temp WAV if an AUDIO input was given ----
        tmp_wav = None
        voice_ref_path = None
        try:
            if voice_sample is not None:
                waveform = voice_sample["waveform"]
                sr = int(voice_sample["sample_rate"])

                # Normalise to [C, S]
                if waveform.dim() == 3:
                    waveform = waveform[0]        # [B, C, S] -> [C, S]
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # [S] -> [1, S]

                tmp_wav = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="dramabox_ref_"
                )
                tmp_wav.close()
                voice_ref_path = tmp_wav.name
                _save_audio(voice_ref_path, waveform.float().cpu(), sr)
                logger.info(
                    f"[DramaBox] Voice reference saved to temp file: {voice_ref_path} "
                    f"({waveform.shape[-1] / sr:.1f}s)"
                )

            # ---- Run inference ----
            waveform_out, sr_out = server.generate(
                prompt=text,
                voice_ref=voice_ref_path,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                seed=seed,
            )

        finally:
            # Clean up temp file regardless of success/failure
            if tmp_wav is not None:
                try:
                    os.unlink(tmp_wav.name)
                except OSError:
                    pass

        # ---- Format output for ComfyUI: {"waveform": [B, C, S], "sample_rate": int} ----
        wav = waveform_out.float().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)   # -> [1, 1, S]
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)                # -> [1, C, S]
        # dim == 3 is already [B, C, S]; leave as-is

        duration = wav.shape[-1] / sr_out
        logger.info(f"[DramaBox] Generated {duration:.1f}s of audio at {sr_out} Hz.")

        return ({"waveform": wav, "sample_rate": sr_out},)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DramaBoxTTS": DramaBoxTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTTS": "DramaBox TTS",
}
