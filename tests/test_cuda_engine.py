"""CUDA backend (int8 engine), its fp32 PyTorch reference and GPU preprocessing vs the official implementation.

Needs an NVIDIA GPU with compute capability 8.0+ and `uv sync --extra cuda --group parity`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
    pytest.skip("needs an NVIDIA GPU with compute capability 8.0+", allow_module_level=True)


@pytest.fixture(scope="module")
def official(official_model_dir, sample_images):
    """(images, reference pixels HWC, reference probabilities) from the official code."""
    from transformers import AutoImageProcessor, AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModel.from_pretrained(official_model_dir, trust_remote_code=True, dtype=torch.float32).eval().cuda()
    proc = AutoImageProcessor.from_pretrained(official_model_dir, trust_remote_code=True)
    with torch.no_grad():
        pixels = [proc(im)["pixel_values"] for im in sample_images]
        probs = [torch.sigmoid(model(p.cuda())).float().cpu().numpy()[0] for p in pixels]
    return sample_images, [p[0].permute(1, 2, 0) for p in pixels], np.stack(probs)


def test_gpu_preprocessing_matches_reference(official):
    from foidfinder.backends.cuda.preprocess import preprocess

    images, ref_pixels, _ = official
    got = preprocess(images).float().cpu()
    for g, r in zip(got, ref_pixels, strict=True):
        assert (g - r).abs().max().item() < 2e-3  # fp16 output step


def test_window_ordered_reference_is_exact(official, official_model_dir):
    from foidfinder.backends.cuda.reference import PixAITorch

    _, pixels, ref = official
    model = PixAITorch(official_model_dir, dtype=torch.float32)
    got = torch.sigmoid(model(torch.stack(pixels).cuda())).cpu().numpy()
    assert np.abs(got - ref).max() < 1e-4


def test_int8_engine_tracks_reference(official, official_model_dir):
    from foidfinder.backends.cuda.engine import PixAICuda

    _, pixels, ref = official
    engine = PixAICuda(official_model_dir)
    got = torch.sigmoid(engine(torch.stack(pixels).cuda().half())).cpu().numpy()
    assert np.isfinite(got).all()
    # Measured on 588 real images: max |dp| 0.08, mean 1e-5.
    assert np.abs(got - ref).max() < 0.15 and np.abs(got - ref).mean() < 1e-4
    # Only tags near the 0.5 decision boundary may flip.
    assert np.all(got[ref > 0.65] > 0.5) and np.all(got[ref < 0.35] < 0.5)
