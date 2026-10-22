"""PyTorch Hub entry points for PVT-GDLA.

    import torch
    model = torch.hub.load("xmindflow/gdla", "pvtgdla", dataset="synapse")
    predictor = torch.hub.load("xmindflow/gdla", "predictor", dataset="ham10000")
    mask = predictor.predict("lesion.jpg")

Weights are downloaded from https://huggingface.co/afshinbigboy/GDLA.
"""
import os

dependencies = ["torch", "timm", "monai", "numpy", "scipy", "huggingface_hub"]

import numpy as np
import torch
import torch.nn.functional as F

HF_REPO = "afshinbigboy/GDLA"
IMG_SIZE = 224

_SYNAPSE = ["background", "aorta", "gallbladder", "left kidney", "right kidney",
            "liver", "pancreas", "spleen", "stomach"]
_ACDC = ["background", "right ventricle", "myocardium", "left ventricle"]

CONFIGS = {
    "synapse":  dict(file="pvtgdla-synapse.ckpt",  in_chs=1, n_classes=9, kind="gray", classes=_SYNAPSE),
    "acdc":     dict(file="pvtgdla-acdc.ckpt",     in_chs=1, n_classes=4, kind="gray", classes=_ACDC),
    "busi":     dict(file="pvtgdla-busi.ckpt",     in_chs=3, n_classes=2, kind="rgb", classes=["background", "lesion"]),
    "ham10000": dict(file="pvtgdla-ham10000.ckpt", in_chs=3, n_classes=2, kind="rgb", classes=["background", "lesion"]),
    "ph2":      dict(file="pvtgdla-ph2.ckpt",      in_chs=3, n_classes=2, kind="rgb", classes=["background", "lesion"]),
}


def pvtgdla(dataset="synapse", pretrained=True, map_location=None):
    """PVT-GDLA for `dataset` in {"synapse", "acdc", "busi", "ham10000", "ph2"}, in eval mode.

    Input: (B, C, 224, 224) float tensor, C=1 for synapse/acdc and C=3 otherwise.
    Output: a list of logits; the last entry is the final (B, n_classes, 224, 224) prediction.
    """
    from models.pvt_gdla import PVTGDLA
    if dataset not in CONFIGS:
        raise ValueError(f"dataset must be one of {list(CONFIGS)}, got {dataset!r}")
    cfg = CONFIGS[dataset]
    net = PVTGDLA(in_chs=cfg["in_chs"], n_classes=cfg["n_classes"], deepsupervision=True, encoder_ckpt=None)
    if pretrained:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=cfg["file"])
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        net.load_state_dict(ckpt["model_state_dict"])
    return net.eval()


class GDLAPredictor:
    """Wraps PVT-GDLA with the preprocessing used during training.

    `predict` takes a file path, PIL image or numpy array and returns an integer mask
    with the same height and width as the input.
      - busi / ham10000 / ph2: an RGB image (H, W, 3); grayscale images are converted to RGB.
      - synapse / acdc: a 2D slice (H, W) or a volume (D, H, W), predicted slice by slice.
    """

    def __init__(self, dataset="synapse", device=None):
        self.dataset = dataset
        self.cfg = CONFIGS[dataset]
        self.classes = self.cfg["classes"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = pvtgdla(dataset).to(self.device)

    @staticmethod
    def _read(image):
        if isinstance(image, (str, os.PathLike)):
            path = str(image)
            if path.endswith((".nii", ".nii.gz")):
                import SimpleITK as sitk
                return sitk.GetArrayFromImage(sitk.ReadImage(path))  # (D, H, W)
            if path.endswith(".npy"):
                return np.load(path)
            from PIL import Image
            image = Image.open(path)
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        return image

    @torch.no_grad()
    def _run(self, x):
        """x: (N, C, 224, 224) float tensor -> (N, 224, 224) labels."""
        return self.model(x.to(self.device))[-1].argmax(dim=1).cpu()

    def _predict_rgb(self, img):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3]
        h, w = img.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        pred = self._run(x)[None].float()
        return F.interpolate(pred, size=(h, w), mode="nearest")[0, 0].numpy().astype(np.uint8)

    def _predict_gray(self, vol, raw_hu):
        from scipy.ndimage import zoom
        if vol.ndim == 3 and vol.shape[-1] in (3, 4):  # RGB(A) image of a slice
            vol = vol[..., :3].mean(-1)
        is_2d = vol.ndim == 2
        vol = (vol[None] if is_2d else vol).astype(np.float32)
        if self.dataset == "synapse" and (vol.min() < 0 if raw_hu is None else raw_hu):
            vol = (np.clip(vol, -125, 275) + 125) / 400  # CT in Hounsfield units, Synapse window
        elif vol.max() > 1 or vol.min() < 0:
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        d, h, w = vol.shape
        x = torch.from_numpy(np.stack([zoom(s, (IMG_SIZE / h, IMG_SIZE / w), order=3) for s in vol]))[:, None]
        preds = torch.cat([self._run(x[i:i + 16]) for i in range(0, d, 16)]).numpy()
        out = np.stack([zoom(p, (h / IMG_SIZE, w / IMG_SIZE), order=0) for p in preds]).astype(np.uint8)
        return out[0] if is_2d else out

    def predict(self, image, raw_hu=None):
        """Return the label mask. For synapse, `raw_hu` says whether the input is CT in
        Hounsfield units (auto-detected from negative values when None)."""
        img = self._read(image)
        if self.cfg["kind"] == "rgb":
            return self._predict_rgb(img)
        return self._predict_gray(img, raw_hu)


def predictor(dataset="synapse", device=None):
    """A GDLAPredictor: `predictor.predict(image)` handles preprocessing and returns a mask."""
    return GDLAPredictor(dataset, device)
