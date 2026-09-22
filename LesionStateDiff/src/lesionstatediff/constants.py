from __future__ import annotations

CHANNEL_ORDER = ["LM", "FC", "LC", "VV"]
CLASS_INDEX = {"background": 0, "LM": 1, "FC": 2, "LC": 3, "VV": 4}
VALID_CLASS_VALUES = {0, 1, 2, 3, 4}
TARGET_CLASSES = ("FC", "LC", "VV")
ORIGINAL_SIZE = 750
PAD = 9
PADDED_SIZE = 768
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMAGE_EXTS | {".npy", ".npz"}
