from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDirDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        root: str | Path,
        *,
        transform,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(str(self.root))
        exts = {e.lower() for e in extensions}
        files: list[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        files.sort()
        if limit is not None:
            files = files[: int(limit)]
        if len(files) == 0:
            raise RuntimeError(f"No images found under: {self.root}")
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x

