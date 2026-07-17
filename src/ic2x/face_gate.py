"""
Local face gate — YuNet detection + SFace embeddings (OpenCV, fully on-device).

Tier-1 localization of the owner-selfie check (2026-07-16): face embeddings
computed from the reference photos in owner_refs/ identify the owner far more
reliably than prompting a VLM, cost nothing per call, run in ~100 ms, and the
winner photo never leaves the Mac for this check.

Decision: "owner is a main subject" when a detected face BOTH matches an owner
reference (SFace cosine ≥ SIM_THRESHOLD) AND is prominent (bbox area ratio ≥
MAIN_FACE_AREA, i.e. selfie/portrait scale — a tiny background appearance
passes, matching the user's rule).

Fail-through: missing ONNX models, no reference photos, unreadable image, any
error → the caller falls back to the cloud VLM check (judge_owner). Model files
(from the OpenCV zoo) live in models/face/ — gitignored, re-downloadable.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("ic2x.face_gate")

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"

SIM_THRESHOLD = 0.36     # SFace cosine — the model's standard verification threshold
MAIN_FACE_AREA = 0.02    # face bbox ≥ 2% of image area → prominent/close-up
DETECT_MAX_EDGE = 1280   # downscale before detection (speed; small faces survive)


class FaceGate:
    def __init__(self, models_dir: Path, refs_dir: Path) -> None:
        self._models_dir = models_dir
        self._refs_dir = refs_dir
        self._rec = None
        self._owner_embs: list = []
        self.available = False
        try:
            import cv2  # noqa: F401 — presence check only
        except Exception:  # noqa: BLE001
            logger.warning("face-gate: OpenCV unavailable")
            return
        det, rec = models_dir / YUNET, models_dir / SFACE
        if not det.is_file() or not rec.is_file():
            logger.info("face-gate: model files missing under %s — gate disabled", models_dir)
            return
        try:
            self._load_owner_embeddings()
            self.available = bool(self._owner_embs)
            if not self.available:
                logger.warning("face-gate: no usable face in owner_refs/ — gate disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("face-gate: init failed (%s) — gate disabled", exc)

    # ── internals ───────────────────────────────────────────────────────────────

    def _detector(self, w: int, h: int):
        import cv2
        det = cv2.FaceDetectorYN.create(str(self._models_dir / YUNET), "", (w, h),
                                        score_threshold=0.7)
        return det

    def _recognizer(self):
        import cv2
        if self._rec is None:
            self._rec = cv2.FaceRecognizerSF.create(str(self._models_dir / SFACE), "")
        return self._rec

    def _read(self, path: Path):
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = DETECT_MAX_EDGE / max(w, h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    def _faces(self, img):
        h, w = img.shape[:2]
        n, faces = self._detector(w, h).detect(img)
        return faces if faces is not None else []

    def _embed(self, img, face_row):
        rec = self._recognizer()
        aligned = rec.alignCrop(img, face_row)
        return rec.feature(aligned).copy()

    def _load_owner_embeddings(self) -> None:
        for ref in sorted(self._refs_dir.glob("*")):
            if ref.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = self._read(ref)
            if img is None:
                continue
            faces = self._faces(img)
            if len(faces) == 0:
                logger.warning("face-gate: no face found in reference %s", ref.name)
                continue
            largest = max(faces, key=lambda f: f[2] * f[3])
            self._owner_embs.append(self._embed(img, largest))
            logger.info("face-gate: loaded owner reference %s", ref.name)

    def _owner_sim(self, emb) -> float:
        import cv2
        rec = self._recognizer()
        return max(rec.match(o, emb, cv2.FaceRecognizerSF_FR_COSINE)
                   for o in self._owner_embs)

    # ── public API ──────────────────────────────────────────────────────────────

    def owner_main_subject(self, image_path: Path) -> tuple[bool, str] | None:
        """(hit, reason) — or None when the gate can't decide (caller falls back
        to the cloud VLM). hit=True only when a face matches the owner AND is
        prominent in frame."""
        if not self.available:
            return None
        try:
            img = self._read(image_path)
            if img is None:
                return None
            h, w = img.shape[:2]
            faces = self._faces(img)
            if len(faces) == 0:
                return False, "no faces detected"
            best_sim, best_area = 0.0, 0.0
            for f in faces:
                area = (f[2] * f[3]) / float(w * h)
                try:
                    sim = self._owner_sim(self._embed(img, f))
                except Exception:  # noqa: BLE001 — one bad crop is not fatal
                    continue
                if sim > best_sim:
                    best_sim, best_area = sim, area
            if best_sim >= SIM_THRESHOLD and best_area >= MAIN_FACE_AREA:
                return True, f"owner face, sim={best_sim:.2f}, area={best_area:.1%}"
            if best_sim >= SIM_THRESHOLD:
                return False, f"owner only in background (sim={best_sim:.2f}, area={best_area:.1%})"
            return False, f"no owner match (best sim={best_sim:.2f}, {len(faces)} faces)"
        except Exception as exc:  # noqa: BLE001
            logger.warning("face-gate: check failed (%s)", exc)
            return None


_GATE: FaceGate | None = None


def get_gate(cfg) -> FaceGate:
    """Process-wide cached gate (owner embeddings computed once)."""
    global _GATE
    if _GATE is None:
        from ic2x.config import _PROJECT_ROOT
        _GATE = FaceGate(_PROJECT_ROOT / "models" / "face", cfg.owner_refs_dir)
    return _GATE
