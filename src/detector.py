"""Pré e pós-processamento para modelos YOLO (ONNX) executados com cv2.dnn."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona mantendo proporção e preenche com cinza até size x size."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return canvas, scale, (pad_left, pad_top)


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Converte a saída bruta em [(classe, confiança, (x, y, w, h))] na imagem original."""
    preds = np.squeeze(output, axis=0)

    # v8/v11: (4+nc, N) -> transpõe para (N, 4+nc). v5: (N, 5+nc) com objectness.
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
    else:
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 5:] * preds[:, 4:5]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thr
    boxes_xywh, class_ids, confidences = boxes_xywh[mask], class_ids[mask], confidences[mask]
    if len(confidences) == 0:
        return []

    # centro (cx, cy, w, h) no espaço do letterbox -> canto (x, y, w, h) na imagem original
    pad_x, pad_y = pad
    cx, cy, bw, bh = boxes_xywh.T
    x = (cx - bw / 2 - pad_x) / scale
    y = (cy - bh / 2 - pad_y) / scale
    boxes = np.stack([x, y, bw / scale, bh / scale], axis=1)

    keep = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thr, iou_thr)
    keep = np.array(keep).flatten()

    return [
        (int(class_ids[i]), float(confidences[i]), tuple(int(v) for v in boxes[i]))
        for i in keep
    ]
