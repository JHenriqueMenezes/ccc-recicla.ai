"""Classificação de material com um modelo YOLO-cls (ONNX) executado com cv2.dnn.

O pré-processamento reproduz o do Ultralytics (classify_transforms): o lado menor vai para
`size`, recorte central de size x size, RGB em [0, 1], sem normalização por média e desvio.
"""

from __future__ import annotations

import cv2
import numpy as np

NOMES = ["metal", "organico", "papel", "plastico", "vidro"]
"""Ordem das classes do Ultralytics-cls: alfabética, pelos nomes das pastas do dataset."""

NOMES_COM_NADA = ["metal", "nada", "organico", "papel", "plastico", "vidro"]
"""Modelos treinados com a classe extra "nada" (cena sem resíduo)."""


def preparar(img: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR (H, W, 3) -> blob (1, 3, size, size)."""
    h, w = img.shape[:2]
    escala = size / min(h, w)
    nh, nw = max(size, round(h * escala)), max(size, round(w * escala))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (nh - size) // 2, (nw - size) // 2
    img = img[top:top + size, left:left + size]
    return cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)


class Classificador:
    def __init__(self, modelo_onnx: str, size: int = 224, nomes: list[str] | None = None):
        self.net = cv2.dnn.readNetFromONNX(modelo_onnx)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.size = size
        self.nomes = nomes
        if self.nomes is None:
            n = len(self.probabilidades(np.zeros((size, size, 3), np.uint8)))
            self.nomes = {len(NOMES): NOMES, len(NOMES_COM_NADA): NOMES_COM_NADA}.get(n)
            if self.nomes is None:
                raise ValueError(f"modelo com {n} classes; informe os nomes em `nomes`")

    def probabilidades(self, img: np.ndarray) -> np.ndarray:
        """Probabilidades por classe (a saída exportada do Ultralytics já passa por softmax)."""
        self.net.setInput(preparar(img, self.size))
        return np.squeeze(self.net.forward()).astype(np.float32)

    def classificar(self, img: np.ndarray) -> tuple[str, float]:
        p = self.probabilidades(img)
        i = int(np.argmax(p))
        return self.nomes[i], float(p[i])
