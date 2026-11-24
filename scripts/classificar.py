"""Classifica o material de imagens com models/cls.onnx (cv2.dnn, CPU).

Uso:
    python scripts/classificar.py foto.jpg outra.png pasta_com_fotos/
    python scripts/classificar.py --centro 0.6 pasta/     # usa só a região central (60%)
    python scripts/classificar.py --camera 0              # webcam; tecla q para sair
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classificador import Classificador  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def centro(img: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 1:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entradas", nargs="*", type=Path)
    p.add_argument("--modelo", default=str(Path(__file__).resolve().parents[1] / "models" / "cls.onnx"))
    p.add_argument("--centro", type=float, default=1.0, help="Fração central da imagem a classificar (0-1)")
    p.add_argument("--camera", type=int, help="Índice da câmera para classificar ao vivo")
    a = p.parse_args()
    clf = Classificador(a.modelo)

    if a.camera is not None:
        cap = cv2.VideoCapture(a.camera)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            nome, conf = clf.classificar(centro(frame, a.centro))
            h, w = frame.shape[:2]
            if a.centro < 1:
                ch, cw = int(h * a.centro), int(w * a.centro)
                cv2.rectangle(frame, ((w - cw) // 2, (h - ch) // 2), ((w + cw) // 2, (h + ch) // 2), (255, 255, 255), 2)
            cv2.putText(frame, f"{nome} {conf:.2f}", (10, 40), 0, 1.2, (0, 0, 0), 6)
            cv2.putText(frame, f"{nome} {conf:.2f}", (10, 40), 0, 1.2, (0, 255, 0), 2)
            cv2.imshow("Recicla.ai", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    arquivos = []
    for e in a.entradas:
        arquivos += sorted(f for f in e.rglob("*") if f.suffix.lower() in EXT) if e.is_dir() else [e]
    if not arquivos:
        p.error("informe imagens, pastas ou --camera")
    for f in arquivos:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: não consegui ler")
            continue
        probs = clf.probabilidades(centro(img, a.centro))
        ordem = probs.argsort()[::-1]
        resto = ", ".join(f"{clf.nomes[i]} {probs[i]:.2f}" for i in ordem[1:3])
        print(f"{f.name}: {clf.nomes[ordem[0]]} {probs[ordem[0]]:.2f}  ({resto})")


if __name__ == "__main__":
    main()
