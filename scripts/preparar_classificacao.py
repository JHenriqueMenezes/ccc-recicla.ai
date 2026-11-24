"""Monta um dataset de CLASSIFICAÇÃO (formato Ultralytics) para o Recicla.ai.

Fontes (todas opcionais, menos --gd):
    1. --gd: Garbage Dataset (GD), espelho maronnello/waste-garbage-management-dataset no
       Hugging Face (MIT): uma pasta por classe.
         biological=organico, paper+cardboard=papel, plastic=plastico, glass=vidro, metal=metal
         (battery, clothes, shoes e trash são ignoradas.)
    2. --realwaste: RealWaste (UCI, CC BY 4.0), espelho shahzaibvohra/realwaste no Hugging Face:
       arquivos "<Classe>_<n>.jpg" numa pasta só.
         Food Organics+Vegetation=organico, Paper+Cardboard=papel, Plastic=plastico,
         Glass=vidro, Metal=metal (Miscellaneous Trash e Textile Trash são ignoradas.)
    3. --nada: pasta com cenas sem resíduo (ex.: cômodos do MIT Indoor-67) -> classe "nada".
    4. --deteccao: recortes das caixas de um dataset de DETECÇÃO já preparado (ex.:
       combinado-v1), com margem em volta. Cada recorte fica no mesmo split da imagem de origem.

Passos das fontes 1 a 3 (juntas):
    - remove quase duplicadas (pHash DCT de 64 bits, Hamming <= 6, incluindo giros e
      espelhamentos; mesmas funções de scripts/preparar_dataset.py);
    - remove imagens quase iguais às de --excluir (para não vazar o teste);
    - divide 80/10/10 por classe, com semente fixa.
Por fim, --max-treino limita o número de imagens de treino por classe (sorteio fixo).

Saída: <output>/{train,val,test}/<classe>/<fonte>_<nome>.jpg

Uso:
    python scripts/preparar_classificacao.py --gd caminho/gd-hf --realwaste caminho/realwaste-hf \
        --nada caminho/indoor-hf/data --deteccao caminho/combinado-v1 \
        --excluir caminho/fotos-internet/candidatas --output caminho/cls-v2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_dataset import HAMMING_MAX, agrupar_duplicatas, hashes_phash  # noqa: E402

NOMES = ["papel", "plastico", "vidro", "metal", "organico"]
CLASSES = NOMES + ["nada"]
MAPA_GD = {
    "paper": "papel", "cardboard": "papel", "plastic": "plastico",
    "glass": "vidro", "metal": "metal", "biological": "organico",
}
MAPA_RW = {
    "Paper": "papel", "Cardboard": "papel", "Plastic": "plastico", "Glass": "vidro",
    "Metal": "metal", "Food Organics": "organico", "Vegetation": "organico",
}
SPLITS_DET = {"train": "train", "valid": "val", "test": "test"}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gd", type=Path, required=True, help="Pasta do Garbage Dataset (uma subpasta por classe)")
    p.add_argument("--realwaste", type=Path, help="Pasta do RealWaste (arquivos <Classe>_<n>.jpg)")
    p.add_argument("--nada", type=Path, help="Pasta com cenas sem resíduo (classe 'nada')")
    p.add_argument("--max-treino", type=int, default=0, help="Máximo de imagens de treino por classe (0 = sem limite)")
    p.add_argument("--deteccao", type=Path, help="Dataset de detecção YOLO já com as 5 classes do Recicla.ai")
    p.add_argument("--excluir", type=Path, nargs="*", default=[], help="Pastas de imagens de teste a manter fora")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margem", type=float, default=0.15, help="Margem do recorte, fração do lado da caixa")
    p.add_argument("--min-lado", type=int, default=48, help="Descarta recortes menores que isso (px)")
    p.add_argument("--max-recortes", type=int, default=3, help="Máximo de recortes por imagem (os maiores)")
    return p.parse_args()


def listar(pasta: Path) -> list[Path]:
    return sorted(f for f in pasta.rglob("*") if f.suffix.lower() in EXT)


def ler(caminho: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)


def gravar(caminho: Path, img: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(caminho))


def coletar_web(args, stats: Counter) -> list[tuple[Path, str, str]]:
    """(arquivo, classe, fonte) de todas as fontes de imagens inteiras."""
    itens: list[tuple[Path, str, str]] = []
    for pasta, classe in MAPA_GD.items():
        fs = listar(args.gd / pasta)
        stats[f"gd: {pasta} -> {classe}"] = len(fs)
        itens += [(f, classe, "gd") for f in fs]
    if args.realwaste:
        for f in listar(args.realwaste):
            origem = f.stem.rsplit("_", 1)[0]
            if origem in MAPA_RW:
                stats[f"rw: {origem} -> {MAPA_RW[origem]}"] += 1
                itens.append((f, MAPA_RW[origem], "rw"))
    if args.nada:
        fs = listar(args.nada)
        stats["nada: cenas"] = len(fs)
        itens += [(f, "nada", "in") for f in fs]
    return itens


def preparar_gd(args, stats: Counter) -> dict[str, list[tuple[Path, str, str]]]:
    itens = coletar_web(args, stats)
    print(f"[web] {len(itens)} imagens nas classes usadas")

    print("[web] calculando hashes perceptuais...")
    hashes, validos = [], []
    for f, c, fonte in itens:
        h = hashes_phash(f)
        if h is None:
            stats[f"{fonte}: ilegível"] += 1
            continue
        hashes.append(h)
        validos.append((f, c, fonte))

    excl = []
    for pasta in args.excluir:
        for f in listar(pasta):
            h = hashes_phash(f)
            if h is not None:
                excl.append(h)
    n_gd = len(validos)
    todos = np.stack(hashes + excl)
    raizes = agrupar_duplicatas(todos, HAMMING_MAX)

    raizes_teste = {raizes[i] for i in range(n_gd, len(todos))}
    grupos: dict[int, list[int]] = defaultdict(list)
    for i in range(n_gd):
        if raizes[i] in raizes_teste:
            stats[f"{validos[i][2]}: parecida com imagem de --excluir"] += 1
            continue
        grupos[raizes[i]].append(i)

    # uma imagem por grupo; descarta grupos com classes conflitantes
    escolhidos: list[tuple[Path, str, str]] = []
    for membros in grupos.values():
        classes = {validos[i][1] for i in membros}
        if len(classes) > 1:
            stats["web: duplicata com classes diferentes (grupo descartado)"] += len(membros)
            continue
        stats["web: quase duplicada removida"] += len(membros) - 1
        escolhidos.append(validos[membros[0]])

    rng = random.Random(args.seed)
    por_classe: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for item in escolhidos:
        por_classe[item[1]].append(item)
    divisao: dict[str, list[tuple[Path, str, str]]] = {"train": [], "val": [], "test": []}
    for c, lista in sorted(por_classe.items()):
        lista.sort(key=lambda x: str(x[0]))
        rng.shuffle(lista)
        n = len(lista)
        n_val = n_test = round(n * 0.1)
        divisao["test"] += lista[:n_test]
        divisao["val"] += lista[n_test:n_test + n_val]
        divisao["train"] += lista[n_test + n_val:]
    return divisao


def recortes_deteccao(args, stats: Counter):
    """Gera (split, classe, nome, recorte) a partir do dataset de detecção."""
    for split_det, split in SPLITS_DET.items():
        pasta_img = args.deteccao / split_det / "images"
        pasta_lbl = args.deteccao / split_det / "labels"
        if not pasta_img.is_dir():
            continue
        for f in listar(pasta_img):
            lbl = pasta_lbl / (f.stem + ".txt")
            if not lbl.exists():
                continue
            caixas = []
            for linha in lbl.read_text().splitlines():
                partes = linha.split()
                if len(partes) == 5:
                    c, cx, cy, w, h = int(partes[0]), *map(float, partes[1:])
                    caixas.append((w * h, c, cx, cy, w, h))
            if not caixas:
                continue
            img = ler(f)
            if img is None:
                stats["recorte: imagem ilegível"] += 1
                continue
            H, W = img.shape[:2]
            caixas.sort(reverse=True)
            for k, (_, c, cx, cy, w, h) in enumerate(caixas[: args.max_recortes]):
                bw, bh = w * W, h * H
                x0 = int(max(0, (cx * W) - bw / 2 - args.margem * bw))
                y0 = int(max(0, (cy * H) - bh / 2 - args.margem * bh))
                x1 = int(min(W, (cx * W) + bw / 2 + args.margem * bw))
                y1 = int(min(H, (cy * H) + bh / 2 + args.margem * bh))
                if min(x1 - x0, y1 - y0) < args.min_lado:
                    stats["recorte: menor que --min-lado"] += 1
                    continue
                yield split, NOMES[c], f"{f.stem}_{k}", f, (x0, y0, x1, y1)
            stats["recorte: além de --max-recortes"] += max(0, len(caixas) - args.max_recortes)


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        sys.exit(f"{args.output} já existe e não está vazia; escolha outra pasta.")
    stats: Counter = Counter()
    contagem: Counter = Counter()

    divisao = preparar_gd(args, stats)
    # (split, classe, fonte, arquivo, caixa de recorte ou None, nome de saída)
    saida: list[tuple[str, str, str, Path, tuple[int, int, int, int] | None, str]] = []
    for split, lista in divisao.items():
        for f, c, fonte in lista:
            nome = f"{fonte}_{f.parent.name}_{f.stem}".replace(" ", "-")
            saida.append((split, c, fonte, f, None, nome))

    if args.deteccao:
        print("[det] listando recortes...")
        for split, c, nome, f, caixa in recortes_deteccao(args, stats):
            saida.append((split, c, "det", f, caixa, f"det_{nome}"))

    if args.max_treino:
        rng = random.Random(args.seed)
        treino: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(saida):
            if item[0] == "train":
                treino[item[1]].append(i)
        fora: set[int] = set()
        for c, idx in treino.items():
            if len(idx) > args.max_treino:
                rng.shuffle(idx)
                fora.update(idx[args.max_treino:])
                stats[f"treino: {c} acima de --max-treino"] = len(idx) - args.max_treino
        saida = [item for i, item in enumerate(saida) if i not in fora]

    print("[saída] gravando...")
    fontes = sorted({item[2] for item in saida})
    for split, c, fonte, f, caixa, nome in saida:
        destino = args.output / split / c / f"{nome}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if caixa is not None:
            x0, y0, x1, y1 = caixa
            gravar(destino, ler(f)[y0:y1, x0:x1])
        elif f.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(f, destino)
        else:
            gravar(destino, ler(f))
        contagem[(split, c, fonte)] += 1

    print("\nRemovidas / ignoradas:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    rot = "/".join(fontes)
    print(f"\n{'classe':10s} " + " ".join(f"{s + ' ' + rot:>26s}" for s in ("train", "val", "test")))
    for c in CLASSES:
        if any(contagem[(s, c, fo)] for s in ("train", "val", "test") for fo in fontes):
            print(f"{c:10s} " + " ".join(
                f"{'/'.join(str(contagem[(s, c, fo)]) for fo in fontes):>26s}" for s in ("train", "val", "test")))
    total = {s: sum(v for (sp, _, _), v in contagem.items() if sp == s) for s in ("train", "val", "test")}
    print(f"{'total':10s} " + " ".join(f"{total[s]:>26d}" for s in ("train", "val", "test")))


if __name__ == "__main__":
    main()
