"""Prepara um dataset YOLO exportado do Roboflow para o treino do Recicla.ai.

Uso:
    python scripts/preparar_dataset.py --input caminho/dataset-roboflow --output caminho/saida

Etapas:
    1. Junta train, valid e test.
    2. Remove caixas com área zero e imagens com mais de 20 caixas.
    3. Mapeia as classes originais para papel, plastico, vidro, metal e organico.
    4. Agrupa imagens quase iguais (hash perceptual DCT de 64 bits, considerando
       giros de 90° e espelhamentos, distância de Hamming <= 6) e mantém uma por grupo.
    5. Refaz a divisão 80/10/10, estratificada pela classe principal de cada imagem.
    6. Gera data.yaml com caminhos relativos.

Os arquivos de --input só são lidos; tudo é gravado em --output.
Rótulos em formato de polígono (segmentação) são convertidos na caixa que os envolve.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SPLITS_ORIGINAIS = ("train", "valid", "test")
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ordem final das classes no data.yaml
CLASSES = ["papel", "plastico", "vidro", "metal", "organico"]

MAPA_CLASSES = {
    "BIODEGRADABLE": "organico",
    "CARDBOARD": "papel",
    "PAPER": "papel",
    "GLASS": "vidro",
    "METAL": "metal",
    "PLASTIC": "plastico",
}

MAX_CAIXAS = 20
HAMMING_MAX = 6
PROPORCOES = (0.8, 0.1, 0.1)  # train, valid, test


@dataclass
class Amostra:
    imagem: Path
    split_original: str
    # (classe_original, cx, cy, w, h)
    caixas: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    nome_saida: str = ""
    # Índice do quadro quando `imagem` é um vídeo (usado por combinar_datasets.py)
    quadro: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara dataset YOLO do Roboflow")
    p.add_argument("--input", required=True, help="Pasta do dataset exportado do Roboflow")
    p.add_argument("--output", required=True, help="Pasta de saída (não pode existir com conteúdo)")
    p.add_argument("--seed", type=int, default=42, help="Semente da nova divisão")
    return p.parse_args()


def ler_nomes_originais(data_yaml: Path) -> list[str]:
    """Lê a lista `names` do data.yaml do Roboflow (formato de lista em uma linha)."""
    for linha in data_yaml.read_text(encoding="utf-8").splitlines():
        if linha.strip().startswith("names:"):
            conteudo = linha.split(":", 1)[1].strip().strip("[]")
            return [n.strip().strip("'\"") for n in conteudo.split(",") if n.strip()]
    raise SystemExit(f"Não encontrei 'names' em {data_yaml}")


def ler_rotulo(caminho: Path, stats: Counter) -> list[tuple[int, float, float, float, float]]:
    caixas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if not partes:
            continue
        valores = [float(v) for v in partes[1:]]
        cls = int(partes[0])
        if len(valores) == 4:
            cx, cy, w, h = valores
        elif len(valores) >= 6 and len(valores) % 2 == 0:
            xs, ys = valores[0::2], valores[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            stats["poligonos_convertidos"] += 1
        else:
            stats["linhas_invalidas"] += 1
            continue
        caixas.append((cls, cx, cy, w, h))
    return caixas


def carregar(input_dir: Path, stats: Counter) -> list[Amostra]:
    amostras = []
    for split in SPLITS_ORIGINAIS:
        pasta_img = input_dir / split / "images"
        if not pasta_img.is_dir():
            continue
        for img in sorted(pasta_img.iterdir()):
            if img.suffix.lower() not in EXTENSOES_IMAGEM:
                continue
            rotulo = input_dir / split / "labels" / f"{img.stem}.txt"
            if not rotulo.is_file():
                stats["img_sem_arquivo_rotulo"] += 1
                continue
            amostras.append(Amostra(img, split, ler_rotulo(rotulo, stats)))
    return amostras


# ---------------------------------------------------------------- hash perceptual

def _bits_para_int(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits.flatten()), 2))


def hashes_phash(imagem: Path) -> np.ndarray | None:
    """Retorna os 8 pHashes (64 bits, DCT) da imagem nas 4 rotações x espelhamento.

    O índice 0 é a imagem original.
    """
    dados = np.fromfile(str(imagem), dtype=np.uint8)
    cinza = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
    if cinza is None:
        return None
    return hashes_phash_array(cinza)


def hashes_phash_array(cinza: np.ndarray) -> np.ndarray:
    """Mesmo que `hashes_phash`, a partir de uma imagem em tons de cinza já carregada."""
    pequena = cv2.resize(cinza, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    hashes = []
    for espelhar in (False, True):
        base = np.fliplr(pequena) if espelhar else pequena
        for k in range(4):
            variante = np.ascontiguousarray(np.rot90(base, k))
            baixa = cv2.dct(variante)[:8, :8]
            hashes.append(_bits_para_int(baixa > np.median(baixa)))
    return np.array(hashes, dtype=np.uint64)


def agrupar_duplicatas(hashes: np.ndarray, limite: int) -> list[int]:
    """Union-find sobre pares com distância de Hamming <= limite. Retorna a raiz de cada item."""
    n = len(hashes)
    pai = list(range(n))

    def raiz(i: int) -> int:
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    originais = hashes[:, 0]
    bloco = 128
    for ini in range(0, n, bloco):
        fim = min(ini + bloco, n)
        # (bloco, n, 8): original de cada linha contra as 8 variantes de todas as imagens
        dist = np.bitwise_count(originais[ini:fim, None, None] ^ hashes[None, :, :]).min(axis=2)
        linhas, colunas = np.nonzero(dist <= limite)
        for a, b in zip(linhas + ini, colunas):
            if a < b:
                ra, rb = raiz(int(a)), raiz(int(b))
                if ra != rb:
                    pai[max(ra, rb)] = min(ra, rb)
    return [raiz(i) for i in range(n)]


# ---------------------------------------------------------------- divisão

def classe_principal(caixas) -> int:
    """Classe com mais caixas; empate decidido pela maior área total e depois pela ordem."""
    contagem = Counter(c for c, *_ in caixas)
    area = defaultdict(float)
    for c, _, _, w, h in caixas:
        area[c] += w * h
    return max(contagem, key=lambda c: (contagem[c], area[c], -c))


def dividir(amostras: list[Amostra], seed: int) -> dict[str, list[Amostra]]:
    por_classe: dict[int, list[Amostra]] = defaultdict(list)
    for a in amostras:
        por_classe[classe_principal(a.caixas)].append(a)

    rng = random.Random(seed)
    divisao = {"train": [], "valid": [], "test": []}
    for cls in sorted(por_classe):
        grupo = por_classe[cls]
        rng.shuffle(grupo)
        n_valid = round(len(grupo) * PROPORCOES[1])
        n_test = round(len(grupo) * PROPORCOES[2])
        divisao["valid"] += grupo[:n_valid]
        divisao["test"] += grupo[n_valid:n_valid + n_test]
        divisao["train"] += grupo[n_valid + n_test:]
    return divisao


# ---------------------------------------------------------------- saída

def ler_quadro(video: Path, indice: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, quadro = cap.read()
    cap.release()
    return quadro if ok else None


def gravar(output: Path, divisao: dict[str, list[Amostra]],
           origem: str = "scripts/preparar_dataset.py") -> None:
    for split, amostras in divisao.items():
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
        for a in amostras:
            if a.quadro is None:
                shutil.copy2(a.imagem, output / split / "images" / f"{a.nome_saida}{a.imagem.suffix}")
            else:
                quadro = ler_quadro(a.imagem, a.quadro)
                if quadro is None:
                    raise SystemExit(f"Não consegui ler o quadro {a.quadro} de {a.imagem}")
                cv2.imwrite(str(output / split / "images" / f"{a.nome_saida}.jpg"), quadro)
            linhas = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in a.caixas]
            (output / split / "labels" / f"{a.nome_saida}.txt").write_text(
                "".join(f"{l}\n" for l in linhas), encoding="utf-8"
            )

    yaml = [
        f"# Gerado por {origem}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASSES)}",
        f"names: [{', '.join(repr(c) for c in CLASSES)}]",
        "",
    ]
    (output / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")


def imprimir_resumo(divisao: dict[str, list[Amostra]], removidas: list[tuple[str, int]],
                    stats: Counter, total_inicial: int) -> None:
    print("\n=== Remoções ===")
    print(f"Imagens no dataset original: {total_inicial}")
    for etapa, qtd in removidas:
        print(f"  - {etapa:<44} {qtd:>6}")
    print(f"Imagens finais: {sum(len(v) for v in divisao.values())}")
    print(f"Caixas com área zero removidas: {stats['caixas_area_zero']}")
    print(f"Polígonos convertidos em caixas: {stats['poligonos_convertidos']}")
    if stats["linhas_invalidas"]:
        print(f"Linhas de rótulo inválidas ignoradas: {stats['linhas_invalidas']}")

    print("\n=== Divisão final (imagens que contêm a classe / caixas) ===")
    cab = f"{'classe':<10}" + "".join(f"{s:>18}" for s in divisao)
    print(cab)
    print("-" * len(cab))
    for i, nome in enumerate(CLASSES):
        linha = f"{nome:<10}"
        for amostras in divisao.values():
            imgs = sum(1 for a in amostras if any(c == i for c, *_ in a.caixas))
            caixas = sum(1 for a in amostras for c, *_ in a.caixas if c == i)
            linha += f"{imgs:>9} /{caixas:>7}"
        print(linha)
    linha = f"{'total':<10}"
    for amostras in divisao.values():
        linha += f"{len(amostras):>9} /{sum(len(a.caixas) for a in amostras):>7}"
    print("-" * len(cab))
    print(linha)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output = Path(args.output).resolve()

    if not (input_dir / "data.yaml").is_file():
        raise SystemExit(f"data.yaml não encontrado em {input_dir}")
    if output == input_dir or input_dir in output.parents:
        raise SystemExit("--output não pode ficar dentro de --input")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"A pasta de saída já existe e não está vazia: {output}")

    nomes_originais = ler_nomes_originais(input_dir / "data.yaml")
    faltando = [n for n in nomes_originais if n not in MAPA_CLASSES]
    if faltando:
        raise SystemExit(f"Classes sem mapeamento: {faltando}")
    novo_id = {i: CLASSES.index(MAPA_CLASSES[n]) for i, n in enumerate(nomes_originais)}

    stats: Counter = Counter()
    removidas: list[tuple[str, int]] = []

    # 1. Juntar train, valid e test
    amostras = carregar(input_dir, stats)
    total_inicial = len(amostras) + stats["img_sem_arquivo_rotulo"]
    if stats["img_sem_arquivo_rotulo"]:
        removidas.append(("sem arquivo de rótulo", stats["img_sem_arquivo_rotulo"]))
    print(f"[1] {len(amostras)} imagens com rótulo em {', '.join(SPLITS_ORIGINAIS)}")

    # 2. Caixas com área zero e imagens com mais de MAX_CAIXAS caixas
    for a in amostras:
        validas = [c for c in a.caixas if c[3] > 0 and c[4] > 0]
        stats["caixas_area_zero"] += len(a.caixas) - len(validas)
        a.caixas = validas
    antes = len(amostras)
    amostras = [a for a in amostras if a.caixas]
    removidas.append(("sem caixas (vazias ou só com área zero)", antes - len(amostras)))
    antes = len(amostras)
    amostras = [a for a in amostras if len(a.caixas) <= MAX_CAIXAS]
    removidas.append((f"mais de {MAX_CAIXAS} caixas", antes - len(amostras)))
    print(f"[2] {len(amostras)} imagens após a limpeza das caixas")

    # 3. Mapear classes
    for a in amostras:
        a.caixas = [(novo_id[c], cx, cy, w, h) for c, cx, cy, w, h in a.caixas]
    print("[3] classes mapeadas: " + ", ".join(f"{k}->{v}" for k, v in MAPA_CLASSES.items()))

    # 4. Duplicatas quase iguais
    print("[4] calculando hashes perceptuais...")
    hashes, legiveis = [], []
    for a in amostras:
        h = hashes_phash(a.imagem)
        if h is not None:
            hashes.append(h)
            legiveis.append(a)
    if len(legiveis) < len(amostras):
        removidas.append(("imagem ilegível", len(amostras) - len(legiveis)))
    amostras = legiveis
    raizes = agrupar_duplicatas(np.stack(hashes), HAMMING_MAX)
    # Mantém a primeira imagem de cada grupo (ordem: train, valid, test; nome do arquivo)
    vistos: set[int] = set()
    unicas = []
    for a, r in zip(amostras, raizes):
        if r not in vistos:
            vistos.add(r)
            unicas.append(a)
    removidas.append((f"quase duplicadas (Hamming <= {HAMMING_MAX})", len(amostras) - len(unicas)))
    amostras = unicas
    print(f"[4] {len(amostras)} imagens únicas")

    # Nomes de saída únicos (o mesmo nome pode existir em splits diferentes)
    usados: Counter = Counter(a.imagem.stem for a in amostras)
    for a in amostras:
        a.nome_saida = a.imagem.stem if usados[a.imagem.stem] == 1 else f"{a.split_original}_{a.imagem.stem}"

    # 5. Nova divisão estratificada
    divisao = dividir(amostras, args.seed)
    print(f"[5] divisão 80/10/10 estratificada (seed={args.seed})")

    # 6. Gravar imagens, rótulos e data.yaml
    gravar(output, divisao)
    print(f"[6] dataset gravado em {output}")

    # 7. Resumo
    imprimir_resumo(divisao, removidas, stats, total_inicial)


if __name__ == "__main__":
    main()
