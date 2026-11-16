"""Combina o Recyclable Waste Detection, a classe orgânica do GARBAGE CLASSIFICATION 3
e imagens de fundo em um único dataset YOLO para o Recicla.ai.

Uso:
    python scripts/combinar_datasets.py --base caminho/recyclable-waste \
        --organico caminho/garbage-3 --backgrounds caminho/fundos --output caminho/saida

Etapas:
    1. Converte polígonos em caixas nos dois datasets (feito na leitura dos rótulos).
    2. Base: Cardboard e Paper=papel, Plastic=plastico, Glass=vidro, Metal=metal.
       Descarta imagens sem rótulo e com mais de uma classe.
    3. Orgânico: só imagens cuja única classe é BIODEGRADABLE, com até 20 caixas e
       nenhuma caixa de área zero.
    4. Remove imagens com alguma caixa de área >= --max-box-area (fração da imagem).
    5. Remove quase duplicatas entre todas as imagens (pHash de 64 bits, giros e
       espelhamentos, Hamming <= 6).
    6. Adiciona fundos com rótulo vazio, até 10% do total final.
    7. Divide 80/10/10 estratificado pela classe principal e gera data.yaml.

Fundos (--backgrounds): cada arquivo de vídeo é amostrado a cada --frame-step quadros
e forma um grupo; imagens dentro de uma subpasta formam um grupo (os quadros de um
vídeo já extraídos); imagens soltas na raiz são grupos de uma imagem só. Um grupo
nunca é dividido entre train, valid e test.

Os arquivos de entrada só são lidos; tudo é gravado em --output.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preparar_dataset import (  # noqa: E402
    CLASSES,
    EXTENSOES_IMAGEM,
    HAMMING_MAX,
    PROPORCOES,
    Amostra,
    agrupar_duplicatas,
    carregar,
    dividir,
    gravar,
    hashes_phash,
    hashes_phash_array,
    imprimir_resumo,
    ler_nomes_originais,
    ler_quadro,
)

MAPA_BASE = {
    "Cardboard": "papel",
    "Paper": "papel",
    "Plastic": "plastico",
    "Glass": "vidro",
    "Metal": "metal",
}
CLASSE_ORGANICO = "BIODEGRADABLE"
MAX_CAIXAS_ORGANICO = 20
FRACAO_FUNDOS = 0.10
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combina datasets para o Recicla.ai")
    p.add_argument("--base", required=True, help="Recyclable Waste Detection (YOLO do Roboflow)")
    p.add_argument("--organico", required=True, help="GARBAGE CLASSIFICATION 3 (YOLO do Roboflow)")
    p.add_argument("--backgrounds", required=True, help="Pasta com fundos (vídeos e/ou imagens)")
    p.add_argument("--output", required=True, help="Pasta de saída (não pode existir com conteúdo)")
    p.add_argument("--max-box-area", type=float, default=0.8,
                   help="Remove imagens com caixa de área >= este valor (fração da imagem)")
    p.add_argument("--frame-step", type=int, default=15,
                   help="Amostra 1 quadro a cada N nos vídeos de fundo")
    p.add_argument("--seed", type=int, default=42, help="Semente das amostragens e da divisão")
    return p.parse_args()


def mapa_ids(input_dir: Path, mapa: dict[str, str]) -> dict[int, int]:
    """id original -> id final, só para as classes presentes em `mapa`."""
    nomes = ler_nomes_originais(input_dir / "data.yaml")
    faltando = [n for n in mapa if n not in nomes]
    if faltando:
        raise SystemExit(f"Classes não encontradas em {input_dir / 'data.yaml'}: {faltando}")
    return {i: CLASSES.index(mapa[n]) for i, n in enumerate(nomes) if n in mapa}


def contar(removidas: list[tuple[str, int]], etapa: str, antes: int, depois: int) -> None:
    removidas.append((etapa, antes - depois))


def carregar_base(input_dir: Path, stats: Counter, removidas: list) -> list[Amostra]:
    ids = mapa_ids(input_dir, MAPA_BASE)
    amostras = carregar(input_dir, stats)
    total = len(amostras) + stats["img_sem_arquivo_rotulo"]
    print(f"[base] {total} imagens")

    for a in amostras:
        validas = [c for c in a.caixas if c[3] > 0 and c[4] > 0]
        stats["caixas_area_zero"] += len(a.caixas) - len(validas)
        a.caixas = validas
    antes = total
    amostras = [a for a in amostras if a.caixas]
    contar(removidas, "base: sem rótulo", antes, len(amostras))

    antes = len(amostras)
    amostras = [a for a in amostras if len({c for c, *_ in a.caixas}) == 1]
    contar(removidas, "base: mais de uma classe", antes, len(amostras))

    antes = len(amostras)
    amostras = [a for a in amostras if a.caixas[0][0] in ids]
    contar(removidas, "base: classe fora do mapeamento", antes, len(amostras))

    for a in amostras:
        a.caixas = [(ids[c], cx, cy, w, h) for c, cx, cy, w, h in a.caixas]
        a.nome_saida = f"base_{a.split_original}_{a.imagem.stem}"
    return amostras


def carregar_organico(input_dir: Path, stats: Counter, removidas: list) -> list[Amostra]:
    ids = mapa_ids(input_dir, {CLASSE_ORGANICO: "organico"})
    (id_bio,) = ids
    amostras = carregar(input_dir, stats)
    total = len(amostras) + stats["img_sem_arquivo_rotulo"]
    print(f"[organico] {total} imagens")

    antes = total
    amostras = [a for a in amostras if a.caixas and all(c == id_bio for c, *_ in a.caixas)]
    contar(removidas, f"orgânico: sem {CLASSE_ORGANICO} como única classe", antes, len(amostras))

    antes = len(amostras)
    amostras = [a for a in amostras if len(a.caixas) <= MAX_CAIXAS_ORGANICO]
    contar(removidas, f"orgânico: mais de {MAX_CAIXAS_ORGANICO} caixas", antes, len(amostras))

    antes = len(amostras)
    amostras = [a for a in amostras if all(w > 0 and h > 0 for _, _, _, w, h in a.caixas)]
    contar(removidas, "orgânico: com caixa de área zero", antes, len(amostras))

    for a in amostras:
        a.caixas = [(ids[c], cx, cy, w, h) for c, cx, cy, w, h in a.caixas]
        a.nome_saida = f"org_{a.split_original}_{a.imagem.stem}"
    return amostras


def candidatos_fundo(pasta: Path, frame_step: int) -> list[tuple[str, Amostra, np.ndarray]]:
    """Lista (grupo, amostra sem caixas, hashes) de todos os fundos candidatos."""
    candidatos = []
    for arq in sorted(p for p in pasta.rglob("*") if p.is_file()):
        rel = arq.relative_to(pasta)
        sufixo = arq.suffix.lower()
        if sufixo in EXTENSOES_VIDEO:
            cap = cv2.VideoCapture(str(arq))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for i in range(0, n, frame_step):
                quadro = ler_quadro(arq, i)
                if quadro is None:
                    continue
                h = hashes_phash_array(cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY))
                nome = f"fundo_{'_'.join(rel.with_suffix('').parts)}_q{i:06d}"
                candidatos.append((str(rel), Amostra(arq, "fundo", [], nome, quadro=i), h))
        elif sufixo in EXTENSOES_IMAGEM:
            h = hashes_phash(arq)
            if h is None:
                continue
            grupo = str(rel.parent) if rel.parent != Path(".") else str(rel)
            nome = f"fundo_{'_'.join(rel.with_suffix('').parts)}"
            candidatos.append((grupo, Amostra(arq, "fundo", [], nome), h))
    return candidatos


def dividir_grupos(grupos: dict[str, list[Amostra]], seed: int) -> dict[str, list[Amostra]]:
    """Distribui grupos inteiros entre as divisões, buscando 80/10/10 em número de imagens."""
    ordem = sorted(grupos)
    random.Random(seed).shuffle(ordem)
    total = sum(len(v) for v in grupos.values())
    alvo = dict(zip(("train", "valid", "test"), (p * total for p in PROPORCOES)))
    divisao = {"train": [], "valid": [], "test": []}
    # Grupos maiores primeiro, cada um na divisão mais longe da sua meta
    for g in sorted(ordem, key=lambda g: -len(grupos[g])):
        split = max(divisao, key=lambda s: alvo[s] - len(divisao[s]))
        divisao[split] += grupos[g]
    return divisao


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base).resolve()
    org_dir = Path(args.organico).resolve()
    fundos_dir = Path(args.backgrounds).resolve()
    output = Path(args.output).resolve()

    for d in (base_dir, org_dir):
        if not (d / "data.yaml").is_file():
            raise SystemExit(f"data.yaml não encontrado em {d}")
    if not fundos_dir.is_dir():
        raise SystemExit(f"Pasta de fundos não encontrada: {fundos_dir}")
    for d in (base_dir, org_dir, fundos_dir):
        if output == d or d in output.parents:
            raise SystemExit(f"--output não pode ficar dentro de {d}")
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise SystemExit("--output não pode ficar dentro do repositório")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"A pasta de saída já existe e não está vazia: {output}")

    stats: Counter = Counter()
    removidas: list[tuple[str, int]] = []

    # 1-3. Leitura (polígonos viram caixas), mapeamento e filtros de cada dataset
    stats_base, stats_org = Counter(), Counter()
    base = carregar_base(base_dir, stats_base, removidas)
    organico = carregar_organico(org_dir, stats_org, removidas)
    stats.update(stats_base)
    stats.update(stats_org)
    total_inicial = (len(base) + len(organico) + sum(qtd for _, qtd in removidas))
    amostras = base + organico
    print(f"[1-3] {len(base)} imagens da base + {len(organico)} do orgânico")

    # 4. Caixas grandes demais
    antes = len(amostras)
    amostras = [a for a in amostras if all(w * h < args.max_box_area for *_, w, h in a.caixas)]
    contar(removidas, f"caixa com área >= {args.max_box_area}", antes, len(amostras))
    print(f"[4] {len(amostras)} imagens")

    # 5. Quase duplicatas entre todas as imagens
    print("[5] calculando hashes perceptuais...")
    hashes, legiveis = [], []
    for a in amostras:
        h = hashes_phash(a.imagem)
        if h is not None:
            hashes.append(h)
            legiveis.append(a)
    contar(removidas, "imagem ilegível", len(amostras), len(legiveis))
    amostras = legiveis
    raizes = agrupar_duplicatas(np.stack(hashes), HAMMING_MAX)
    vistos: set[int] = set()
    unicas, hashes_unicas = [], []
    for a, r, h in zip(amostras, raizes, hashes):
        if r not in vistos:
            vistos.add(r)
            unicas.append(a)
            hashes_unicas.append(h)
    contar(removidas, f"quase duplicadas (Hamming <= {HAMMING_MAX})", len(amostras), len(unicas))
    amostras = unicas
    print(f"[5] {len(amostras)} imagens únicas")

    # 6. Fundos: sem duplicatas entre si nem com as imagens rotuladas
    print("[6] lendo fundos...")
    candidatos = candidatos_fundo(fundos_dir, args.frame_step)
    fundos_por_grupo: dict[str, list[Amostra]] = defaultdict(list)
    if candidatos:
        todos = np.stack(hashes_unicas + [h for *_, h in candidatos])
        raizes = agrupar_duplicatas(todos, HAMMING_MAX)
        n_rot = len(amostras)
        grupos_com_rotulo = set(raizes[:n_rot])
        vistos = set()
        sem_dup = []
        for (grupo, a, _), r in zip(candidatos, raizes[n_rot:]):
            if r not in grupos_com_rotulo and r not in vistos:
                vistos.add(r)
                sem_dup.append((grupo, a))
        limite = int(len(amostras) * FRACAO_FUNDOS / (1 - FRACAO_FUNDOS))
        rng = random.Random(args.seed)
        escolhidos = sem_dup if len(sem_dup) <= limite else rng.sample(sem_dup, limite)
        for grupo, a in escolhidos:
            fundos_por_grupo[grupo].append(a)
        print(f"[6] fundos: {len(candidatos)} candidatos, {len(sem_dup)} sem duplicatas, "
              f"{len(escolhidos)} usados (limite {limite}) em {len(fundos_por_grupo)} grupo(s)")
    else:
        print("[6] nenhum fundo encontrado")

    # 7. Divisão e gravação
    divisao = dividir(amostras, args.seed)
    for split, lista in dividir_grupos(fundos_por_grupo, args.seed).items():
        divisao[split] += lista
    print(f"[7] divisão 80/10/10 estratificada (seed={args.seed})")
    gravar(output, divisao, origem="scripts/combinar_datasets.py")
    print(f"[7] dataset gravado em {output}")

    imprimir_resumo(divisao, removidas, stats, total_inicial)
    fundos = {s: sum(1 for a in v if not a.caixas) for s, v in divisao.items()}
    print(f"{'fundos':<10}" + "".join(f"{n:>9} /{0:>7}" for n in fundos.values()))
    print(f"Fundos: {sum(fundos.values())} de {sum(len(v) for v in divisao.values())} imagens "
          f"({sum(fundos.values()) / max(1, sum(len(v) for v in divisao.values())):.1%})")


if __name__ == "__main__":
    main()
