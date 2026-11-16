# 01 — Primeiro treino (baseline `gc3-y11n`)

> **Data:** início de novembro de 2026
> **Objetivo:** ter um primeiro modelo, mesmo fraco, para validar o fluxo dataset → treino → inferência e descobrir onde ele falha em imagens reais.

## Dataset

**Fonte:** *Garbage Classification 3* (Roboflow Universe), versão **v2 (GC1)**, formato YOLOv11.

A v1 (17.365 imagens) foi descartada porque tem cópias geradas por giros e espelhamentos. Elas aumentam o volume sem trazer conteúdo novo, e o YOLO já faz esse aumento durante o treino.

### Mapeamento de classes

| Classe original | Classe Recicla.ai | Cor (CONAMA 275/2001) |
|---|---|---|
| CARDBOARD, PAPER | `papel` | azul |
| PLASTIC | `plastico` | vermelho |
| GLASS | `vidro` | verde |
| METAL | `metal` | amarelo |
| BIODEGRADABLE | `organico` | marrom |

### Limpeza (`scripts/preparar_dataset.py`)

O script junta train, valid e test, limpa os dados e refaz a divisão. Os arquivos originais não são alterados.

| Etapa | Imagens |
|---|---|
| Original (train + valid + test) | 10.464 |
| Imagens com mais de 20 caixas | −801 |
| Quase duplicadas (hash perceptual DCT de 64 bits, Hamming ≤ 6, incluindo giros e espelhamentos) | −417 |
| **Final** | **9.246** |

Também foram removidas 17 caixas com área zero, sem esvaziar nenhuma imagem.

**Nova divisão:** 80/10/10, estratificada pela classe principal de cada imagem (a classe com mais caixas), com semente 42.

| Classe | train (imagens / caixas) | valid | test |
|---|---|---|---|
| papel | 2445 / 5413 | 306 / 739 | 303 / 619 |
| plastico | 1060 / 2941 | 136 / 315 | 129 / 346 |
| vidro | 1945 / 4330 | 244 / 512 | 246 / 529 |
| metal | 1396 / 3644 | 169 / 391 | 186 / 427 |
| organico | 1306 / 6747 | 161 / 701 | 162 / 779 |
| **total** | **7394 / 23075** | **926 / 2658** | **926 / 2700** |

Uma imagem pode conter várias classes, por isso as linhas somam mais que o total.

## Treino

```bash
yolo detect train model=yolo11n.pt data=<dataset>/data.yaml imgsz=416 epochs=100 patience=20 batch=16 device=0
```

| Parâmetro | Valor | Motivo |
|---|---|---|
| Modelo | YOLO11n (2,6 M parâmetros, 6,4 GFLOPs) | Versão nano, a menor da família YOLO11 |
| `imgsz` | 416 | As imagens do dataset já vêm nesse tamanho |
| `patience` | 20 | Parar quando a validação estabilizar |

- **Hardware:** treinado em GPU NVIDIA RTX 5060 Ti (16 GB), com CUDA.
- **Duração:** ~35 min. O *early stopping* parou na época 100, e o melhor resultado foi na **época 80**.
- **Velocidade na GPU:** ~0,4 ms de inferência por imagem na validação e ~0,8 ms no teste.
- **Pesos:** `best.pt` (5,5 MB), guardado **fora do repositório**.

## Resultados no dataset

| Classe | P | R | mAP50 (val) | mAP50 (test) | mAP50-95 (test) |
|---|---|---|---|---|---|
| papel | 0,851 | 0,648 | 0,762 | 0,774 | 0,628 |
| plastico | 0,792 | 0,679 | 0,830 | 0,750 | 0,583 |
| vidro | 0,861 | 0,844 | 0,882 | 0,904 | 0,735 |
| metal | 0,821 | 0,739 | 0,821 | 0,833 | 0,644 |
| organico | 0,807 | 0,556 | 0,745 | 0,694 | 0,498 |
| **all** | **0,826** | **0,693** | **0,808** | **0,791** | **0,618** |

P e R são do conjunto de teste.

**Leitura:** validação e teste ficaram próximos, então o modelo não se ajustou demais à validação. O ponto fraco é o *recall* de `papel` e `organico`: o modelo deixa muitos objetos dessas classes passar.

## Resultado em vídeo real (celular)

O modelo **não generalizou** para a cena real, com itens na mão sobre um fundo doméstico.

| Situação | Resultado |
|---|---|
| Sem item (chão, pernas) | ❌ `papel` em tela cheia, com confiança de até 0,98 |
| Garrafa PET | ⚠️ `plastico` em ~60% dos quadros, muitas vezes só na tampa |
| Garrafa de vidro | ❌ não detectou (mesmo com mAP50 de 0,90 no teste) |
| Aerossol | ❌ `metal` em ~2% dos quadros |
| Recibo de papel, tampa, cortador de unha | ❌ não detectou |
| Caixas de papelão ao fundo | ✅ `papel` com confiança de 0,80 a 0,90 |
| Ventilador, lixeira | ❌ falso `metal` |

## Causas prováveis

1. **Caixas de tela cheia:** 24% das caixas de papel/papelão ocupam 70% ou mais da imagem (nas outras classes, de 2% a 5%). O modelo aprendeu que "superfície clara e texturizada = papel".
2. **Nenhuma imagem de fundo** (`0 backgrounds`): o modelo nunca viu uma cena sem lixo.
3. **Diferença de domínio:** no dataset, os objetos aparecem centralizados e em fundo limpo. No vídeo, estão na mão, menores e em ambiente bagunçado.

## Lições

- Métricas altas no dataset **não preveem** o desempenho nas imagens do professor. As duas partes vêm da mesma distribuição.
