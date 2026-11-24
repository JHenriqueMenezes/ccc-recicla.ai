# 02 — Segundo treino (`comb`, dataset `combinado-v1`)

> **Data:** 23 de novembro de 2026
> **Objetivo:** ter um modelo que acerte o material em imagens reais.

## Dataset: `combinado-v1`

Gerado por `scripts/combinar_datasets.py`, a partir de:

- **Recyclable Waste Detection** (prefixo `base_`): Cardboard e Paper → `papel`, Plastic → `plastico`, Glass → `vidro`, Metal → `metal`.
- **Garbage Classification 3** (prefixo `org_`): só imagens em que `BIODEGRADABLE` é a única classe, com até 20 caixas → `organico`.
- **Fundos:** nenhum. A pasta `fundos` estava vazia, e o log do treino registra `0 backgrounds` em train e em valid.

| Split | Imagens (`base_` / `org_`) | Caixas papel | plastico | vidro | metal | organico |
|---|---|---|---|---|---|---|
| train | 6209 (5090 / 1119) | 1945 | 1295 | 1026 | 1047 | 5673 |
| valid | 777 (637 / 140) | 249 | 159 | 128 | 130 | 702 |
| test | 777 (637 / 140) | 253 | 159 | 128 | 130 | 668 |

Um run anterior com esse dataset (`runs/combinado-v1`, de 03/11) tinha parado na época 1 de 100. Ele não foi usado.

## Treino

```bash
yolo detect train data=C:\Users\hique\recicla-dados\combinado-v1\data.yaml model=yolo11n.pt imgsz=640 epochs=60 patience=15 batch=32 scale=0.9 device=0 project=C:\Users\hique\recicla-dados\runs name=comb
```

| Parâmetro | Valor |
|---|---|
| Modelo | YOLO11n (2,59 M parâmetros, 6,5 GFLOPs) |
| `imgsz` | 640 |
| `epochs` / `patience` | 60 / 15 |
| `batch` | 32 |
| `scale` | 0,9 |
| Demais | padrão do Ultralytics 8.4.201 (`optimizer=auto` → AdamW lr=0,001111; `mosaic=1.0`, `close_mosaic=10`, `seed=0`, `deterministic=true`, AMP ligado) |

- **Hardware:** RTX 5060 Ti 16 GB, torch 2.15.0+cu134.
- **Duração:** 60 épocas em 0,474 h (~28 min). O *early stopping* não foi acionado, e a melhor época foi a **60**.
- **Pesos:** `runs/comb/weights/best.pt` (5,5 MB), fora do repositório.
- **Observação de ambiente:** chamado pelo Git Bash, o `yolo` falhou ao carregar `torch\lib\shm.dll` (WinError 4551, bloqueio da política de Controle de Aplicativo do Windows). Chamado pelo PowerShell, funcionou.

## Resultados no dataset

| Classe | P (val) | R (val) | mAP50 (val) | mAP50-95 (val) | P (test) | R (test) | mAP50 (test) | mAP50-95 (test) |
|---|---|---|---|---|---|---|---|---|
| papel | 0,935 | 0,863 | 0,944 | 0,733 | 0,904 | 0,877 | 0,942 | 0,735 |
| plastico | 0,901 | 0,859 | 0,901 | 0,590 | 0,857 | 0,870 | 0,907 | 0,590 |
| vidro | 0,967 | 0,945 | 0,988 | 0,764 | 0,976 | 0,965 | 0,969 | 0,736 |
| metal | 0,975 | 0,992 | 0,994 | 0,793 | 0,928 | 0,946 | 0,977 | 0,754 |
| organico | 0,802 | 0,668 | 0,762 | 0,547 | 0,830 | 0,681 | 0,802 | 0,568 |
| **all** | **0,916** | **0,865** | **0,918** | **0,685** | **0,899** | **0,868** | **0,919** | **0,677** |

## Exportação

`yolo export model=best.pt format=onnx imgsz=640 opset=12` → `best.onnx` (10,2 MB), copiado para `models/best.onnx`. O git não rastreia esse arquivo por causa da regra `*.onnx` do `.gitignore`. Durante a exportação, o Ultralytics instalou `onnx` 1.23.2, `onnxruntime` 1.32.0 e `onnxslim` 0.1.99 no venv.

## Resultado em imagens reais

**Material:** não havia a pasta `fotos-teste`. Foram usados 15 quadros, em intervalos iguais, de `videos/original-20261103-210330.mp4` (1280×720, 221 s). É o mesmo vídeo do primeiro treino: objeto na mão, fundo doméstico. O gabarito foi definido visualmente. O quadro 00 (tampa vermelha) ficou de fora porque não dá para saber se o material é plástico ou metal.

**Inferência:** `models/best.onnx` com `cv2.dnn` (OpenCV 5.0.0, CPU), usando `letterbox` e `postprocess` de `src/detector.py`, NMS IoU 0,45. A classe considerada é a da detecção de maior confiança. Imagens anotadas em `recicla-dados/resultados/conf_025` e `conf_05`.

**Controle da cadeia de inferência:** em 40 imagens do split de teste, a classe de maior confiança estava entre as do rótulo em 38. O `best.pt` rodando pelo Ultralytics deu o mesmo resultado que o ONNX nos 15 quadros (só `papel`, ou nada).

| Quadro | Real | conf 0,25 | conf 0,5 |
|---|---|---|---|
| 01 | plastico (garrafa PET) | papel 0,36 | — |
| 02 | plastico (garrafa PET) | papel 0,31 | — |
| 03 | plastico (garrafa PET) | papel 0,53 | papel 0,53 |
| 04 | sem objeto | papel 0,37 | — |
| 05 | metal (aerossol) | papel 0,48 | — |
| 06 | metal (aerossol) | papel 0,59 | papel 0,59 |
| 07 | metal (cortador de unha) | papel 0,77 | papel 0,77 |
| 08 | metal (cortador de unha) | papel 0,37 | — |
| 09 | metal (arame) | papel 0,36 | — |
| 10 | vidro (garrafa pequena) | papel 0,57 | papel 0,57 |
| 11 | vidro (garrafa pequena) | — | — |
| 12 | sem objeto | papel 0,28 | — |
| 13 | papel (recibo) | papel 0,53 + papel 0,30 | papel 0,53 |
| 14 | sem objeto | papel 0,45 | — |

| Material | conf 0,25: acertou / errou / nada | conf 0,5: acertou / errou / nada |
|---|---|---|
| papel (1) | 1* / 0 / 0 | 1* / 0 / 0 |
| plastico (3) | 0 / 3 / 0 | 0 / 1 / 2 |
| vidro (2) | 0 / 1 / 1 | 0 / 1 / 1 |
| metal (5) | 0 / 5 / 0 | 0 / 2 / 3 |
| sem objeto (3) | 0 ok / 3 falsos positivos | 3 ok / 0 falsos positivos |

\* No quadro 13, as duas caixas `papel` estão sobre as caixas de papelão do fundo, não sobre o recibo na mão. Nenhum objeto segurado na mão foi detectado com a classe correta, em nenhum dos dois limiares.

- Todas as detecções nos 15 quadros são da classe `papel`.
- Nos quadros 01 a 12 e 14, as caixas cobrem de 55% a 95% da área do quadro.

## Tempo de inferência em CPU

`cv2.dnn`, backend OpenCV, target CPU, sem CUDA (o cv2 vê 0 dispositivos CUDA). AMD Ryzen, 16 threads. 20 imagens (5 do split de teste e 15 quadros 1280×720), mais 1 de aquecimento.

| Etapa | Média | Mín. | Máx. |
|---|---|---|---|
| `net.forward()` | 16,5 ms | 15,9 ms | 17,8 ms |
| letterbox + forward + pós-processamento | 19,8 ms (~50 FPS) | — | — |
