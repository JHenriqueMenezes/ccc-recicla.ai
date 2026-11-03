# Recicla.ai

Detecção de resíduos recicláveis com visão computacional. O projeto usa modelos
YOLO leves (nano) exportados para ONNX e executados com OpenCV (`cv2.dnn`), com
foco em rodar bem em CPU, sem depender de GPU.

O app detecta os materiais presentes na imagem, indica se cada um é reciclável e
destaca cada item com a cor da coleta seletiva correspondente, conforme a
[Resolução CONAMA nº 275/2001](https://conama.mma.gov.br/) (por exemplo: azul
para papel, vermelho para plástico, verde para vidro e amarelo para metal).

## 🎥 Vídeo de demonstração

Ainda não

## Estrutura

```
ccc-recicla.ai/
├── src/          # Código-fonte do projeto
├── scripts/      # Scripts utilitários (treino, exportação, etc.)
├── models/       # Pesos dos modelos (.pt / .onnx) — não versionados
├── requirements.txt
└── README.md
```

## Instalação

Requisitos: Python 3.14 (testado com 3.14.6).

```bash
git clone https://github.com/JHenriqueMenezes/ccc-recicla.ai.git
cd ccc-recicla.ai

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Obtendo o modelo ONNX

Os pesos não ficam no Git. Para gerar um YOLO nano pré-treinado em ONNX, instale
o Ultralytics (só é necessário para a exportação) e exporte:

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12
move yolo11n.onnx models\   # Windows  (Linux/macOS: mv yolo11n.onnx models/)
```
