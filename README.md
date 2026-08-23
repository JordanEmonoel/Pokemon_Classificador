# Pokemon Classifier

Uma CNN treinada do zero e um modelo de transfer learning (ResNet18), competindo
para reconhecer as 151 espécies da primeira geração de Pokémon a partir de uma
imagem.

Projeto pessoal de aprendizado prático em visão computacional: curadoria de
dataset, arquitetura de rede neural, comparação de abordagens e avaliação
honesta dos resultados (incluindo um bug de vazamento de dados encontrado e
corrigido no meio do caminho).

<p align="center">
  <img src="docs/images/demo_ensemble_pikachu.png" width="270">
  <img src="docs/images/demo_transfer_gengar.png" width="270">
  <img src="docs/images/demo_scratch_dragonite.png" width="270">
</p>

## Motivação

Ideia inspirada [neste vídeo](https://www.youtube.com/watch?v=IoM5zUI8oFc), sobre
alguém treinando uma IA do zero para jogar Super Mario. Comecei pequeno, com 4
espécies fáceis de distinguir só pela cor (Bulbasaur, Charmander, Pikachu,
Squirtle), para aprender o básico, e depois escalei para as 151 espécies
reais da primeira geração, onde o problema fica bem mais difícil (Charmander
e Charizard, por exemplo, são quase gêmeos de cor para um modelo simples).

## Resultados

Avaliados no conjunto de teste (2.478 imagens, sem sobreposição com treino/validação):

| Modelo | Acurácia | Top-5 |
|---|---|---|
| CNN do zero (residual + SE blocks, Focal Loss, Mixup/CutMix) | 82.8% | 94.6% |
| Transfer Learning (ResNet18, fine-tuning parcial do backbone) | 94.4% | 98.8% |
| **Ensemble (20% CNN do zero + 80% transfer learning)** | **94.7%** | n/d |

<p align="center">
  <img src="docs/images/transfer_curves.png" width="420">
  <img src="docs/images/transfer_confusion_matrix.png" width="360">
</p>

Um exemplo real do comportamento diferente das duas abordagens: pedindo a
classificação de um Dragonite, a CNN do zero acerta (82.5%), mas coloca
Charizard como segunda opção (11.9%). Os dois são dragões alaranjados da
primeira geração. É um bom retrato da limitação conhecida de uma rede menor
sem conhecimento visual prévio: ela se apoia mais em cor do que forma, algo
que o transfer learning, por já ter aprendido conceitos visuais gerais no
ImageNet, praticamente não sofre.

### O bug que quase inflou os números

No meio do processo, o script de geração dos splits treino/val/teste não
limpava a pasta de saída antes de gerar de novo. Como o dataset cresceu várias
vezes ao longo do projeto, arquivos de execuções antigas ficaram acumulados.
Resultado: 1.617 imagens apareciam tanto em treino quanto em teste ao mesmo
tempo, inflando artificialmente a acurácia relatada (95.8% em vez dos 94.4%
reais). Encontrado, corrigido e todos os modelos reavaliados no conjunto de
teste limpo antes de fechar os números acima.

## O que foi testado

- **CNN do zero**: 4 blocos residuais com Squeeze-and-Excitation, Focal Loss
  ponderada por classe (ataca o desbalanceamento entre espécies), Mixup/CutMix,
  learning rate com `ReduceLROnPlateau`, early stopping dinâmico (para só
  quando o ganho vira marginal, não por teto fixo de épocas).
- **Transfer Learning**: ResNet18 pré-treinada no ImageNet, com fine-tuning
  parcial (último bloco residual descongelado, learning rate diferenciado
  entre backbone e cabeça nova).
- **Ensemble**: combinação ponderada das probabilidades dos dois modelos.
- **Test-Time Augmentation**: múltiplas variações aleatórias (crop, rotação,
  cor) por imagem de teste, com a predição final sendo a média.
- **Monte Carlo Dropout**: testado como alternativa ao TTA, mas não superou
  as outras técnicas (fica registrado como experimento honesto, nem tudo que
  se testa funciona).

## Estrutura do repositório

```
pokemon-classifier/
├── data_pipeline.py   # curadoria (corrompidos, bordas pretas, dedup) + geração de splits
├── models.py           # dataloaders, arquiteturas, losses, augmentation, loop de treino
├── train.py             # treino via CLI (--arch scratch|transfer)
├── app.py                # avaliação, ensemble, predição de imagem, servidor web
├── docs/images/           # gráficos de treino, matrizes de confusão e prints da interface
├── results/<variante>/     # histórico de treino, curvas e matriz de confusão de cada rodada
├── requirements.txt
└── README.md
```

## Modelos treinados

Os pesos dos dois modelos de 151 classes estão no repositório (45 MB no total),
em `results/transfer_151/best_model.pt` e `results/scratch_151/best_model.pt`.
Ou seja: dá para clonar e classificar imagens na hora, sem treinar nada e sem
precisar do dataset.

O histórico de cada rodada de treino também está versionado em
`results/<variante>/`: `history.json` com as métricas por época, `curves.png`,
`confusion_matrix.png` e `class_to_idx.json`. As variantes `_5class` e as de 4
classes são as etapas iniciais do projeto e estão ali como registro; só o
histórico delas foi mantido, não os pesos.

## Como rodar

Para brincar com o app não precisa de mais nada além de clonar o repositório:

```bash
git clone https://github.com/JordanEmonoel/Pokemon_Classificador.git
cd Pokemon_Classificador
pip install -r requirements.txt

# interface web: sobe em http://localhost:5000, arrasta uma imagem e escolhe
# entre CNN do zero, transfer learning ou o ensemble dos dois
python app.py serve

# ou pela linha de comando
python app.py predict caminho/da/imagem.jpg --model transfer_151
```

Os comandos abaixo mexem com treino e avaliação, e esses sim precisam do
dataset organizado em `data/splits_151/{train,val,test}/<classe>/`:

```bash
# treino
python train.py --arch transfer --splits-dir data/splits_151 --results-dir results/transfer_151

# avaliação
python app.py evaluate --model transfer_151 --splits-dir data/splits_151 --tta-n 10

# comparação de pesos do ensemble entre os dois modelos
python app.py ensemble --splits-dir data/splits_151
```

O dataset de imagens não está incluído neste repositório. Foi raspado de
várias fontes públicas, majoritariamente arte oficial e fan art de Pokémon,
e não é redistribuído aqui por questão de direitos autorais. O código em
`data_pipeline.py` documenta o processo de curadoria usado.

## Stack

Python, PyTorch, torchvision, Flask, scikit-learn, matplotlib, imagehash.
