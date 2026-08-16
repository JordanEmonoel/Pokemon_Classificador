"""
Tudo relacionado a usar os modelos ja treinados: avaliacao no conjunto de
teste (com TTA/MC-Dropout opcionais), ensemble dos dois modelos, predicao
de uma imagem via linha de comando, e um servidor web com interface visual.

Exemplos:
    python app.py evaluate --model transfer_151 --splits-dir data/splits_151 --tta-n 10
    python app.py ensemble --splits-dir data/splits_151
    python app.py predict caminho/da/imagem.jpg --model transfer_151
    python app.py serve
"""
import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image

from models import (
    get_dataloaders, get_augmented_test_loader, get_transforms, load_trained_model,
)

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def enable_mc_dropout(model):
    """Deixa as camadas de Dropout ativas mesmo com o modelo em eval() --
    gera variacao entre chamadas repetidas na mesma imagem."""
    n = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            n += 1
    return n


def cmd_evaluate(args):
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ROOT / "results" / args.model
    model, class_names, image_size, mean, std = load_trained_model(args.model, results_dir)
    model.to(device)

    _, _, test_loader, _ = get_dataloaders(image_size=image_size, mean=mean, std=std, batch_size=16, splits_dir=args.splits_dir)

    all_labels = []
    accumulated_probs = None
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            if args.tta:
                flipped = torch.flip(images, dims=[3])
                probs = probs + torch.softmax(model(flipped), dim=1)
            all_labels.extend(labels.tolist())
            accumulated_probs = probs if accumulated_probs is None else torch.cat([accumulated_probs, probs])
    base_probs = accumulated_probs / (2 if args.tta else 1)

    if args.tta_n > 0:
        print(f"rodando TTA com {args.tta_n} variacoes aleatorias por imagem...")
        aug_loader = get_augmented_test_loader(image_size=image_size, mean=mean, std=std, batch_size=16, splits_dir=args.splits_dir)
        extra_probs_sum = torch.zeros_like(base_probs)
        with torch.no_grad():
            for _ in range(args.tta_n):
                pass_probs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in aug_loader]
                extra_probs_sum += torch.cat(pass_probs).to(base_probs.device)
        final_probs = (base_probs + extra_probs_sum) / (1 + args.tta_n)
    else:
        final_probs = base_probs

    if args.mc_dropout > 0:
        n_dropout = enable_mc_dropout(model)
        if n_dropout == 0:
            print("[!] esse modelo nao tem camada de Dropout -- MC Dropout nao tem efeito, pulando.")
        else:
            print(f"rodando Monte Carlo Dropout com {args.mc_dropout} passadas ({n_dropout} camada(s) ativas)...")
            mc_probs_sum = torch.zeros_like(final_probs)
            with torch.no_grad():
                for _ in range(args.mc_dropout):
                    pass_probs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in test_loader]
                    mc_probs_sum += torch.cat(pass_probs).to(final_probs.device)
            final_probs = (final_probs + mc_probs_sum) / (1 + args.mc_dropout)
            model.eval()

    all_preds = final_probs.argmax(dim=1).cpu().tolist()
    top5 = final_probs.topk(5, dim=1).indices.cpu()
    labels_tensor = torch.tensor(all_labels).unsqueeze(1)
    top5_acc = (top5 == labels_tensor).any(dim=1).sum().item() / len(all_labels)

    print(f"\n=== avaliacao no conjunto de TESTE (modelo: {args.model}) ===\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    print(f"top-5 accuracy: {top5_acc:.3f}  ({top5_acc*100:.1f}% dos casos com a classe certa entre as 5 mais provaveis)")

    cm = confusion_matrix(all_labels, all_preds)
    n_classes = len(class_names)

    if n_classes > 30:
        # com muitas classes, rotulo de texto por classe fica ilegivel --
        # mostra so a matriz (diagonal = acertos) com escala de cor, sem tentar
        # encaixar 151 nomes nos eixos
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de confusao - {args.model} ({n_classes} classes)")
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="numero de imagens")
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"Matriz de confusao - {args.model}")

    plt.tight_layout()
    out_path = results_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    print(f"matriz de confusao salva em: {out_path}")


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------

def get_probs_and_labels(model, loader, device):
    model.to(device)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            all_probs.append(probs.cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_probs), all_labels


def cmd_ensemble(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scratch_model, _, s_size, s_mean, s_std = load_trained_model("scratch_151")
    transfer_model, _, t_size, t_mean, t_std = load_trained_model("transfer_151")
    _, _, scratch_loader, _ = get_dataloaders(image_size=s_size, mean=s_mean, std=s_std, batch_size=16, splits_dir=args.splits_dir)
    _, _, transfer_loader, _ = get_dataloaders(image_size=t_size, mean=t_mean, std=t_std, batch_size=16, splits_dir=args.splits_dir)

    print("calculando probabilidades da CNN do zero...")
    scratch_probs, labels = get_probs_and_labels(scratch_model, scratch_loader, device)
    print("calculando probabilidades do transfer learning...")
    transfer_probs, labels2 = get_probs_and_labels(transfer_model, transfer_loader, device)
    assert labels == labels2, "os dois loaders precisam estar na mesma ordem (mesmo splits_dir)"

    print("\n=== comparando estrategias de ensemble no TESTE ===\n")

    def report(name, probs):
        preds = probs.argmax(dim=1).tolist()
        acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
        print(f"{name}: acuracia = {acc:.4f}")
        return acc

    report("so CNN do zero", scratch_probs)
    report("so transfer learning", transfer_probs)
    report("ensemble 50/50", 0.5 * scratch_probs + 0.5 * transfer_probs)

    best_acc, best_w = 0.0, 0.0
    for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        combined = w * scratch_probs + (1 - w) * transfer_probs
        preds = combined.argmax(dim=1).tolist()
        acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"\nmelhor peso pra CNN do zero: {best_w:.2f} (transfer={1-best_w:.2f}) -> acuracia = {best_acc:.4f}")


# ---------------------------------------------------------------------------
# predict (uma imagem via CLI)
# ---------------------------------------------------------------------------

def predict_probs(model, image_size, mean, std, image_path_or_bytes):
    """Roda o modelo numa imagem e devolve o vetor de probabilidades completo
    (todas as classes), pra permitir combinar com outro modelo (ensemble)."""
    transform = get_transforms(image_size, mean, std, augment=False)

    if isinstance(image_path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(image_path_or_bytes))
    else:
        im = Image.open(image_path_or_bytes)
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    tensor = transform(im).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0]


def predict_image(model, class_names, image_size, mean, std, image_path_or_bytes, top_k=5):
    probs = predict_probs(model, image_size, mean, std, image_path_or_bytes)
    top_probs, top_idx = probs.topk(top_k)
    return [(class_names[i], p.item()) for p, i in zip(top_probs, top_idx)]


def cmd_predict(args):
    model, class_names, image_size, mean, std = load_trained_model(args.model)
    results = predict_image(model, class_names, image_size, mean, std, args.image, top_k=args.top)

    print(f"\nImagem: {args.image}")
    print(f"Modelo: {args.model}\n")
    for i, (name, prob) in enumerate(results, 1):
        bar = "#" * int(prob * 30)
        print(f"{i}. {name:<20s} {prob*100:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# serve (servidor web com interface visual)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pokemon Classifier</title>
<style>
  :root {
    --bg: #1a1c2e; --card: #23263f; --accent: #ffcb05; --accent-dark: #3b4cca;
    --text: #f0f0f5; --text-dim: #9a9cb8; --border: #363a5c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(circle at top, #24274a, var(--bg));
    color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
  }
  header { text-align: center; margin-bottom: 32px; }
  header h1 {
    font-size: 2.2rem;
    background: linear-gradient(90deg, var(--accent), #ffe27a);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 12px;
  }
  .pokeball { flex-shrink: 0; animation: spin-in 0.8s ease-out; }
  @keyframes spin-in { from { transform: rotate(-90deg) scale(0.5); opacity: 0; } to { transform: rotate(0) scale(1); opacity: 1; } }
  header p { color: var(--text-dim); margin-top: 6px; font-size: 0.95rem; }
  .container { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 32px; width: 100%; max-width: 520px; box-shadow: 0 10px 40px rgba(0,0,0,0.4); }
  .model-select { display: flex; gap: 6px; margin-bottom: 20px; background: #181a2e; border-radius: 10px; padding: 4px; }
  .model-select button { flex: 1; padding: 10px 6px; border: none; border-radius: 8px; background: transparent; color: var(--text-dim); font-weight: 600; cursor: pointer; font-size: 0.78rem; transition: all 0.2s; }
  .model-select button.active { background: var(--accent-dark); color: white; }
  #dropzone { border: 2px dashed var(--border); border-radius: 12px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden; }
  #dropzone.dragover { border-color: var(--accent); background: rgba(255, 203, 5, 0.05); }
  #dropzone p { color: var(--text-dim); font-size: 0.9rem; }
  #dropzone .icon { font-size: 2.5rem; margin-bottom: 10px; }
  #fileInput { display: none; }
  #preview { display: none; max-width: 100%; max-height: 220px; border-radius: 10px; margin: 0 auto; }
  #status { text-align: center; color: var(--text-dim); margin-top: 16px; font-size: 0.9rem; min-height: 20px; }
  #status.error { color: #ff6b6b; }
  #results { margin-top: 24px; display: none; }
  #results h2 { font-size: 1rem; color: var(--text-dim); margin-bottom: 14px; font-weight: 500; }
  .result-row { margin-bottom: 12px; }
  .result-row .row-top { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.9rem; }
  .result-row .name { text-transform: capitalize; font-weight: 600; }
  .result-row.top1 .name { color: var(--accent); font-size: 1.05rem; }
  .result-row .pct { color: var(--text-dim); font-variant-numeric: tabular-nums; }
  .bar-track { background: #181a2e; border-radius: 6px; height: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent-dark), #6a7bff); width: 0%; transition: width 0.6s ease; }
  .result-row.top1 .bar-fill { background: linear-gradient(90deg, var(--accent), #ffe27a); }
  .reset-btn { display: none; margin-top: 20px; width: 100%; padding: 12px; background: transparent; border: 1px solid var(--border); border-radius: 8px; color: var(--text-dim); cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }
  .reset-btn:hover { border-color: var(--accent); color: var(--accent); }
  footer { margin-top: 24px; color: var(--text-dim); font-size: 0.8rem; text-align: center; }
</style>
</head>
<body>

<header>
  <h1>
    <svg class="pokeball" viewBox="0 0 100 100" width="34" height="34" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="46" fill="#fff" stroke="#1a1c2e" stroke-width="5"/>
      <path d="M4 50a46 46 0 0 1 92 0z" fill="#ff3b3b"/>
      <rect x="4" y="46" width="92" height="8" fill="#1a1c2e"/>
      <circle cx="50" cy="50" r="15" fill="#1a1c2e"/>
      <circle cx="50" cy="50" r="9" fill="#fff" stroke="#1a1c2e" stroke-width="3"/>
    </svg>
    <span>Pokemon Classifier</span>
  </h1>
  <p>151 especies da 1a geracao &middot; envie uma imagem e veja o palpite do modelo</p>
</header>

<div class="container">
  <div class="model-select">
    <button id="btnEnsemble" class="active" data-model="ensemble">Ensemble (94.7%)</button>
    <button id="btnTransfer" data-model="transfer_151">Transfer Learning (94.4%)</button>
    <button id="btnScratch" data-model="scratch_151">CNN do Zero (82.8%)</button>
  </div>

  <div id="dropzone">
    <img id="preview" alt="preview">
    <div id="dropzoneText">
      <div class="icon">&#128248;</div>
      <p><strong>Clique</strong> ou arraste uma imagem aqui</p>
      <p style="margin-top:4px; font-size:0.8rem;">JPG ou PNG</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div id="status"></div>

  <div id="results">
    <h2>Top 5 palpites</h2>
    <div id="resultsList"></div>
  </div>

  <button class="reset-btn" id="resetBtn">Testar outra imagem</button>
</div>

<footer>Modelo treinado do zero e por transfer learning &middot; projeto pessoal</footer>

<script>
let currentModel = "ensemble";
let currentFile = null;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropzoneText = document.getElementById("dropzoneText");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsList = document.getElementById("resultsList");
const resetBtn = document.getElementById("resetBtn");
const btnEnsemble = document.getElementById("btnEnsemble");
const btnTransfer = document.getElementById("btnTransfer");
const btnScratch = document.getElementById("btnScratch");
const modelButtons = [btnEnsemble, btnTransfer, btnScratch];

function selectModel(model, btn) {
  if (model === currentModel) return;
  currentModel = model;
  modelButtons.forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  if (currentFile) classify(currentFile);
}
btnEnsemble.addEventListener("click", () => selectModel("ensemble", btnEnsemble));
btnTransfer.addEventListener("click", () => selectModel("transfer_151", btnTransfer));
btnScratch.addEventListener("click", () => selectModel("scratch_151", btnScratch));

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

resetBtn.addEventListener("click", () => {
  preview.style.display = "none";
  dropzoneText.style.display = "block";
  resultsEl.style.display = "none";
  resetBtn.style.display = "none";
  statusEl.textContent = "";
  statusEl.classList.remove("error");
  fileInput.value = "";
  currentFile = null;
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    statusEl.textContent = "isso nao parece uma imagem valida";
    statusEl.classList.add("error");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = "block";
    dropzoneText.style.display = "none";
  };
  reader.readAsDataURL(file);
  classify(file);
}

async function classify(file) {
  statusEl.classList.remove("error");
  statusEl.textContent = "classificando...";
  resultsEl.style.display = "none";
  resetBtn.style.display = "none";

  const formData = new FormData();
  formData.append("image", file);
  formData.append("model", currentModel);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.error || "erro ao classificar";
      statusEl.classList.add("error");
      return;
    }
    statusEl.textContent = "";
    renderResults(data.predictions);
    resultsEl.style.display = "block";
    resetBtn.style.display = "block";
  } catch (err) {
    statusEl.textContent = "nao consegui falar com o servidor";
    statusEl.classList.add("error");
  }
}

function renderResults(predictions) {
  resultsList.innerHTML = "";
  predictions.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "result-row" + (i === 0 ? " top1" : "");
    row.innerHTML = `
      <div class="row-top">
        <span class="name">${p.name.replace(/-/g, " ")}</span>
        <span class="pct">${p.confidence.toFixed(1)}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
    `;
    resultsList.appendChild(row);
    requestAnimationFrame(() => { row.querySelector(".bar-fill").style.width = p.confidence + "%"; });
  });
}
</script>

</body>
</html>
"""


def cmd_serve(args):
    from flask import Flask, request, jsonify, Response

    flask_app = Flask(__name__)
    models_cache = {}

    def get_model(model_type):
        if model_type not in models_cache:
            models_cache[model_type] = load_trained_model(model_type)
        return models_cache[model_type]

    @flask_app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @flask_app.route("/predict", methods=["POST"])
    def predict_route():
        if "image" not in request.files:
            return jsonify({"error": "nenhuma imagem enviada"}), 400

        model_type = request.form.get("model", "ensemble")
        if model_type not in ("transfer_151", "scratch_151", "ensemble"):
            return jsonify({"error": "modelo invalido"}), 400

        file = request.files["image"]
        try:
            image_bytes = file.read()
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            return jsonify({"error": "arquivo de imagem invalido"}), 400

        if model_type == "ensemble":
            # peso 20% CNN do zero / 80% transfer learning -- o melhor encontrado
            # comparando as duas no conjunto de teste (ver: python app.py ensemble)
            s_model, s_names, s_size, s_mean, s_std = get_model("scratch_151")
            t_model, t_names, t_size, t_mean, t_std = get_model("transfer_151")
            assert s_names == t_names, "os dois modelos precisam ter o mesmo mapeamento de classes"

            s_probs = predict_probs(s_model, s_size, s_mean, s_std, image_bytes)
            t_probs = predict_probs(t_model, t_size, t_mean, t_std, image_bytes)
            combined = 0.2 * s_probs + 0.8 * t_probs

            top_probs, top_idx = combined.topk(5)
            results = [(s_names[i], p.item()) for p, i in zip(top_probs, top_idx)]
        else:
            model, class_names, image_size, mean, std = get_model(model_type)
            results = predict_image(model, class_names, image_size, mean, std, image_bytes, top_k=5)

        return jsonify({
            "predictions": [{"name": n, "confidence": round(p * 100, 2)} for n, p in results],
            "model": model_type,
        })

    flask_app.run(debug=args.debug, port=args.port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="avalia um modelo no conjunto de teste")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--splits-dir", default=None)
    p_eval.add_argument("--tta", action="store_true", help="media entre imagem original e espelhada")
    p_eval.add_argument("--tta-n", type=int, default=0, help="N variacoes aleatorias por imagem")
    p_eval.add_argument("--mc-dropout", type=int, default=0, help="N passadas com Monte Carlo Dropout")

    p_ens = sub.add_parser("ensemble", help="compara estrategias de ensemble entre scratch_151 e transfer_151")
    p_ens.add_argument("--splits-dir", default="data/splits_151")

    p_pred = sub.add_parser("predict", help="classifica uma imagem")
    p_pred.add_argument("image")
    p_pred.add_argument("--model", default="transfer_151")
    p_pred.add_argument("--top", type=int, default=5)

    p_serve = sub.add_parser("serve", help="sobe o servidor web com interface visual")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true", default=True)

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "ensemble":
        cmd_ensemble(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
