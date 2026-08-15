"""
Pipeline de preparacao de dados: checagem de arquivos corrompidos, deteccao
e corte de bordas pretas (letterbox/pillarbox), merge de datasets com
deduplicacao por hash perceptual, e geracao dos splits treino/val/teste.

Exemplos:
    python data_pipeline.py check --data-dir data/cleaned
    python data_pipeline.py crop --src-dir data/raw/algum_dataset --dst-dir data/cleaned
    python data_pipeline.py merge --src-dir data/raw/outro_dataset --cleaned-dir data/cleaned --prefix outro
    python data_pipeline.py splits --cleaned-dir data/cleaned --out-dir data/splits_151
"""
import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Checagem de integridade
# ---------------------------------------------------------------------------

def check_corrupt_images(data_dir):
    """Varre todas as imagens de data_dir (uma subpasta por classe) e reporta
    quais falham ao decodificar por completo (nao so o header)."""
    data_dir = Path(data_dir)
    bad_files, total = [], 0
    for cls_folder in sorted(data_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        for f in cls_folder.iterdir():
            if not f.is_file():
                continue
            total += 1
            try:
                with Image.open(f) as im:
                    im.load()
            except Exception as e:
                bad_files.append((f, str(e)))

    print(f"total verificado: {total}")
    print(f"arquivos com problema: {len(bad_files)}")
    for f, err in bad_files:
        print(f"  {f}: {err}")
    return bad_files


# ---------------------------------------------------------------------------
# Deteccao/corte de barras pretas (letterbox/pillarbox de thumbnails)
# ---------------------------------------------------------------------------

def flatten_to_white(im):
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def find_crop_box(arr, black_thresh=25, row_col_black_frac=0.9, max_crop_frac=0.45):
    import numpy as np
    h, w, _ = arr.shape
    black_mask = np.all(arr < black_thresh, axis=-1)
    col_black_frac = black_mask.mean(axis=0)
    row_black_frac = black_mask.mean(axis=1)

    left = 0
    while left < w * max_crop_frac and col_black_frac[left] > row_col_black_frac:
        left += 1
    right = w
    while right > w * (1 - max_crop_frac) and col_black_frac[right - 1] > row_col_black_frac:
        right -= 1
    top = 0
    while top < h * max_crop_frac and row_black_frac[top] > row_col_black_frac:
        top += 1
    bottom = h
    while bottom > h * (1 - max_crop_frac) and row_black_frac[bottom - 1] > row_col_black_frac:
        bottom -= 1
    return left, top, right, bottom


def auto_crop_dataset(src_dir, dst_dir):
    """Corta bordas pretas de todas as imagens de src_dir (uma subpasta por
    classe), salvando o resultado (cortado ou intacto) em dst_dir."""
    import numpy as np
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    total_cropped = total_files = 0

    for cls_folder in sorted(src_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        files = [f for f in cls_folder.iterdir() if f.is_file()]
        cropped_count = 0
        for f in files:
            total_files += 1
            with Image.open(f) as im:
                flat = flatten_to_white(im)
                arr = np.array(flat)
                left, top, right, bottom = find_crop_box(arr)
                if right - left < 20 or bottom - top < 20:
                    left, top, right, bottom = 0, 0, arr.shape[1], arr.shape[0]
                cropped = flat.crop((left, top, right, bottom))
                out_path = dst_dir / cls_folder.name / f.stem
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(out_path.with_suffix(".png"))
                if (left, top, right, bottom) != (0, 0, arr.shape[1], arr.shape[0]):
                    cropped_count += 1
        total_cropped += cropped_count
        print(f"{cls_folder.name}: {cropped_count}/{len(files)} cortadas")

    print(f"\nTotal: {total_cropped}/{total_files} cortadas -> {dst_dir}")


# ---------------------------------------------------------------------------
# Merge de datasets com deduplicacao por hash perceptual
# ---------------------------------------------------------------------------

def slugify(name):
    name = name.lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def hash_folder(folder, hash_size=8):
    import imagehash
    hashes = {}
    for f in sorted(Path(folder).iterdir()):
        if not f.is_file():
            continue
        try:
            with Image.open(f) as im:
                hashes[f] = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        except Exception:
            pass
    return hashes


def merge_dataset(src_dir, cleaned_dir, prefix, threshold=5):
    """Adiciona imagens de src_dir/<classe>/ em cleaned_dir/<classe>/, pulando
    duplicatas/quase-duplicatas (hash perceptual) que ja existem em cleaned_dir.
    Classes de src_dir sao casadas por slug com as pastas ja existentes em cleaned_dir."""
    src_dir, cleaned_dir = Path(src_dir), Path(cleaned_dir)
    total_added = 0

    for src_folder in sorted(src_dir.iterdir()):
        if not src_folder.is_dir():
            continue
        slug = slugify(src_folder.name)
        dst_folder = cleaned_dir / slug
        if not dst_folder.exists():
            continue  # nao e uma classe que ja temos

        existing = hash_folder(dst_folder)
        incoming = hash_folder(src_folder)
        added = 0
        for k_path, k_hash in incoming.items():
            if any((k_hash - e_hash) <= threshold for e_hash in existing.values()):
                continue
            dst_name = f"{prefix}_{k_path.name}".replace(" ", "-")
            shutil.copy2(k_path, dst_folder / dst_name)
            existing[dst_folder / dst_name] = k_hash
            added += 1

        if added:
            print(f"{slug}: +{added}")
        total_added += added

    print(f"\ntotal adicionado: {total_added}")


# ---------------------------------------------------------------------------
# Geracao dos splits treino/val/teste
# ---------------------------------------------------------------------------

def make_splits(cleaned_dir, out_dir, train_frac=0.70, val_frac=0.15, seed=42):
    cleaned_dir, out_dir = Path(cleaned_dir), Path(out_dir)

    # limpa a saida antes de gerar de novo -- se nao, execucoes anteriores
    # (com um cleaned_dir menor) deixam arquivos parados que podem acabar
    # em train E test ao mesmo tempo quando o dataset cresce e o split muda
    if out_dir.exists():
        print(f"limpando splits antigos em {out_dir} antes de gerar de novo...")
        shutil.rmtree(out_dir)

    random.seed(seed)
    classes = sorted(p.name for p in cleaned_dir.iterdir() if p.is_dir())
    print(f"{len(classes)} classes encontradas em {cleaned_dir}")

    totals = {"train": 0, "val": 0, "test": 0}
    for cls in classes:
        files = sorted(f for f in (cleaned_dir / cls).iterdir() if f.is_file())
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits = {"train": files[:n_train], "val": files[n_train:n_train + n_val], "test": files[n_train + n_val:]}
        for split_name, split_files in splits.items():
            dst = out_dir / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst / f.name)
            totals[split_name] += len(split_files)

    print(f"total treino: {totals['train']} | val: {totals['val']} | teste: {totals['test']}")
    print(f"Saida em: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="verifica imagens corrompidas")
    p_check.add_argument("--data-dir", required=True)

    p_crop = sub.add_parser("crop", help="corta bordas pretas (letterbox/pillarbox)")
    p_crop.add_argument("--src-dir", required=True)
    p_crop.add_argument("--dst-dir", required=True)

    p_merge = sub.add_parser("merge", help="mescla um dataset novo, pulando duplicatas")
    p_merge.add_argument("--src-dir", required=True)
    p_merge.add_argument("--cleaned-dir", required=True)
    p_merge.add_argument("--prefix", required=True, help="prefixo pros nomes de arquivo novos (evita colisao)")

    p_splits = sub.add_parser("splits", help="gera os splits treino/val/teste")
    p_splits.add_argument("--cleaned-dir", required=True)
    p_splits.add_argument("--out-dir", required=True)
    p_splits.add_argument("--train-frac", type=float, default=0.70)
    p_splits.add_argument("--val-frac", type=float, default=0.15)
    p_splits.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "check":
        check_corrupt_images(args.data_dir)
    elif args.command == "crop":
        auto_crop_dataset(args.src_dir, args.dst_dir)
    elif args.command == "merge":
        merge_dataset(args.src_dir, args.cleaned_dir, args.prefix)
    elif args.command == "splits":
        make_splits(args.cleaned_dir, args.out_dir, args.train_frac, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
