"""Auditoria rápida de datasets YOLO antes de iniciar um treino.

Uso:
    python auditar_dataset.py
    python auditar_dataset.py --data dados/oculos.yaml --json saidas/auditoria.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp"}


def _resolver_raiz(data_yaml: Path, valor: str) -> Path:
    raiz = Path(valor).expanduser()
    if raiz.is_absolute():
        return raiz.resolve()
    return (data_yaml.parent / raiz).resolve()


def _resolver_split(raiz: Path, valor: str | list[str]) -> list[Path]:
    valores = valor if isinstance(valor, list) else [valor]
    return [(raiz / item).resolve() for item in valores]


def _pasta_labels(pasta_imagens: Path) -> Path:
    partes = list(pasta_imagens.parts)
    if "images" in partes:
        indice = len(partes) - 1 - partes[::-1].index("images")
        partes[indice] = "labels"
        return Path(*partes)
    return pasta_imagens.parent / "labels"


def _id_fonte(stem: str) -> str:
    """Remove o fingerprint de uma variante exportada pelo Roboflow."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)


def auditar(data_yaml: Path) -> dict:
    data_yaml = data_yaml.expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raiz = _resolver_raiz(data_yaml, str(config.get("path", ".")))
    nomes_cfg = config.get("names", {})
    if isinstance(nomes_cfg, list):
        nomes = {i: nome for i, nome in enumerate(nomes_cfg)}
    else:
        nomes = {int(i): nome for i, nome in nomes_cfg.items()}

    relatorio: dict = {
        "data_yaml": str(data_yaml),
        "raiz": str(raiz),
        "classes": nomes,
        "splits": {},
        "erros": [],
        "avisos": [],
    }
    ids_por_split: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        if split not in config:
            if split in {"train", "val"}:
                relatorio["erros"].append(f"Split obrigatório ausente no YAML: {split}")
            continue

        imagens: list[Path] = []
        for pasta in _resolver_split(raiz, config[split]):
            if not pasta.exists():
                relatorio["erros"].append(f"Pasta do split {split} não existe: {pasta}")
                continue
            imagens.extend(
                arquivo
                for arquivo in pasta.rglob("*")
                if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM
            )

        contagem_classes: Counter[int] = Counter()
        imagens_sem_objeto = 0
        labels_ausentes = 0
        labels_orfaos = 0
        labels_invalidos = 0
        caixas_pequenas = 0
        caixas_muito_pequenas = 0
        dimensoes: Counter[str] = Counter()

        for imagem in imagens:
            ids_por_split[split].add(_id_fonte(imagem.stem))
            hashes[hashlib.sha256(imagem.read_bytes()).hexdigest()].append((split, imagem.name))
            try:
                with Image.open(imagem) as aberta:
                    dimensoes[f"{aberta.width}x{aberta.height}"] += 1
            except Exception as exc:
                relatorio["erros"].append(f"Imagem inválida {imagem}: {exc}")

            label = _pasta_labels(imagem.parent) / f"{imagem.stem}.txt"
            if not label.exists():
                labels_ausentes += 1
                continue
            linhas = [linha for linha in label.read_text(encoding="utf-8").splitlines() if linha.strip()]
            if not linhas:
                imagens_sem_objeto += 1
            for numero_linha, linha in enumerate(linhas, 1):
                partes = linha.split()
                try:
                    if len(partes) != 5:
                        raise ValueError("esperados 5 campos")
                    classe = int(partes[0])
                    x, y, largura, altura = map(float, partes[1:])
                    if classe not in nomes:
                        raise ValueError(f"classe {classe} não declarada")
                    if not all(0.0 <= valor <= 1.0 for valor in (x, y, largura, altura)):
                        raise ValueError("coordenada fora de 0..1")
                    if largura <= 0 or altura <= 0:
                        raise ValueError("caixa sem área")
                except ValueError as exc:
                    labels_invalidos += 1
                    relatorio["erros"].append(f"Label inválido {label}:{numero_linha}: {exc}")
                    continue

                contagem_classes[classe] += 1
                area = largura * altura
                caixas_pequenas += area < 0.01
                caixas_muito_pequenas += area < 0.0025

        for pasta in _resolver_split(raiz, config[split]):
            if not pasta.is_dir():
                continue
            pasta_labels = _pasta_labels(pasta)
            if not pasta_labels.is_dir():
                continue
            stems_imagens = {
                arquivo.stem
                for arquivo in pasta.rglob("*")
                if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM
            }
            labels_orfaos += sum(
                label.stem not in stems_imagens for label in pasta_labels.rglob("*.txt") if label.is_file()
            )

        relatorio["splits"][split] = {
            "imagens": len(imagens),
            "fontes_unicas": len(ids_por_split[split]),
            "variantes_por_fonte": round(len(imagens) / max(1, len(ids_por_split[split])), 2),
            "instancias": sum(contagem_classes.values()),
            "instancias_por_classe": {
                nomes.get(classe, str(classe)): quantidade
                for classe, quantidade in sorted(contagem_classes.items())
            },
            "imagens_sem_objeto": imagens_sem_objeto,
            "labels_ausentes": labels_ausentes,
            "labels_orfaos": labels_orfaos,
            "labels_invalidos": labels_invalidos,
            "caixas_area_menor_1pct": caixas_pequenas,
            "caixas_area_menor_0_25pct": caixas_muito_pequenas,
            "dimensoes": dict(dimensoes),
        }
        if not imagens:
            relatorio["erros"].append(f"Split {split} não contém imagens.")
        if labels_ausentes:
            relatorio["erros"].append(f"Split {split} tem {labels_ausentes} imagem(ns) sem arquivo de label.")
        if labels_orfaos:
            relatorio["erros"].append(f"Split {split} tem {labels_orfaos} label(s) sem imagem correspondente.")

    vazamento: list[dict] = []
    splits = sorted(ids_por_split)
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            repetidos = sorted(ids_por_split[split_a] & ids_por_split[split_b])
            if repetidos:
                vazamento.append(
                    {"splits": [split_a, split_b], "quantidade": len(repetidos), "amostra": repetidos[:10]}
                )
    relatorio["fontes_em_mais_de_um_split"] = vazamento
    if vazamento:
        relatorio["erros"].append("Há imagens da mesma fonte em splits diferentes (vazamento de dados).")

    duplicatas_exatas = [itens for itens in hashes.values() if len(itens) > 1]
    duplicatas_cruzadas = [itens for itens in duplicatas_exatas if len({split for split, _ in itens}) > 1]
    relatorio["grupos_duplicados_exatos"] = len(duplicatas_exatas)
    relatorio["grupos_duplicados_entre_splits"] = len(duplicatas_cruzadas)
    if duplicatas_cruzadas:
        relatorio["erros"].append("Há arquivos idênticos em splits diferentes.")

    treino = relatorio["splits"].get("train", {})
    if treino.get("fontes_unicas", 0) < 1000:
        relatorio["avisos"].append(
            "O treino tem menos de 1.000 fontes independentes; priorize novos dados reais antes de aumentar épocas."
        )
    if treino.get("variantes_por_fonte", 1) > 1.5:
        relatorio["avisos"].append(
            "O treino contém muitas variantes pré-aumentadas; elas não equivalem a novas cenas independentes."
        )

    return relatorio


def _imprimir(relatorio: dict) -> None:
    print(f"Dataset: {relatorio['raiz']}")
    print("Classes:", ", ".join(f"{i}={nome}" for i, nome in relatorio["classes"].items()))
    for split, dados in relatorio["splits"].items():
        print(
            f"[{split}] {dados['imagens']} imagens | {dados['fontes_unicas']} fontes | "
            f"{dados['instancias']} caixas | {dados['imagens_sem_objeto']} fundos vazios"
        )
        print("  classes:", dados["instancias_por_classe"])
        print(
            f"  caixas pequenas: {dados['caixas_area_menor_1pct']} abaixo de 1% da imagem; "
            f"{dados['caixas_area_menor_0_25pct']} abaixo de 0,25%"
        )
    print("Duplicatas exatas:", relatorio["grupos_duplicados_exatos"])
    print("Duplicatas exatas entre splits:", relatorio["grupos_duplicados_entre_splits"])
    print("Fontes em mais de um split:", relatorio["fontes_em_mais_de_um_split"] or "nenhuma")
    for aviso in relatorio["avisos"]:
        print(f"[AVISO] {aviso}")
    for erro in relatorio["erros"]:
        print(f"[ERRO] {erro}")
    print("Resultado:", "REPROVADO" if relatorio["erros"] else "APROVADO")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita estrutura, labels, duplicatas e splits de um dataset YOLO.")
    parser.add_argument("--data", type=Path, default=Path("dados/oculos.yaml"))
    parser.add_argument("--json", type=Path, help="Também grava o relatório em JSON.")
    args = parser.parse_args()

    relatorio = auditar(args.data)
    _imprimir(relatorio)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Relatório JSON: {args.json}")
    return 1 if relatorio["erros"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
