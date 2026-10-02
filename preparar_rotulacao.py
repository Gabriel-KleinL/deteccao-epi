"""Seleciona capturas reais diversas para rotulagem, sem alterar os originais.

O nome da captura contém apenas o alerta produzido pelo modelo antigo. Ele não
é um rótulo confiável e nunca deve ser importado como anotação automática.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


def _classe_alerta(arquivo: Path) -> str:
    partes = arquivo.stem.split("_", 2)
    return partes[2] if len(partes) == 3 else "DESCONHECIDO"


def _amostra_uniforme(itens: list[Path], limite: int) -> list[Path]:
    if len(itens) <= limite:
        return itens
    if limite <= 1:
        return [itens[len(itens) // 2]]
    return [itens[round(i * (len(itens) - 1) / (limite - 1))] for i in range(limite)]


def _dhash(arquivo: Path) -> int:
    with Image.open(arquivo) as imagem:
        cinza = imagem.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = cinza.tobytes()
    valor = 0
    for y in range(8):
        for x in range(8):
            valor = (valor << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return valor


def _distancia(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _materializar(origem: Path, destino: Path, modo: str) -> str:
    if modo == "copiar":
        shutil.copy2(origem, destino)
        return "copia"
    try:
        os.link(origem, destino)
        return "hardlink"
    except OSError:
        shutil.copy2(origem, destino)
        return "copia_fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cria um lote deduplicado de capturas para rotulagem manual.")
    parser.add_argument("--origem", type=Path, default=Path("resultados/capturas"))
    parser.add_argument("--destino", type=Path, default=Path("dados/rotulacao_pendente"))
    parser.add_argument("--limite", type=int, default=400, help="Quantidade máxima de imagens no lote.")
    parser.add_argument(
        "--distancia-hash",
        type=int,
        default=5,
        help="Distância perceptual mínima; aumente para remover mais quadros parecidos.",
    )
    parser.add_argument("--modo", choices=("hardlink", "copiar"), default="hardlink")
    args = parser.parse_args()

    if args.limite < 1:
        parser.error("--limite precisa ser maior que zero")
    if not args.origem.is_dir():
        raise FileNotFoundError(f"Pasta de capturas não encontrada: {args.origem}")

    por_classe: dict[str, list[Path]] = defaultdict(list)
    for arquivo in sorted(args.origem.glob("*.jpg")):
        por_classe[_classe_alerta(arquivo)].append(arquivo)
    if not por_classe:
        raise RuntimeError(f"Nenhuma captura JPG encontrada em {args.origem}")

    lote = args.destino / f"lote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    imagens_destino = lote / "images"
    imagens_destino.mkdir(parents=True, exist_ok=False)

    # O pool uniforme evita decodificar dezenas de milhares de frames quase iguais.
    por_classe_pool = {}
    for classe, arquivos in sorted(por_classe.items()):
        pool = _amostra_uniforme(arquivos, min(len(arquivos), args.limite * 5))
        # Ordem aleatória determinística: cobre todo o período sem perder a
        # reprodutibilidade do lote.
        random.Random(f"deteccao-epi:{classe}").shuffle(pool)
        por_classe_pool[classe] = pool
    cursores = {classe: 0 for classe in por_classe_pool}
    hashes_escolhidos: list[int] = []
    registros: list[dict[str, str]] = []
    classes = list(por_classe_pool)

    while len(registros) < args.limite:
        houve_candidato = False
        for classe in classes:
            pool = por_classe_pool[classe]
            while cursores[classe] < len(pool):
                origem = pool[cursores[classe]]
                cursores[classe] += 1
                houve_candidato = True
                try:
                    hash_perceptual = _dhash(origem)
                except Exception as exc:
                    print(f"[IGNORADA] {origem.name}: {exc}")
                    continue
                if any(_distancia(hash_perceptual, anterior) < args.distancia_hash for anterior in hashes_escolhidos):
                    continue

                hash_curto = hashlib.sha256(origem.read_bytes()).hexdigest()[:12]
                destino = imagens_destino / f"{origem.stem}_{hash_curto}{origem.suffix.lower()}"
                materializacao = _materializar(origem, destino, args.modo)
                hashes_escolhidos.append(hash_perceptual)
                registros.append(
                    {
                        "arquivo": destino.name,
                        "original": str(origem.resolve()),
                        "alerta_modelo_antigo": classe,
                        "rotulo_status": "PENDENTE",
                        "materializacao": materializacao,
                        "observacao": "",
                    }
                )
                break
            if len(registros) >= args.limite:
                break
        if not houve_candidato or all(cursores[c] >= len(por_classe_pool[c]) for c in classes):
            break

    if not registros:
        raise RuntimeError("Nenhuma imagem passou pelo filtro de diversidade.")

    with (lote / "manifesto.csv").open("w", newline="", encoding="utf-8-sig") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=list(registros[0]))
        writer.writeheader()
        writer.writerows(registros)

    (lote / "INSTRUCOES.md").write_text(
        "# Lote para rotulagem manual\n\n"
        "O campo `alerta_modelo_antigo` e o nome do arquivo são apenas pistas do modelo atual. "
        "Eles não são verdade-terreno. Revise cada imagem e anote todas as pessoas/áreas oculares visíveis.\n\n"
        "Para óculos, use `Oculos de Protecao` somente quando o EPI estiver visualmente confirmado. "
        "Óculos comum, ausência de óculos e proteção inadequada pertencem a "
        "`SEM-Oculos de Protecao`. Rostos sem informação visual suficiente devem ser ignorados, "
        "não rotulados por suposição.\n",
        encoding="utf-8",
    )

    print(f"Lote criado: {lote.resolve()}")
    print(f"Imagens selecionadas: {len(registros)} de {sum(map(len, por_classe.values()))} capturas")
    print("Por alerta antigo:")
    for classe in classes:
        quantidade = sum(r["alerta_modelo_antigo"] == classe for r in registros)
        print(f"  {classe}: {quantidade}")
    print("Os originais não foram alterados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
