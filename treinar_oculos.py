"""Treina um candidato de óculos e compara com o modelo em produção.

Este script nunca substitui ``modelos/oculos.pt``. A promoção só deve ocorrer
depois da validação offline e de um teste com cenas reais da câmera de destino.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from auditar_dataset import auditar


PROJETO = Path(__file__).resolve().parent
CLASSE_ALERTA = "SEM-Oculos de Protecao"
ALIASES = {"SEM-Oculos": CLASSE_ALERTA}


def escolher_device(solicitado: str) -> str | int:
    if solicitado != "auto":
        return int(solicitado) if solicitado.isdigit() else solicitado
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _metricas(resultado) -> dict:
    classes = {}
    for item in resultado.summary():
        classe = ALIASES.get(str(item["Class"]), str(item["Class"]))
        classes[classe] = {
            "imagens": int(item["Images"]),
            "instancias": int(item["Instances"]),
            "precision": float(item["Box-P"]),
            "recall": float(item["Box-R"]),
            "f1": float(item["Box-F1"]),
            "map50": float(item["mAP50"]),
            "map50_95": float(item["mAP50-95"]),
        }
    return {"classes": classes, "map50_95_macro": float(resultado.box.map)}


def avaliar(
    caminho_modelo: Path,
    data: Path,
    split: str,
    device: str | int,
    nome: str,
    imgsz: int,
    conf_operacional: float,
) -> dict:
    modelo = YOLO(caminho_modelo)
    argumentos = dict(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=16,
        device=device,
        verbose=False,
        project=str(PROJETO / "saidas" / "avaliacoes_oculos"),
        exist_ok=True,
    )
    padrao = modelo.val(**argumentos, name=f"{nome}_padrao", plots=True)
    operacional = modelo.val(
        **argumentos,
        name=f"{nome}_conf_{str(conf_operacional).replace('.', '_')}",
        conf=conf_operacional,
        iou=0.3,
        plots=False,
    )
    return {
        "padrao": _metricas(padrao),
        "operacional": {"conf": conf_operacional, **_metricas(operacional)},
    }


def comparar(baseline: dict, candidato: dict) -> dict:
    base_alerta = baseline["operacional"]["classes"].get(CLASSE_ALERTA)
    novo_alerta = candidato["operacional"]["classes"].get(CLASSE_ALERTA)
    if not base_alerta or not novo_alerta:
        return {"apto_offline": False, "motivo": f"Classe obrigatória ausente: {CLASSE_ALERTA}"}

    deltas = {
        "precision_alerta": novo_alerta["precision"] - base_alerta["precision"],
        "recall_alerta": novo_alerta["recall"] - base_alerta["recall"],
        "f1_alerta": novo_alerta["f1"] - base_alerta["f1"],
        "map50_95_macro": (
            candidato["padrao"]["map50_95_macro"] - baseline["padrao"]["map50_95_macro"]
        ),
    }
    sem_regressao_grave = (
        deltas["precision_alerta"] >= -0.03
        and deltas["recall_alerta"] >= -0.02
        and deltas["map50_95_macro"] >= -0.01
    )
    houve_ganho = deltas["f1_alerta"] >= 0.01 or deltas["map50_95_macro"] >= 0.01
    return {
        "deltas": deltas,
        "sem_regressao_grave": sem_regressao_grave,
        "houve_ganho_minimo": houve_ganho,
        "apto_offline": sem_regressao_grave and houve_ganho,
        "observacao": "Apto offline não significa aprovado em campo.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina e avalia um candidato sem alterar o modelo atual.")
    parser.add_argument("--data", type=Path, default=PROJETO / "dados/oculos.yaml")
    parser.add_argument("--baseline", type=Path, default=PROJETO / "modelos/oculos.pt")
    parser.add_argument(
        "--pesos-iniciais",
        type=Path,
        help="Pesos para fine-tuning. O padrão é o próprio baseline; use yolov8n.pt para um treino limpo.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--horas",
        type=float,
        help="Limite de duração do treino em horas; quando informado, prevalece sobre epochs.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--conf-operacional",
        type=float,
        default=0.35,
        help="Limiar usado para medir precision/recall reais da classe de alerta.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps ou índice CUDA, por exemplo 0")
    parser.add_argument("--sem-teste", action="store_true", help="Não usa o split test neste experimento.")
    parser.add_argument("--somente-auditar", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.conf_operacional < 1.0:
        parser.error("--conf-operacional precisa estar entre 0 e 1")
    if min(args.epochs, args.batch, args.imgsz, args.patience) < 1:
        parser.error("epochs, batch, imgsz e patience precisam ser maiores que zero")
    if args.horas is not None and args.horas <= 0:
        parser.error("--horas precisa ser maior que zero")

    data = args.data.expanduser().resolve()
    baseline = args.baseline.expanduser().resolve()
    pesos_iniciais = (args.pesos_iniciais or baseline).expanduser().resolve()
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline não encontrado: {baseline}")
    if not pesos_iniciais.exists():
        raise FileNotFoundError(f"Pesos iniciais não encontrados: {pesos_iniciais}")

    auditoria = auditar(data)
    print(
        f"[AUDITORIA] {sum(s['imagens'] for s in auditoria['splits'].values())} imagens; "
        f"{len(auditoria['erros'])} erro(s); {len(auditoria['avisos'])} aviso(s)."
    )
    for aviso in auditoria["avisos"]:
        print(f"[AVISO] {aviso}")
    if auditoria["erros"]:
        for erro in auditoria["erros"]:
            print(f"[ERRO] {erro}")
        raise RuntimeError("Dataset reprovado pela auditoria; treino cancelado.")
    if args.somente_auditar:
        return 0

    device = escolher_device(args.device)
    if device == "cpu":
        print("[AVISO] CUDA/MPS indisponível: o treino será executado em CPU e pode demorar bastante.")

    id_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_execucao = f"oculos_candidato_{id_execucao}"
    print(f"[TREINO] device={device} | início={pesos_iniciais.name} | execução={nome_execucao}")

    modelo = YOLO(pesos_iniciais)
    parametros_treino = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cache="ram",
        device=device,
        workers=4,
        project=str(PROJETO / "saidas"),
        name=nome_execucao,
        exist_ok=False,
        save_period=10,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=8,
        translate=0.1,
        scale=0.35,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
    )
    if args.horas is not None:
        parametros_treino["time"] = args.horas
    treino = modelo.train(**parametros_treino)

    melhor = Path(treino.save_dir) / "weights/best.pt"
    if not melhor.exists():
        raise FileNotFoundError(f"Treino terminou sem gerar {melhor}")

    pasta_candidatos = PROJETO / "modelos/candidatos"
    pasta_candidatos.mkdir(parents=True, exist_ok=True)
    candidato = pasta_candidatos / f"oculos_{id_execucao}.pt"
    shutil.copy2(melhor, candidato)

    splits = ["val"] if args.sem_teste else ["val", "test"]
    relatorio = {
        "execucao": id_execucao,
        "data": str(data),
        "baseline": str(baseline),
        "pesos_iniciais": str(pesos_iniciais),
        "candidato": str(candidato),
        "device": str(device),
        "horas_planejadas": args.horas,
        "conf_operacional": args.conf_operacional,
        "auditoria": auditoria,
        "avaliacoes": {},
        "promovido_automaticamente": False,
    }
    for split in splits:
        print(f"[VALIDAÇÃO] Comparando baseline e candidato no split {split}...")
        metricas_base = avaliar(
            baseline,
            data,
            split,
            device,
            f"{id_execucao}_baseline_{split}",
            args.imgsz,
            args.conf_operacional,
        )
        metricas_novas = avaliar(
            candidato,
            data,
            split,
            device,
            f"{id_execucao}_candidato_{split}",
            args.imgsz,
            args.conf_operacional,
        )
        relatorio["avaliacoes"][split] = {
            "baseline": metricas_base,
            "candidato": metricas_novas,
            "comparacao": comparar(metricas_base, metricas_novas),
        }

    splits_reprovados = [
        split
        for split, avaliacao in relatorio["avaliacoes"].items()
        if not avaliacao["comparacao"].get("apto_offline", False)
    ]
    relatorio["decisao_offline"] = {
        "apto_em_todos_os_splits": not splits_reprovados,
        "splits_avaliados": splits,
        "splits_reprovados": splits_reprovados,
        "promover_em_producao": False,
        "motivo": (
            "Ainda exige validação em um lote de campo rotulado."
            if not splits_reprovados
            else "O candidato não passou em todos os splits offline."
        ),
    }

    caminho_relatorio = Path(treino.save_dir) / "comparacao.json"
    caminho_relatorio.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio["avaliacoes"], indent=2, ensure_ascii=False))
    print(f"\n[CANDIDATO] {candidato}")
    print(f"[RELATÓRIO] {caminho_relatorio}")
    print("[PRODUÇÃO] modelos/oculos.pt não foi alterado. Faça primeiro o teste de campo rotulado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
