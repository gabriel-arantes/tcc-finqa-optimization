"""
Coleta, persistência e exportação de resultados experimentais.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.metrics import EvaluationReport, PredictionResult


def save_report(report: EvaluationReport, results_dir: str = "results"):
    """
    Salva relatório de avaliação em JSON + CSV.

    Gera:
      - results/<pipeline>_summary.json  → métricas agregadas
      - results/<pipeline>_predictions.csv → predições individuais
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = report.pipeline_name

    # Sumário JSON
    summary = report.summary()
    summary["timestamp"] = timestamp
    summary_path = Path(results_dir) / f"{name}_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Predições CSV
    csv_path = Path(results_dir) / f"{name}_predictions_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "question",
                "gold_exe_ans",
                "predicted_answer",
                "exec_acc",
                "prog_acc",
                "input_tokens",
                "output_tokens",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        for p in report.predictions:
            writer.writerow({
                "example_id": p.example_id,
                "question": p.question[:100],
                "gold_exe_ans": p.gold_exe_ans,
                "predicted_answer": p.predicted_answer,
                "exec_acc": p.exec_acc,
                "prog_acc": p.prog_acc,
                "input_tokens": p.input_tokens,
                "output_tokens": p.output_tokens,
                "latency_seconds": round(p.latency_seconds, 3),
            })

    print(f"Resultados salvos em:")
    print(f"  Sumário: {summary_path}")
    print(f"  Predições: {csv_path}")
    return summary_path, csv_path


def load_all_summaries(results_dir: str = "results") -> list[dict]:
    """Carrega todos os sumários de resultados."""
    summaries = []
    for path in sorted(Path(results_dir).glob("*_summary_*.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def generate_comparison_table(results_dir: str = "results") -> str:
    """
    Gera tabela comparativa entre todos os pipelines avaliados.

    Formato LaTeX para inclusão direta no artigo SBC.
    """
    summaries = load_all_summaries(results_dir)
    if not summaries:
        return "Nenhum resultado encontrado."

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparação entre engenharia de prompts manual e otimização automática de prompts.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Pipeline} & \textbf{Exec. Acc. (\%)} & \textbf{Tokens} & \textbf{Latência (s)} \\",
        r"\hline",
    ]

    for s in summaries:
        name = s["pipeline"].replace("_", r"\_")
        lines.append(
            f"  {name} & {s['execution_accuracy']:.2f} & "
            f"{s['total_tokens']:,} & {s['avg_latency_s']:.2f} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
