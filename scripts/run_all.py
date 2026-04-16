#!/usr/bin/env python3
"""
Script principal: Executa TODOS os experimentos do TCC.

Fluxo:
  1. Carrega dataset FinQA
  2. Executa baseline (engenharia de prompts manual)
  3. Executa otimizadores DSPy (BootstrapFewShot, MIPROv2, GEPA)
  4. Gera tabela comparativa LaTeX
  5. Salva todos os resultados

Uso:
    # Experimento completo
    python scripts/run_all.py

    # Teste rápido com 20 exemplos
    python scripts/run_all.py --eval_subset 20 --train_subset 100

    # Múltiplas rodadas para variância
    python scripts/run_all.py --num_runs 3
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_finqa
from src.llm_client import LLMClient
from src.baseline_manual import ManualBaselinePipeline
from src.dspy_module import finqa_to_dspy_examples
from src.dspy_pipelines import (
    configure_dspy_lm,
    optimize_bootstrap_few_shot,
    optimize_miprov2,
    optimize_gepa,
    evaluate_dspy_module,
    save_optimized_module,
)
from src.results_collector import save_report, generate_comparison_table


def run_baseline(eval_examples, args):
    """Executa o pipeline baseline."""
    print(f"\n{'#'*60}")
    print("# BASELINE — Engenharia de Prompts Manual")
    print(f"{'#'*60}\n")

    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
        seed=args.seed,
    )

    pipeline = ManualBaselinePipeline(llm_client=llm)
    report = pipeline.evaluate(eval_examples, verbose=True)
    print(report)
    save_report(report, args.results_dir)
    return report


def run_optimizer(name, train_dspy, val_dspy, eval_examples, args):
    """Executa um otimizador DSPy."""
    print(f"\n{'#'*60}")
    print(f"# OTIMIZADOR — {name}")
    print(f"{'#'*60}\n")

    start = time.time()

    if name == "bootstrap_few_shot":
        optimized = optimize_bootstrap_few_shot(
            trainset=train_dspy,
            max_bootstrapped_demos=5,
            max_labeled_demos=5,
            max_rounds=3,
        )
    elif name == "miprov2":
        optimized = optimize_miprov2(
            trainset=train_dspy,
            valset=val_dspy,
            num_candidates=10,
            max_bootstrapped_demos=5,
            max_labeled_demos=5,
            num_trials=30,
            seed=args.seed,
        )
    elif name == "gepa":
        optimized = optimize_gepa(
            trainset=train_dspy,
            valset=val_dspy,
            seed=args.seed,
        )

    opt_time = time.time() - start
    print(f"Otimização {name} concluída em {opt_time:.1f}s")

    save_optimized_module(optimized, f"{args.results_dir}/optimized_{name}")

    report = evaluate_dspy_module(
        module=optimized,
        examples=eval_examples,
        pipeline_name=f"dspy_{name}",
        split=args.eval_split,
        verbose=True,
    )
    print(report)
    save_report(report, args.results_dir)
    return report, opt_time


def main():
    parser = argparse.ArgumentParser(description="Executa todos os experimentos do TCC")
    parser.add_argument("--eval_subset", type=int, default=None)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Pipelines para pular: baseline bootstrap miprov2 gepa"
    )
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("=" * 60)
    print("TCC — Otimização Automática de Prompts vs. Engenharia Manual")
    print(f"Modelo: {args.provider}/{args.model}")
    print(f"Avaliação: {args.eval_split} (subset={args.eval_subset or 'todos'})")
    print(f"Treino: subset={args.train_subset or 'todos'}")
    print(f"Runs: {args.num_runs} | Seed: {args.seed}")
    print("=" * 60)

    # ── Carregar dados ──
    print("\nCarregando dataset FinQA...")
    eval_examples = load_finqa(args.eval_split, subset=args.eval_subset, seed=args.seed)
    train_raw = load_finqa("train", subset=args.train_subset, seed=args.seed)
    print(f"  Eval: {len(eval_examples)} | Train: {len(train_raw)}")

    train_dspy = finqa_to_dspy_examples(train_raw)
    val_split = int(len(train_dspy) * 0.8)
    val_dspy = train_dspy[val_split:]
    train_dspy_opt = train_dspy[:val_split]

    # ── Configurar DSPy (necessário para otimizadores) ──
    needs_dspy = any(
        opt not in args.skip
        for opt in ["bootstrap", "miprov2", "gepa"]
    )
    if needs_dspy:
        configure_dspy_lm(
            provider=args.provider,
            model=args.model,
        )

    # ── Executar experimentos ──
    all_results = {}

    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n{'*'*60}")
            print(f"* RUN {run_idx + 1}/{args.num_runs}")
            print(f"{'*'*60}")
            run_seed = args.seed + run_idx
        else:
            run_seed = args.seed

        # Baseline
        if "baseline" not in args.skip:
            report = run_baseline(eval_examples, args)
            all_results.setdefault("manual_baseline", []).append(report.summary())

        # Otimizadores DSPy
        for opt_name in ["bootstrap_few_shot", "miprov2", "gepa"]:
            short_name = opt_name.replace("_few_shot", "").replace("ootstrap", "")
            if short_name in args.skip or opt_name in args.skip:
                print(f"\nPulando {opt_name}...")
                continue

            report, opt_time = run_optimizer(
                opt_name, train_dspy_opt, val_dspy, eval_examples, args
            )
            summary = report.summary()
            summary["optimization_time_seconds"] = round(opt_time, 1)
            all_results.setdefault(f"dspy_{opt_name}", []).append(summary)

    # ── Resultados finais ──
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("RESULTADOS FINAIS")
    print(f"{'='*60}")
    print(f"Tempo total: {total_time:.1f}s ({total_time/60:.1f}min)\n")

    # Tabela comparativa
    print(f"{'Pipeline':<28} {'Exec Acc':>10} {'Tokens':>12} {'Latência':>10}")
    print("-" * 65)
    for pipeline_name, runs in all_results.items():
        # Média entre runs
        avg_acc = sum(r["execution_accuracy"] for r in runs) / len(runs)
        avg_tokens = sum(r["total_tokens"] for r in runs) / len(runs)
        avg_lat = sum(r["avg_latency_s"] for r in runs) / len(runs)
        print(f"  {pipeline_name:<26} {avg_acc:>8.2f}% {avg_tokens:>12,.0f} {avg_lat:>8.2f}s")

    # Salvar consolidado
    consolidated_path = Path(args.results_dir) / "consolidated_results.json"
    with open(consolidated_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados consolidados: {consolidated_path}")

    # Tabela LaTeX
    latex_table = generate_comparison_table(args.results_dir)
    latex_path = Path(args.results_dir) / "comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"Tabela LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
