#!/usr/bin/env python3
"""
Script: Executa os pipelines de otimização automática de prompts via DSPy.

Uso:
    python scripts/run_optimizers.py [--optimizer all|bootstrap|miprov2|gepa] [--subset N]

Variáveis de ambiente necessárias:
    OPENAI_API_KEY   (ou ANTHROPIC_API_KEY se provider=anthropic)
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_finqa
from src.dspy_module import finqa_to_dspy_examples, FinQAModule
from src.dspy_pipelines import (
    configure_dspy_lm,
    optimize_bootstrap_few_shot,
    optimize_miprov2,
    optimize_gepa,
    optimize_knn_few_shot,
    optimize_simba,
    evaluate_dspy_module,
    save_optimized_module,
)
from src.results_collector import save_report


def run_single_optimizer(
    name: str,
    train_dspy: list,
    val_dspy: list,
    eval_examples: list,
    args: argparse.Namespace,
):
    """Executa otimização + avaliação para um otimizador."""
    print(f"\n{'='*60}")
    print(f"Otimizador: {name}")
    print(f"{'='*60}")

    # ── Otimização ──
    print(f"\n[1/3] Otimizando com {name}...")
    start = time.time()

    if name == "bootstrap_few_shot":
        optimized = optimize_bootstrap_few_shot(
            trainset=train_dspy,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            max_rounds=args.max_rounds,
        )
    elif name == "miprov2":
        optimized = optimize_miprov2(
            trainset=train_dspy,
            valset=val_dspy,
            auto_level=args.auto_level,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            seed=args.seed,
        )
    elif name == "gepa":
        optimized = optimize_gepa(
            trainset=train_dspy,
            valset=val_dspy,
            auto_level=args.auto_level,
            seed=args.seed,
        )
    elif name == "knn_few_shot":
        optimized = optimize_knn_few_shot(
            trainset=train_dspy,
            k=args.max_bootstrapped_demos,
        )
    elif name == "simba":
        optimized = optimize_simba(
            trainset=train_dspy,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Otimizador desconhecido: {name}")

    optimization_time = time.time() - start
    print(f"  Otimização concluída em {optimization_time:.1f}s")

    # ── Salvar módulo otimizado ──
    save_path = f"results/optimized_{name}_seed{args.seed}"
    try:
        save_optimized_module(optimized, save_path)
    except Exception as e:
        print(f"  ⚠ Não foi possível salvar módulo ({name}): {e}")

    # ── Inspecionar prompt otimizado ──
    print(f"\n[2/3] Inspecionando prompt otimizado...")
    try:
        # DSPy armazena demos e instruções nos predictors
        for pred_name, predictor in optimized.named_predictors():
            print(f"\n  Predictor: {pred_name}")
            if hasattr(predictor, "demos") and predictor.demos:
                print(f"    Demos selecionados: {len(predictor.demos)}")
                for j, demo in enumerate(predictor.demos):
                    q = getattr(demo, "question", "N/A")
                    print(f"      Demo {j+1}: {q[:80]}...")
            if hasattr(predictor, "signature"):
                sig = predictor.signature
                if hasattr(sig, "instructions") and sig.instructions:
                    print(f"    Instrução otimizada: {str(sig.instructions)[:200]}...")
    except Exception as e:
        print(f"  Erro ao inspecionar: {e}")

    # ── Avaliação ──
    print(f"\n[3/3] Avaliando {name} no split de avaliação...")
    report = evaluate_dspy_module(
        module=optimized,
        examples=eval_examples,
        pipeline_name=f"dspy_{name}",
        split=args.eval_split,
        verbose=True,
    )

    # Adicionar tempo de otimização ao relatório
    report_summary = report.summary()
    report_summary["optimization_time_seconds"] = round(optimization_time, 1)

    print(report)
    save_report(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Executa pipelines de otimização automática de prompts via DSPy"
    )
    parser.add_argument(
        "--optimizer", type=str, default="all",
        choices=["all", "bootstrap", "miprov2", "gepa", "knn", "simba"],
        help="Qual otimizador executar"
    )
    parser.add_argument(
        "--train_subset", type=int, default=None,
        help="Subsample do train para otimização"
    )
    parser.add_argument(
        "--eval_subset", type=int, default=None,
        help="Subsample do split de avaliação"
    )
    parser.add_argument(
        "--eval_split", type=str, default="dev", choices=["dev", "test"],
        help="Split para avaliação final"
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="Provider do LLM"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Modelo a utilizar"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Hiperparâmetros dos otimizadores
    parser.add_argument("--max_bootstrapped_demos", type=int, default=5)
    parser.add_argument("--max_labeled_demos", type=int, default=5)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument(
        "--auto_level", type=str, default="medium",
        choices=["light", "medium", "heavy"],
        help="Nível de otimização para MIPROv2/GEPA (light/medium/heavy)"
    )

    args = parser.parse_args()

    # ── Configurar DSPy ──
    print("Configurando DSPy LM...")
    configure_dspy_lm(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
    )

    # ── Carregar dados ──
    print("\nCarregando dataset FinQA...")
    train_raw = load_finqa("train", subset=args.train_subset, seed=args.seed)
    eval_raw = load_finqa(args.eval_split, subset=args.eval_subset, seed=args.seed)
    print(f"  Train: {len(train_raw)} exemplos")
    print(f"  Eval ({args.eval_split}): {len(eval_raw)} exemplos")

    # Converter para formato DSPy
    train_dspy = finqa_to_dspy_examples(train_raw)

    # Separar validação para MIPROv2/GEPA (últimos 20% do train)
    val_split = int(len(train_dspy) * 0.8)
    val_dspy = train_dspy[val_split:]
    train_dspy_opt = train_dspy[:val_split]
    print(f"  Train (otimização): {len(train_dspy_opt)} exemplos")
    print(f"  Validação: {len(val_dspy)} exemplos")

    # ── Executar otimizadores ──
    optimizers_to_run = {
        "all": ["bootstrap_few_shot", "miprov2", "gepa", "knn_few_shot", "simba"],
        "bootstrap": ["bootstrap_few_shot"],
        "miprov2": ["miprov2"],
        "gepa": ["gepa"],
        "knn": ["knn_few_shot"],
        "simba": ["simba"],
    }[args.optimizer]

    reports = {}
    for opt_name in optimizers_to_run:
        # KNNFewShot usa trainset completo (busca KNN em tempo de inferência)
        train_for_opt = train_dspy if opt_name == "knn_few_shot" else train_dspy_opt
        report = run_single_optimizer(
            name=opt_name,
            train_dspy=train_for_opt,
            val_dspy=val_dspy,
            eval_examples=eval_raw,
            args=args,
        )
        reports[opt_name] = report

    # ── Sumário comparativo ──
    print(f"\n{'='*60}")
    print("SUMÁRIO COMPARATIVO — Otimização Automática de Prompts")
    print(f"{'='*60}")
    print(f"{'Pipeline':<25} {'Exec Acc':>10} {'Tokens':>10} {'Latência':>10}")
    print("-" * 60)
    for name, r in reports.items():
        s = r.summary()
        print(
            f"  dspy_{name:<21} {s['execution_accuracy']:>8.2f}% "
            f"{s['total_tokens']:>10,} {s['avg_latency_s']:>8.2f}s"
        )


if __name__ == "__main__":
    main()