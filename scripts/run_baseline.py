#!/usr/bin/env python3
"""
Script: Executa o pipeline baseline de engenharia de prompts manual.

Uso:
    python scripts/run_baseline.py [--subset N] [--split dev|test]

Variáveis de ambiente necessárias:
    OPENAI_API_KEY   (ou ANTHROPIC_API_KEY se provider=anthropic)
"""

import argparse
import sys
import os

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_finqa
from src.llm_client import LLMClient
from src.baseline_manual import ManualBaselinePipeline
from src.results_collector import save_report


def main():
    parser = argparse.ArgumentParser(
        description="Executa pipeline baseline — engenharia de prompts manual"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Número de exemplos (None = todos)"
    )
    parser.add_argument(
        "--split", type=str, default="dev", choices=["dev", "test"],
        help="Split do FinQA para avaliação"
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="Provider do LLM: openai | anthropic"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Modelo a utilizar"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reprodutibilidade"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Pipeline: Engenharia de Prompts Manual (Baseline)")
    print(f"Modelo: {args.provider}/{args.model}")
    print(f"Split: {args.split} | Subset: {args.subset or 'todos'}")
    print(f"Retoma automaticamente de checkpoint se interrompido")
    print(f"{'='*60}\n")

    # 1. Carregar dados
    print("Carregando dataset FinQA...")
    examples = load_finqa(
        split=args.split,
        subset=args.subset,
        seed=args.seed,
    )
    print(f"  {len(examples)} exemplos carregados.\n")

    # 2. Inicializar LLM
    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
        seed=args.seed,
    )

    # 3. Executar pipeline
    pipeline = ManualBaselinePipeline(llm_client=llm)
    print("Executando avaliação...\n")
    report = pipeline.evaluate(examples, verbose=True)

    # 4. Resultados
    print(report)
    save_report(report)


if __name__ == "__main__":
    main()
