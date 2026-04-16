"""
Carregamento e pré-processamento do dataset FinQA.

O FinQA fornece contexto pré-extraído (tabelas + texto) para cada pergunta.
NÃO há etapa de retrieval — a tarefa é puramente raciocínio numérico.
"""

import json
import random
import urllib.request
from dataclasses import dataclass
from typing import Optional


# ── Constantes ──────────────────────────────────────────────
FINQA_BASE_URL = (
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
)
SPLITS = ("train", "dev", "test")


# ── Estrutura de dados ──────────────────────────────────────
@dataclass
class FinQAExample:
    """Um par pergunta-resposta do FinQA com contexto pré-extraído."""

    id: str
    question: str
    context: str          # Texto + tabela formatados
    gold_answer: str      # Resposta textual anotada
    exe_ans: float | str  # Resultado numérico da execução do programa
    program: str          # Programa DSL (representação nested)
    table_raw: list       # Tabela original (lista de listas)
    pre_text: list        # Texto antes da tabela
    post_text: list       # Texto depois da tabela


# ── Formatação de contexto ──────────────────────────────────
def format_table(table: list[list[str]]) -> str:
    """Formata a tabela do FinQA como texto legível."""
    if not table:
        return ""

    # Calcular larguras das colunas
    col_widths = [0] * len(table[0])
    for row in table:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    lines = []
    for row_idx, row in enumerate(table):
        formatted = " | ".join(
            str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        lines.append(formatted)
        if row_idx == 0:
            lines.append("-" * len(formatted))

    return "\n".join(lines)


def build_context(example_raw: dict) -> str:
    """Constrói o contexto textual a partir dos campos do FinQA."""
    parts = []

    # Texto pré-tabela
    if example_raw.get("pre_text"):
        parts.append("### Relevant Text (Before Table)")
        parts.append("\n".join(example_raw["pre_text"]))

    # Tabela
    if example_raw.get("table"):
        parts.append("\n### Financial Table")
        parts.append(format_table(example_raw["table"]))

    # Texto pós-tabela
    if example_raw.get("post_text"):
        parts.append("\n### Relevant Text (After Table)")
        parts.append("\n".join(example_raw["post_text"]))

    return "\n\n".join(parts)


# ── Carregamento ────────────────────────────────────────────
def _download_split(split: str) -> list[dict]:
    """Baixa um split do FinQA do GitHub."""
    url = f"{FINQA_BASE_URL}/{split}.json"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def _parse_example(raw: dict) -> FinQAExample:
    """Converte um exemplo bruto do FinQA em FinQAExample."""
    qa = raw["qa"]
    return FinQAExample(
        id=raw.get("id", "unknown"),
        question=qa["question"],
        context=build_context(raw),
        gold_answer=qa.get("answer", ""),
        exe_ans=qa.get("exe_ans", ""),
        program=qa.get("program_re", qa.get("program", "")),
        table_raw=raw.get("table", []),
        pre_text=raw.get("pre_text", []),
        post_text=raw.get("post_text", []),
    )


def load_finqa(
    split: str = "dev",
    subset: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[FinQAExample]:
    """
    Carrega o dataset FinQA.

    Args:
        split: "train", "dev" ou "test"
        subset: Se não None, retorna apenas N exemplos (amostra aleatória)
        seed: Seed para reprodutibilidade do subsample
        cache_dir: Diretório para cache local (futuro)

    Returns:
        Lista de FinQAExample
    """
    assert split in SPLITS, f"Split deve ser um de {SPLITS}"

    raw_data = _download_split(split)
    examples = [_parse_example(raw) for raw in raw_data]

    if subset is not None and subset < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, subset)

    return examples


# ── Utilitários ─────────────────────────────────────────────
def get_few_shot_examples(
    train_data: list[FinQAExample],
    indices: list[int],
) -> list[FinQAExample]:
    """Seleciona exemplos demonstrativos por índice."""
    return [train_data[i] for i in indices if i < len(train_data)]


# ── Teste rápido ────────────────────────────────────────────
if __name__ == "__main__":
    print("Carregando FinQA dev split...")
    dev = load_finqa("dev", subset=5)
    for ex in dev:
        print(f"\n{'='*60}")
        print(f"ID: {ex.id}")
        print(f"Q:  {ex.question}")
        print(f"A:  {ex.gold_answer} (exe_ans={ex.exe_ans})")
        print(f"Program: {ex.program}")
        print(f"Context (primeiros 200 chars): {ex.context[:200]}...")
