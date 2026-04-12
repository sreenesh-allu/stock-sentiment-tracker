import csv
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FINBERT_MODEL = "ProsusAI/finbert"
_HEADLINE_COL_WIDTH = 48
_COL_MANUAL = 10
_COL_PRED = 12
_COL_MATCH = 12


def _compound_to_label(compound: float) -> str:
    """Match sentiment buckets used in app.py /sentiments."""
    if compound > 0.05:
        return "positive"
    if compound < -0.05:
        return "negative"
    return "neutral"


def _load_labeled_rows(csv_path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            headline = (row.get("headline") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if headline:
                rows.append((headline, label))
    return rows


def _finbert_predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    headline: str,
) -> str:
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(dim=-1).item())
    return model.config.id2label[pred_id].lower()


def _build_results(rows: list[tuple[str, str]]) -> list[dict]:
    analyzer = SentimentIntensityAnalyzer()
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()

    results: list[dict] = []
    for headline, manual in rows:
        v_pred = _compound_to_label(analyzer.polarity_scores(headline)["compound"])
        f_pred = _finbert_predict(model, tokenizer, headline)
        results.append(
            {
                "headline": headline,
                "manual": manual,
                "vader": v_pred,
                "vader_ok": v_pred == manual,
                "finbert": f_pred,
                "finbert_ok": f_pred == manual,
            }
        )
    return results


def _print_comparison_table(results: list[dict]) -> None:
    header = (
        f"{'#':<4}"
        f"{'Headline':<{_HEADLINE_COL_WIDTH}}"
        f"{'Manual':<{_COL_MANUAL}}"
        f"{'VADER':<{_COL_PRED}}"
        f"{'VADER':<{_COL_MATCH}}"
        f"{'FinBERT':<{_COL_PRED}}"
        f"{'FinBERT':<{_COL_MATCH}}"
    )
    sub = (
        f"{'':<4}"
        f"{'':<{_HEADLINE_COL_WIDTH}}"
        f"{'label':<{_COL_MANUAL}}"
        f"{'prediction':<{_COL_PRED}}"
        f"{'match':<{_COL_MATCH}}"
        f"{'prediction':<{_COL_PRED}}"
        f"{'match':<{_COL_MATCH}}"
    )
    rule = "-" * len(header)

    print()
    print("Detailed comparison")
    print(rule)
    print(header)
    print(sub)
    print(rule)

    for i, r in enumerate(results, start=1):
        h_lines = textwrap.wrap(
            r["headline"].replace("\n", " "),
            width=_HEADLINE_COL_WIDTH,
        ) or [""]
        v_match = "correct" if r["vader_ok"] else "incorrect"
        f_match = "correct" if r["finbert_ok"] else "incorrect"

        line0 = (
            f"{i:<4}"
            f"{h_lines[0]:<{_HEADLINE_COL_WIDTH}}"
            f"{r['manual']:<{_COL_MANUAL}}"
            f"{r['vader']:<{_COL_PRED}}"
            f"{v_match:<{_COL_MATCH}}"
            f"{r['finbert']:<{_COL_PRED}}"
            f"{f_match:<{_COL_MATCH}}"
        )
        print(line0)
        for hl in h_lines[1:]:
            print(f"{'':<4}{hl}")

    print(rule)
    print()


def evaluate_labeled_headlines(csv_path: str) -> tuple[float, float]:
    """
    Load labeled_headlines.csv, run VADER and FinBERT on each headline,
    print a detailed comparison table and accuracy summary.

    Returns (vader_accuracy_pct, finbert_accuracy_pct) in [0, 100].
    """
    rows = _load_labeled_rows(csv_path)
    if not rows:
        print("No labeled headlines to evaluate.")
        return 0.0, 0.0

    results = _build_results(rows)
    _print_comparison_table(results)

    v_ok = sum(1 for r in results if r["vader_ok"])
    f_ok = sum(1 for r in results if r["finbert_ok"])
    n = len(results)
    vader_pct = 100.0 * v_ok / n
    finbert_pct = 100.0 * f_ok / n

    print(f"VADER accuracy:    {vader_pct:.1f}% ({v_ok}/{n} correct)")
    print(f"FinBERT accuracy: {finbert_pct:.1f}% ({f_ok}/{n} correct)")

    return vader_pct, finbert_pct


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent / "labeled_headlines.csv"
    evaluate_labeled_headlines(str(default_csv))
