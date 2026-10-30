"""Stage 3 (text): self-verification, and LLM-as-judge evaluation.

Both modes ask an LLM to label every triple Correct/Incorrect against the
source article; they differ in what is kept:

  verify  the pipeline step -- writes the Correct triples to --verified-output
  eval    the measurement step -- writes labels only, so a stronger judge model
          can score an already-finished triple set

Judgement files are keyed by the triple's 0-based position in the article's
triple list, which is what extract_from_text/calc_eval_scores.py and
cross_model_verify.py expect.
"""

import argparse
import inspect
import io
import json
import re
import sys
import time

import prompts
from pydantic import BaseModel

from llm.openai_chat import chat_deepseek, chat_qwen, chat_structured
from utils import pipeline
from utils.pmc_xml import (
    build_xml_index,
    count_tokens,
    find_xml_file,
    get_article_text,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)

# Models without structured-output support; their replies are parsed as markdown.
UNSTRUCTURED_MODELS = {"deepseek": chat_deepseek, "qwen": chat_qwen}


class ResultForm(BaseModel):
    Result: str


class EvaluationForm(BaseModel):
    Evaluations: list[ResultForm]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Triples to judge.")
    p.add_argument("--judgement-output", required=True, help="Per-triple labels.")
    p.add_argument("--verified-output", default="", help="Required for --mode verify.")
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--mode", required=True, choices=["verify", "eval"])
    p.add_argument("--llm", required=True)
    p.add_argument("--effort", default="minimal")
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--evaluation-prompt", default="evaluation_prompt_structured")
    p.add_argument("--title-file", default="")
    args = p.parse_args()
    if args.mode == "verify" and not args.verified_output:
        p.error("--verified-output is required for --mode verify")
    return args


def parse_json_markdown(content):
    """Best-effort JSON extraction from a possibly fenced LLM reply."""
    if not isinstance(content, str):
        print("❌ Input is not a string")
        return None

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON in code block: {e}")
                return None

    print("❌ No valid JSON found")
    return None


def judge_article(article_text, triples, args):
    """Label every triple of one article. Returns {index: label} or None."""
    triples_text = "".join(
        f"{i + 1}. [ {t['head']['name']} | {t['relation']['name']} | {t['tail']['name']} ]\n"
        for i, t in enumerate(triples)
    )
    if not triples_text:
        return None, 0

    prompt = (
        getattr(prompts, args.evaluation_prompt)
        .replace("<<text>>", article_text)
        .replace("<<triples>>", triples_text)
    )

    if args.llm in UNSTRUCTURED_MODELS:
        response = parse_json_markdown(UNSTRUCTURED_MODELS[args.llm](prompt))
    else:
        response = chat_structured(args.llm, prompt, EvaluationForm, args.effort)

    tokens_count = count_tokens(prompt)
    print(f"Prompt has {tokens_count} tokens")

    # A judgement list of the wrong length cannot be aligned to the triples.
    if response is None or len(response["Evaluations"]) != len(triples):
        raise ValueError("Invalid judgement response or mismatched triple count")

    labels = {i: r["Result"] for i, r in enumerate(response["Evaluations"])}
    return labels, tokens_count


def run(args, records):
    done = pipeline.open_output(args.judgement_output)
    if args.mode == "verify":
        pipeline.open_output(args.verified_output)

    print(f"Loaded {len(records)} triple records from {args.input}")
    print(f"Building XML index under {args.xml_dir} ...")
    xml_index = build_xml_index(args.xml_dir)
    print(f"✅ Indexed {len(xml_index)} XML files")

    valid_articles = 0
    total_tokens = 0
    start = time.time()

    for record in records:
        file_name = next(iter(record.keys()))
        if file_name in done:
            continue

        print(f"Evaluating: {file_name}")
        triples = record[file_name]

        xml_path = find_xml_file(file_name, xml_index)
        if xml_path is None:
            raise FileNotFoundError(f"XML not found for {file_name} under {args.xml_dir}")

        try:
            article_text = get_article_text(xml_path, file_name, args.title_file)
            if not article_text:
                continue

            labels, tokens_count = judge_article(article_text, triples, args)
            if labels is None:
                continue

            if args.mode == "verify":
                verified = [triples[i] for i, v in labels.items() if v == "Correct"]
                if verified:
                    pipeline.append_records(
                        args.verified_output, [{file_name: verified}]
                    )
                    pipeline.append_records(
                        args.judgement_output, [{file_name: labels}]
                    )
                    print(
                        f"Verified {len(verified)}/{len(triples)} triples for {file_name}"
                    )
            else:
                pipeline.append_records(args.judgement_output, [{file_name: labels}])

            valid_articles += 1
            total_tokens += tokens_count
            print(
                f"Average tokens per valid article: {total_tokens / valid_articles:.2f}"
            )
        except Exception as e:
            print(f"❌ {e}")
            print(inspect.currentframe().f_lineno)
            raise

    runtime = (time.time() - start) / 60
    print(f"\n====================== {args.mode} Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = parse_args()

    all_records = pipeline.read_records(args.input)
    records = pipeline.shard_items(all_records, args.shard, args.num_shards)

    print(f"Mode: {args.mode}, judge: {args.llm}")
    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(records)} of {len(all_records)} articles"
    )
    print(f"Input: {args.input}")
    print(f"Judgement output: {args.judgement_output}")
    if args.mode == "verify":
        print(f"Verified output:  {args.verified_output}")

    run(args, records)
