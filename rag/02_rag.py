"""Run graph-augmented, chain-of-thought or direct question answering."""

import argparse
import json
import os
import re
from pathlib import Path

from rag.core import parse_answer, read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["pubmedqa", "bioasq"], default="pubmedqa")
    parser.add_argument("--data-path")
    parser.add_argument("--method", choices=["rag", "cot", "none"], default="rag")
    parser.add_argument("--llm", default="gpt-5")
    parser.add_argument("--effort", default="minimal")
    parser.add_argument("--params", type=int, choices=[1], default=1)
    parser.add_argument("--score", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--output-dir", default="rag/save")
    parser.add_argument("--kg-dir", default=os.path.join(
        os.environ.get("UKB_KG_ROOT", "save"), os.environ.get("UKB_KG_RUN", "sample")))
    parser.add_argument("--triple-csv-path")
    parser.add_argument("--ukb-node-emb")
    parser.add_argument("--data-entity-emb")
    parser.add_argument("--biobert", default=os.environ.get("UKB_BIOBERT", "model/biobert_v1.2"))
    args = parser.parse_args()
    args.data_path = args.data_path or f"rag/data/{args.data}/test_set.json"
    args.data_entity_emb = args.data_entity_emb or f"rag/data/{args.data}/test_set_entity_map_emb.npz"
    args.triple_csv_path = args.triple_csv_path or str(Path(args.kg_dir) / "csv/triples_with_attributes.csv")
    args.ukb_node_emb = args.ukb_node_emb or str(Path(args.kg_dir) / "csv/node_emb.npz")
    args.neo4j_uri = os.environ.get("NEO4J_URI")
    args.neo4j_user = os.environ.get("NEO4J_USER")
    args.neo4j_password = os.environ.get("NEO4J_PASSWORD")
    args.neo4j_database = os.environ.get("NEO4J_DATABASE")
    if args.method == "rag" and not all(
        [args.neo4j_uri, args.neo4j_user, args.neo4j_password]
    ):
        parser.error("Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD.")
    return args


def run(args):
    import prompts
    import tiktoken
    from llm.openai_chat import chat_wo_structure

    with open(args.data_path, encoding="utf-8") as stream:
        items = list(json.load(stream).items())

    outputs = []
    for shot in ([""] if args.method == "none" else ["_0shot", "_1shot"]):
        prompt_text = getattr(prompts, f"{args.data}_{args.method}{shot}")
        name = f"{args.llm}_{args.effort}_{args.data}_{args.method}{shot}"
        if args.method == "rag":
            name += f"_param-{args.params}_score-{args.score}"
        path = Path(args.output_dir) / f"{name}.json"
        outputs.append((prompt_text, path, read_json(path)))

    encoder = tiktoken.encoding_for_model("gpt-4o")
    retriever = None
    try:
        if args.method == "rag":
            from rag.retrieval import GraphRetriever
            retriever = GraphRetriever(args)
        for qid, details in items:
            pending = [(prompt_text, path, results) for prompt_text, path, results in outputs
                       if qid not in results]
            question = details.get("QUESTION", "")
            if not pending or not question:
                continue
            try:
                context = ""
                if retriever is not None:
                    context = retriever.retriever(qid, question)
                    context = encoder.decode(encoder.encode(context)[:10000])
                for prompt_text, path, results in pending:
                    prompt = prompt_text.replace("<<question>>", question)
                    if args.method == "rag":
                        prompt = re.sub(r"\n{3,}", "\n\n", prompt.replace("<<context>>", context))
                    response = chat_wo_structure(args.llm, prompt, args.effort)
                    answer = parse_answer(response, args.data)
                    if answer is None:
                        raise ValueError(f"{qid}: invalid answer for {path.name}; not saved.")
                    results[qid] = answer
                    write_json(path, results)
            except Exception as exc:
                print(f"{qid}: {exc}")
                raise
    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    run(parse_args())
