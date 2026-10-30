# RAG

Answer questions using neighboring KG triples and graph paths.

## Inputs

- Question JSON: question IDs mapped to objects containing QUESTION.
- KG CSV: HeadName, TailName, RelName, PMCID, Confidence_Score, Frequency_Score.
- Graph NPZ: entities and embeddings arrays.
- The same KG in Neo4j.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL. RAG also requires
NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD; NEO4J_DATABASE is optional.

## Run

Before these commands, build the KG with `run_extraction.sh` and
`run_postprocess.sh`, then run `python -m disease_prediction.03_get_entity_emb`
to create `node_emb.npz`. Set `UKB_KG_ROOT`/`UKB_KG_RUN` to the same
`SAVE_ROOT`/`RUN_NAME` used for extraction, and use the same `UKB_BIOBERT`
checkpoint for graph and question embeddings. Separately import the final CSV
into Neo4j, with node `name` equal to `HeadName`/`TailName` and directed
relationship `name` equal to `RelName`.

From the repository root:

    python -m rag.01_get_umls_map_emb --data pubmedqa --device cpu
    python -m rag.02_rag --data pubmedqa --method rag

Inputs and generated entity caches default to `rag/data/<dataset>/`.
KG files default to `save/sample/csv/{triples_with_attributes.csv,node_emb.npz}`;
change `--kg-dir` as needed. UKB_BIOBERT selects the local BioBERT checkpoint.

For custom data, set `--data-path` and `--cache-dir` during preparation, then
`--data-path`, `--data-entity-emb` and `--output-dir` during QA. Use separate
cache/output directories for different datasets. `--data` selects the answer
format: pubmedqa allows yes/no/maybe, and bioasq allows yes/no.

Use `--method cot` or `--method none` for the baselines. RAG/CoT save separate
0-shot and 1-shot predictions in `rag/save`. Runs resume saved progress.
All questions in the input file are processed with the provided prompts.
See --help for model, retrieval and path options.

## Evaluate

    python -m rag.evaluation pubmedqa path/to/predictions.json

