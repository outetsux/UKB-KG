# UKB-KG

**UKB-KG: Knowledge Graph for Integrating and Enhancing Biomedical Insights from the UK Biobank**

- **Paper Code Repository:** [UKB-KG](https://github.com/outetsux/UKB-KG.git)
- **Platform Repository:** [UKB-KG-App](https://github.com/outetsux/ukb-kg-app.git)
- **UKB-KG Platform:** [Access the Online Platform](https://ukb-kg-app.streamlit.app)

---

## Contents

Code for the paper *UKB-KG: Knowledge Graph for Integrating and Enhancing Biomedical Insights from the UK Biobank*. Each part of the repository corresponds to a numbered section of the paper:

| Directory | Paper section                                                      |
| --- |--------------------------------------------------------------------|
| `extract_from_text/`, `extract_from_table/` | Section 2; Supplementary B.2 (Triple Extraction)                   |
| `Proprocess/` | Supplementary B.3 (Triple Refinement), B.4 (Contextual Enrichment) |
| `extract_from_text/calc_eval_*.py` | Section 3.1 (Graph Evaluation)                                     |
| `disease_prediction/` | Section 3.2 (Multi-Disease Prediction)                             |
| `rag/` | Section 3.3 (Retrieval-Augmented Generation)                       |
| `prompts/` | Supplementary E                                 |

## UKB-KG Construction

The pipeline is driven by two scripts. `run_extraction.sh` takes a list of articles to their verified triples; `run_postprocess.sh` takes those triples to the final knowledge graph. Both read their settings from a single config file and are resumable, so an interrupted run is continued by re-issuing the same command.

Use Python 3.10 and prepare the external data and models listed under Data Requirements.

```bash
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
export OPENAI_API_KEY=...                          # see LLM Settings below

cp configs/default.sh configs/my_run.sh            # edit RUN_NAME, LLM, NUM_SHARDS, paths
./run_extraction.sh   --config configs/my_run.sh
./run_postprocess.sh  --config configs/my_run.sh
```

Run outputs, including selected tables, are written under `$SAVE_ROOT/$RUN_NAME/`; parsed tables are cached in `TABLE_DIR`. Extraction settings are recorded in `run_config.json`. The outputs are:


| Artifact                                          | Path                                             |
| ------------------------------------------------- | ------------------------------------------------ |
| Triple records before CSV cleanup                      | `save/$RUN_NAME/merged/10_triples.json`          |
| Knowledge graph, edge list with confidence scores | `save/$RUN_NAME/csv/triples_with_attributes.csv` |


`configs/default.sh` is the only file that needs editing for a normal run.

Once the required data and models are prepared, the two driver scripts run the construction stages.

### LLM Settings

`llm/openai_chat.py` reads credentials from the environment; set the ones for the providers you use:


| Variable           | Used for                                       | Default base URL           |
| ------------------ | ---------------------------------------------- | -------------------------- |
| `OPENAI_API_KEY`   | GPT-5 series, extraction and self-verification | OpenAI                     |
| `DEEPSEEK_API_KEY` | DeepSeek judge, cross-model verification only  | `https://api.deepseek.com` |
| `QWEN_API_KEY`     | Qwen judge, cross-model verification only      | Alibaba DashScope          |


`OPENAI_BASE_URL`, `DEEPSEEK_BASE_URL` and `QWEN_BASE_URL` override the endpoints for proxies or self-hosted deployments. Clients are constructed lazily, so a missing key only matters if that provider is actually called. Both structured (schema-constrained) and unstructured outputs are supported; the structured prompts in `prompts/` are the ones used in the paper.

### Stages

`run_extraction.sh` runs the text chain and then the table chain. `--source text` or `--source table` runs one of them, `--from <stage>` resumes at a stage, and `--only <stage>` runs a single stage.


| #   | Text stage              | Output                     | Table stage                    | Output                      |
| --- | ----------------------- | -------------------------- | ------------------------------ | --------------------------- |
| 0   |                         |                            | markdown tables from XML       | `PMC_Data/tables/`          |
| 0   |                         |                            | select relation-bearing tables | `table/00_select.json`      |
| 1   | NER + triple extraction | `text/01_extract.json`     | triple extraction from tables  | `table/01_extract.json`     |
| 2   | LLM triple filtering    | `text/02_llm_filter.json`  | same                           | `table/02_llm_filter.json`  |
| 3   | LLM triple revision     | `text/03_refine.json`      | same                           | `table/03_refine.json`      |
| 4   | rule-based filtering    | `text/04_rule_filter.json` | same                           | `table/04_rule_filter.json` |
| 5   | self-verification       | `text/05_verify.json`      | same                           | `table/05_verify.json`      |


`run_postprocess.sh` then continues, with the same `--from` and `--only` flags:


| #   | Stage                                                 | Script                                              | Output                                      |
| --- | ----------------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| 6   | merge text and table triples                          | `Proprocess/00_merge_triples.py`                    | `merged/06_merged.json`                     |
| 7   | entity alignment to UMLS                              | `Proprocess/01_umls_link.py`                        | `merged/07_umls_linked.json`                |
| 8   | case unification, dedup, entity typing                | `Proprocess/03_process.py`                          | `merged/08_processed.json`                  |
| 9   | KG fusion with BIOS, and typing of the added entities | `Proprocess/02_bios_completion.py`, `03_process.py` | `merged/09_bios_completion_typed.json`      |
| 10  | flatten to an edge list                               | `Proprocess/04_json_to_csv.py`                      | `merged/10_triples.json`, `csv/triples.csv` |
| 11  | per-edge confidence scores                            | `Proprocess/05_add_attributes.py`                   | `csv/triples_with_attributes.csv`           |


Stage 9 is skipped by setting `ENABLE_BIOS_COMPLETION=0`, which builds the graph from literature triples alone.

#### Parallelism

LLM stages run across `NUM_SHARDS` concurrent processes. Each shard writes `<stage>/shard_001_of_004.json` (for example), or `<stage>/all.json` for one shard; the driver concatenates the files into `<stage>.json`. A stage skips articles already recorded in its shard output. Use a new `RUN_NAME` for different settings, and keep the configuration, inputs and `NUM_SHARDS` unchanged when resuming.

`utils/pipeline.py` also runs standalone for inspecting or repairing a run:

```bash
python -m utils.pipeline stats        --input save/my_run/text/05_verify.json
python -m utils.pipeline merge-shards --stage-dir save/my_run/text/01_extract \
                                      --output    save/my_run/text/01_extract.json
```



### Evaluation

Extraction quality is measured by an LLM judge stronger than the extraction model (`EVAL_LLM` in the config), on a sample of articles:

```bash
source configs/my_run.sh
python -m extract_from_text.03_verify_or_eval --mode eval \
    --input "$SAVE_ROOT/$RUN_NAME/text/05_verify.json" \
    --judgement-output "$SAVE_ROOT/$RUN_NAME/eval/text_judgements.json" \
    --xml-dir "$XML_DIR" --llm "$EVAL_LLM" --effort "$EVAL_EFFORT"
python -m extract_from_text.calc_eval_scores \
    --input "$SAVE_ROOT/$RUN_NAME/eval/text_judgements.json"
```

`calc_eval_scores.py` reports precision. For recall, `recall_eval.py` and `calc_eval_recall_scores.py` compare the extracted triples against a reference set. `cross_model_verify.py` combines judgements from several models by majority vote, as an alternative to single-model self-verification.

### Baseline Extraction

Study-cohort demographics come from the tables that `table_select` routed to `BASELINE_TABLE_DIR`:

```bash
./run_extraction.sh --config configs/my_run.sh --source table --only baseline
```

This reuses the table selection from the triple run; `--only select` first if the extraction chain has not been run.

### Data Requirements

Included in this repository:


| Data                      | Path                                     | Notes                                                       |
| ------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| Sample article list       | `PMC_Data/pmcid-ukb-xml-sample.txt`      | 50 PMCIDs                                                   |
| Sample full-text XML      | `PMC_Data/xml/`                          | the corresponding PMC Open Access articles                  |
| UMLS semantic-group table | `Proprocess/entity_type_tag/umls/SemGroups_UKB.txt` | the type vocabulary of the KG                               |
| Prompts                   | `prompts/`                               | every prompt used in the paper, including ablation variants |


Not included, and where to get it:


| Data                            | Config setting                   | Needed for                                   | Source                                                                                             |
| ------------------------------- | -------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| The full UKB literature corpus  | `PAPER_LIST`, `XML_DIR`          | reproducing the full KG rather than a sample | PMC Open Access subset; the shipped list is a 50-article sample of it                              |
| Publication metadata            | `PUBLICATION_METADATA`           | stage 11 confidence scores                   | PMCID, PMID, date, journal, title, keywords, ... from PubMed                                       |
| BIOS v3 core data               | `BIOS_CONCEPTS`, `BIOS_TRIPLETS` | stage 9 KG fusion                            | [BIOS](https://huggingface.co/datasets/THUMedInfo/BIOS_v3). Set `ENABLE_BIOS_COMPLETION=0` to skip |
| UMLS `MRCONSO.RRF`, `MRSTY.RRF` | —                                | training the entity-typing model             | UMLS Metathesaurus, requires a UTS license. Place under `Proprocess/entity_type_tag/umls/`                    |
| Entity-typing model             | `ENTITY_TYPE_MODEL`              | stages 8 and 9                               | prepare BioBERT in `model/biobert_v1.0_pubmed_pmc`; run `Proprocess/entity_type_tag/01_generate_data.py`, then `bash Proprocess/entity_type_tag/02_train.sh` (set the GPU in that script)                                |
| scispaCy UMLS KB                | —                                | stage 7 entity alignment                     | downloaded automatically by scispaCy on first use                                                  |
| BioBERT                         | `UKB_BIOBERT`                    | disease prediction, RAG                      | biobert_v1.2 (disease prediction)                                                                                       |


## Disease Prediction

The downstream disease-prediction experiments live in `disease_prediction/`: the numbered scripts prepare KG inputs and patient features, with an external KGE training step required before `05_to_ene.py`. `train.sh` and `test.sh` separately train and evaluate the predictor.

These experiments use individual-level UK Biobank cohort data, which cannot be redistributed. A small synthetic cohort with the same file layout, shapes and dtypes is therefore included for a runnable classifier demo; see [disease_prediction/data/README.md](disease_prediction/data/README.md) for what it contains and how to run the pipeline on it.

## Retrieval-Augmented Generation

See [rag/README.md](rag/README.md) for data preparation and RAG commands.
