# Project Structure

```text
jepa_novel_lab/
  app.py
  requirements.txt
  AGENTS.md
  README.md
  TASKS.md
  .env.example

  configs/
    default.yaml

  data/
    raw/
    synthetic/
    filtered/
    embeddings/
    indexes/

  checkpoints/
    predictor/

  reports/
    runs/

  prompts/
    00_project_scaffold.txt
    01_data_generation.txt
    02_embedding_and_training.txt
    03_generation_and_evaluation.txt
    04_streamlit_oneclick_ui.txt
    05_hardening_and_reports.txt

  src/
    llm/
      ollama_client.py
      prompts.py
    data/
      generate_synthetic.py
      validate_jsonl.py
      filter_dataset.py
      scene_schema.py
    embedding/
      embed_scenes.py
      vector_store.py
    planner/
      dataset.py
      model.py
      train.py
      predict.py
    generation/
      generate_baseline.py
      generate_with_rag.py
      generate_with_jepa.py
    evaluation/
      metrics.py
      judge.py
      report.py
    utils/
      config.py
      logging.py
      paths.py
```
