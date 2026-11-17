# Dataset Datasheet (Datasheets for Datasets - Gebru et al.)

## 1. Motivation
- **For what purpose was the dataset created?**  
  To evaluate whether Small Language Models (0.5B–8B parameters) fine-tuned via LoRA/QLoRA can match or exceed large general-purpose models (and RAG pipelines) in task accuracy, strict formatting, latency, and operational inference cost on specialized domain tasks.
- **Who created the dataset?**  
  Project research team.
- **Who funded the creation of the dataset?**  
  Self-funded / academic research.

## 2. Composition
- **What do the instances that comprise the dataset represent?**  
  Standard multi-turn or single-turn conversational instances formatted as `messages`:
  ```json
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```
- **How many instances are there in total?**  
  Targeting 2,000 to 10,000 curated instances split 80% train, 10% validation, 10% frozen test.
- **Does the dataset contain all possible instances or is it a sample?**  
  It is a focused sample representing typical domain queries and instructions.
- **What data does each instance consist of?**  
  Instruction prompt, optional system instruction, and ground-truth verified completion.

## 3. Collection Process
- **How was the data acquired?**  
  Curated from authorized public repositories, open government databases, domain documents, or human expert demonstrations.
- **What mechanisms or procedures were used to collect the data?**  
  Document chunking, prompt-response pairing, and expert verification.
- **Who was involved in the data collection process and what were their roles?**  
  Research annotators and domain reviewers.

## 4. Preprocessing, Cleaning, and Labeling
- **Was any preprocessing/cleaning done?**  
  - SHA-256 / MinHash exact and near-duplicate removal (`src/data/preprocessor.py`).
  - Strict chat schema validation (`src/data/dataset_loader.py`).
  - Document-level split grouping to prevent cross-partition contamination.
  - Personal identifiable information (PII) scrubbing.

## 5. Uses
- **Has the dataset been used for specific tasks already?**  
  Used to train and evaluate parameter-efficient fine-tuning (LoRA/QLoRA) models and RAG baselines.
- **What tasks should this dataset NOT be used for?**  
  Clinical diagnostic decisions without physician oversight, high-stakes legal binding contracts without human counsel.

## 6. Distribution & Licensing
- **How will the dataset be distributed?**  
  Via Hugging Face Datasets Hub and GitHub repository releases.
- **License:**  
  CC-BY-4.0 / Apache 2.0.
