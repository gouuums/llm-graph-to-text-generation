# Graph-to-Text Generation with Large Language Models: A brief analysis

**Authors:** Thomas Goumont, Melvin Bazeille, Vojtech John, Kassem Anis Bouali, Romita Pawar, Ines Ben Moussa 
**Institution:** Charles University 
**Supervisor:** Ondřej Dušek 
**Course:** NPFL140 - Large Language Models (LLMs) 

## Abstract
This repository contains the code, data processing scripts, and evaluation metrics for investigating the use of Large Language Models (LLMs) in graph-to-text (G2T) generation tasks. By utilizing the WebNLG dataset from the GEM benchmark, we evaluate how advanced LLMs interpret structured data (RDF triplets) and generate coherent, semantically precise natural language. 

## Methodology

### Dataset
*   We use the English portion of the WebNLG dataset, which contains approximately 13,000 graph-text training pairs derived from DBpedia.
*   The data covers domains such as Airports, Astronauts, Universities, and Monuments. 
*   Each graph contains between 1 to 7 Resource Description Framework (RDF) triplets
*   The dataset provides original triple sets (e.g., `Aarhus_Airport | cityServed | "Aarhus, Denmark"`) and human-annotated lexicalisations (e.g., "Aarhus Airport serves the city of Aarhus, Denmark.").

### Model architecture
The methodology employs a dual-encoder architecture that processes both text and graph topology:
*   **Text Encoder:** A Transformer-based language model (such as LLaMA2-7B or TinyLlama) encodes the natural language query to capture semantic meaning.
*   **Graph Encoder:** A Graph Neural Network (GNN), such as TransformerConv or GATConv, processes the relevant subgraph to create a graph-aware embedding that preserves structural relationships.
*   **Decoder:** The embeddings from the text and graph encoders are concatenated and passed to a decoder to generate the final response.

### Training protocol
*   The models are trained using supervised fine-tuning with the AdamW optimizer.
*   The training loop includes gradient clipping, learning rate adjustments, and Weights & Biases (`wandb`) integration for logging metrics.
*   We utilize Parameter-Efficient Fine-Tuning, specifically Low-Rank Adaptation (LoRA), to ensure stable and efficient training.
*   Early stopping is implemented to monitor validation loss and prevent overfitting, saving the best checkpoint for inference.

## Evaluation and results

Model performance is automatically evaluated using a custom evaluation script (`eval.py`). The metrics utilized include:
*   **BLEU:** Measures n-gram precision.
*   **Exact Match:** Checks for exact string matches between predictions and references.
*   **BERTScore:** Computes cosine similarity between token embeddings (requires the `-gpu` flag to run efficiently).
*   **METEOR:** Evaluates semantic similarity considering synonymy and stemming.

### Key findings
*   The scale of the base LLM was found to be more critical for success than the specialized graph-aware components on the WebNLG dataset.
*   The **LLaMA2-7B + TransformerConv** model outperformed all other configurations, achieving a BLEU score of ~0.499, an Exact Match of ~0.063, and a BERTScore of ~0.958.
*   Among the smaller GNN models, the **TinyLlama + TransformerConv + LoRA** variant achieved a highly competitive BLEU score of ~0.487 and a BERTScore of ~0.954. 
*   The LoRA-tuned variants consistently demonstrated superior stability and slightly better performance than their fully fine-tuned counterparts.
*   Models with fewer parameters (e.g., the 3B model) struggled significantly, dropping to a BLEU score of ~0.199 and an Exact Match of 0.0.

## Usage

**1. Training**
Run the main training script. Ensure your dataset is stored in `dataset/webnlg/`.
```bash
python3 src/main.py --model_name [MODEL] --batch_size 32
```
2. Evaluation
To evaluate the generated .jsonl prediction files, use the evaluation script:

```
./run_eval.sh output/llama2-7b/graphllm_pred_test2.jsonl
```
Note: The shell script automatically installs requirements and passes the -gpu flag for BERTScore computation
