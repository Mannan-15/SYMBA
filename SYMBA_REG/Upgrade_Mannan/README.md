# Symba 2026: Next-Gen Transformers for Symbolic Regression

**Read the Full GSoC 2026 Proposal:** [Symba 2026 Proposal (Google Docs)](https://docs.google.com/document/d/1WaaLbe_9OeSylVhjENV5TdaDmsZNnZ357YENC6tNcUw/edit?usp=sharing)

Welcome to my experimental repository for the ML4SCI Symba project. This codebase contains the Proof of Concept (PoC) experiments and architectural upgrades designed for my GSoC 2026 proposal. 

This repository directly addresses the following ML4SCI Common Tasks:

> **Common Task 1.1: Dataset Preprocessing**
> **Dataset:** [AI Feynman](https://space.mit.edu/home/tegmark/aifeynman.html) *(Note: Authors not affiliated with ML4SCI)*
> **Description:** Download the `Feynman_with_units.tar.gz` features and corresponding `FeynmanEquations.csv` targets. Preprocess and tokenize the target data and document your rationale for the choice of tokenization.

> **Common Task 2.6: Next-Gen Transformers Seeding Generative Models**
> **Description:** Use a Transformer model to seed a generative technique for symbolic regression. Build on previous code solutions from ML4SCI GSoC 2025 combining models and generative frameworks.

---

## How to Run the Models

To replicate my experiments and run the proposed architectures locally, follow these steps:

**1. Clone the repository and navigate to the upgrade directory:**
```bash
git clone [https://github.com/Mannan-15/SYMBA.git](https://github.com/Mannan-15/SYMBA.git)
cd ./SYMBA_REG/Upgrade_Mannan
```

2. Run the Proposed Postfix Decoder (Standard Inference):

```bash
python3 src/models/sliding_window.py
```

3. Run the Beam Search Inference (Generative Seeding):

To run the Top-K beam search decoder used to uncover exposure bias and seed the M

```bash
python3 src/baselines/old_sliding_window.py
```

4. Run the Legacy Prefix Baseline (For Comparison):

To run the original 2024/2025 baseline model and observe the sequence bloat and baseline accuracy:

```bash
python3 src/models/updated_sliding_window.py
```

## Key Experiments & Architectural Updates

To fulfill these tasks, I audited the 2024/2025 ML4SCI baselines and engineered three major architectural upgrades:

### 1. Tokenization Rationale: Postfix + `<C>` (Task 1.1)
The legacy baseline relied on bloated Prefix notation and discrete digit prediction, which caused massive sequence lengths and severe hallucination of physical constants.

* **The Upgrade:** I built a mathematically enforced Postfix tokenizer that completely removes redundant parentheses and tokens. Furthermore, I replaced all discrete floating-point numbers with a continuous `<C>` embedding token (xVal).
* **Result:** The maximum sequence length dropped from 67 tokens to 48 tokens.
* **Impact:** This structural compression nearly doubled the Exact Match accuracy from **25.7% (Baseline)** to **47.4% (Proposed)** on the test split.

### 2. Next-Gen Generative Core: KAN vs. MLP (Task 2.6)
To push the generative seeding capabilities further, I experimented with replacing the standard linear MLPs inside the Transformer blocks with Kolmogorov-Arnold Networks (KANs).

* **The Experiment:** Located in `src/models/kan_mlp.py`, I benchmarked both layers.
* **Result:** By using learnable B-splines on the edges instead of fixed linear node activations, the KAN converged significantly faster and achieved a lower MSE floor, proving its superiority for mapping continuous physical geometries.

### 3. Inference Upgrades: Beam Search & Exposure Bias (Task 2.6)
To properly seed a generative Monte Carlo Tree Search (MCTS), a model must output Top-K candidate skeletons rather than a single greedy prediction.

* **The Upgrade:** I upgraded the baseline decoder to utilize Beam Search (`src/models/updated_sliding_window.py`).
* **Discovery:** Running this revealed severe **Exposure Bias** in the baseline models (collapsing to 0.00% exact match during beam search), proving that standard teacher-forcing is insufficient for generative seeding. This directly motivates my 2026 proposal to align the architecture using Group Relative Policy Optimization (GRPO).

---

## Repository Navigation

* `src/baselines/`: The original 2024/2025 Prefix & Greedy decoding scripts used for benchmarking.
* `src/models/`: **[My Contributions]** The updated Postfix Decoder, Beam Search logic, and KAN experiments.
* `src/parser/`: **[My Contributions]** The custom Postfix AST and continuous token masking logic.
* `src/embeddings/` & `src/labels/`: Custom continuous embeddings and newly generated Postfix JSON parse trees.
