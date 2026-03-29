<h1>Symba 2026: Next-Gen Transformers for Symbolic Regression</h1>

<p><strong>Read the Full GSoC 2026 Proposal:</strong> <a href="https://docs.google.com/document/d/1WaaLbe_9OeSylVhjENV5TdaDmsZNnZ357YENC6tNcUw/edit?usp=sharing">Symba 2026 Proposal (Google Docs)</a></p>

<p>Welcome to my experimental repository for the ML4SCI Symba project. This codebase contains the Proof of Concept (PoC) experiments and architectural upgrades designed for my GSoC 2026 proposal.</p>

<p>This repository directly addresses the following ML4SCI Common Tasks:</p>

<blockquote>
  <p><strong>Common Task 1.1: Dataset Preprocessing</strong><br>
  <strong>Dataset:</strong> <a href="https://space.mit.edu/home/tegmark/aifeynman.html">AI Feynman</a> <em>(Note: Authors not affiliated with ML4SCI)</em><br>
  <strong>Description:</strong> Download the <code>Feynman_with_units.tar.gz</code> features and corresponding <code>FeynmanEquations.csv</code> targets. Preprocess and tokenize the target data and document your rationale for the choice of tokenization.</p>
</blockquote>

<blockquote>
  <p><strong>Common Task 2.6: Next-Gen Transformers Seeding Generative Models</strong><br>
  <strong>Description:</strong> Use a Transformer model to seed a generative technique for symbolic regression. Build on previous code solutions from ML4SCI GSoC 2025 combining models and generative frameworks.</p>
</blockquote>

<hr>

<h2>Quick Links for Evaluators</h2>
<p>To make reviewing the evaluation tasks as seamless as possible, here is exactly where the relevant code and data for each task are located:</p>

<h3>Task 1.1: Tokenization &amp; Parsing</h3>
<ul>
  <li><strong>The Parser:</strong> <a href="https://github.com/Mannan-15/SYMBA/blob/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan/src/parser/symbolic_parser.py">src/parser/symbolic_parser.py</a></li>
  <li><strong>Tokenized Target Data:</strong> The fully parsed and tokenized postfix equations are stored in <a href="https://github.com/Mannan-15/SYMBA/blob/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan/src/parser/symbolic_parser.py">src/parser/feynman_parse_trees_postfix.json</a></li>
  <li><strong>Vocab Builder:</strong> The Postfix vocabulary builder logic is located in <a href="https://github.com/Mannan-15/SYMBA/blob/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan/src/models/vocab_builder_postfix.py">src/models/vocab_builder_postfix.py</a></li>
</ul>

<h3>Task 2.6: Next-Gen Transformer Models</h3>
<p>All architectural experiments and model scripts are located in the <code>src/models/</code> directory.<br>
<em>(Note: Any file with "old" in the name, such as <code>old_sliding_window.py</code>, refers to the previous 2024/2025 baseline from Krish Malik's proposal and is included purely for benchmarking comparisons).</em></p>
<ul>
  <li><code>sliding_window.py</code>: The proposed upgraded model utilizing TNet embeddings, continuous <code>&lt;C&gt;</code> tokenization, and Postfix encoding.</li>
  <li><code>updated_sliding_window.py</code>: The proposed model equipped with Beam Search inference to output Top-K candidate skeletons.</li>
  <li><code>kan_mlp.py</code>: The isolated experiment benchmarking Kolmogorov-Arnold Networks (KAN) against standard MLPs for the Transformer blocks.</li>
</ul>

<hr>

<h2>How to Run the Models</h2>
<p>To replicate my experiments and run the proposed architectures locally, follow these steps:</p>

<p><strong>1. Clone the repository and navigate to the upgrade directory:</strong></p>
<pre><code>git clone -b Mannan-upgrade --single-branch https://github.com/Mannan-15/SYMBA.git
cd ./SYMBA_REG/Upgrade_Mannan</code></pre>

<p><strong>2. Parse the AI Feynman Dataset (Task 1.1):</strong><br>
First, execute the symbolic parser to convert the raw equations into mathematically compressed Postfix JSON parse trees:</p>
<pre><code>python3 src/parser/symbolic_parser.py</code></pre>

<p><strong>3. Build the Postfix Vocabulary (Task 1.1):</strong><br>
Next, generate the continuous <code>&lt;C&gt;</code> token mappings and the final vocabulary file required by the encoders:</p>
<pre><code>python3 src/models/vocab_builder_postfix.py</code></pre>

<p><strong>4. Run the Proposed Postfix Decoder (Standard Inference):</strong></p>
<pre><code>python3 src/models/sliding_window.py</code></pre>

<p><strong>5. Run the Beam Search Inference (Generative Seeding):</strong><br>
To run the Top-K beam search decoder used to uncover exposure bias and seed the MCTS:</p>
<pre><code>python3 src/models/updated_sliding_window.py</code></pre>

<p><strong>6. Run the KAN vs. MLP Experiment:</strong><br>
Benchmarks the Kolmogorov-Arnold Network against standard linear MLPs for physical geometries:</p>
<pre><code>python3 src/models/kan_mlp.py</code></pre>

<p><strong>7. Run the Legacy Prefix Baseline (For Comparison):</strong><br>
To run the original 2024/2025 baseline model and observe the sequence bloat and baseline accuracy:</p>
<pre><code>python3 src/baselines/old_sliding_window.py</code></pre>

<hr>

<h2>Key Experiments &amp; Architectural Updates</h2><b>(for more details, check the proposal's section 2)</b>
<p>To fulfill these tasks, I audited the 2024/2025 ML4SCI baselines and engineered three major architectural upgrades:</p>

<h3>1. Tokenization Rationale: Postfix + <code>&lt;C&gt;</code> (Task 1.1)</h3>
<p>The legacy baseline relied on bloated Prefix notation and discrete digit prediction, which caused massive sequence lengths and severe hallucination of physical constants.</p>
<ul>
  <li><strong>The Upgrade:</strong> I built a mathematically enforced Postfix tokenizer that completely removes redundant parentheses and tokens. Furthermore, I replaced all discrete floating-point numbers with a continuous <code>&lt;C&gt;</code> embedding token (xVal).</li>
  <li><strong>Result:</strong> The maximum sequence length (vocab size) dropped from 67 tokens to 48 tokens.</li>
  <li><strong>Impact:</strong> This structural compression nearly doubled the Exact Match accuracy from <strong>25.7% (Baseline)</strong> to <strong>47.4% (Proposed)</strong> on the test split.</li>
</ul>

<h3>2. Next-Gen Generative Core: KAN vs. MLP (Task 2.6)</h3>
<p>To push the generative seeding capabilities further, I experimented with replacing the standard linear MLPs inside the Transformer blocks with Kolmogorov-Arnold Networks (KANs).</p>
<ul>
  <li><strong>The Experiment:</strong> Located in <code>src/models/kan_mlp.py</code>, I benchmarked both layers.</li>
  <li><strong>Result:</strong> By using learnable B-splines on the edges instead of fixed linear node activations, the KAN converged significantly faster and achieved a lower MSE floor, proving its superiority for mapping continuous physical geometries.</li>
</ul>

<h3>3. Inference Upgrades: Beam Search &amp; Exposure Bias (Task 2.6)</h3>
<p>To properly seed a generative Monte Carlo Tree Search (MCTS), a model must output Top-K candidate skeletons rather than a single greedy prediction.</p>
<ul>
  <li><strong>The Upgrade:</strong> I upgraded the baseline decoder to utilize Beam Search (<code>src/models/updated_sliding_window.py</code>).</li>
  <li><strong>Discovery:</strong> Running this revealed severe <strong>Exposure Bias</strong> in the baseline models (collapsing to 0.00% exact match during beam search), proving that standard teacher-forcing is insufficient for generative seeding. This directly motivates my 2026 proposal to align the architecture using Group Relative Policy Optimization (GRPO).</li>
</ul>

<hr>

<h2>Repository Navigation</h2>
<ul>
  <li><code>src/baselines/</code>: The original 2024/2025 Prefix &amp; Greedy decoding scripts used for benchmarking.</li>
  <li><code>src/models/</code>: <strong>[My Contributions]</strong> The updated Postfix Decoder, Beam Search logic, and KAN experiments.</li>
  <li><code>src/parser/</code>: <strong>[My Contributions]</strong> The custom Postfix AST and continuous token masking logic.</li>
  <li><code>src/embeddings/</code> &amp; <code>src/labels/</code>: Custom continuous embeddings and newly generated Postfix JSON parse trees.</li>
</ul>
