<h1 align="center">Symba 2026: LM-JEPA for Symbolic Regression</h1>

<p align="center"><strong>Read the Full GSoC 2026 Proposal:</strong></p>

<p>This codebase contains the Proof of Concept (PoC) experiments, continuous tokenization pipeline, and Joint-Embedding architectures designed for my GSoC 2026 proposal.</p>

<p>This repository directly addresses the following ML4SCI Common Tasks:</p>

<blockquote>
  <p><strong>Common Task 1.1: Dataset Preprocessing</strong><br>
  <strong>Dataset:</strong> <a href="https://space.mit.edu/home/tegmark/aifeynman.html">AI Feynman</a> <em>(Note: Authors not affiliated with ML4SCI)</em><br>
  <strong>Description:</strong> Download the <code>Feynman_with_units.tar.gz</code> features and corresponding <code>FeynmanEquations.csv</code> targets. Preprocess and tokenize the target data and document your rationale for the choice of tokenization.</p>
</blockquote>

<blockquote>
  <p><strong>Common Task 2.7: LM-JEPA for Symbolic Regression</strong><br>
  <strong>Description:</strong> Transformer model integrated with LM-JEPA pretraining.</p>
</blockquote>

<hr>

<h2>Quick Links for Evaluators</h2>

<h3>Task 1.1: Tokenization &amp; Parsing</h3>
<ul>
  <li><strong>The Parser:</strong> <a href="src/parser/symbolic_parser.py">src/parser/symbolic_parser.py</a></li>
  <li><strong>Tokenized Target Data:</strong> The fully parsed and tokenized postfix equations are stored in <a href="data/feynman_parse_trees.json">data/feynman_parse_trees.json</a></li>
  <li><strong>Vocab Builder:</strong> The Postfix vocabulary builder logic is located in <a href="src/models/vocab_builder_postfix.py">src/models/vocab_builder_postfix.py</a></li>
</ul>

<h3>Task 2.7: LM-JEPA Architecture</h3>
<p>All continuous-space architectural experiments and model scripts are located in the <code>src/</code> directory.<br>
<em>(Note: Any file with "old" or "prefix" in the name, such as <code>old_sliding_window_prefix.py</code>, refers to the previous 2024/2025 generative baselines and is included purely for benchmarking comparisons).</em></p>
<ul>
  <li><strong>LM-JEPA Core Modules:</strong> <a href="src/models/jepa_modules.py">src/models/jepa_modules.py</a> (Contains the Context Encoder, Target Encoder, and Predictor).</li>
  <li><strong>Full Inference Pipeline:</strong> <a href="src/models/sliding_window.py">src/models/sliding_window.py</a> (The complete end-to-end generative inference pipeline integrating T-Net inputs, continuous <code>&lt;C&gt;</code> tokens, and Sparse Transformer blocks).</li>
  <li><strong>Pretraining Loop &amp; VICReg Loss:</strong> <a href="src/train_jepa_poc.py">src/train_jepa_poc.py</a> (The main execution script proving representation stability).</li>
  <li><strong>KAN Upgrades:</strong> <a href="src/models/kan_mlp.py">src/models/kan_mlp.py</a> (Comparison between Kolmogorov-Arnold Networks and MLP for Symbolic Regression task).</li>
</ul>

<hr>

<h2>How to Run the Models</h2>
<p>To replicate my experiments and run the proposed LM-JEPA architecture locally, follow these steps:</p>

<p><strong>1. Directly clone the LM-JEPA branch and navigate to the working directory:</strong></p>
<pre><code>git clone -b LM-JEPA --single-branch https://github.com/Mannan-15/SYMBA.git
cd ./SYMBA/SYMBA_REG/LM-JEPA_Mannan</code></pre>

<p><strong>2. Parse the AI Feynman Dataset (Task 1.1):</strong><br>
First, execute the symbolic parser to convert the raw equations into mathematically compressed Postfix JSON parse trees:</p>
<pre><code>python3 src/parser/symbolic_parser.py</code></pre>

<p><strong>3. Build the Postfix Vocabulary (Task 1.1):</strong><br>
Next, generate the continuous <code>&lt;C&gt;</code> token mappings and the final vocabulary file required by the encoders:</p>
<pre><code>python3 src/models/vocab_builder_postfix.py</code></pre>

<p><strong>4. Run the LM-JEPA Pretraining Pipeline (Task 2.7):</strong><br>
Executes the joint-embedding forward pass and calculates the VICReg loss to prevent representation collapse:</p>
<pre><code>python3 src/train_jepa_poc.py</code></pre>

<p><strong>5. Run the Postfix + &lt;C&gt; Tokenizer Baseline (Task 1.1):</strong><br>
Runs the standard sliding-window inference using the optimized tokenization scheme:</p>
<pre><code>python3 src/models/sliding_window.py</code></pre>

<p><strong>6. Run the Beam Search Inference (Generative Seeding):</strong><br>
To run the Top-K beam search decoder used to uncover exposure bias:</p>
<pre><code>python3 src/models/updated_sliding_window.py</code></pre>

<p><strong>7. Run the KAN vs. MLP Experiment:</strong><br>
Benchmarks the Kolmogorov-Arnold Network against standard linear MLPs for physical geometries:</p>
<pre><code>python3 src/models/kan_mlp.py</code></pre>

<p><strong>8. Run the Legacy Prefix Baseline (For Comparison with Postfix):</strong><br>
To run the original 2024/2025 baseline model and observe the sequence bloat and baseline accuracy:</p>
<pre><code>python3 src/baselines/old_sliding_window.py</code></pre>

<hr>

<h2>Key Experiments &amp; Architectural Updates</h2>
<p><b>(For deep technical details and loss landscape graphs, please refer to Section 2 of my written proposal)</b></p>
<p>To fulfill these tasks, I audited the 2024/2025 ML4SCI baselines and engineered three major architectural upgrades to shift Symba from <em>syntactic text generation</em> to <em>semantic physics prediction</em>:</p>

<h3>1. Tokenization Rationale, T-Net, &amp; Inference Upgrades (Task 1.1)</h3>
<p>The legacy baseline relied on bloated Prefix notation and discrete digit prediction, which caused massive sequence lengths and severe hallucination of physical constants.</p>
<ul>
  <li><strong>Data Encoding (T-Net):</strong> I integrated and retained the T-Net encoder for the physical tabular data <code>(x)</code>. Its permutation-invariant architecture is mathematically required to handle unordered sets of physical observations without injecting artificial sequence biases.</li>
  <li><strong>Postfix vs. Prefix + <code>&lt;C&gt;</code> Tokenization:</strong> Prefix notation caused severe sequence bloat. I engineered a mathematically enforced Postfix tokenizer that completely removes redundant parentheses. Additionally, I replaced all discrete floating-point numbers with a continuous <code>&lt;C&gt;</code> embedding token.</li>
  <li><strong>Beam Search &amp; Exposure Bias:</strong> I implemented a Top-K Beam Search inference script to test the generative capabilities. Testing revealed severe <strong>Exposure Bias</strong> in the baseline models (collapsing to 0.00% exact match during autoregressive rollout). This proved that standard teacher-forcing is highly brittle for math, directly motivating my shift to a JEPA continuous-space architecture.</li>
  <li><strong>Impact:</strong> The maximum sequence length dropped from 67 tokens down to 48. This structural compression nearly doubled the greedy Exact Match accuracy from <strong>25.7% (Baseline)</strong> to <strong>47.4% (Proposed)</strong> on my PoC test split.</li>
</ul>

<h3>2. The LM-JEPA Core, Architecture, &amp; VICReg (Task 2.7)</h3>
<p>A standard L2 prediction loss in a Joint-Embedding environment causes both networks to instantly collapse and output vectors of all zeros. To prevent this, I engineered a stabilized continuous-space pipeline.</p>
<ul>
  <li><strong>The Architecture:</strong> The pipeline pairs a Context Encoder (utilizing Sparse Attention Transformer blocks + an MLP projector to process physics data) with a frozen Math Target Encoder (utilizing a GRU + an MLP projector to process the Postfix AST).</li>
  <li><strong>VICReg Optimization:</strong> To actively prevent representation collapse during this PoC phase, I implemented Variance-Invariance-Covariance Regularization (VICReg) to explicitly sculpt and penalize the latent space, forcing the network to maintain information density.</li>
  <li><strong>Metrics &amp; Evaluation:</strong> I benchmarked the pre-training loop using a comprehensive suite of metrics: L2 MSE (Prediction Loss), total VICReg Loss, Variance Penalty, Latent Cosine Similarity, and Kernel Density Estimation (KDE) distributions.</li>
  <li><strong>Result:</strong> The PoC training dynamics confirm the architecture maps continuous physical data to discrete mathematical structures without collapsing. The <strong>Variance Penalty</strong> flatlined at 0.000, the <strong>Latent Cosine Similarity</strong> aligned to a perfect 1.000, and the KDE plots verify a dense, decorrelated distribution of latent concepts.</li>
</ul>

<h3>3. Next-Gen Feature Encoders: KAN vs. MLP</h3>
<p>To push the physical mapping capabilities further, I experimented with replacing the standard linear MLPs inside the Transformer blocks with Kolmogorov-Arnold Networks (KANs).</p>
<ul>
  <li><strong>The Experiment:</strong> Located in <code>src/models/kan_mlp.py</code>, I benchmarked both layers on a noisy, non-linear dataset.</li>
  <li><strong>Result:</strong> By using learnable B-splines on the edges instead of fixed linear node activations, the KAN converged significantly faster and achieved a lower MSE floor, proving its superiority for the upcoming Phase 1 scaling of LM-JEPA.</li>
</ul>

<hr>

<h2>Repository Navigation</h2>
<ul>
  <li><code>data/</code>: Processed <code>FeynmanEquations.csv</code> targets and generated JSON parse trees.</li>
  <li><code>src/baselines/</code>: The original Prefix &amp; Greedy decoding scripts used for benchmarking.</li>
  <li><code>src/embeddings/</code>: Legacy and updated T-Net physical embedding scripts.</li>
  <li><code>src/library/</code>: <strong>[Previous Project's]</strong> The LASR Concept Library mapping logic.</li>
  <li><code>src/models/</code>: JEPA module definitions, continuous sliding window decoders, and KAN scripts.</li>
  <li><code>src/parser/</code>: The custom Postfix AST and symbolic parsing logic.</li>
  <li><code>src/train_jepa_poc.py</code>: The master training loop for the LM-JEPA task.</li>
</ul>
