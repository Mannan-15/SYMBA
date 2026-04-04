<h1 align="center">Symba 2026: LM-JEPA for Symbolic Regression</h1>

<p align="center"><strong><h4 align="center">Read the Full GSoC 2026 Proposal: <a href="https://docs.google.com/document/d/1j8NAA6b-6zpPlglhggNHf5iv9rBzBXs1lFekACsmZC0/edit?usp=sharing">LM-JEPA Proposal (Google Docs)</a></strong></p></h4>

<div align="center">
  <p><em><h5><strong>Note on Parallel Submission:</strong> While this repository branch focuses on continuous latent-space prediction via LM-JEPA (Task 2.7), I have also architected and submitted a highly synergistic parallel proposal for <strong>Using Next-Gen Transformers to Seed Generative Models for Symbolic Regression </strong>(Task 2.6). That proposal focuses on search-augmented text generation using Next-Gen Transformers and generative algorithms (like GP, MCTS, KANs).<br></h5>
<a href="https://docs.google.com/document/d/1WaaLbe_9OeSylVhjENV5TdaDmsZNnZ357YENC6tNcUw/edit?usp=sharing">Read my Next-Gen Transformers Proposal Here</a> | <a href="https://github.com/Mannan-15/SYMBA/tree/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan">View the Next-Gen Code Branch</a></em></p>
</div>

<hr>

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
  <li><strong>T-Net Embeddings: </strong><a href="https://github.com/Mannan-15/SYMBA/blob/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/src/embeddings/t_net_embeddings.py">src/embeddings/t_net_embeddings.py</a></li>
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
<p>To fulfill these tasks, I audited the 2024/2025 ML4SCI baselines and engineered three major architectural upgrades to shift Symba from <em>syntactic text generation</em> to <em>semantic physics prediction</em>. <b>(For deep technical proofs and loss landscape graphs, please refer to Section 2 of the proposal)</b>.</p>

<h3>1. Tokenization Rationale, T-Net, &amp; Inference Upgrades (Task 1.1)</h3>
<p>The legacy baseline relied on bloated Prefix notation and discrete digit prediction, which caused massive sequence lengths and severe hallucination of physical constants. To fix this, I engineered a mathematically enforced <strong>Postfix + <code>&lt;C&gt;</code> Tokenizer</strong>.</p>
<ul>
  <li><strong>Data Encoding (T-Net):</strong> I integrated the T-Net encoder for the physical tabular data <code>(x)</code>. Its permutation-invariant architecture handles unordered sets of physical observations without injecting artificial sequence biases.</li>
  <li><strong>Postfix vs. Prefix + <code>&lt;C&gt;</code> Tokenization:</strong> I engineered a mathematically enforced Postfix tokenizer that completely removes redundant parentheses and replaces discrete floating-point numbers with a continuous embedding token.</li>
  <li><strong>Compute Reinvestment:</strong> The maximum sequence length dropped from 67 tokens down to 48. This ~30% compression drastically reduces the <code>O(N^2)</code> self-attention compute cost, buying back architectural headroom to learn deeper physics.</li>
  <li><strong>Impact:</strong> The maximum sequence length dropped from 67 tokens down to 48. Because Postfix removes redundant tokens that artificially inflate training metrics, it forces the model to learn true mathematical generalization. This structural compression improved <strong>Validation Accuracy from 52.1% to 56.3%</strong> and nearly doubled the greedy <strong>Exact Match accuracy from 25.7% (Baseline) to 47.4% (Proposed)</strong>.</li>
</ul>
<p>Furthermore, the training curves reveal that while the Prefix baseline flatlined after ~40 epochs, the Postfix loss remained dynamic indicating that training beyond this 50 epochs PoC will yield even higher ultimate accuracy.</p>
<p align="center">
  <b>Left:</b> Prefix + <code>&lt;C&gt;</code> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Right:</b> Proposed Postfix + <code>&lt;C&gt;</code>
</p>
<br>
<p align="center">
<img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan/plots/Prefix_exactmatch.png" width="45%" /> &nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/Mannan-upgrade/SYMBA_REG/Upgrade_Mannan/plots/Postfix_exactmatch.png" width="45%" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/prefix_parse.png" width="45%" /> &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/postfix_parse.png" width="45%" />
</p>

<p align="center">
  
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/prefix_data.png" height="350" width="49%" /> &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/postfix_data.png" height="350" width="49%" />
</p>

<h3>2. The LM-JEPA Core Architecture &amp; Pipeline (Task 2.7)</h3>
<p>A standard L2 prediction loss in a continuous Joint-Embedding environment causes both networks to instantly collapse and output vectors of all zeros. To prevent this, I engineered a stabilized continuous-space pipeline.</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/lm_jepa_architecture.png" alt="LM-JEPA Architecture Pipeline" height="650" width="500" />
  <br><em>Figure: The LM-JEPA Pipeline bridging continuous physics (Context) and discrete math (Target).</em>
</p>

<h4>Architecture Breakdown:</h4>
<ul>
  <li><strong>Context Encoder (Physics via Sparse-Transformer):</strong> Raw physical point clouds (<code>x</code>) are processed through a T-Net to generate 128D invariant embeddings. A Sparse Attention Transformer captures global physical relationships, outputting a highly dense context embedding (<code>s_x</code>).</li>
  <li><strong>Target Encoder (Math via Sequence Modeling):</strong> The target equation (<code>y</code>) is parsed into an AST and tokenized into Postfix notation. A GRU + MLP projector ingests these variable-length tokens to output a fixed-size mathematical latent vector (<code>s_y</code>).</li>
  <li><strong>The Predictor:</strong> A lightweight neural network bridges the modalities, taking the physics context (<code>s_x</code>) and predicting the mathematical target (<code>s_y_hat</code>) purely in latent space.</li>
  <li><strong>VICReg Optimization:</strong> To prevent representation collapse, Variance-Invariance-Covariance Regularization (VICReg) applies a hinge loss to maintain embedding variance, explicitly sculpting the latent space to remain information-dense.</li>
</ul>

<h4>Pretraining Results:</h4>
<p>The training dynamics mathematically prove that the Context Encoder successfully maps continuous physical data to discrete mathematical structures without collapsing. The <strong>Variance Penalty</strong> flatlined at 0.000, the <strong>Latent Cosine Similarity</strong> aligned to a perfect 1.000, and the KDE plots verify a dense, decorrelated distribution of latent concepts.</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/jepa_alignments_result.png" height="500" width="900" />
</p>

<h3>3. Next-Gen Feature Encoders: KAN vs. MLP</h3>
<p>To push the physical mapping capabilities further, I experimented with replacing the standard linear MLPs inside the Transformer blocks with Kolmogorov-Arnold Networks (KANs).</p>
<ul>
  <li><strong>The Experiment:</strong> Located in <code>src/models/kan_mlp.py</code>, I benchmarked both layers on a noisy, non-linear dataset <code>(y = e^(-0.1x) * sin(3x) + noise)</code>.</li>
  <li><strong>Result:</strong> By using learnable B-splines on the edges instead of fixed linear node activations, the KAN converged significantly faster and achieved a lower MSE floor, proving its superiority for the upcoming Phase 1 scaling of LM-JEPA.</li>
</ul>
<p align="center">
  <img src="https://raw.githubusercontent.com/Mannan-15/SYMBA/LM-JEPA/SYMBA_REG/LM-JEPA_Mannan/plots/kan_mlp.png" height="350" />
</p>
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
