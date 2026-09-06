# FoeGlass: Faithful Reproduction & DRAO Enhancement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source, faithful reproduction and architectural enhancement of **FoeGlass** — an adversarial framework designed to audit and probe state-of-the-art neural speech deepfake detectors using feedback-driven text-to-speech (TTS) optimization and distributional alignment.

This repository provides:
1. **`foeglass/`**: Baseline reproduction of the core FoeGlass iterative prompt feedback pipeline.
2. **`foeglass-DRAO/`**: An enhanced framework introducing **Distributional Realism Adversarial Optimization (DRAO)** via Maximum Mean Discrepancy (MMD) distribution alignment, spectral smoothness regularization, and LLM-guided prompt mutation (Groq/Ollama).
3. **`foeglass2 kaggle.ipynb`**: Kaggle/Colab notebook optimized for GPU environments (e.g. dual Tesla T4s) with cross-dataset evaluation across ASVspoof 2019, In-the-Wild (RITW), and WaveFake.

---

## Architecture Overview

```
                         +-----------------------------+
                         |     Target Detector f(x)    |
                         +--------------+--------------+
                                        | (Detection Score / Logit)
                                        v
+-----------------------+     +------------------------+     +-----------------------+
|  Real Audio Baseline  | --> |     DRAO Objective     | <-- |   Acoustic Features   |
|   (Wav2Vec-2 / MMD)   |     |    Loss Formulation    |     | (Pitch, Energy, Spec) |
+-----------------------+     +-----------+------------+     +-----------------------+
                                          | (Feedback Vector)
                                          v
+-----------------------+     +------------------------+     +-----------------------+
|   Synthesized Audio   | <-- |   TTS Audio Engine     | <-- |  LLM Prompt Mutator   |
|         x_t           |     |     (Coqui TTS)        |     | (Groq / Llama / Rules)|
+-----------------------+     +------------------------+     +-----------------------+
```

### Key Innovations in DRAO

In the enhanced `foeglass-DRAO` formulation, prompts are not merely optimized against single-point detector misclassification, but guided by a multi-objective loss $J_t$:

$$J_t = \alpha \cdot \Delta f_t + \beta \cdot f(x_t) - \lambda_{\text{mmd}} \cdot \text{MMD}^2(P_g, P_r) - \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}}$$

- **Attack Gain ($\Delta f_t$) & Target Margin**: Rewards iterations that increase false acceptance while maintaining a safety margin against detector thresholds.
- **Distributional Realism (MMD)**: Enforces that the rolling buffer of generated attack embeddings $P_g$ remains distributionally indistinguishable from genuine human speech embeddings $P_r$ under a Gaussian RBF kernel.
- **Spectral Smoothness ($\mathcal{L}_{\text{smooth}}$)**: Penalizes high-frequency acoustic discontinuities and phase artifacts often left by naive TTS manipulations.
- **Momentum Memory Bank**: Maintains top-$k$ performing prompts and scores to balance exploration vs. exploitation across search trajectories.

---

## Repository Structure

```
.
├── .gitignore
├── README.md
├── foeglass/                            # Baseline reproduction codebase
│   ├── main.py                          # Main iterative optimization pipeline
│   ├── train_detector.py                # Trains the MLP deepfake detector
│   ├── evaluation.py                    # Evaluation metrics (ASR, FNR, ROC, EER)
│   ├── build_real_embeddings.py         # Extracts reference Wav2Vec embeddings
│   ├── statistical_analysis.py          # Statistical hypothesis testing
│   ├── requirements.txt                 # Baseline dependencies
│   ├── src/                             # Core modules
│   │   ├── audio_features.py            # Acoustic prosody & feature extraction
│   │   ├── compliance.py                # Dataset integrity verification
│   │   ├── config.py                    # Hyperparameter & path configuration
│   │   ├── detector.py                  # PyTorch detector architectures
│   │   ├── distance.py                  # Cosine & Euclidean embedding distance
│   │   ├── diversity.py                 # Lexical & acoustic diversity tracking
│   │   ├── embedding.py                 # Wav2Vec-2 feature extraction
│   │   ├── feedback.py                  # Feedback prompt constructor
│   │   ├── memory.py                    # Iteration memory buffer
│   │   ├── pipeline.py                  # Orchestration pipeline
│   │   ├── prompt_generator.py          # Heuristic prompt generator
│   │   ├── run_logging.py               # Structured logging & run artifacts
│   │   └── tts_engine.py                # Text-to-speech synthesis wrapper
│   └── reference output for limitations/# Sample baseline evaluation plots
│       ├── asr_success_by_iteration.png
│       ├── classification_metrics.png
│       ├── confusion_matrix.png
│       ├── cumulative_asr.png
│       └── scores.png
│
├── foeglass-DRAO/                       # Enhanced DRAO framework
│   ├── main.py                          # DRAO optimization runner
│   ├── run_ablations.py                 # Automated ablation experiments
│   ├── train_detector.py                # Detector training with validation early stopping
│   ├── build_real_embeddings.py         # Builds real reference dataset embeddings
│   ├── build_generated_embeddings.py    # Builds generated attack embeddings
│   ├── config.py                        # Dataclass-based central configuration
│   ├── requirements.txt                 # DRAO dependencies (Groq, Seaborn, etc.)
│   ├── .env.example                     # Environment template for API keys
│   ├── data/                            # Machine-generated runtime artifacts
│   │   └── README.md                    # Data directory documentation
│   ├── src/                             # DRAO modular components
│   │   ├── detector.py                  # Wav2Vec2 + MLP classifier
│   │   ├── embedding.py                 # Acoustic feature & embedding extractor
│   │   ├── llm_generator.py             # LLM prompt mutation (Groq API / Ollama)
│   │   ├── logging_utils.py             # Run logs and metric serialization
│   │   ├── mmd.py                       # Maximum Mean Discrepancy computation
│   │   ├── pipeline.py                  # Main DRAO iterative search loop
│   │   ├── smoothness.py                # Acoustic spectral smoothness metric
│   │   └── tts_engine.py                # Coqui TTS wrapper with fallback
│   └── output plots samples/            # Empirical demonstration plots
│       └── run_20260427_021150_seed42_metrics.png
│
└── foeglass2 kaggle.ipynb               # Kaggle GPU notebook for ASVspoof/RITW/WaveFake
```

---

## Getting Started

### 1. Prerequisites & Environment

Python 3.10+ and a CUDA-capable GPU are recommended (CPU is supported for testing).

```bash
# Clone the repository
git clone https://github.com/sahilpatil6305/Foeglass.git
cd Foeglass

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

For the baseline reproduction:
```bash
pip install -r foeglass/requirements.txt
```

For the enhanced DRAO framework:
```bash
pip install -r foeglass-DRAO/requirements.txt
```

### 3. Configure Environment Variables (Optional for DRAO)

If using Groq for LLM-assisted prompt generation in `foeglass-DRAO`:
```bash
cp foeglass-DRAO/.env.example foeglass-DRAO/.env
# Edit foeglass-DRAO/.env and add your GROQ_API_KEY
```

---

## Running the Experiments

### Baseline FoeGlass

1. **Extract Reference Embeddings:**
   ```bash
   python foeglass/build_real_embeddings.py
   ```

2. **Train the Deepfake Detector:**
   ```bash
   python foeglass/train_detector.py
   ```

3. **Run Iterative Attack Pipeline:**
   ```bash
   python foeglass/main.py --mode baseline --iterations 20 --seed 42
   ```

### Enhanced DRAO Framework

1. **Run DRAO Optimization Loop:**
   ```bash
   python foeglass-DRAO/main.py --mode drao --iterations 20 --seed 42
   ```

2. **Run Systematic Ablation Studies:**
   Evaluate the contribution of MMD distribution matching, smoothness penalties, and memory reuse:
   ```bash
   python foeglass-DRAO/run_ablations.py --iterations 20 --seeds 42 123 999
   ```

### Kaggle / Cloud GPU Execution

Open `foeglass2 kaggle.ipynb` in Kaggle or Google Colab:
- Pre-configured to mount `ASVspoof-2019`, `Release-In-The-Wild (RITW)`, and `WaveFake`.
- Extracts `microsoft/wavlm-large` and `facebook/wav2vec2-base` embeddings.
- Trains an XGBoost / MLP forensic detector and computes EER, ROC-AUC, and Attack Success Rate (ASR).

---

## Reference Evaluation Results

| Metric | Description | Target |
|---|---|---|
| **ASR (Attack Success Rate)** | Percentage of synthetic samples classified as genuine bonafide speech | Higher is more effective attack |
| **FNR (False Negative Rate)** | Detector vulnerability rate under adversarial perturbation | Higher indicates detector failure |
| **MMD Score** | Discrepancy between generated attack samples and genuine human audio | Lower indicates higher acoustic realism |
| **Spectral Smoothness** | Measure of temporal and frequency continuity across frames | Lower error represents natural speech prosody |

Empirical plots comparing baseline vs. DRAO runs are located in:
- `foeglass/reference output for limitations/`
- `foeglass-DRAO/output plots samples/`

---

## Citation & Acknowledgements

This implementation reproduces and builds upon research in adversarial audio generation and synthetic speech detection auditing:
- **FoeGlass**: Adversarial audio prompt auditing against deepfake detectors.
- **Wav2Vec 2.0 / WavLM**: Self-supervised representations for speech processing.
- **Coqui TTS**: Open-source neural speech synthesis.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
