<div align="center">
  <h1>LMUnit: Fine-grained Evaluation with Natural Language Unit Tests
</h1>
 </p>
</div>
<br>

![LMUnit Workflow](/assets/figure1.png)
This repository provides code for evaluation and reproduction of our results in [LMUnit: Fine-grained Evaluation with Natural Language Unit Tests](https://arxiv.org/abs/2412.13091).


## 🚀 Recent Updates

- **2025-07**: Open-Sourced LMUnit checkpoints [**Llama 3.1-70B**](https://huggingface.co/ContextualAI/LMUnit-llama3.1-70b) and [**Qwen 2.5-72B**](https://huggingface.co/ContextualAI/LMUnit-qwen2.5-72b) models. Besides, evaluation framework for reproducing our direct scoring and preference tasks results

## 📋 Table of Contents

- [🔧 LMUnit Inference Example](#-lmunit-inference-example)
- [📁 Repository Structure](#-repository-structure)
- [📄 Artifacts](#-artifacts)
  - [📋 Paper](#-paper)
  - [🤗 HuggingFace Collection](#-huggingface-collection)
  - [💾 Checkpoints](#-checkpoints)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Run Evaluation Task](#run-evaluation-task)
- [Citation](#citation)

## 🔧 LMUnit Inference Example

#### Install LMunit
```bash
pip install lmunit
```

#### Run Inference Example Script

```python
from lmunit import LMUnit
from vllm import SamplingParams

# Initialize LMUnit with your model
model = LMUnit(
    model_path="ContextualAI/LMUnit-llama3.1-70b", 
    tp_size=4
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=10, logprobs=20)
# Define your evaluation prompt

unit_test = "Does the response specify minimum system logs required for compliance?"

query =  "What security validation protocols ensure HIPAA compliance when connecting research databases to clinical systems?"



response = """Customer Concentration Risk Comparison:
Snowflake's Platform segment shows higher concentration risk than Product segment:

Product: 15% (Customer X), 8% (Customer Y), 77% others
Platform: 20% (Customer Z), 5% (Customer W), 75% others

HIPAA Technical Controls:
Data Encryption Requirements:

Regulation: 45 CFR § 164.312(a)(2)(iv)
Validation: All data encrypted at rest/transit using FIPS 140-2 algorithms
Testing: Vulnerability scanning and penetration testing for encryption weaknesses

Additional Compliance Measures:

Risk analysis for security threats
Access controls for PHI authorization
Incident response planning
Required logs: encryption key management, data access, security incidents"""

prompt = f"Query: {query}\n\nResponse: {response}\n\nUnit Test: {unit_test}"

output = model.generate(prompt, sampling_params)
```


## 📁 Repository Structure

```
lmunit/
├── assets/                 # Documentation assets and figures
├── eval/                   # Evaluation scripts and benchmarks
│   ├── eval.py             # Main evaluation script
│   └── reward_bench2.py    # Reward benchmarking utilities
├── lmunit/                 # Core LMUnit package
│   ├── __init__.py         # Package initialization
│   ├── constants.py        # Framework constants
│   ├── lmunit.py           # Main LMUnit class implementation
│   ├── metrics.py          # Evaluation metrics
│   └── tasks.py            # Task definitions and utilities
├── requirements/           # Dependencies
    ├── requirements.txt    # Main dependencies  
    └── dev.txt             # Development dependencies
```

## 📄 Artifacts

### 📋 Paper
- [LMUnit: Fine-grained Evaluation with Natural Language Unit Tests](https://arxiv.org/abs/2412.13091)

### 🤗 HuggingFace Collection
- [LMUnit Models Collection](https://huggingface.co/collections/ContextualAI/lmunit-6879d97293090553a9300abe) - Pre-trained models and evaluation datasets

### 💾 Checkpoints


| Model | Flask | BiGGen-Bench | Human-Internal | InfoBench | RB | LFQA | RB2 |
|-------|-------|--------------|----------------|-----------|-------------|------|--------------|
| [**LMUnit-LLaMA-3.1-70B**](https://huggingface.co/ContextualAI/LMUnit-llama3.1-70b) | 72.03 | 67.69 | 93.63 | 89.00 | 91.56 | 76.15 | 80.5 |
| [**LMUNIT_Qwen2.5-72B**](https://huggingface.co/ContextualAI/LMUnit-qwen2.5-72b) | 73.85 | 69.56 | 94.44 | 88.67 | 91.13 | 73.85 | 82.1 |


## 🚀 Quick Start

###  **Installation**

```bash
pip install lmunit
```

### **Run Evaluation Task**

For running an specific task on an LMUnit model
```bash
python eval/eval.py --task <task> --model-path <lmunit-model> --tensor-parallel-size <tp-size>
```

For running rewardbench2 results:
```bash
python eval/reward_bench2.py --model-path <lmunit-model> --tensor-parallel-size <tp-size>
```

### **Reproduce LMUnit evaluation suite**
```bash
./scripts/run_all_evaluations.sh <model_path> <tensor_parallel_size> [output_dir]
```

## Citation

```bash
@misc{saadfalcon2024lmunitfinegrainedevaluationnatural,
      title={LMUnit: Fine-grained Evaluation with Natural Language Unit Tests}, 
      author={Jon Saad-Falcon* and Rajan Vivek* and William Berrios* and Nandita Shankar Naik and Matija Franklin and Bertie Vidgen and Amanpreet Singh and Douwe Kiela and Shikib Mehri},
      year={2024},
      eprint={2412.13091},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13091},
      note={*Equal contribution}
}
```