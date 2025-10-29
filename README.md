# RL Task: Conditional Co-Mixup Implementation

This repository contains an RL task designed for training an LLM in ML engineering skills, as per the assignment guidelines.

## 🤖 Task Objective

The task requires the model to act as an ML engineer implementing a complex, deterministic data augmentation algorithm called "Conditional Co-Mixup" with adaptive parameters.

The model is given a prompt describing a 3-stage algorithm with conditional logic:

1. **Stage 1 - Initial Pairwise Mixing:** Perform two mixup operations on multi-view data samples
2. **Stage 2 - Conditional Final Mix:** Calculate L2 norm of intermediate result and conditionally choose λ parameter (IF norm > 0.8 THEN λ=0.3 ELSE λ=0.6)
3. **Stage 3 - Post-Processing:** Apply L2 normalization to feature vectors and conditional label smoothing

The task tests multi-step reasoning, conditional logic implementation, mathematical formula execution, and numerical precision with multi-view data structures.

## 📁 Task Files

* **`main.py`**: The complete, runnable Python script containing the test harness (`run_agent_loop`, `run_single_test`) and the `main` function with task prompt, multi-view data, and expected answer.
* **`.env.example`**: Template for environment variables. Copy to `.env` and add your API key.
* **`pyproject.toml`**: UV package manager configuration

## 🚀 How to Run

1. **Clone the repo**
2. **Set up your API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```
3. **Install dependencies using UV:**
   ```bash
   uv install
   ```
4. **Run the test:**
   ```bash
   uv run main.py
   ```
   The script runs multiple iterations and prints the final pass rate.

---

## 📋 Task Design & Expected Challenges

### 🎯 Problem Statement
The task requires implementing a **Conditional Co-Mixup** algorithm with adaptive parameters that tests:
- Multi-step algorithmic reasoning across 3 distinct stages
- Conditional branching based on computed intermediate values
- Mathematical precision with floating-point calculations
- Multi-view data structure manipulation

### 🧠 Cognitive Challenges
The algorithm specifically targets known LLM weaknesses:

**Conditional Branching:** Models must compute L2 norm of intermediate results and conditionally select λ parameters based on threshold comparison (norm > 0.8)

**Stateful Multi-Stage Processing:** Each stage depends on outputs from previous stages, requiring careful state management and intermediate value tracking

**Numerical Precision:** Requires exact floating-point calculations rounded to 5 decimal places with label sum validation

### 📊 Expected Failure Modes

Based on the algorithm design, models typically fail in these patterns:

**Timeout/Complexity Issues:** Models may return `None` when overwhelmed by the multi-stage conditional logic, especially during the critical L2 norm evaluation and λ selection step

**Copy-Paste Errors:** Models often correctly implement the first calculation but inappropriately reuse values for subsequent independent calculations (e.g., copying `mixed_view1` results to `mixed_view2`)

**Conditional Logic Errors:** Models may implement the mixup formula correctly but fail the conditional check (L2 norm > 0.8), leading to wrong λ parameter selection and cascading errors

**Numerical Precision Issues:** Models may use correct algorithms but fail due to rounding differences or label normalization errors

### 🔧 Tool Robustness
The test harness includes comprehensive error handling to ensure failures reflect algorithmic understanding rather than tool usage issues. Invalid inputs receive helpful error messages instead of crashes.

---

## 🔍 Multi-View Data Structure

The task uses realistic multi-view data common in contrastive learning:

```python
[
    {"sample_id": 0, "view1": [0.2, 0.4], "view2": [0.3, 0.5], "label": [1, 0, 0]},
    {"sample_id": 1, "view1": [0.6, 0.8], "view2": [0.7, 0.9], "label": [0, 1, 0]}, 
    {"sample_id": 2, "view1": [0.1, 0.3], "view2": [0.2, 0.4], "label": [0, 0, 1]},
    {"sample_id": 3, "view1": [0.4, 0.6], "view2": [0.5, 0.7], "label": [1, 0, 0]}
]
```

Each sample contains two feature views and a one-hot label, testing the model's ability to handle structured data representations used in modern ML.

---

## 📐 Expected Answer Format

The deterministic algorithm produces a single correct answer:
```python
{
    "mixed_view1": [0.53281, 0.84623], 
    "mixed_view2": [0.56653, 0.82404], 
    "mixed_label": [0.36, 0.36, 0.28]
}
```

This validates correct implementation of all three stages: initial pairwise mixing, conditional λ selection, and post-processing normalization.