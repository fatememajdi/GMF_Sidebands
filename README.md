# Generalized Mechanical Frequency (GMF) & Sideband Analysis

## Introduction
This project provides signal processing tools to detect **Generalized Mechanical Frequencies (GMFs)** 
and evaluate their **sideband patterns** in vibration signals.  
It is primarily designed for **rotating machinery diagnostics**, using spectral peak analysis, harmonic 
alignment, and sideband scoring.

### Key Features
- Automatic GMF detection across single or multiple signals
- Harmonic and sideband evaluation with scoring metrics
- Aggregation of results from multiple signals
- Stepwise sideband search with fixed spacing

---

## Prerequisites
- **Python version**: 3.11+
- **Libraries**:
  - numpy
  - scipy
  - collections
  - matplotlib (optional, for visualization)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure
```
GMF_Sidebands/
|— src/
|    |— gmf_sidebands.py    # Core implementation
|— notebooks/               # Example notebooks
|— requirements.txt
|— README.md
```

---

## Code Overview

### GMF Detection
- `detect_final_gmfs3(ts, …)` → Detect GMFs in a single signal
- `detect_final_gmfs_across_signals_ref(ts_list, …)` → Detect GMFs across multiple signals

### Sideband Scoring
- `calculate_score_iso(value, iso=[1,2])` → Normalize against ISO limits
- `evaluate_sideband_set(harm_amp, sideband_amps, …)` → Evaluate sideband quality

### Sideband Detection
- `find_sidebands_harmonics_filtered(ts, gmfs, …)` → Detect sidebands
- `aggregate_from_results(list_of_results, …)` → Aggregate spacing estimates
- `evaluate_sidebands_with_final_spacing_stepwise(ts, gmfs_final, spacing_map, …)` → Stepwise search for sidebands

### Pipeline
- `find_signals_sideband(path)` → End-to-end pipeline:
  1. Load signals  
  2. Detect GMFs  
  3. Detect sidebands  
  4. Aggregate spacings  
  5. Return structured results  

---

## How to Run
1. Clone the repository
```bash
git clone https://github.com/yourusername/GMF_Sidebands.git
cd GMF_Sidebands
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Place signal files in `data/`
4. Run the notebook:
```bash
jupyter notebook notebooks/evaluate.ipynb
```

---

## Outputs
The module produces:
- **Detected GMFs (Hz)** → Dominant mechanical frequencies
- **Sideband Spacing (Hz)** → Estimated spacing per GMF
- **Sideband Scores (0–100)** → Quality metric
- **Final Dictionary** → Example:

```python
{
    "SignalName1": [
        {"gmf": 31.2, "score": 78.5, "sidebands": {100: [168,211]}, "spacing": 3.4},
        ...
    ],
    "SignalName2": [...]
}
```
