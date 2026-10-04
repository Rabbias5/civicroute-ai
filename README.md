# CivicRoute AI

CivicRoute AI is a small TensorFlow/Keras classifier that recommends a category
for a council service request: broken streetlight, illegal dumping, pothole or
water leak. It uses the supplied synthetic educational dataset. A council
employee remains responsible for reviewing the recommendation and routing the case.
Repository reviewed for assessment demonstration.
## Results

The model correctly classified **68 of 90 test requests**, giving **75.56% accuracy**
and a test loss of **0.5730**. Illegal dumping had the highest F1 score, at 0.909.
Potholes were the weakest category: only 10 of 23 were recognised, with seven
classified as water leaks. The notebook and `outputs/` contain the evidence.

## Reproduce the results

Use Python 3.12. Open a terminal in this folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies and execute the notebook code:

```bash
python -m pip install -r requirements.txt
python run_assessment.py
```

This regenerates the result files in `outputs/`. It does not update the saved
notebook displays. To update those, open the notebook in JupyterLab, restart its
Python kernel and run all cells:

```bash
python -m ipykernel install --user --name civicroute --display-name "CivicRoute Python"
python -m jupyterlab
```

The supplied notebook already contains executed outputs. The tested environment
used Python 3.12 and TensorFlow CPU 2.20.0 on Linux. Hardware or library changes
can affect numerical reproducibility. If `tensorflow-cpu` is unavailable for
your platform, use a compatible Linux environment or the matching
`tensorflow==2.20.0` distribution.

## Approach

The CSV contains 600 records, 13 fields and four balanced categories. Nine
input fields are selected. The request identifier, derived night flag and
neighbourhood are excluded. Excluding neighbourhood does not itself establish
fairness; other features may still reflect reporting patterns.

Rows are split into 420 training, 90 validation and 90 test examples using
stratification and seed 2026. Missing numerical values use training medians,
and categorical gaps use the training mode. One-hot encoding and NumPy
standardisation produce 15 model inputs. All fitted preprocessing statistics
and categorical layouts use only training data.

The network has Dense layers with 16 and 8 ReLU units, followed by four softmax
outputs: 428 trainable parameters. It trains for 30 epochs in batches of 32,
using Adam at 0.001 and sparse categorical cross-entropy. Training choices were
fixed before inspecting the test results. Seeds and deterministic TensorFlow
operations are configured for CPU execution.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `civicroute_assessment.ipynb` | Preparation, model, executed outputs and interpretation |
| `run_assessment.py` | Runs the notebook code cells |
| `requirements.txt` | Python dependencies |
| `data/` | Unchanged supplied dataset |
| `outputs/` | Prepared data, preprocessing, metrics, predictions, charts and saved model |

The saved model expects the saved preprocessing and exact feature order.
It is not an end-to-end application. No explanation library or graph neural
network has been implemented.

## Limitations and next steps

Pothole recall is only 43.48%, so the overall accuracy hides an important weakness.
The synthetic test set is small and contains no raw complaint text or images.
Softmax scores are not calibrated guarantees. Historical features such as
`previous_resolution_days` must be verified as available when a request arrives.

Further evaluation should use representative real requests, reliable labels,
later time periods and checks by category, neighbourhood and submission channel.
Any future pilot should retain human review for uncertain, unusual or high-risk
cases. The evidence supports further investigation rather than automatic routing.

## Sources

The CSV, starter notebook and data dictionary were supplied with the course
assessment. This notebook adapts the supplied starter. Technical references:

- [Keras loss documentation](https://keras.io/api/losses/probabilistic_losses/)
- [TensorFlow deterministic operations](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)
- Ribeiro, Singh and Guestrin (2016), [LIME](https://arxiv.org/abs/1602.04938)
- Selvaraju et al. (2017), [Grad-CAM](https://arxiv.org/abs/1610.02391)
