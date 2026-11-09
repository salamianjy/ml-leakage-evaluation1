# Model Leakage Lab — scikit-learn

**Question:** Can a model look excellent on a held-out test set and still be invalid for real-world use?

Yes. This experiment creates a synthetic classification problem and compares logistic regression using (a) valid features and (b) an extra post-outcome signal that is **not known at prediction time**. Even an ordinary random train/test split will not reveal this kind of feature leakage.

## Skills demonstrated
- Reproducible synthetic classification and stratified train/test splits
- Scaling and logistic regression inside a `Pipeline` to prevent preprocessing leakage
- ROC AUC, accuracy, precision, recall
- Critical feature-availability review, rather than accepting high test metrics at face value

## Run
```bash
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

The project prints a metrics comparison and writes `outputs/metrics.json`. The leakage case should show dramatically higher ROC AUC, *but is not a valid deployable model*. Exact numeric outputs depend on your scikit-learn version.

## AI-model evaluation task idea
Present both metric reports to an AI without labeling the leakage column. Ask which model should be deployed and why. A robust evaluator must flag that post-outcome fields are unavailable at inference time, even if the test ROC AUC is outstanding.

## Limits and next steps
Synthetic data only, no causal claims. Extend with temporal splits, leakage via group IDs, calibration curves, and an explicit feature-availability registry.
