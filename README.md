# MuleNet-Z: cross-bank money-mule detection on IBM Z (IBM Z Datathon 2026, SSF2026)
## Data

### Source dataset

MuleNet-Z uses the **IBM Transactions for Anti-Money Laundering (AML)** dataset, specifically the `HI-Small_Trans.csv` file distributed through Kaggle: [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

The processed full dataset contains 515,088 accounts and 5,078,345 transactions. An account is labeled as a mule if it appears as the sender or receiver in at least one transaction marked as laundering. This produces 6,357 labeled mule accounts (approximately 1.23% of accounts).

### Consortium construction

The dataset's 30,470 bank IDs are grouped into five consortium members, `B1`–`B5`, using greedy transaction-volume balancing. Each member represents a bank-level view of the data. This provides five simulated institutions for evaluating collaborative AML detection.

### Labels and data splits

Account labels follow the transaction-touch rule: an account is labeled positive if it is the sender or receiver of any transaction marked as laundering. The label is a prediction target and must not be used as an input feature.

Accounts are divided into training, validation, and test splits using stratified 60/20/20 splitting with random seed `42`, preserving the label distribution as closely as possible across splits.

### Institution-specific views and features

Features are computed using only the transactions available in the view being evaluated. For a single-institution view, feature calculations use only that institution's available transactions; for the combined view, they use the combined transaction set. Features must not use transactions from outside the selected view. The mule label is never included as a feature.

### Building the processed data

From the repository root, run the project's `build_processed()` function from `mulenet.build` to generate the processed full dataset and its associated graph and feature artifacts.

For example, in Python:

```python
from mulenet.build import build_processed

build_processed()
```

Run the build only when sufficient compute and memory are available; it can process millions of transactions. The robustness dataset `LI-Small_Trans.csv` is a separate input and is not built as part of this step unless explicitly configured.
