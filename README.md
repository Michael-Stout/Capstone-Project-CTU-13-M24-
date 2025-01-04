# README.md

## Overview

This project focuses on analyzing and detecting botnet network traffic using the **CTU-13**-style dataset(s). It walks through several major steps:

1. **Section 1**: Logging Setup and Importing Libraries  
2. **Section 2**: Loading and Exploring the Data  
3. **Section 3**: Basic Data Cleaning & Feature Engineering  
4. **Section 4**: Visualizations of Traffic Patterns  
5. **Section 5**: Train–Test Split and Multi-Model Pipeline  
6. **Section 6**: Model Comparison  
7. **Section 7**: Evaluating KNN on Multiple Datasets  

Below is a summary of each step, expanded modeling insights, and final findings and recommendations.

---

## Section 1: Logging Setup and Libraries

- Configured logging to capture output in a log file while optionally printing to console.
- Imported and declared project-wide constants (e.g., **`RANDOM_STATE`**, **`TEST_SIZE`**, etc.).
- Provided a helper function `log_and_print(msg, logger, level='info')` to synchronize logs and console prints.

**Key Point**: This ensures consistent, well-formatted logs for each section’s output, improving traceability and reproducibility.

---

## Section 2: Loading & Exploring Data

- Successfully loaded the main dataset (e.g., `11-Rbot-20110818-2.binetflow.csv`) of shape **(107251, 15)**.
- Displayed a sample of rows and the dataset’s basic info, including column names and dtypes.

The dataset includes:
- **Network flow details** like `Proto`, `SrcAddr`, `DstAddr`, `TotBytes`, etc.
- A `Label` column identifying traffic as `Background`, `Botnet`, or normal.

Through initial exploration, we observed:
- Mixed columns: floats (`Dur`, `sTos`, `dTos`), ints (`TotPkts`, `TotBytes`, etc.), and objects (`Proto`, `Label`, etc.).
- Some missing values (e.g., `Sport` and `Dport` columns are not complete).

---

## Section 3: Basic Data Cleaning & Feature Engineering

1. **Removed** rows labeled `Background`.
2. **Created** a binary `Botnet` column (`1` if flow is botnet, `0` otherwise), dropping `Label`.
3. **Dropped** columns `sTos`, `dTos`, `StartTime`.
4. **Engineered**:
   - `BytesPerSecond = TotBytes / Dur`
   - `PktsPerSecond = TotPkts / Dur`
   - `SrcAddrEntropy` & `DstAddrEntropy` (Shannon entropy of IP addresses)
   - `SportRange` & `DportRange` (e.g., `WellKnown`, `Registered`, `Ephemeral`).
5. **Label-encoded** categorical columns like `Proto`, `Dir`, `State`, etc.

**Result**:  
- After cleaning, the feature matrix shape was **(10882, 15)**, focusing on relevant columns.
- A quick `.head()` shows new columns and transformations, verifying correct data types.

---

## Section 4: Visualizations

We generated and saved several plots in the `plots/` directory:

1. **Bar & Pie Chart** of Botnet vs Normal — reveals the proportion of each label.
2. **Correlation Heatmap** of numeric columns — highlights relationships (e.g., `PktsPerSecond` vs. `BytesPerSecond`).
3. **Additional Plots** for distribution insights:
   - **Count plot** of port ranges vs. botnet label.
   - **Box plot** of `PktsPerSecond` by Botnet label.
   - **Strip plot** (replacing a swarm) for `BytesPerSecond` by Botnet.

These helped confirm certain features (like packet or byte rates) strongly differ between normal and botnet flows.

---

## Section 5: Train–Test Split and Multi-Model Pipeline

- **Split** the data: ~80% train, ~20% test.
- Built a pipeline for **scaling** (StandardScaler) + each classifier.
- Ran **GridSearchCV** on multiple models: 
  - RandomForest  
  - DecisionTree  
  - NaiveBayes  
  - KNN  
  - SVM  
  - LogisticRegression  
  - GradientBoosting  
- Computed performance metrics:
  - **Accuracy**, **Precision**, **Recall**, **F1**, **ROC AUC**, **Log Loss**, **mAP**.
- Saved confusion matrices, ROC curves, top-features charts.

**Observation**: Models like **RandomForest**, **DecisionTree**, **KNN**, **GradientBoosting** often achieve near-perfect scores on this data. 

---

## Modeling: Deeper Interpretation & Evaluation Metrics

### Deeper Interpretation of Models

1. **Tree-Based Models (RandomForest, DecisionTree, GradientBoosting)**  
   - Identify key features like **`SrcBytes`**, **`TotBytes`**, or **`PktsPerSecond`**.  
   - Provide interpretability via feature importances (visible in top-10 bar charts).  
   - Indicate that a few core features are enough to robustly separate botnet from normal flows.

2. **NaiveBayes**  
   - Simple and fast but less robust if feature independence assumptions are violated.  
   - Still performs well, though not as perfectly as the ensemble methods.

3. **KNN**  
   - Distances in scaled feature space prove highly separable for botnet vs. normal.  
   - Achieved near-perfect classification with moderate training time.

4. **SVM**  
   - With `rbf` kernel, also excels at separating the classes.  
   - Tends to be more computationally expensive depending on data size and hyperparameters.

5. **LogisticRegression**  
   - Interpretable coefficients revealing how features push predictions toward botnet vs. normal.  
   - Slower in training with L1 penalty, but still effective.

### Deeper Interpretation of Evaluation Metric

**F1 Score** was the primary metric used in GridSearchCV. We also looked at **ROC AUC**, **Log Loss**, and **mAP**:

- **F1 Score**: 
  - The harmonic mean of precision and recall.  
  - Particularly relevant when we want to avoid both false positives (precision) and false negatives (recall).  
  - Especially crucial in security contexts where both false alarms (too many false positives) and missed threats (false negatives) are costly.

- **ROC AUC & mAP**: 
  - Evaluate the ranking quality of predicted probabilities, indicating how well the model separates classes overall.  
  - High AUC (>0.99) signals the model is extremely capable of distinguishing botnet from normal traffic across probability thresholds.

### Rationale for Use of These Metrics

1. **Security Risk**: Missing a botnet threat is severe; thus, recall is critical. But we also want high precision so analysts aren’t flooded with false positives. **F1** balances both.
2. **Operational Impact**: A high **ROC AUC** and **mAP** confirm that if we adjust thresholds, the model can still effectively separate normal vs. botnet. This is crucial for different security sensitivities.
3. **Business Context**: Minimizing downtime or investigations of benign traffic is essential. The chosen metrics highlight overall detection accuracy and the cost of errors from multiple angles.

---

## Section 6: Model Comparison

A table summarized each model’s performance:

| Model              | CV F1   | Test Accuracy | Test Precision | Test Recall | Test F1  | ROC AUC  | Train Time (s) |
|--------------------|---------|--------------|---------------|------------|----------|----------|----------------|
| RandomForest       | 0.9992  | 0.9986       | 0.9988        | 0.9994     | 0.9991   | ~1.0000  | ~2.14          |
| DecisionTree       | 0.9991  | 0.9991       | 0.9988        | 1.0000     | 0.9994   | 0.9982   | ~0.09          |
| NaiveBayes         | 0.9865  | 0.9816       | 0.9778        | 0.9982     | 0.9879   | 0.9963   | ~0.04          |
| **KNN**            | 0.9988  | 0.9991       | 1.0000        | 0.9988     | 0.9994   | 1.0000   | ~0.20          |
| SVM                | 0.9953  | 0.9963       | 0.9957        | 0.9994     | 0.9976   | 0.9999   | ~3.75          |
| LogisticRegression | 0.9917  | 0.9890       | 0.9867        | 0.9988     | 0.9927   | 0.9997   | ~21.13         |
| GradientBoosting   | 0.9992  | 0.9986       | 0.9988        | 0.9994     | 0.9991   | 1.0000   | ~1.37          |

**Observation**: KNN edges out or ties for top performance across multiple metrics, with near-perfect F1 and AUC. Tree-based ensembles also excel. Logistic regression is strong yet more time-consuming with certain hyperparameters.

---

## Section 7: Evaluating KNN on Multiple Datasets

- We used a function (`load_and_prepare_data`) replicating the cleaning & engineering steps for each dataset.
- Tested **KNN** with best hyperparams (e.g., `n_neighbors=5, weights='distance'`) across multiple `.binetflow` files from CTU-13.
- Found that KNN typically scored extremely high (often near 1.0 in Accuracy, F1, etc.) but occasionally dropped slightly on certain data (e.g., ~98% on “9-Neris-20110817”).

Overall, **KNN** validated well across these additional sets, reinforcing its robustness as a go-to model for botnet detection in this environment.

---

## Findings & Recommendations

### Business Understanding & Significance

- We used the **CTU-13** dataset to identify malicious botnet traffic.  
- Botnets pose a **significant risk** to businesses: they can exfiltrate data, disrupt operations, or launch DDoS attacks.  
- **Detection is crucial** to mitigate potential financial and reputational damage.  
- AI/ML solutions **assist** security personnel by automatically flagging suspicious traffic so analysts can focus on **zero-day threats** and higher-level incident response instead of sifting benign flows.

### Clean & Organized Notebook

- Each section systematically handles data exploration, cleaning, feature engineering, and modeling.  
- The notebook (or `.py` file) logs each step for reproducibility and clarity.

### Correct & Concise Interpretation of Statistics

- Descriptive stats confirm the distribution of normal vs. botnet flows.  
- Inferential results (F1, AUC) confirm that certain features strongly separate classes.

### Key Findings for a Non-Technical Audience

- **We can detect botnet traffic with near 100% accuracy** across several major threats (Rbot, Neris, etc.).  
- The top features revolve around **bytes, packets, and addresses**, highlighting that malicious flows can be spotted by behavior patterns (e.g., extremely high or low rates).
- Automated ML detection drastically reduces manpower needed to filter out benign traffic.

### Next Steps & Recommendations

1. **Deploy** the KNN or similarly high-performing model in a production environment—possibly as a real-time intrusion detection module.
2. **Run** against new PCAP files (unseen traffic) to confirm generalization beyond CTU-13.
3. **Tune** the model for specific botnet families or custom threshold adjustments, especially if zero-day variants appear with new behaviors.
4. **Incorporate** additional features (e.g., DNS queries, domain-based features, time-series patterns) to improve coverage of emerging threats.
5. **Expand** to a full pipeline that integrates with SIEM systems, alerting security teams only when anomalous traffic crosses defined thresholds.

**Bottom Line**: By combining thorough data cleaning, robust feature engineering, and extensive model comparisons, we achieved a state-of-the-art detection pipeline. **KNN** is singled out as the best overall model in these experiments, delivering near‑perfect detection across multiple CTU-13 `.binetflow` files, thereby significantly reducing security staff workload and enhancing protection against botnet threats.
