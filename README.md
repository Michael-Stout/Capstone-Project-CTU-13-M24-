# Network Traffic Analysis for Botnet Detection
**Author:** Michael Stout

## Executive Summary
This project analyzes network traffic data to detect and classify botnet activities using various machine learning techniques. Through processing over 107,000 network traffic records from the CTU-13 dataset (Scenario 11), we identified clear distinctions between normal and malicious (botnet) traffic. Multiple models (Random Forest, Decision Tree, Naive Bayes, K-Nearest Neighbors (KNN), SVM, Logistic Regression, and Gradient Boosting) were trained, tuned, and evaluated.  
Notably, **KNN** demonstrated exceptional performance, achieving near-perfect classification metrics both on Scenario 11 and when tested against other CTU-13 scenarios.

## Rationale
Modern cyber threats (particularly botnets and zero-day exploits) create significant risks to network security. Signature-based detection methods often fail to catch new or evolving threats, whereas machine learning offers an adaptive, proactive defense mechanism. By employing classification models and engineered network traffic features, organizations can more effectively detect suspicious behavior in real time.

## Research Question
**How can machine learning techniques enhance the detection of zero-day exploits and botnet activities within network traffic?**  
Specifically, **which network traffic patterns and features** most indicate botnet activity?

## Data Sources
This analysis used the [CTU-13 Dataset](https://www.stratosphereips.org/datasets-ctu13#:~:text=The%20CTU%2D13%20is%20a,normal%20traffic%20and%20background%20traffic.), focusing on Scenario 11. Key details include:

- **Total records:** 107,251  
- **Botnet traffic:** 8,164  
- **Normal traffic:** 2,718  
- **Background traffic:** 96,369  

Scenario 11 captures the **Neris botnet**, known for spamming and click fraud.  
Reference:  
Sebastian Garcia, Martin Grill, Jan Stiborek and Alejandro Zunino. "An empirical comparison of botnet detection methods," *Computers and Security Journal*, Elsevier. 2014. Vol 45, pp 100-123.

## Methodology
1. **Data Processing & Cleaning**  
   - Loaded raw network traffic data (107,251 records).  
   - Handled missing values for ports, states, and traffic characteristics.  
   - Grouped traffic types into **Background**, **Botnet**, and **Normal** categories.  
   - Created enhanced features, such as `BytePktRatio`, entropy metrics, traffic flow statistics, and duration categories.

2. **Feature Engineering**  
   - **Duration categories** (`very_short`, `short`, `medium`, `long`).  
   - **Byte-per-packet ratio** and **bytes/packets per second**.  
   - **Entropy metrics** for source and destination IP addresses.  
   - **Categorical encoding** for protocol, direction, port ranges, etc.

3. **Model Development**  
   - **Multiple Classifiers:**  
     - Random Forest  
     - Decision Tree  
     - Naive Bayes  
     - **KNN**  
     - SVM  
     - Logistic Regression  
     - Gradient Boosting  
   - **Hyperparameter Tuning:** Used GridSearchCV.  
   - **Cross-Validation:** Validated each model’s performance using multiple folds.

4. **Analysis & Visualization**  
   - Explored distribution of botnet vs. normal traffic.  
   - Created correlation heatmaps, bar charts, scatter/strip plots, pair plots, etc.  
   - Generated metrics including F1 Score, Precision, Recall, ROC AUC, Log Loss, and Mean Average Precision (mAP).

## Results
### Overall Model Performance
Across all models, classification metrics (Accuracy, F1, Precision, Recall) were **extremely high**. Each model achieved an F1 > 0.998 on the test set. Below is a brief summary:

| Model               | Test Accuracy | Test F1   | ROC AUC | Log Loss  |
|---------------------|--------------:|----------:|--------:|----------:|
| RandomForest        | 0.9986       | 0.9991    | 1.0000  | 0.0024    |
| DecisionTree        | 0.9995       | 0.9997    | 0.9991  | 0.0166    |
| NaiveBayes          | 0.9986       | 0.9991    | 0.9986  | 0.0497    |
| **KNN**             | **0.9991**   | **0.9994**| **1.0000** | **0.0016** |
| SVM                 | 0.9991       | 0.9994    | 0.9989  | 0.0097    |
| LogisticRegression  | 0.9986       | 0.9991    | 0.9999  | 0.0059    |
| GradientBoosting    | 0.9986       | 0.9991    | 1.0000  | 0.0030    |

All models reached near-perfect scores. **KNN** stood out for its perfect ROC AUC = 1.0000 and highest F1 Score on the test set.  

**Sample Plots**  
- Precision-Recall Curve for RandomForest: [S5_pr_curve_RandomForest](./plots/S5_pr_curve_RandomForest.png)  
- Gains Chart for DecisionTree: [S5_gains_DecisionTree](./plots/S5_gains_DecisionTree.png)  
- Confusion Matrix (KNN): [S5_confusion_matrix_KNN](./plots/S5_confusion_matrix_KNN.png)  

### Model-Specific Insights
Below, each model is grouped into its own subsection with key plots, strengths, and weaknesses.

#### 1. Random Forest
**Key Plots**  
- Precision-Recall: [S5_pr_curve_RandomForest](./plots/S5_pr_curve_RandomForest.png)  
- ROC Curve: [S5_roc_RandomForest](./plots/S5_roc_RandomForest.png)  
- Feature Importances: [S5_top10_features_RandomForest](./plots/S5_top10_features_RandomForest.png)  

**Strengths**  
- Handles high-dimensional data and complex interactions well.  
- Robust to outliers and missing data.  

**Weaknesses**  
- Larger memory footprint.  
- Can be slower to train with large parameter grids.

#### 2. Decision Tree
**Key Plots**  
- Precision-Recall: [S5_pr_curve_DecisionTree](./plots/S5_pr_curve_DecisionTree.png)  
- ROC Curve: [S5_roc_DecisionTree](./plots/S5_roc_DecisionTree.png)  
- Feature Importances: [S5_top10_features_DecisionTree](./plots/S5_top10_features_DecisionTree.png)  

**Strengths**  
- Very interpretable with straightforward rules.  
- Fast to train.  

**Weaknesses**  
- Prone to overfitting if not well-pruned.  
- Less robust for complex boundary classifications without ensemble methods.

#### 3. Naive Bayes
**Key Plots**  
- Precision-Recall: [S5_pr_curve_NaiveBayes](./plots/S5_pr_curve_NaiveBayes.png)  
- ROC Curve: [S5_roc_NaiveBayes](./plots/S5_roc_NaiveBayes.png)  

**Strengths**  
- Extremely fast to train.  
- Theoretically optimal for independent features.  

**Weaknesses**  
- Assumes (often unrealistic) independence among features.  
- No direct measure of feature importance available (in the standard approach).

#### 4. K-Nearest Neighbors (KNN)
**Key Plots**  
- Precision-Recall: [S5_pr_curve_KNN](./plots/S5_pr_curve_KNN.png)  
- ROC Curve: [S5_roc_KNN](./plots/S5_roc_KNN.png)  
- Confusion Matrix: [S5_confusion_matrix_KNN](./plots/S5_confusion_matrix_KNN.png)  

**Strengths**  
- Simple, highly effective for well-separated classes.  
- Easy to implement, no explicit training step beyond storing data.  

**Weaknesses**  
- Computation cost grows with dataset size.  
- Performance can degrade with high-dimensional data if not carefully tuned.

#### 5. Support Vector Machine (SVM)
**Key Plots**  
- Precision-Recall: [S5_pr_curve_SVM](./plots/S5_pr_curve_SVM.png)  
- ROC Curve: [S5_roc_SVM](./plots/S5_roc_SVM.png)  
- Feature Weights: [S5_top10_features_SVM](./plots/S5_top10_features_SVM.png)  

**Strengths**  
- Effective in high dimensional spaces.  
- Many kernel functions for different data complexities.  

**Weaknesses**  
- Can be slower in large datasets.  
- Needs careful tuning of kernel parameters.

#### 6. Logistic Regression
**Key Plots**  
- Precision-Recall: [S5_pr_curve_LogisticRegression](./plots/S5_pr_curve_LogisticRegression.png)  
- ROC Curve: [S5_roc_LogisticRegression](./plots/S5_roc_LogisticRegression.png)  
- Feature Coefficients: [S5_top10_features_LogisticRegression](./plots/S5_top10_features_LogisticRegression.png)  

**Strengths**  
- Offers clear, interpretable coefficients.  
- Fast training times.  

**Weaknesses**  
- Assumes a linear decision boundary.  
- Susceptible to underfitting if relationships are highly non-linear.

#### 7. Gradient Boosting
**Key Plots**  
- Precision-Recall: [S5_pr_curve_GradientBoosting](./plots/S5_pr_curve_GradientBoosting.png)  
- ROC Curve: [S5_roc_GradientBoosting](./plots/S5_roc_GradientBoosting.png)  
- Feature Importances: [S5_top10_features_GradientBoosting](./plots/S5_top10_features_GradientBoosting.png)  

**Strengths**  
- Often delivers state-of-the-art predictive performance.  
- Can handle diverse data types and reduce bias from weaker learners.  

**Weaknesses**  
- Tends to overfit if not tuned carefully.  
- Longer training times than simpler models.

---

## Detailed KNN Performance (Section 7)
After selecting **KNN** as our primary model, we evaluated it across all **13 scenarios** of the CTU-13 dataset. This cross-scenario validation highlights how well KNN generalizes:

- **Perfect accuracy (1.0)** achieved on 6 scenarios (e.g., Rbot Scenarios 3,4,10,11 and Virut Scenarios 5,13).  
- **Above 0.99 accuracy** on an additional 5 scenarios.  
- **Notable Challenges**:  
  - **Scenario 9 (Neris)**: 0.977 accuracy  
  - **Scenario 12 (NsisAy)**: 0.978 accuracy  

Despite these slight drops, KNN consistently maintained extremely high performance (F1 often > 0.99). The minor decreases in accuracy highlight how some botnet families (e.g., Neris) produce more varied traffic patterns, thus slightly challenging the distance-based classification.  

| Scenario       | Accuracy  | Comments                                    |
|--------------- |----------:|--------------------------------------------|
| 1-Neris        | 0.9998    | Nearly perfect performance                 |
| 2-Neris        | 0.9997    | Perfect recall in this scenario            |
| 3-Rbot         | 1.0000    | Perfect detection of Rbot traffic          |
| 4-Rbot         | 1.0000    | 100% accuracy, all classes                 |
| 5-Virut        | 1.0000    | Perfect classification                     |
| 6-Menti        | 1.0000    | Perfect classification                     |
| 7-Sogou        | 1.0000    | Perfect classification                     |
| 8-Murlo        | 0.9999    | Nearly perfect detection                   |
| 9-Neris        | 0.9770    | Most challenging Neris scenario            |
| 10-Rbot        | 0.9993    | Achieved near-perfect detection            |
| 11-Rbot        | 0.9991    | Very high accuracy                         |
| 12-NsisAy      | 0.9780    | Next most challenging scenario (NsisAy)    |
| 13-Virut       | 0.9997    | Excellent detection accuracy               |

### Key Takeaways
- **Generalization:** KNN’s ability to classify diverse botnet families (Neris, Rbot, Virut, etc.) with near-perfect results underscores its robustness.  
- **Challenging Scenarios:** Some families, like Neris (Scenario 9) and NsisAy (Scenario 12), slightly reduced performance due to more complex distribution of features.  
- **Conclusion:** KNN stands out as a top choice for real-world detection across multiple botnet families if computational resources are sufficient.

---

## Next Steps
1. **Model Deployment**  
   - Deploy KNN as the primary classifier within a real-time pipeline.  
   - Integrate alerts and dashboards for immediate threat detection.

2. **Feature Enhancement**  
   - Explore additional protocol-level features, deep packet inspection data, and advanced entropy-based metrics.  
   - Incorporate time-series or streaming analysis for real-time monitoring.

3. **Performance Optimization**  
   - Scale KNN for real-time predictions (e.g., approximate nearest neighbors).  
   - Automate data preprocessing and model retraining as new threats emerge.

4. **Operational Integration**  
   - Design and implement a robust, scalable architecture for deployment.  
   - Implement automated mitigation or response mechanisms for detected threats.

---

## Outline of Project

### 1. Network Analysis
- **Section 1:** [Import Libraries & Logging](./plots/S2_time_based_totpkts_all_scaled.png)  
  Sets up logging, imports standard libraries, and configures global variables.  
- **Section 2:** [Data Loading & Exploration](./plots/S2_target_distribution_combined.png)  
  Loads dataset(s), explores columns, checks missing values, provides summary stats.

### 2. Feature Engineering
- **Section 3:** [Data Cleaning & Feature Engineering](./plots/S3_durcategory_distribution.png)  
  Removes redundant columns, creates derived features (`BytesPerSecond`, `PktsPerSecond`, IP entropy), applies categorical encoding.  
- **Section 4:** [Visualizations](./plots/S4_botnet_distribution.png)  
  Shows distribution of botnet vs normal traffic, correlation heatmap, and other analyses.

### 3. Model Development
- **Section 5:** [Train-Test Split & Multi-Model Pipeline](./plots/S5_confusion_matrix_KNN.png)  
  Prepares data splits and builds multiple classifiers with GridSearchCV. Logs performance metrics.  
- **Section 6:** [Model Evaluations](./plots/S6_model_metrics_scaled_line.png)  
  Compares model metrics, outputs scaled comparison charts, logs summary table of performance.

### 4. KNN Evaluation
- **Section 7:** [Evaluate KNN on Multiple Datasets](./plots/S5_pr_curve_KNN.png)  
  Demonstrates generalization by applying the KNN model to multiple external CTU-13 scenario files. Logs final performance metrics.

> **Note:** All referenced plot images are saved in the `plots/` directory.  

---

**Thank you for reviewing this comprehensive analysis.**  
Should you have questions or need additional details on methodology or results, please feel free to reach out!  
