Domain: Recruitment

## How to Run
Run the following commands in order from a clean clone of the repository:
```bash
pip install -r requirements.txt
python generate_data.py
python part1.py #Estimation, inference and a regression baseline
python part2.py #Classification and evaluation
python part3.py #Segmentation and dimensionality reduction
python part4.py #Model-choice and architecture memo

##Part 1 Findings
    R1.1: The null hypothesis $H_0: \mu = 8.0$ was rejected ($z\_stat = 6.2418$, absolute value exceeds critical value $1.96$). A Type I error in this domain means incorrectly concluding that the true average time to an offer decision differs from 8.0 days when it actually equals 8.0 days. The gap between the mean value ($8.9380$) and the planning assumption ($8.0$) is nearly one full day, which matters in practice because recruitment pipelines rely heavily on accurate planning assumptions to avoid staffing bottlenecks.
    R1.2: $n\_required = 138$ compared to 800 available rows. This means our sample size is more than sufficient to estimate the mean within the required margin of error ($E = 0.5$).
    R1.3: Gradient descent reached a slope gap of $0.0000$ (exact match) because linear regression with MSE cost is a strictly convex optimization problem, and the standardisation of $X_4$ ensured smooth, stable gradient updates.
    R1.4: The raw model exhibited a funnel pattern with resid_sd_ratio_raw = 1.3421, showing that residual spread grows with fitted values. The log-log transformation successfully removed this funnel, bringing resid_sd_ratio_log closer to 1.0.
    R1.5: Predictors in Task 1.7 with coefficients above $0.05$ absolute value include $\ln(X_4)$ and $X_3$. Comparing adjusted $R^2$ shows whether the extra predictors earn their place by improving explanatory power beyond penalizing complexity.
##Part 2 Findings
    R2.1: The learning rates ordered by loss from lowest to highest are $\eta = 0.1$, $\eta = 0.01$, and $\eta = 1.0$. A learning rate that is too small ($\eta = 0.01$) under-updates the weights in 200 epochs, while a learning rate that is too large ($\eta = 1.0$) overshoots the minimum and causes divergence or high loss.
    R2.2: The three predictors with the largest absolute weights are salary gap ratio, interview reschedules, and experience. An odds ratio greater than 1 for salary gap indicates that higher expected pay gaps substantially increase the odds of a candidate declining an offer.
    R2.3: Blind accuracy is $0.7875$, whereas $\tau = 0.5$ accuracy is higher, demonstrating why accuracy alone is misleading under class imbalance. Lowering $\tau$ from $0.5$ to $0.3$ increased recall significantly while trading off precision.
    R2.4: The logistic regression model at $\tau = 0.3$ outperformed KNN in recall and F1-score by capturing linear boundaries effectively across standardised features.
    R2.5: The logistic model at cost-optimized threshold $\tau = 0.35$ is recommended because it minimizes financial loss (false alarms cost INR 50, missed positives cost INR 400). Thresholds are chosen on training rows to prevent data leakage and overfitting to the test set.
##Part 3 Findings
    R3.1: WCSS drops across $K$ support $K=3$ as the optimal elbow point where marginal variance reduction levels off.
    R3.2: Segment A represents "Experienced & Stable Candidates" (low friction, high tenure), Segment B represents "Moderate Friction Candidates", and Segment C represents "High-Friction, High-Gap Candidates" with the highest outcome decline rate of $45\%$. Action: provide immediate recruiter engagement for Segment C.
    R3.3: Single and complete linkage yield different cluster structures due to chaining versus compact diameter sensitivities.
    R3.4: Principal Component 1 is primarily driven by experience and salary expectations. Two components capture over $80\%$ of cumulative variance. Segment mean PC1 scores clearly separate candidate profiles.
##Part 4 Findings
    R4.1: Split S3 ($X_3 > 0.5$) is the best split because it achieves the highest information gain. Growing a tree until every leaf is pure causes severe overfitting; max depth constraints or minimum samples per leaf prevent this.
    R4.2: Logistic regression and soft voting ensembles provide robust recall and F1-scores, confirming that combining models stabilizes decision boundaries.
    R4.3: After round 1, the total weight of misclassified points increased for subsequent re-weighting, allowing round 2 to focus on hard examples.
    R4.4: Convolutional filters maintain spatial dimensions while extracting local features, whereas fully connected layers scale quadratically. Skip connections mitigate vanishing gradients in deep networks.

##Model-choice Memo
To: Head of Talent-Acquisition Operations
From: Talent-Analytics Team
Subject: Model Deployment Recommendation for Offer Decline Prediction

We recommend deploying the Logistic Regression model with threshold $\tau = 0.30$ for production use. Based on our evaluation test rows (cost_test_at_tau), this configuration minimizes financial exposure, yielding a total cost of INR 6,950, compared to INR 11,200 for flagging nobody (cost_test_flag_nobody) and INR 8,400 at $\tau = 0.5$. The primary limitation of this evidence is the synthetic nature of the dataset and the limited positive test sample size ($n = 51$). Building Model A (image processing for scanned forms) is not recommended before the structured pipeline is fully operational, as structured resume attributes drive immediate operational decisions.

##Limitations
1- The dataset is entirely synthetic and generated via a seeded script, meaning real-world noise and    unmodeled human behaviors are absent.
2- A single 70/30 stratified split was used for evaluation without cross-validated hyperparameter testing across multiple random seeds.
3 -The count of positive decline outcomes in the test set ($n = 51$) is relatively small, leading to potential variance in precision and recall metrics.