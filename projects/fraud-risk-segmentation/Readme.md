## Motivation: 

Traditional fraud detection systems typically focus on binary classification, where claims are labelled as either fraudulent or legitimate. While effective for identifying suspicious cases, this approach does not capture the underlying structure and variability present within insurance claims data.

In practice, claims differ significantly in terms of vehicle value, repair complexity, labour intensity, and cost patterns. Treating all claims uniformly can lead to inefficient investigation processes and a limited understanding of how different types of claims relate to fraud risk.

This project is motivated by the need to move beyond binary classification and develop a **data-driven segmentation approach** using clustering techniques. By grouping claims into distinct clusters based on their characteristics, the analysis aims to uncover hidden patterns in claim behaviour and identify segments associated with varying levels of fraud risk.

## Objective:

The objective of this project is to apply clustering techniques to segment insurance claims into distinct groups based on their underlying characteristics, and to analyse how these segments relate to fraud risk.

The analysis focuses on key variables derived from the dataset, including vehicle price, repair cost, repair duration, labour intensity, and the price-to-cost ratio. By grouping claims with similar patterns, the project aims to identify meaningful differences in claim behaviour that are not captured by traditional binary classification approaches.

In addition, the project evaluates the fraud rate within each cluster to determine whether certain segments exhibit higher levels of suspicious activity. This enables a deeper understanding of how fraud risk varies across different types of claims and supports the development of more targeted and efficient fraud detection strategies.

## System Pipeline:

The project follows a structured data science pipeline to transform raw insurance claim data into meaningful risk-based segments.

---

### 1. Data Cleaning

The raw dataset was first processed to remove invalid and unrealistic values that could distort the analysis. This included filtering out negative or extreme values in key variables such as vehicle price, mileage, repair cost, and repair hours. Domain-informed thresholds were applied to ensure that the data reflects realistic insurance claim scenarios, resulting in a high-quality dataset suitable for further analysis.

---

### 2. Exploratory Data Analysis

#### 📊 Exploratory Visualisations

The following visualisations highlight key patterns observed during exploratory data analysis, focusing on engineered features that are central to clustering.

---

**Price-to-Cost Ratio Distribution**
The distribution of the price-to-cost ratio shows significant variation across claims, with a right-skewed pattern indicating that a subset of claims have disproportionately high repair costs relative to vehicle value. This feature serves as a key indicator of potential anomalies.

![Ratio Distribution](outputs/figures/ratio_distribution.png)

---

**Cost per Repair Hour Distribution**
The cost per repair hour reveals variability in labour intensity, with some claims exhibiting unusually high costs that may indicate inefficiencies or suspicious activity.

![Cost per Hour](outputs/figures/cost_per_hour.png)

---

**Repair Lag Distribution**
The time between breakdown and repair highlights operational patterns, where delays may reflect differences in claim complexity or processing behaviour.

![Repair Lag](outputs/figures/repair_lag.png)

---

**Feature Correlation Matrix**
The correlation matrix illustrates relationships between key variables used in clustering. Low to moderate correlations suggest that the selected features provide complementary information, supporting their use in segmentation.

![Correlation](outputs/figures/correlation.png)

---

### 3. Feature Engineering

New features were created to capture meaningful patterns in the data. The price-to-cost ratio was used to measure repair cost relative to vehicle value, while cost per repair hour captured labour intensity. In addition, the time between breakdown and repair was calculated to represent operational delays. These engineered features form the basis for clustering and are central to identifying distinct claim profiles.

---

### 4. Clustering Model:


To segment insurance claims into meaningful groups, the K-Means clustering algorithm was applied using a set of engineered features that capture key aspects of claim behaviour, including vehicle value, repair cost, labour intensity, and cost efficiency.

---

#### Feature Selection and Scaling

The clustering model was built using both raw and engineered variables, including:

* Vehicle price
* Repair cost
* Repair hours
* Price-to-cost ratio
* Cost per repair hour
* Repair lag (days)

Since these variables exist on different scales, standardisation was applied prior to clustering to ensure that each feature contributes equally to the distance calculations used by the algorithm.

---

#### Determining the Number of Clusters

The optimal number of clusters was identified using the **Elbow Method**, which evaluates the within-cluster sum of squares (inertia) for different values of *k*.

As the number of clusters increases, the model’s inertia decreases, but with diminishing returns. The “elbow point” represents a balance between model complexity and explanatory power.

![Elbow Plot](outputs/figures/elbow.png)

The plot shows a clear inflection point around **k = 6**, beyond which the reduction in inertia becomes marginal.
The inertia decreases steeply from K=2 to K=5, after which the rate of reduction slows noticeably. 
The inertia values at K=5 and K=6 are almost identical, suggesting both are equally valid choices. K=6 was selected over K=5 because the additional cluster revealed a distinct fraud pattern, a group with 100% fraud rate characterised by extremely high repair hours (292 hours) and near-zero cost per hour (£1.29),  that was not identifiable at K=5. This additional cluster provides valuable insight into a specific fraud mechanism that would otherwise be merged into a larger, less interpretable group.

---

####  Final Model Selection

Based on the elbow method and consistency with observed data patterns, a model with **six clusters** was selected.

This choice provides a balance between:

* capturing meaningful variation in claim characteristics
* avoiding over-segmentation
* maintaining interpretability of cluster profiles

---

#### Rationale

The selected clustering approach allows the dataset to be partitioned into distinct groups that reflect underlying differences in claim behaviour. By incorporating both financial and operational features, the model captures multiple dimensions of variation, enabling more nuanced segmentation compared to traditional rule-based methods.

This forms the foundation for analysing how fraud risk varies across different claim types, which is explored in the subsequent cluster interpretation.

---

### Cluster Analysis and Visualisation

The resulting clusters were analysed by comparing their average characteristics, including vehicle price, repair cost, repair hours, and price-to-cost ratio. Fraud rates were calculated for each cluster to assess differences in risk levels. Visualisations were created to highlight variations across clusters, enabling clear identification of segments associated with higher or lower fraud risk.

## 🔍 Cluster Interpretation and Fairness Analysis

The clustering model segments insurance claims into six distinct groups, each representing a unique combination of vehicle characteristics, repair behaviour, and cost structure. Analysis of these clusters reveals clear differences in fraud risk and provides deeper insight into how claim behaviour—not just vehicle value—drives fraudulent activity.

---

### 📊 Cluster Profiles

![Cluster Profiles](outputs/figures/cluster_profiles.png)

---

### Cluster-Level Insights:

**Cluster 0 — Premium High-Cost Fraud (13,042 claims, 32% fraud)**
This cluster consists of high-value, newer vehicles (mean £32,557, average age 2.13 years) with extremely high labour costs (£413.91 per hour) and low mileage. The combination of elevated fraud rates and inflated labour costs suggests a pattern of **premium billing fraud**, where legitimate repairs are significantly overcharged. This cluster is dominated by Premium and Luxury vehicles (80%).

---

**Cluster 1 — Standard Legitimate Claims (123,392 claims, 17% fraud)**
The largest cluster, representing mid-range vehicles (£14,480) with moderate mileage and typical labour costs (£61.29 per hour). The fraud rate is below the dataset average, indicating this segment reflects **normal claim behaviour** and serves as a baseline for comparison.

---

**Cluster 2 — Aged Budget High-Ratio Claims (9,612 claims, 29% fraud)**
This cluster contains older, high-mileage budget vehicles (mean £4,715, ~81k miles, age 9.6 years) with elevated price-to-cost ratios (28.28). The high fraud rate suggests that repair costs are disproportionately large relative to vehicle value, indicating **cost inflation behaviour in low-value vehicles**. The cluster is predominantly composed of Budget segment vehicles (65%).

---

**Cluster 3 — Complex Legitimate Premium Claims (26,443 claims, 15% fraud)**
High-value vehicles (£37,968) with the highest repair complexity (3.03) but reasonable labour costs (£62.53 per hour). Despite high absolute costs, the low fraud rate suggests that these are **genuinely complex repairs rather than fraudulent claims**.

---

**Cluster 4 — Aged Budget Low-Cost Claims (83,099 claims, 15% fraud)**
Similar to Cluster 2 in terms of vehicle type (budget, high mileage), but characterised by low labour costs (£38.07 per hour) and low price-to-cost ratios (3.18). The low fraud rate indicates **genuine, low-cost repairs**, demonstrating that not all budget vehicle claims are high risk.

---

**Cluster 5 — Pure Labour Fraud (3,471 claims, 100% fraud)**
This cluster is defined by extreme repair hours (mean 292 hours) combined with near-zero labour cost (£1.29 per hour). This pattern represents a clear case of **labour inflation fraud**, where repair hours are artificially inflated while keeping hourly rates low to avoid detection based on cost thresholds.

---

## Fairness Assessment

The clustering results provide a critical comparison to the bias identified in earlier analysis. In particular, clusters dominated by Budget vehicles (Clusters 2 and 4) exhibit significantly different fraud rates (29% vs 15%), despite similar vehicle characteristics.

This distinction is important:

* Cluster 4 represents **low-risk budget claims**, characterised by low labour cost and balanced repair behaviour
* Cluster 2 represents **high-risk budget claims**, driven by inflated repair costs relative to vehicle value

This demonstrates that clustering successfully separates claims based on **behavioural patterns rather than vehicle price alone**.

---

### Comparison with Rule-Based Bias

In earlier analysis (RQ1), a uniform threshold approach disproportionately flagged Budget vehicles, despite their lower actual fraud rate. In contrast, the clustering approach distributes fraud risk across multiple segments:

* Cluster 0 (high fraud) → predominantly Premium and Luxury (80%)
* Cluster 2 (high fraud) → predominantly Budget (65%)

This cross-segment distribution shows that **fraud risk is not tied to a single socioeconomic group**, but instead depends on claim behaviour.

---

###  Key Insight

Clustering enables a more nuanced understanding of fraud risk by incorporating multiple behavioural features simultaneously. High-risk clusters are identified based on patterns such as inflated labour costs, excessive repair hours, or abnormal cost ratios, rather than relying on simple thresholds.

---

### Limitations

Despite these improvements, some risk of imbalance remains. Cluster 2, which is dominated by budget vehicles and exhibits high fraud rates, could still disproportionately impact lower-value vehicle owners if used directly for decision-making.

This reinforces the need to combine clustering with **segment-aware thresholds and fairness considerations**, ensuring that fraud detection systems remain both accurate and equitable.

## Key Results:

The clustering analysis reveals several important findings regarding fraud behaviour and claim segmentation:

* Fraud risk varies significantly across clusters, ranging from **15% to 100%**, demonstrating that claim behaviour is a strong determinant of fraud likelihood.

* **Cluster 5 (Pure Labour Fraud)** exhibits extreme characteristics, with artificially inflated repair hours and near-zero labour cost, resulting in a **100% fraud rate**. This represents a clear and distinct fraud pattern.

* **Cluster 0 (Premium High-Cost Fraud)** shows a high fraud rate of **32%**, driven by unusually high labour costs in premium and luxury vehicles, indicating overbilling behaviour.

* **Cluster 2 (Aged Budget High-Ratio Claims)** has a fraud rate of **29%**, where repair costs are disproportionately high relative to vehicle value, highlighting risk in low-value vehicles with inflated claims.

* In contrast, **Clusters 3 and 4** exhibit lower fraud rates (~15%), despite differences in vehicle value and repair complexity, suggesting that not all high-cost or budget claims are inherently risky.

* The largest segment, **Cluster 1**, represents standard claim behaviour with a fraud rate of **17%**, closely aligned with the dataset average (~18.5%).

---

### Key Takeaway:

The results demonstrate that fraud detection cannot rely on single variables such as vehicle price or repair cost alone. Instead, fraud risk emerges from **combinations of behavioural patterns**, including labour cost anomalies, repair duration, and cost-to-value relationships.

This reinforces the importance of segmentation-based approaches in identifying high-risk claims more accurately than traditional rule-based systems.

