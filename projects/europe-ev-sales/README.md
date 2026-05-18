# EV Adoption in Europe: An Exploratory Data Analysis

> Investigating the socio-economic and infrastructural drivers of electric vehicle adoption across European markets, and visualising regional growth patterns from 2015 to 2024.

---

## Overview

This project analyses Battery Electric Vehicle (BEV) adoption trends across 17 European countries using data from the International Energy Agency (IEA) Global EV Data Explorer. It was completed as part of the Operational Data Science module (MA7441) at the University of Leicester.

The analysis focuses on three core questions:

- **H1** — Does higher median income positively correlate with EV adoption?
- **H2** — Does greater density of public charging stations lead to higher EV adoption?
- **H3** — Do regions with higher fuel prices show faster EV uptake?

The preliminary analysis focuses on H2, establishing a strong positive relationship between charging infrastructure and BEV stock across leading European markets, with H1 and H3 planned as extensions.

---

## Key Findings

### Charging Infrastructure vs EV Sales (H2)

Linear regression fitted per country on EV stock vs electric charging points (2015–2024):

| Country | R² | β (slope) | p-value |
|---|---|---|---|
| Germany | 0.9806 | 13.31 | 3.9e-08 |
| France | 0.9827 | 7.95 | 2.5e-08 |
| United Kingdom | 0.9817 | 16.81 | 3.1e-08 |
| Netherlands | 0.9883 | 3.57 | 5.2e-09 |

- All four countries show very strong positive linear relationships between charging infrastructure and BEV stock (R² ≥ 0.98, p < 0.001).
- The UK has the steepest slope, suggesting the highest marginal EV adoption per charging point added.
- Residual analysis reveals non-linearity in early years (2015–2019), where an exponential model provides a better fit than linear regression.

### Regional Patterns

- Western Europe (Germany, France, UK, Netherlands) dominates BEV adoption with the steepest growth curves, particularly after 2020.
- The Nordic region (Norway, Sweden, Denmark) shows strong adoption at a lower absolute scale.
- Southern and Eastern Europe exhibit slower, later-starting growth from much smaller baselines.
- The gap between regions widens over time, suggesting that policy maturity, charging infrastructure density, and economic capacity are key structural drivers.

### Distribution

- BEV sales data is highly right-skewed across all countries, consistent with exponential adoption growth.
- A normal distribution does not fit the data well — an F-distribution with adjusted degrees of freedom provides a better approximation.
- Germany (1.9M), UK (1.4M), and France (1.1M) were the top three markets by 2024 BEV stock.

---

## Methodology

```
IEA raw data  →  Cleaning & filtering  →  EDA  →  Statistical modelling  →  Visualisation  →  Insights
```

**Data source:**
- IEA Global EV Data Explorer (EVDataExplorer2025.xlsx)
- Source: https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer
- License: CC BY 4.0
- Version 1.2, last updated 31 July 2025, retrieved 15 October 2024

**Processing steps:**
- Filtered to 17 European countries, historical data only (2015–2024)
- Extracted BEV stock, electric charging points, FCEV, and PHEV as separate datasets
- Pivot-wide transformation for year-on-year country comparisons
- Processed files saved to `data/processed/`

**Statistical methods:**
- Descriptive statistics: mean, median, SD, IQR per country
- Distribution fitting: normal, F-distribution, kernel density estimates
- Simple linear regression: EV sales ~ charging infrastructure, per country
- Exponential regression (NLS): EV sales ~ year, for growth modelling
- Residual analysis to assess model fit and non-linearity

**Visualisation techniques:**
- Line plots with log scale for cross-country growth comparison
- Horizontal boxplots with log-scale x-axis for distribution comparison
- Faceted scatter plots with region-level LOESS smoothing
- Area charts for top/bottom country comparison

---

## Tech Stack

| Tool | Purpose |
|---|---|
| R | All analysis and modelling |
| `tidyverse` / `dplyr` | Data wrangling and transformation |
| `ggplot2` | Static visualisations |
| `plotly` | Interactive 3D visualisation |
| `scales` | Axis formatting |
| Quarto (HTML) | Reproducible notebook output |

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── EVDataExplorer2025.xlsx        # Raw IEA dataset
│   └── processed/
│       ├── EV_Europe_BEV_Stock_2015_2024.csv
│       ├── EC_Europe_BEV_Stock_2015_2024.csv
│       ├── EV_Europe_FEV_Stock_2015_2024.csv
│       └── EV_Europe_PHEV_Stock_2015_2024.csv
├── Week2_Data_Wrangling_EDA.qmd           # Data pipeline and EDA
├── Week4_Statistics_First_Model.qmd       # Distributions and regression
├── Week7_Visualisation_Storytelling.qmd  # Visualisation and narrative
├── session-info.txt                       # Reproducibility log
└── README.md
```

---

## Limitations

- Analysis covers 17 European countries — smaller and Eastern European markets are underrepresented, which may inflate observed correlations.
- Only H2 (charging infrastructure) is tested in this phase; income levels (H1) and fuel prices (H3) are planned extensions.
- Charging infrastructure and EV sales likely have a bidirectional relationship — causal claims should not be drawn from regression results alone.
- With only 10 data points per country (2015–2024), regression results should be interpreted with caution despite high R² values.

---

## Ethical Notes

- **Representation fairness:** Large markets (Germany, France, UK) have more complete data and disproportionately influence model outcomes. Results are reported per country rather than generalised across Europe to mitigate this.
- **Data provenance:** IEA data is aggregated from national agencies using potentially inconsistent collection standards. Comparability between countries carries measurement uncertainty.
- **Reproducibility:** Random seeds documented in `session-info.txt`. All preprocessing steps are logged in the weekly notebooks.

---

## Data Attribution

IEA (2024); Global EV Data Explorer, https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer, License: CC BY 4.0

---

