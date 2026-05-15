# CO₂ Emissions & Climate Drivers: A Statistical Analysis of the Netherlands

> Investigating whether rising CO₂ emissions drive regional warming, and identifying what structural factors explain emission changes in the Netherlands across six decades of data.

---

## Overview

This project analyses the relationship between carbon dioxide emissions and temperature trends in the Netherlands using live data from [Our World in Data](https://ourworldindata.org/). It was built as a group presentation for the Foundations of Data Science module (MA7419) at the University of Leicester.

The work is split into two research questions:

- **RQ1** — How strongly is Netherlands annual temperature increase correlated with CO₂ emissions, and how has this changed over time?
- **RQ2** — What are the largest contributors to CO₂ emission changes in the Netherlands: energy efficiency, GDP growth, or fossil fuel mix?

---

## Key Findings

### RQ1 — Temperature & CO₂

| Analysis | Result | p-value |
|---|---|---|
| NL temperature linear trend (R²) | 0.494 | 0.00131 |
| NL ↔ Global detrended covariance (ρ) | 0.324 | 0.00247 |
| Global CO₂ ↔ Temperature (R²) | 0.840 | < 2.2e-16 |

- Netherlands temperatures show a statistically significant warming trend since the 1970s, tracking and slightly exceeding European and global rates.
- After removing long-term trends, Dutch year-to-year temperature swings show a modest but significant co-variation with global anomalies (ρ ≈ 0.32), indicating synchronized variability beyond just the shared trend.
- Global CO₂ emissions explain 84% of the variance in global mean surface temperature — strongly consistent with the established physical link.

### RQ2 — CO₂ Drivers (Multivariate Regression)

Regression model fitted in log-differences (growth rates):

```
Δln(CO₂ₜ) = β₀ + β₁ Δln(GDPₜ) + β₂ Δln(FossilShareₜ) + β₃ Δln(EnergyIntensityₜ) + εₜ
```

| Driver | Pre-1990 β | Pre-1990 p | Post-1990 β | Post-1990 p |
|---|---|---|---|---|
| GDP Growth | 2.141 | 0.00778 * | 0.967 | 4.3e-05 * |
| Fossil Fuel Share | 4.035 | 0.514 (ns) | 1.400 | 0.00353 * |
| Energy Intensity | 0.678 | 0.02726 * | 0.861 | 6.2e-07 * |

- GDP growth is a strong, significant driver of CO₂ in both periods — but the effect size roughly halves after 1990, suggesting the Dutch economy became less carbon-intensive as it grew.
- Fossil fuel share was not a significant driver before 1990, but becomes strongly significant after — a cleaner energy mix begins to matter more once efficiency improvements plateau.
- Energy intensity (energy used per unit of GDP) is a consistent and significant driver in both periods, with the effect strengthening post-1990.

The structural break around 1990 aligns with major European energy and climate policy shifts following the formation of the EU's internal energy market and early carbon reduction commitments.

---

## Methodology

```
Live API data  →  Cleaning & formatting  →  Transformation  →  Statistical modelling  →  Period comparison  →  Insights
```

**Data sources** (all fetched live via Our World in Data CSV API):
- Average monthly surface temperature — Netherlands, Europe (NIAID), World
- Annual CO₂ emissions per country — Netherlands, Europe, World
- CO₂ emissions per capita — Netherlands, EU27, World
- GDP per capita (Maddison Project Database) — Netherlands, UK, Germany, France
- Fossil fuel share of primary energy — Netherlands, Europe, World
- Primary energy consumption by source — Netherlands, UK, Germany, France

**Processing steps:**
- Missing value removal and column formatting
- Date parsing with `lubridate`; aggregation to annual means
- Detrending via linear model residuals for covariance analysis
- Log-differencing (`Δln`) to compute growth rates for regression
- Energy intensity derived as total primary energy divided by GDP per capita
- Pre/post-1990 period split to capture structural shifts

**Statistical methods:**
- Polynomial trend fitting (degree 2) with R² and significance tests
- Detrended Pearson correlation (ρ) between NL and global temperature anomalies
- Simple linear regression: global CO₂ vs global temperature
- Multivariate OLS regression for CO₂ drivers, fitted separately for pre- and post-1990 periods

---

## Tech Stack

| Tool | Purpose |
|---|---|
| R | All analysis and modelling |
| `tidyverse` / `dplyr` | Data wrangling and transformation |
| `lubridate` | Date parsing |
| `ggplot2` | All visualisations |
| Quarto (RevealJS) | Presentation output |

---

## Output

The project is delivered as a Quarto RevealJS presentation (`PG36_MA7419_CW2.qmd`). Render with:

```bash
quarto render PG36_MA7419_CW2.qmd
```

No local data files are required — all datasets are fetched live from the Our World in Data API at render time.

---

## Project Structure

```
├── PG36_MA7419_CW2.qmd     # Main Quarto source file
├── public.avif              # Figure 1 — GHG emissions by sector
└── README.md
```

---

## References

1. Visual Capitalist. A global breakdown of greenhouse gas emissions by sector. 2023.
2. Our World in Data. Average monthly surface temperature. 2025.
3. Our World in Data. Annual CO₂ emissions per country. 2025.
4. Our World in Data. GDP per capita — Maddison Project Database. 2024.
5. Our World in Data. Fossil fuels share of primary energy. 2025.
6. Our World in Data. Primary energy consumption by source. 2025.
7. von Storch H, Zwiers FW. Statistical Analysis in Climate Research. Cambridge University Press; 1999.
8. Shepherd TG. Atmospheric circulation as a source of uncertainty in climate change projections. Phil Trans R Soc A. 2016;374(2080):20140426.

---

## Authors

Group 36 — MA7419 Foundations of Data Science, University of Leicester, 2025.
