# 🚀 PROJECT LOG: 4-Week UK Energy Churn Build

Steve Mathew | Day 0: Dec 13, 2025

## Week 1 (Dec 13–19): Data + EDA

- [x] Portfolio live (flagship + 2 minis)  
- [x] UK Energy dataset in `data/raw/` + basic checks (rows, cols, churn rate)  
- [x] Initial EDA: churn by geography, tenure, age, gender etc. 
- [x] Write 3–5 bullets: key churn patterns from EDA  
- [ ] Week 1 reveal.js deck: `slides/week1_eda.html`  

## Week 2 (Dec 20–26): Modeling + Features

- [x] Logistic baseline + tree model (XGBoost or Random Forest)  
- [x] Metric focus: high recall on churners with reasonable precision  
- [x] Compare models and pick “champion” for app + SHAP
      - Champion model: ranger Random Forest, class.weights = c("0"=1,"1"=4), mtry = 3, num.trees = 300, threshold = 0.3 (recall ≈ 0.65, precision ≈ 0.59).
- [ ] Week 2 reveal.js deck: `slides/week2_models.html`  

## Week 3 (Dec 27–Jan 2): Streamlit + SHAP

- [ ] Interactive churn dashboard in Streamlit (risk score + drivers)  
- [ ] SHAP explanations: global top drivers + a few example customers  
- [ ] Refine business case (simple £ impact story)  
- [ ] Week 3 reveal.js deck: `slides/week3_explain_app.html`  

## Week 4 (Jan 3–10): Deploy + Apply

- [ ] Polish README with results, screenshots, and links (app + decks)  
- [ ] Light deployment notes (how to run app; ideas for batch scoring)  
- [ ] Optional survival analysis notebook (time‑to‑churn experiment)  
- [ ] Use project in 10 targeted Energy/Automotive applications  
- [ ] Week 4 reveal.js deck: `slides/week4_business.html`  

