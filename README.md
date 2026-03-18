# Aadhaar Identity Maintenance Risk Framework — UIDAI Hackathon Project

**Built for UIDAI Data Hackathon 2026**  
District-level prioritization engine using Aadhaar enrolment/update data to identify high-risk areas for stale records and authentication failures.


### Problem
High-enrolment districts suffer low update rates → stale biometric/demographic data → authentication failures, exclusion, governance risk.

### Solution
- **District-level aggregation** via pincode mapping
- **Composite Risk Index**: Enrolment volume + update rate + age-group balance
- **K-means clustering** → 4 archetypes (High-Growth Low-Maintenance, etc.)
- **NITI Aayog validation** - correlation with Financial Inclusion progress
- **Prophet forecasting** for future backlog trends
- **Interactive dashboard** (ipywidgets + Plotly choropleth)

### Visual Analysis
#### 1. District Risk Map (Choropleth)
![District Risk Map](plots/maintenance_risk.png)
*Geographic distribution of the Composite Risk Index across analyzed districts.*

#### 2. NITI Aayog Correlation & Trends
![NITI Correlation](plots/niti_scatter.png)
*Correlation between financial inclusion progress and Aadhaar maintenance risk.*

#### 3. Backlog Forecasting (Prophet Model)
![Prophet Forecast](plots/prophet_plot.png)
*Forecasting future update demand to predict potential system backlogs.*

### Key Insights
- 61% of analysed enrolments in priority archetypes needing urgent action
- Higher risk in Aspirational Districts (0.7904 vs 0.7819 avg risk score)
- Negative correlation: Faster financial inclusion progress → lower maintenance risk
- Forecast: Update demand growing → backlog risk rising without intervention

### Tech Stack
- Python: pandas, numpy, scikit-learn, Prophet, Plotly, ipywidgets
- Jupyter Notebook: `UIDAI Project.ipynb`
- Data: UIDAI anonymized enrolment/update CSVs + NITI Champions of Change (Financial Inclusion)

### Repository Structure
```
Uidai-project/
├── UIDAI Project.ipynb          # Main analysis notebook
├── financial_inclusion.csv      # NITI Aayog data for validation
├── pincode_directory.csv        # Pincode → district mapping
├── india_districts.json         # GeoJSON for choropleth
├── my_plot.png                  # Sample visualization
├── utils/                       # Helper functions (config.py, helpers.py)
└── README.md
```

### How to Run
1. Clone repo: `git clone https://github.com/Vatsalsrivastava1209/Uidai-project.git`
2. Install dependencies: `pip install -r requirements.txt` 
3. Open `UIDAI Project.ipynb` in Jupyter/Colab
4. Run all cells (data files included)

### Results & Impact
- Targeted policy blueprint for UIDAI: Mobile camps in high-risk archetypes
- Scalable to other India Stack layers (GSTN, UPI, Ayushman Bharat)
- Demonstrates end-to-end data science: cleaning → modeling → visualization → policy recommendation

### Future Work
- Full national-scale deployment
- Real-time monitoring dashboard
- Anomaly detection for fraud patterns

Feedback / collaboration welcome!  
VaTsaL (@Codat_V) On X | Open to data analytics/science roles

#data-science #aadhaar #DataAnalytics #hackathon #python #plotly
