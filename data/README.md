# 📊 Dataset Documentation

## 1️⃣ Dataset Overview

### What This Dataset Represents
This dataset contains customer-level information from a **telecommunications company**, capturing demographic attributes, service subscriptions, account details, and churn behavior. It is commonly known as the **Telco Customer Churn** dataset.

### Unit of Observation
**Each row represents a single customer account** with their complete profile at the time of data extraction, including:
- Personal demographics
- Service subscriptions (phone, internet, add-ons)
- Contract and billing details
- Customer support interactions
- Churn status

### Time Granularity
This is a **point-in-time snapshot** dataset — it captures the state of each customer at a specific moment, not a longitudinal history. There is no time-series dimension tracking changes over time.

### Dataset Size
| Metric | Value |
|--------|-------|
| Total Records | 7,043 |
| Total Features | 23 |
| File Format | CSV |
| File Location | `data/raw/churn.csv` |

---

## 2️⃣ Target Variable Definition

> [!CAUTION]
> **Getting this wrong will invalidate your entire analysis.** Ensure you understand the target definition thoroughly.

| Attribute | Details |
|-----------|---------|
| **Column Name** | `Churn` |
| **Data Type** | Categorical (Yes/No) |
| **Positive Class (1)** | `Yes` — Customer has **churned** (left the company) |
| **Negative Class (0)** | `No` — Customer is **retained** (still active) |

### When Churn is Measured
Churn is measured **at the snapshot date**. If a customer has cancelled their service or failed to renew before the data extraction date, they are marked as `Churn = Yes`. 

### Target Distribution (Expected)
| Class | Label | Approx. % |
|-------|-------|-----------|
| No | Retained | ~73% |
| Yes | Churned | ~27% |

> [!IMPORTANT]
> This is a **moderately imbalanced** classification problem. Consider using appropriate techniques like SMOTE, class weights, or stratified sampling.

---

## 3️⃣ Feature Groups

### 🧑 Demographics
| Column | Description | Type |
|--------|-------------|------|
| `customerID` | Unique customer identifier | Identifier |
| `gender` | Customer gender (Male/Female) | Categorical |
| `SeniorCitizen` | Whether customer is 65+ years old (1=Yes, 0=No) | Binary |
| `Partner` | Whether customer has a partner (Yes/No) | Categorical |
| `Dependents` | Whether customer has dependents (Yes/No) | Categorical |

---

### 📋 Account Information
| Column | Description | Type |
|--------|-------------|------|
| `tenure` | Number of months the customer has been with the company | Numeric |
| `Contract` | Type of contract (Month-to-month, One year, Two year) | Categorical |
| `PaperlessBilling` | Whether billing is paperless (Yes/No) | Categorical |
| `PaymentMethod` | Payment method (Electronic check, Mailed check, Bank transfer, Credit card) | Categorical |

---

### 📱 Service Subscriptions
| Column | Description | Type |
|--------|-------------|------|
| `PhoneService` | Whether customer has phone service (Yes/No) | Categorical |
| `MultipleLines` | Whether customer has multiple lines (Yes/No/No phone service) | Categorical |
| `InternetService` | Type of internet service (DSL, Fiber optic, No) | Categorical |
| `OnlineSecurity` | Whether customer has online security add-on | Categorical |
| `OnlineBackup` | Whether customer has online backup add-on | Categorical |
| `DeviceProtection` | Whether customer has device protection add-on | Categorical |
| `TechSupport` | Whether customer has tech support add-on | Categorical |
| `StreamingTV` | Whether customer has streaming TV add-on | Categorical |
| `StreamingMovies` | Whether customer has streaming movies add-on | Categorical |

---

### 💰 Pricing / Billing
| Column | Description | Type |
|--------|-------------|------|
| `MonthlyCharges` | Current monthly charge amount ($) | Numeric |
| `TotalCharges` | Total amount charged over lifetime ($) | Numeric* |

> [!NOTE]
> `TotalCharges` contains some blank/whitespace values for customers with `tenure = 0`. These need to be handled during preprocessing.

---

### 🎫 Customer Support / Behavior
| Column | Description | Type |
|--------|-------------|------|
| `numAdminTickets` | Number of administrative support tickets raised | Numeric |
| `numTechTickets` | Number of technical support tickets raised | Numeric |

---

## 4️⃣ Data Assumptions

> [!IMPORTANT]
> These assumptions must be validated or acknowledged before building models.

### ✅ Confirmed Assumptions
| Assumption | Rationale |
|------------|-----------|
| **Snapshot data** | No temporal dimension; all features represent state at one point in time |
| **No future information** | Features were captured before or at the same time as churn status |
| **Customer-level granularity** | Each `customerID` appears only once |

### ⚠️ Potential Leakage Risks
| Feature | Risk Level | Explanation |
|---------|------------|-------------|
| `TotalCharges` | 🟡 Medium | Highly correlated with `tenure`; be cautious if used with `tenure` |
| `numTechTickets` | 🟢 Low | Could be predictive, but timing unclear (before/after churn decision?) |
| `numAdminTickets` | 🟢 Low | Same as above — unclear temporal relationship |

### ❓ Missing Business Context
- **No competitor pricing data** — We don't know if customers left for better offers elsewhere
- **No customer satisfaction scores** — NPS, CSAT, or survey data not available
- **No marketing touchpoints** — Campaign exposure, retention offers not captured
- **No account activity logs** — Login frequency, app usage, etc.

---

## 5️⃣ Known Limitations

### 🔴 Critical Limitations
| Limitation | Impact |
|------------|--------|
| **No reason-for-churn labels** | Cannot perform root cause analysis or understand *why* customers left |
| **No competitor information** | Cannot model competitive dynamics or price sensitivity vs. market |
| **No temporal history** | Cannot track behavioral trends over time (e.g., declining usage before churn) |

### 🟡 Moderate Limitations
| Limitation | Impact |
|------------|--------|
| **No customer tenure history** | Cannot analyze cohort effects or vintage behavior patterns |
| **Single snapshot** | Model may not generalize to different time periods (seasonality) |
| **No geographic data** | Cannot account for regional variations in service quality or competition |

### 🟢 Minor Limitations
| Limitation | Impact |
|------------|--------|
| **Binary churn only** | No distinction between voluntary vs. involuntary churn |
| **No revenue/profitability data** | Cannot prioritize customers by business value for retention efforts |

---

## 📁 File Structure

```
data/
├── raw/
│   └── churn.csv          # Original unprocessed dataset
├── processed/              # Cleaned and feature-engineered data (if created)
└── README.md               # This documentation file
```

---

## 📝 Changelog

| Date | Change |
|------|--------|
| 2026-01-07 | Initial documentation created |

---

## 🔗 References

- Original Source: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- IBM Sample Data: Watson Analytics sample dataset
