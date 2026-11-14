# CoralGuard Power BI Data Model Specification

## 1. Architectural Pattern: Star Schema
The semantic model follows an enterprise **Star Schema** architecture optimized for VertiPaq compression, rapid DAX aggregations, and intuitive business self-service:
- **Dimensions (`1`):** `dim_site`, `dim_date`
- **Facts (`*`):** `fact_risk_predictions`, `fact_sst`, `fact_acoustic`, `fact_diver_reports`
- **Disconnected Reference Tables:** `model_metrics`, `feature_importance`
- **Filter Direction:** Strictly **Single (1 -> *)**, pointing from Dimensions down to Fact tables.
- **Date Table:** `dim_date` is configured as the official Date Table via column `dim_date[date]`.

---

## 2. Mermaid Entity-Relationship Diagram

```mermaid
erDiagram
    dim_site ||--o{ fact_risk_predictions : "site_id (1:*)"
    dim_site ||--o{ fact_sst : "site_id (1:*)"
    dim_site ||--o{ fact_acoustic : "site_id (1:*)"
    dim_site ||--o{ fact_diver_reports : "site_id (1:*)"

    dim_date ||--o{ fact_risk_predictions : "date_id (1:*)"
    dim_date ||--o{ fact_sst : "date_id (1:*)"
    dim_date ||--o{ fact_acoustic : "date_id (1:*)"
    dim_date ||--o{ fact_diver_reports : "date_id (1:*)"

    dim_site {
        string site_id PK
        string site_name
        string region
        string country
        float latitude
        float longitude
        string reef_type
    }

    dim_date {
        int date_id PK
        date date
        int year
        int month
        string month_name
        int week
        string season
    }

    fact_risk_predictions {
        string site_id FK
        int date_id FK
        string model_name
        int lead_time_days
        int pred_class
        float prob_none
        float prob_mild
        float prob_severe
        string alert_level
        int alert_sort_order
        int actual_class
    }

    fact_sst {
        string site_id FK
        int date_id FK
        float sst
        float ssta
        float dhw
        float days_above_threshold
    }

    fact_acoustic {
        string site_id FK
        int date_id FK
        float aci
        float adi
        float shrimp_band_energy
        float fish_band_energy
        float acoustic_health_index
    }

    fact_diver_reports {
        string report_id PK
        string site_id FK
        int date_id FK
        float lat
        float lon
        float depth
        float bleaching_percent
        string severity_class
        float observer_confidence
        string source
    }
```

---

## 3. Relationships Definition Matrix

| From Dimension (1) | Primary Key | To Fact Table (*) | Foreign Key | Cardinality | Cross-filter | State |
|---|---|---|---|---|---|---|
| `dim_site` | `site_id` | `fact_risk_predictions` | `site_id` | 1 to Many (`1:*`) | Single (`dim_site` -> Fact) | Active |
| `dim_site` | `site_id` | `fact_sst` | `site_id` | 1 to Many (`1:*`) | Single (`dim_site` -> Fact) | Active |
| `dim_site` | `site_id` | `fact_acoustic` | `site_id` | 1 to Many (`1:*`) | Single (`dim_site` -> Fact) | Active |
| `dim_site` | `site_id` | `fact_diver_reports` | `site_id` | 1 to Many (`1:*`) | Single (`dim_site` -> Fact) | Active |
| `dim_date` | `date_id` | `fact_risk_predictions` | `date_id` | 1 to Many (`1:*`) | Single (`dim_date` -> Fact) | Active |
| `dim_date` | `date_id` | `fact_sst` | `date_id` | 1 to Many (`1:*`) | Single (`dim_date` -> Fact) | Active |
| `dim_date` | `date_id` | `fact_acoustic` | `date_id` | 1 to Many (`1:*`) | Single (`dim_date` -> Fact) | Active |
| `dim_date` | `date_id` | `fact_diver_reports` | `date_id` | 1 to Many (`1:*`) | Single (`dim_date` -> Fact) | Active |

---

## 4. Modeling Best Practices Implemented
1. **Integer Date Keys:** All fact-to-calendar relationships leverage 8-digit integer keys (`YYYYMMDD`), minimizing memory footprint and maximizing VertiPaq query throughput.
2. **Sort by Column:** 
   - `fact_risk_predictions[alert_level]` sorted by `fact_risk_predictions[alert_sort_order]`.
   - `dim_date[month_name]` sorted by `dim_date[month]`.
3. **Data Categories:**
   - `dim_site[latitude]` categorized as **Latitude**.
   - `dim_site[longitude]` categorized as **Longitude**.
   - `dim_site[country]` categorized as **Country/Region**.
4. **Mark as Date Table:** `dim_date` marked as Date Table using column `date`.

