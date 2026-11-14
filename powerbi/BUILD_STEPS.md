# CoralGuard Power BI Dashboard Assembly Guide (Under 30 Minutes)

Follow these numbered steps in **Power BI Desktop** to assemble the production report.

---

### Step 1: Open Power BI & Import Theme (1 min)
1. Launch **Power BI Desktop**.
2. Go to **View** ribbon > **Themes** dropdown > **Browse for themes...**.
3. Select and apply `powerbi/coral_theme.json`.

---

### Step 2: Get Data & Load Star Schema Tables (5 mins)
1. Go to **Home** ribbon > **Transform Data** (opens Power Query Editor).
2. Create a Text Parameter:
   - Click **Manage Parameters** > **New Parameter**.
   - Name: `DataFolderPath` | Type: `Text` | Value: `c:\Users\harih\OneDrive\Documents\Desktop\project\powerbi\data`.
3. In the left query panel, right-click and choose **New Query** > **Blank Query** for each of the 8 tables:
   - Open **Advanced Editor** and paste the M script from `powerbi/queries/`:
     - `dim_site.pq` -> name query `dim_site`
     - `dim_date.pq` -> name query `dim_date`
     - `fact_risk_predictions.pq` -> name query `fact_risk_predictions`
     - `fact_sst.pq` -> name query `fact_sst`
     - `fact_acoustic.pq` -> name query `fact_acoustic`
     - `fact_diver_reports.pq` -> name query `fact_diver_reports`
     - `model_metrics.pq` -> name query `model_metrics`
     - `feature_importance.pq` -> name query `feature_importance`
4. Click **Close & Apply**.

---

### Step 3: Establish Relationships & Mark Date Table (4 mins)
1. Navigate to **Model view** (left icon).
2. Drag and drop to create 1-to-many (`1:*`), single cross-filter relationships:
   - `dim_site[site_id]` ───> `fact_risk_predictions[site_id]`
   - `dim_site[site_id]` ───> `fact_sst[site_id]`
   - `dim_site[site_id]` ───> `fact_acoustic[site_id]`
   - `dim_site[site_id]` ───> `fact_diver_reports[site_id]`
   - `dim_date[date_id]` ───> `fact_risk_predictions[date_id]`
   - `dim_date[date_id]` ───> `fact_sst[date_id]`
   - `dim_date[date_id]` ───> `fact_acoustic[date_id]`
   - `dim_date[date_id]` ───> `fact_diver_reports[date_id]`
3. Mark Date Table:
   - Right-click `dim_date` > **Mark as date table** > select column `date`.
4. Set Sort By Column:
   - Select `fact_risk_predictions[alert_level]` > **Column tools** > **Sort by column** > select `alert_sort_order`.
   - Select `dim_date[month_name]` > **Sort by column** > select `month`.

---

### Step 4: Create DAX Measures Table (5 mins)
1. Go to **Home** ribbon > click **Enter Data**.
2. Name the table `_Measures` and click **Load**.
3. Open `powerbi/measures.dax`.
4. Right-click `_Measures` > **New measure**, and paste each measure:
   - `[Avg Bleaching Risk]`, `[Sites at Alert]`, `[Sites at Warning]`, `[Max DHW]`, `[Avg SSTA]`, `[Acoustic Health Index (avg)]`, `[Diver Reports Count]`, `[Avg Reported Bleaching %]`, `[Risk Change vs Previous Period]`, `[Model PR-AUC (Severe)]`, `[Model Severe Bleaching Recall]`, `[Model Macro F1 Score]`.
5. Hide the blank column in `_Measures`.

---

### Step 5: Build Report Pages (12 mins)
Reference `powerbi/report_layout.md` for exact grid positions and formatting:
1. **Page 1: Overview**
   - Add KPI Cards: `[Sites at Alert]`, `[Sites at Warning]`, `[Max DHW]`, `[Avg Bleaching Risk]`, `[Unstressed Sites]`.
   - Add **Map**: Latitude=`dim_site[latitude]`, Longitude=`dim_site[longitude]`, Bubble size=`[Avg Bleaching Risk]`, Legend=`alert_level`.
   - Add **Line Chart**: X-axis=`dim_date[date]`, Values=`[Avg Bleaching Risk]`.
   - Add Slicers: `dim_site[region]`, `fact_risk_predictions[lead_time_days]`, `dim_date[date]`.
2. **Page 2: Satellite SST**
   - Add Line & Clustered Column: X-axis=`dim_date[date]`, Column=`fact_sst[dhw]`, Line=`fact_sst[sst]`.
   - Add Scatter Chart: X-axis=`fact_sst[dhw]`, Y-axis=`fact_risk_predictions[prob_severe]`, Details=`dim_site[site_name]`.
   - Add Matrix Heatmap: Rows=`dim_site[site_name]`, Columns=`dim_date[month_name]`, Values=`[Max DHW]`.
3. **Page 3: Acoustic Soundscape**
   - Add Line Chart: X-axis=`dim_date[date]`, Y-axis=`[Acoustic Health Index (avg)]`, Legend=`dim_site[region]`.
   - Add Scatter Chart: X-axis=`fact_acoustic[fish_band_energy]`, Y-axis=`fact_acoustic[shrimp_band_energy]`.
4. **Page 4: Diver Reports**
   - Add Map: Lat/Lon=`fact_diver_reports[lat/lon]`, Size=`fact_diver_reports[bleaching_percent]`.
   - Add Stacked Column: X-axis=`dim_date[month_name]`, Y-axis=Count of `report_id`, Legend=`severity_class`.
5. **Page 5: Model Performance**
   - Add Clustered Bar Charts: Y-axis=`ablation_set`, Values=`severe_recall` and `macro_f1`.
   - Add Horizontal Bar Chart: Y-axis=`feature_importance[feature]`, X-axis=`shap_importance`, Legend=`modality`.

---

### Step 6: Save, Refresh & Publish (2 mins)
1. Click **Save** as `CoralGuard_Early_Warning.pbix`.
2. Click **Refresh** to verify all M queries reload with zero errors.
3. Click **Publish** to deploy to Power BI Service / Fabric Workspace.

