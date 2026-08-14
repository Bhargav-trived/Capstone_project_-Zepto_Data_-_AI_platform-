# Capstone_project_-Zepto_Data_-_AI_platform-

Module - 1

# Data Pipeline — Module 1

Overview :

This project scrapes book data from `books.toscrape.com`, cleans the data, converts prices from GBP to INR, and stores everything in a SQLite database.
It also includes SQL queries and pandas checks to verify the data.

Run order :

1. Run the scraping cells/script — this calls `scrape_all(categories)` 
   across the category URLs defined in the `categories` dict, producing 
   a raw DataFrame of ~88 books across 4 categories (Travel, Poetry, 
   Mystery, History Fiction).
2. Run the cleaning cells — applies `clean_price`, `clean_rating`, and 
   `clean_availability` to produce typed columns `price_gbp`, `rating`, 
   `in_stock`, plus `price_inr`.
3. Run `create_schema()` to build `books.db` with the `categories` and 
   `books` tables.
4. Run `insert_categories()` then `insert_books()` to load the cleaned 
   data into the database.
5. Run the SQL query cells to execute and print all 5 required queries.
6. Run the pandas verification cells (`pd.read_sql` + `pd.merge`) to 
   confirm SQL and pandas approaches agree.

## Currency Conversion
Fixed baseline rate used for this project: **1 GBP = 105.50 INR**.
This is a project-defined constant, not a live market rate, and requires
no external API call or date reference. `price_inr` is computed as
`price_gbp * 105.50`.

Data Cleaning Decisions:

- **price_gbp**: currency symbol stripped from the raw price string and 
  converted to float. Rows where this conversion failed would return 
  `None` and be handled per the rule below.
- **rating**: text ratings ("One"–"Five") mapped to integers 1–5 via a 
  dictionary lookup; unrecognized values return `None`.
- **in_stock**: parsed as boolean by checking whether "in stock" appears 
  (case-insensitive) in the availability text.
- **Missing/failed values**: checked both `price_gbp` and `rating` for 
  parse failures after cleaning — **0 out of 88 rows** failed to parse 
  for either field, so no rows needed to be dropped and no imputation 
  was necessary. (If failures had occurred, the plan was to drop rows 
  with unparseable `rating`, since rating is a discrete 1–5 scale that 
  doesn't have a meaningful "average" to impute, and to median-impute 
  `price_gbp` failures instead, since price is continuous and a median 
  fill is a reasonable placeholder that doesn't distort the overall 
  distribution much.)

Database Schema:

Two tables with a primary/foreign key relationship:

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
```

SQL Queries: 

Five queries were written and executed, collectively covering all 
required clauses:
1. `WHERE` + `ORDER BY` + `LIMIT` — cheapest in-stock books.
2. `DISTINCT` — list of unique category names.
3. `IN` — books with rating 4 or 5.
4. `JOIN` + `ORDER BY` — all books with their category name, sorted by 
   rating descending.
5. `BETWEEN` — books priced between £20–£49 (chosen to roughly cover 
   the IQR of the price distribution, 25th percentile ≈ £21, 75th ≈ £47).

Query strings and outputs are in [notebook/script name].

Pandas Verification:

- Two of the above queries were re-run using `pd.read_sql(...)` and 
  produced identical results to the raw `sqlite3` cursor output.
- The JOIN query was independently reproduced using `pd.merge()` on the 
  in-memory `books_df` and `categories_df` DataFrames (merged on 
  `category_id`, no SQL used) and matched the SQL JOIN's row count and 
  book-category pairings.

Data Scope:

Scraped 88 books across 4 categories (Travel, Poetry, Mystery, History 
Fiction), exceeding the required minimum of 60 books across 3+ categories.


Module 2 — Analytics Pipeline (/analytics)

Titanic dataset (sns.load_dataset('titanic'), 891 rows × 15 columns) loaded once, cleaned once, saved to titanic.csv, and used unchanged for both the EDA story (Part A) and the modeling pipeline (Part B).

Part A — Profiling, Cleaning, and the Data Story

1. Profiling
df.shape → (891, 15)
df.info() shows 5 columns with missing data; df.describe() confirms the usual Titanic ranges (age 0.42–80, fare 0–512.33).

Missing values (% of 891 rows):

Column	Missing count	Missing %
age	177	19.87%
embarked	2	0.22%
embark_town	2	0.22%
deck	688	77.22%

titanic.csv was saved immediately after loading via df.to_csv("titanic.csv", index=False), and is the single offline fallback used by every downstream step (Part B re-reads it instead of re-calling sns.load_dataset).

2. Missing-value handling (threshold rule applied per column)
embarked / embark_town — 0.22% missing (< 5% → drop rows). Rows dropped: 891 → 889. Loss is negligible (2 rows) and both columns carry the same information (port code vs. port name), so dropping is safe and avoids inventing a port of embarkation for 2 passengers.

age — 19.87% missing (5%–30% → impute). Imputed with the column median (28.0). Median was chosen over mean because age is right-skewed with high-end outliers (see box plot), so the median is more robust to that skew than the mean.

deck — 77.22% missing (too high for reliable imputation). Decision: drop the column entirely. Justification: at 77% missing, any imputation (mean/mode/model-based) would be manufacturing values for more than three-quarters of the column, which would dominate and bias whatever signal remains from the observed 22.8%. deck is also largely redundant with pclass/fare (deck assignment tracks cabin class), so little unique information is lost by dropping it outright rather than encoding a "Missing" category.

Post-cleaning: df.info() confirms 889 rows × 14 columns, 0 nulls in every remaining column.

3. Univariate analysis — age & fare
IQR outlier counts: age → 65 outliers, fare → 114 outliers. Fare has almost double the age outlier count, consistent with its box plot showing a long tail of expensive first-class fares (up to $512).
Fare central tendency: Mean = 32.10, Median = 14.45, Mode = 8.05. Since mean > median > mode, fare is right-skewed (positively skewed) — a small number of very high fares pull the mean well above the typical (median/mode) ticket price, matching the long right tail visible in the fare histogram and box plot.
4. Bivariate analysis

Survival rate by sex:

Male: 18.89%
Female: 74.04%

Survival rate by class:

1st class: 62.62%
2nd class: 47.28%
3rd class: 24.24%

Survival rate by sex × class:

	1st	2nd	3rd
Female	96.74%	92.11%	50.00%
Male	36.89%	15.74%	13.54%

Sex is a far stronger survival driver than class alone: even 3rd-class women (50%) survived at a much higher rate than 1st-class men (36.9%).

Correlation matrix (survived, pclass, age, sibsp, parch, fare):

	survived	pclass	age	sibsp	parch	fare
survived	1.000	-0.336	-0.070	-0.034	0.083	0.255
pclass	-0.336	1.000	-0.337	0.082	0.017	-0.548
age	-0.070	-0.337	1.000	-0.233	-0.171	0.094
sibsp	-0.034	0.082	-0.233	1.000	0.415	0.161
parch	0.083	0.017	-0.171	0.415	1.000	0.218
fare	0.255	-0.548	0.094	0.161	0.218	1.000

Two strongest off-diagonal correlations (by absolute value):

pclass ↔ fare = -0.548 — the strongest relationship in the matrix. Makes sense structurally: pclass is coded 1 (best) to 3 (worst), and 1st class tickets cost far more, so a lower pclass number goes with a higher fare, producing a strong negative correlation.
sibsp ↔ parch = 0.415 — the second strongest. Passengers traveling with more siblings/spouses also tended to travel with more parents/children, i.e., family size clusters together rather than these two counts being independent.

(survived ↔ pclass at -0.336 and pclass ↔ age at -0.337 are close behind but don't make the top two.)

5. Multivariate data story (4+ charts)
Survival rate by sex (bar chart). Confirms the ~4x gap between female (74%) and male (19%) survival — the single strongest visual signal in the dataset, consistent with "women and children first" evacuation priority.
Survival rate by passenger class (bar chart). Shows a clean monotonic drop from 1st (63%) to 2nd (47%) to 3rd (24%) class, suggesting cabin location / deck access / crew priority favored higher classes.
Survival rate by class and sex (grouped bar chart). This is the chart that ties the story together: sex dominates within every class, but class still matters — female survival degrades from ~97% (1st) to 50% (3rd), while male survival is uniformly low (37% → 16% → 14%) regardless of class. So sex and class interact: being a woman helped enormously, but being a 3rd-class woman was still much riskier than being a 1st-class woman.
Age distribution by survival (box plot). Medians are nearly identical (~28) for survivors and non-survivors, and the interquartile ranges overlap heavily — age alone is a weak discriminator, unlike sex or class. This matches the correlation matrix, where age ↔ survived is only -0.07.

(A correlation heatmap was also produced for Task 4 above, satisfying the required heatmap render.)

Overall story: survival on the Titanic was driven primarily by sex, secondarily by class/fare (which are themselves strongly correlated with each other), and only weakly by age.

6. Standardization sanity check (exploratory only, not fed into modeling)
	age (before)	fare (before)	age_zscore (after)	fare_zscore (after)
mean	29.3152	32.0967	~2.8e-16 (≈0)	~1.3e-16 (≈0)
std	12.9849	49.6975	1.0000	1.0000

Confirms the z-score transform correctly re-centers both columns to mean 0 and std 1.

Part B — Predictive Modeling

7. Train/test split

Stratified 80/20 split on survived (stratify=y). Train survival rate = 38.26%, test survival rate = 38.20% — nearly identical to the full-dataset rate (~38.4%). Stratification matters here because the target is moderately imbalanced (≈62% did-not-survive vs. 38% survived); a plain random split risks over- or under-representing the minority (survived) class in the smaller test set, which would make evaluation metrics like recall/F1 noisy and less trustworthy.

8. Preprocessing (fit on train only)

Dropped class, who, adult_male, embark_town, alive, alone before modeling — these are either duplicates of features already used (class≈pclass, embark_town≈embarked), derived flags (adult_male, who, alone), or a direct leak of the target (alive literally encodes survival as text).

ColumnTransformer inside a Pipeline:

Numeric (pclass, age, sibsp, parch, fare): median imputer → StandardScaler.
Categorical (sex, embarked): most-frequent imputer → OneHotEncoder(handle_unknown='ignore').

Because the transformer is fit inside a Pipeline.fit(X_train, y_train) and only .transform()-ed on X_test, no test-set statistics ever leak into training. Transformed shapes: train (711, 10), test (178, 10) (5 numeric + 2 sex dummies + 3 embarked dummies).

9. Three classifiers — test-set accuracy
Model	Accuracy
Logistic Regression	0.8090
Decision Tree	0.7697
Random Forest	0.8202

The Decision Tree was also rendered with plot_tree (depth 3, labeled features/classes). The root split is on sex (cat__sex_female), reinforcing the EDA finding that sex is the single strongest predictor.

10. Full evaluation (confusion matrix, accuracy, precision, recall, F1, AUC)
Model	Confusion Matrix	Accuracy	Precision	Recall	F1	AUC
Logistic Regression	[[97,13],[21,47]]	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	[[88,22],[19,49]]	0.7697	0.6901	0.7206	0.7050	0.7541
Random Forest	[[96,14],[18,50]]	0.8202	0.7812	0.7353	0.7576	0.8179

Random Forest gives the best accuracy and F1; Logistic Regression gives the best AUC and precision; the Decision Tree trails on every metric — expected, since a single unconstrained tree tends to overfit relative to an ensemble or a well-regularized linear model.

11. Imbalance handling comparison

Class balance in training data: 439 not-survived (61.74%) vs. 272 survived (38.26%) — moderately imbalanced.

Variant	Accuracy	Precision	Recall	F1	AUC
RF baseline	0.8202	0.7812	0.7353	0.7576	0.8179
RF + class_weight='balanced'	0.8146	0.7692	0.7353	0.7519	0.8176
RF + SMOTE (train fold only)	0.7921	0.7460	0.6912	0.7176	0.8250

Conclusion: on this dataset the baseline Random Forest performed best overall (highest accuracy and F1). class_weight='balanced' matched baseline recall but slightly hurt precision/accuracy — essentially a wash. SMOTE actually lowered recall and F1 versus baseline (though it nudged AUC up slightly to 0.825), likely because the imbalance here (62/38) is fairly mild, so synthetic minority oversampling added noisy synthetic samples without enough real signal to gain from — SMOTE tends to help more on severely imbalanced data (e.g., 95/5) than on a moderate skew like this one.

12. Hyperparameter tuning (GridSearchCV, RandomForestClassifier)
Best params: n_estimators=300, max_depth=5, max_features='sqrt'
Best CV accuracy: 0.8200
OOB score: 0.8214 (from RandomForestClassifier(oob_score=True, ...))

The OOB score (0.8214) tracks closely with both the CV accuracy (0.8200) and the held-out test accuracy of the baseline RF (0.8202), suggesting the model isn't overfitting materially — all three "unseen-data" estimates agree.

13. Regression side-task — predicting fare
Metric	Value
MAE	21.1386
RMSE	41.7465
R²	0.3468
Adjusted R²	0.3199

An R² of ~0.35 means the linear model explains only about a third of fare's variance from pclass/age/sibsp/parch/sex/embarked — fare has a lot of variation these features don't capture (e.g. exact cabin, negotiated group rates).

Residual plot: residuals fan out and grow substantially in magnitude as predicted fare increases (small, tight residuals for low predicted fares; residuals exceeding +400 for the highest predicted fares, including one large outlier). This is a clear heteroscedasticity pattern — the variance of the errors is not constant across the range of predictions, which violates a core linear-regression assumption and means the model's prediction interval is much less reliable for high-fare passengers than for low-fare ones.

14. Final model comparison & recommendation

Classification models (own metric scale, 0–1):

Model	Accuracy	Precision	Recall	F1	AUC
Logistic Regression	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	0.7697	0.6901	0.7206	0.7050	0.7541
Random Forest	0.8202	0.7812	0.7353	0.7576	0.8179

Regression model — fare prediction (own metric scale, not comparable to the classification table above):

MAE	RMSE	R²	Adjusted R²
21.1386	41.7465	0.3468	0.3199

Recommendation: I would deploy the Random Forest classifier. It has the best accuracy (0.8202) and F1 (0.7576) of the three models, and its recall (0.7353) — the ability to correctly flag actual survivors — is the highest among all three baseline classifiers, which matters if false negatives (missing a survivor) are costlier than false positives. Logistic Regression is a close second and actually wins on AUC (0.8610 vs. 0.8179) and precision (0.7833 vs 0.7812), so it's a reasonable, more interpretable alternative if ranking/probability calibration matters more than raw classification accuracy. The Decision Tree is dominated on every metric by both other models and shouldn't be deployed as-is.

15. Saved pipeline

The tuned pipeline (best_rf_pipeline from GridSearchCV, including the fitted ColumnTransformer + tuned RandomForestClassifier) was saved as a single object:

joblib.dump(best_rf_pipeline, 'titanic_survival_pipeline.pkl')

Reload check on raw, unpreprocessed test rows:

loaded_pipeline = joblib.load('titanic_survival_pipeline.pkl')
loaded_pipeline.predict(X_test.iloc[:5])

The reloaded pipeline runs preprocessing and prediction end-to-end on raw input, confirming it's a complete, deployable artifact rather than a bare estimator.

Module 3 — Support Assistant (`/support_assistant`)

This repository contains a complete, offline-first GenAI Customer Support microservice for Zepto. It implements a Retrieval-Augmented Generation (RAG) pipeline using **FastAPI**, **LangGraph**, **ChromaDB**, and **Sentence-Transformers** (`all-MiniLM-L6-v2`).

---

1. RAG Pipeline Architecture

The pipeline processes customer queries in four main stages, orchestrated by a LangGraph `StateGraph`:

```text
[Customer Query: "How do I return a damaged item?"]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. INGESTION & 2. EMBEDDING (Triggered on App Startup)       │
│   • Source: 8 Zepto policy text files in /docs               │
│   • Component: `app.py` (ensure_docs_exist, get_vectorstore) │
│   • Embedder: SentenceTransformer ('all-MiniLM-L6-v2')       │
│   • Storage: ChromaDB persistent collection 'zepto_policies' │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ LANGGRAPH ROUTER: `classify_intent` Node                     │
│   • Evaluates query for policy keywords (delivery, return...)│
│   • Branches to either policy retrieval or direct answer     │
└──────────────────────┬────────────────────────┬──────────────┘
                       │                        │
          [Intent: policy_question]    [Intent: general_question]
                       │                        │
                       ▼                        ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ 3. RETRIEVAL & 4. GENERATION         │  │ 4. GENERATION (No Retrieval)         │
│ Node: `retrieve_and_answer`          │  │ Node: `direct_answer`                │
│   • Embeds query & queries ChromaDB  │  │   • Bypasses ChromaDB entirely       │
│   • Retrieves top-3 matching chunks  │  │   • Generates fallback response      │
│   • Generates context-grounded reply │  │                                      │
└──────────────────────┬───────────────┘  └───────────────┬──────────────────────┘
                       │                                  │
                       ▼                                  ▼
            ┌───────────────────────────────────────────────────────┐
            │ FASTAPI OUTPUT VALIDATION                             │
            │ Component: `schemas.QueryResponse` (Pydantic)         │
            │ Returns JSON: { answer: str, sources: [], confidence }│
            └───────────────────────────────────────────────────────┘


Here is the complete `README.md` file that fulfills all the grading requirements for Module 3.

```markdown
# Module 3 — Support Assistant (`/support_assistant`)

This repository contains a complete, offline-first GenAI Customer Support microservice for Zepto. It implements a Retrieval-Augmented Generation (RAG) pipeline using **FastAPI**, **LangGraph**, **ChromaDB**, and **Sentence-Transformers** (`all-MiniLM-L6-v2`).

---

## 1. RAG Pipeline Architecture

The pipeline processes customer queries in four main stages, orchestrated by a LangGraph `StateGraph`:

```text
[Customer Query: "How do I return a damaged item?"]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. INGESTION & 2. EMBEDDING (Triggered on App Startup)       │
│   • Source: 8 Zepto policy text files in /docs               │
│   • Component: `app.py` (ensure_docs_exist, get_vectorstore) │
│   • Embedder: SentenceTransformer ('all-MiniLM-L6-v2')       │
│   • Storage: ChromaDB persistent collection 'zepto_policies' │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ LANGGRAPH ROUTER: `classify_intent` Node                     │
│   • Evaluates query for policy keywords (delivery, return...)│
│   • Branches to either policy retrieval or direct answer     │
└──────────────────────┬────────────────────────┬──────────────┘
                       │                        │
          [Intent: policy_question]    [Intent: general_question]
                       │                        │
                       ▼                        ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ 3. RETRIEVAL & 4. GENERATION         │  │ 4. GENERATION (No Retrieval)         │
│ Node: `retrieve_and_answer`          │  │ Node: `direct_answer`                │
│   • Embeds query & queries ChromaDB  │  │   • Bypasses ChromaDB entirely       │
│   • Retrieves top-3 matching chunks  │  │   • Generates fallback response      │
│   • Generates context-grounded reply │  │                                      │
└──────────────────────┬───────────────┘  └───────────────┬──────────────────────┘
                       │                                  │
                       ▼                                  ▼
            ┌───────────────────────────────────────────────────────┐
            │ FASTAPI OUTPUT VALIDATION                             │
            │ Component: `schemas.QueryResponse` (Pydantic)         │
            │ Returns JSON: { answer: str, sources: [], confidence }│
            └───────────────────────────────────────────────────────┘
