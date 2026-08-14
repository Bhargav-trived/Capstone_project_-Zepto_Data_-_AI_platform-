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
