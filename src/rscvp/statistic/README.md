Stat (dev)
===

## Module for statistic purpose

* For individual dataset i.e, `main_corr_matrix.py`, `main_normality.py`, `main_ses_stat.py`
* For population dataset, based on concat csv from multiple dataset

## Data Structure (for population dataset)

```
 Analysis SRC/
 │
 ├── ED_ID*/
 │
 └── summary/
        ├── ..._stat/ -- (1)
        └── {page_name}_summary.pickle -- (2)
             
```

# Steps

## 1. generate token, and service account json

- Generate token via [GoogleAPI](https://docs.gspread.org/en/latest/oauth2.html#enable-api-access),
- Put `service_account.json` into `~/.rscvp/.config/gspread`
- Share worksheet with `gservice.account.com` in browser
- check the worksheet name in `src.util.query.GoogleWorkSheet.__gspread_name`

## 2. Aggregate (agg)

- Note that each columns in spreadsheet used either str/numeric type. other auto-casting problem
- cell mask should be selected in this process

## 3. based on parquet to plotting & statistic (alternatively, directly load from gspread , GSP)

## Google Spreadsheet and local parquet files (GSP)

(1) Google Spreadsheet should be first *set index (data_info)* and *column (variable/header)*

(3) `src.stat.csvfinder.CSVCollector` to concat csv based on analysis *code* type

(4) `src.stat.csvfinder.CSVStat` help for synchronize google spreadsheet and pickled file while running statistic

** Each *code* for analysis should inherit `CSVStat`
