# SQL/gspread Statistic Pipeline

1. First run analysis and populate data into the *local* *.db
    - See [db_update.sh](../../../../bash/vop/db_update.sh)
2. Push tables to the Google spreadsheet (Optional, depending on the `load_source`)
    - See [main_push_gspread.py](main_push_gspread.py)
3. Run the statistic (use either `gspread`, `db` as `load_source`)