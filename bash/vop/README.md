# Bash pipeline for visual open-loop(vop) calcium dataset

- Order specific due to the dependencies between analysis
- Modify `OUTPUT` related variables based on the running machine, to redirect all script output (stdout and stderr) to
  the log file

## Calcium physiological data

- Follow the orders

    1. [visual.sh](visual.sh) - Preselection and visual-related analysis
    2. [spatial.sh](spatial.sh) - Spatial-related analysis
    3. [bayes_sb15.sh](bayes_sb15.sh) - Bayes position decoding analysis
    4. [generic.sh](generic.sh) - Generic analysis (i.e., signal, FOV-related)
    5. [stat.sh](stat.sh) - TODO

- For different optic planes (ETL) data concatenation (after running above for each plane)

    1. [concat.sh](concat.sh) - Concat csv files across optic planes to one single csv
    2. [volumetric.sh](volumetric.sh) - Run analysis for all the optic planes (based on persistence cache/concat csv)

- For only update the database for statistic, run [db_update.sh](db_update.sh)
