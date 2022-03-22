# Prosjektoppgave
Test 
As the datafiles are to big for github repo, you have to download it yourself to use the code :D

The CYGNSS data can easily be fetched using code here, but the rest of the data are fetch manually from:

Wind Data : https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

Current Data : https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg


How to reproduce results:

1)
Fetch some CYGNSS data using first blocks of this messy code: fetch_cygnss_data_and_old_unsed_code.ipynb

2) 
Get ERA5 and Oscar data manually.

3) 
Calculate MSS values using this script : calculate_mss.py (which uses functions from functions.py)

4)
Play around with some of the codes here to inspect your results : analysis.ipynb



All code are developed by Syver during winter semester 2021 ( Except some codelines from Andreas/Mads to fetch CYGNSS data)
