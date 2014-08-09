""""
  Respondent data structure:
  
  respondents = { for each id in dataset: {"Age": ##, "Gender": #,
                       "Income": #, "Education": #, "Arrested": ##,
                       "Jailed": ##, "Tobacco": #,  "Alcohol": ##,
                       "IllegalDrugs": #, "Depression": ##, 
                       "AlcoholOrDrugTreatment": ##, 
                       "MentalHealthTreatment": ##}}
"""

import time
start_time = time.perf_counter()

#open CSV containing data
import csv
with open('data-subset.csv') as f:
  reader = csv.reader(f)
  #load data into dictionary
  respondents = {rows[0]: {"Age": rows[1], "Gender": rows[2],
                       "Income": rows[3], "Education": rows[4],
                       "Arrested": rows[5], "Jailed": rows[6],
                       "Tobacco": rows[7],  "Alcohol": rows[8],
                       "IllegalDrugs": rows[9], "Depression": rows[10], 
                       "AlcoholOrDrugTreatment": rows[11], 
                       "MentalHealthTreatment": rows[12]} for rows in reader}

print(time.perf_counter() - start_time)