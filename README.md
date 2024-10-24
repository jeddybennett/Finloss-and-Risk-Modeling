# Finloss-and-Risk-Modeling
Code to calculate the probability of a breach, or the risk a company has, and then calculate the expected financial output of that model

To generate the needed historical data that is currently being used in the finloss_model.py first:

## Steps to Generate Data
1) Clone into the repository
2) Navigate into the Data_Organization folder
3) run python data_extractor.py

That will now create a plethora of csv files

4) Navigate out of Data_Organization into the main folder of the repository
5) mkdir Data
6) Navigate back into Data_Organization
7) mv *.csv ../Data

Your csv files should now all be located in the Data Folder

## Note
- finloss_model.py expects a csv file called ***questionnaire_data.csv***
- This data comes from the online tool that will be deployed, and the p_value.py and finloss_model.py will be deployed in the backend
- Expects:
  - Company Name
  - Number of Employees
  - Annual Revenue, and past 2 years revenue
  - Financial Industry
  - Regulations that the could incur fines on the company, E.g: FTC, SEC
  - E,T and M values to calculate vulnerability

## Current Package Requirements
- The model currently leverages pymc3 to get bayesian updates for μ and σ values of a lognormal distribution
- This can cause issues with different versions of python, as the most recent update of pymc3 doesn't work well with python 3.12
- version 1 of the model was developed with:
  - Python 3.10.0
  - Numpy 1.25.2
  - Pandas 2.0.2
  - Pymc3 3.11.5
  - Scipy 1.13.1
  - Matplotlib 3.8.0

***EXTRA INFORMATION***
  - These are currently subject to change as we move from ver1 of the finloss model
  - Working on coming up with formula for M value
  - Integrate weights based on financial industry, as well as integrating insurance deductibles and security measures a company has in place
