import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import scipy.stats as stats
import pymc3 as pm
import warnings
import logging
import json
# Suppress all warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
from p_value import get_p_value

################################### Read Json File Outputs ##################################

def convert_shorthand(value):
    value = value.strip()
    if value.endswith('K'):
        return float(value[:-1]) * 1_000
    elif value.endswith('M'):
        return float(value[:-1]) * 1_000_000
    else:
        return float(value)
    
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.float)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

def parse_number(number_str):
    number_str = number_str.replace('$', '').replace(',', '').strip()
    if 'million' in number_str.lower():
        number = float(number_str.lower().replace('million', '').strip()) * 1_000_000
    elif 'billion' in number_str.lower():
        number = float(number_str.lower().replace('billion', '').strip()) * 1_000_000_000
    else:
        number = float(number_str)
    return number

def parse_revenue(revenue_str):
    revenue_str = revenue_str.replace('$', '').replace(',', '').strip()
    if 'Less than' in revenue_str:
        return 500_000  # Approximation for "Less than $1 million"
    elif '-' in revenue_str:
        low_str, high_str = revenue_str.split('-')
        low = parse_number(low_str.strip())
        high = parse_number(high_str.strip())
        average_revenue = (low + high) / 2  # Or apply other adjustments as needed
        return average_revenue
    else:
        return parse_number(revenue_str)

def parse_employee_count(employee_str):
    employee_str = employee_str.replace(',', '').strip()
    if '-' in employee_str:
        low_str, high_str = employee_str.split('-')
        low = int(low_str.strip())
        high = int(high_str.strip())
        average_employees = (low + high) / 2
        return average_employees
    elif '+' in employee_str:
        return 5500
    else:
        return int(employee_str)

################################### Numerical Calculations for P-values ##################################

def derive_E(incident_count):
    try:
        count = int(incident_count)
        return max(1, min(10, count))
    except:
        return 10.0 # Default value

def derive_T(data_exposed):
    if isinstance(data_exposed, list):
        return len(data_exposed) * 50  # Example scaling
    else:
        return 500  # Default value

def derive_M(budget):
    try:
        budget_value = parse_number(budget)
        if budget_value < 100_000:
            return 4
        elif budget_value < 500_000:
            return 7
        else:
            return 9
    except:
        return 7
    

################################### Load Historical and JSON data ##################################

def load_questionnaire_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_historical_data():
    # Load affected count ranges
    affected_counts_df = pd.read_csv('../Data/affected_count_range.csv')

    # Load subcategory data (response costs, litigated cases, fines & penalties)
    response_costs_df = pd.read_csv('../Data/response_costs.csv')
    litigated_cases_df = pd.read_csv('../Data/litigated_cases.csv')
    fines_penalties_df = pd.read_csv('../Data/fines_penalties.csv')
    economic_loss_df = pd.read_csv('../Data/economic_loss.csv')

    return affected_counts_df, response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df

def prepare_affected_counts(affected_counts_df):
    # Remove 'Total' row if present
    affected_counts_df = affected_counts_df[affected_counts_df['Affected Count Range'] != 'Total']

    # Calculate probabilities
    total_cases = affected_counts_df['Case Count'].sum()
    affected_counts_df['Probability'] = affected_counts_df['Case Count'] / total_cases

    # Prepare ranges and probabilities
    ranges = affected_counts_df['Affected Count Range'].values
    probabilities = affected_counts_df['Probability'].values

    # Parse ranges into numerical values
    range_tuples = []
    for r in ranges:
        if r.strip() == '(1M, )':
            low = 1_000_000
            high = 10_000_000  # Assume an upper bound
        else:
            r = r.strip('()[]')
            low_str, high_str = r.split(',')
            low = convert_shorthand(low_str.strip()) 
            high = convert_shorthand(high_str.strip()) 
        range_tuples.append((low, high))

    return range_tuples, probabilities


################################### Numerical Calculations ##################################

def calculate_lambda(p_value):
    p_value = np.clip(p_value, 0, 1 - 1e-10)
    lambda_value = -np.log(1 - p_value)
    return lambda_value

def simulate_num_records_lost():
    # Define the ranges and probabilities based on the data
    ranges = [
        (0, 10),
        (10, 100),
        (100, 1000),
        (1000, 10000),
        (10000, 100000),
        (100000, 1000000),
        (1000000, 1500000)
    ]
    case_counts = [58269, 14872, 20437, 14143, 6651, 2660, 1726]
    total_cases = sum(case_counts)
    probabilities = [count / total_cases for count in case_counts]

    # Choose a range based on probabilities
    selected_range = np.random.choice(len(ranges), p=probabilities)
    low, high = ranges[selected_range]
    
    # Simulate a number within the selected range
    num_records = np.random.uniform(low, high)
    return num_records

def calculate_expected_loss():
    affected_counts_df, response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df = load_historical_data()

    # Exclude 'Total' rows
    economic_loss_df = economic_loss_df[economic_loss_df['Case Type'] != 'Total']
    response_costs_df = response_costs_df[response_costs_df['Case Type'] != 'Total']
    litigated_cases_df = litigated_cases_df[litigated_cases_df['Case Type'] != 'Total']
    fines_penalties_df = fines_penalties_df[fines_penalties_df['Case Type'] != 'Total']

    # Initialize a list to hold all loss amounts
    all_loss_amounts = []

    # List of datasets and corresponding columns
    datasets = [
        (economic_loss_df, 'Loss Amount (Economic Loss)', 'Case Count (Economic Loss)'),
        (response_costs_df, 'Loss Amount (Response Costs)', 'Case Count (Response Costs)'),
        (litigated_cases_df, 'Loss Amount (Litigated Cases)', 'Case Count (Litigated Cases)'),
        (fines_penalties_df, 'Loss Amount (Fines & Penalties)', 'Case Count (Fines & Penalties)')
    ]

    # Iterate through datasets to collect loss amounts
    for df, loss_col, count_col in datasets:
        # Ensure loss amounts and case counts are valid numbers
        df = df.dropna(subset=[loss_col, count_col])
        df[loss_col] = df[loss_col].astype(float)
        df[count_col] = df[count_col].astype(int)

        # Calculate average loss per case type
        avg_losses = df[loss_col] / df[count_col]

        # Append individual losses based on the number of cases
        for avg_loss, count in zip(avg_losses, df[count_col]):
            # Repeat the average loss 'count' times
            losses = [avg_loss] * count
            all_loss_amounts.extend(losses)

    # Convert to numpy array
    all_loss_amounts = np.array(all_loss_amounts)

    # Remove non-positive values
    all_loss_amounts = all_loss_amounts[all_loss_amounts > 0]

    return all_loss_amounts

def estimate_mu_sigma_mle(all_loss_amounts):
    # Fit the lognormal distribution using MLE
    shape, loc, scale = lognorm.fit(all_loss_amounts, floc=0)
    sigma_mle = shape  # 'shape' parameter is sigma in lognorm
    mu_mle = np.log(scale)
    return mu_mle, sigma_mle

def calculate_lognormal_params(num_samples):
    # Collect all loss amounts
    all_loss_amounts = calculate_expected_loss()
    
    # Perform Bayesian estimation
    # trace = bayesian_estimate_mu_sigma(all_loss_amounts, num_samples)
    
    # # Extract posterior samples for mu and sigma
    # mu_samples = trace.posterior['mu'].values.flatten()
    # sigma_samples = trace.posterior['sigma'].values.flatten()
    mu_mle, sigma_mle = estimate_mu_sigma_mle(all_loss_amounts)
    mu_samples = np.full(num_samples, mu_mle)
    sigma_samples = np.full(num_samples, sigma_mle)
    
    # Return the posterior samples
    return mu_samples, sigma_samples

def bayesian_estimate_mu_sigma(all_loss_amounts, num_samples):
    # Take the logarithm of loss amounts
    log_loss_amounts = np.log(all_loss_amounts)
    
    # Define the model
    with pm.Model() as model:
        # Priors for mu and sigma
        mu_prior = pm.Normal('mu', mu=np.mean(log_loss_amounts), sigma=1)
        sigma_prior = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood
        observed_data = pm.Normal('observed_data', mu=mu_prior, sigma=sigma_prior, observed=log_loss_amounts)
        
        # Sample from the posterior
        trace = pm.sample(draws=num_samples, tune=1000, chains=2, cores=1, return_inferencedata=True)
    
    return trace


def calculate_subcategory_proportions(response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df):
    # Sum loss amounts for each subcategory
    response_cost_total = response_costs_df.loc[response_costs_df['Case Type'] == 'Total', 'Loss Amount (Response Costs)'].values[0]
    litigated_cases_total = litigated_cases_df.loc[litigated_cases_df['Case Type'] == 'Total', 'Loss Amount (Litigated Cases)'].values[0]
    fines_penalties_total = fines_penalties_df.loc[fines_penalties_df['Case Type'] == 'Total', 'Loss Amount (Fines & Penalties)'].values[0]
    economic_loss_total = economic_loss_df.loc[economic_loss_df['Case Type'] == 'Total', 'Loss Amount (Economic Loss)'].values[0]

    # Convert to floats
    response_cost_total = float(response_cost_total)
    litigated_cases_total = float(litigated_cases_total)
    fines_penalties_total = float(fines_penalties_total)
    economic_loss_total = float(economic_loss_total)

    # Calculate total loss
    total_loss = response_cost_total + litigated_cases_total + fines_penalties_total + economic_loss_total

    # Calculate proportions
    subcategory_totals = {
        'Response Costs': response_cost_total,
        'Litigated Cases': litigated_cases_total,
        'Fines & Penalties': fines_penalties_total,
        'Economic Loss': economic_loss_total
    }

    subcategory_proportions = {k: v / total_loss for k, v in subcategory_totals.items()}

    return subcategory_proportions


def calculate_fines(regulations, num_records_lost, annual_revenue, weight=0.1):
    fines = 0

    for regulation in regulations:
        if regulation == 'FTC':
            fine = 43792 * weight * num_records_lost
        elif regulation == 'SEC':
            fine = 0.04 * annual_revenue
        elif regulation == 'HIPAA':
            fine = max(25000 * num_records_lost, 1500000)
        elif regulation == 'OCC':
            fine = 0.04 * annual_revenue
        elif regulation == 'CCPA':
            fine = 5000 + 750 * weight * num_records_lost
        elif regulation == 'BIPA':
            fine = 2500 * weight * num_records_lost
        elif regulation == 'GDPR':
            fine = max(20000000, 0.04 * annual_revenue)
        elif regulation == 'UK ICO':
            fine = max(175000000, 0.04 * annual_revenue)
        elif regulation == 'OPC CA':
            fine = max(10000, 0.04 * annual_revenue)
        elif regulation == 'NYDFS Part 500':
            fine = max(250000, 0.04 * annual_revenue)
        else:
            fine = 0  # Unknown regulation
        fines += fine
    return fines


def determine_company_size(num_employees, current_revenue, prev_year_revenue1, prev_year_revenue2):


    revenue_vals = np.array([prev_year_revenue2, prev_year_revenue1, current_revenue])

    #check if value is a linear regression
    growth_rate_1 = (prev_year_revenue1 - prev_year_revenue2) / prev_year_revenue2
    growth_rate_2 = (current_revenue - prev_year_revenue1) / prev_year_revenue1
    average_growth_rate = (growth_rate_1 + growth_rate_2) / 2
    predicted_revenue = current_revenue * (1 + average_growth_rate)

    #use a linear regression model to predict the revenue for the next year
    #use the predicted revenue to determine the company size

    if num_employees < 0:
        raise ValueError('Number of employees must be a positive integer.')
    if any(revenue_vals < 0):
        raise ValueError('Revenue values must be positive.')
    
    if 51 <= num_employees <= 250 or (10_000_000 < predicted_revenue <= 100_000_000):
        return 'medium'
    elif 11 <= num_employees <= 50 or (250000 < predicted_revenue <= 27_000_000):
        return 'small'
    elif num_employees <= 10 or predicted_revenue <= 250000:
        return 'micro'
    else:
        return 'large'
    

################################### Visualizations ##################################

def plot_loss_exceedance_curve(total_losses):
    sorted_losses = np.sort(total_losses)[::-1]
    exceedance_probabilities = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

    plt.figure(figsize=(10,6))
    plt.plot(sorted_losses, exceedance_probabilities)
    plt.xlabel('Total Loss Over Simulation Period')
    plt.ylabel('Exceedance Probability')
    plt.title('Loss Exceedance Curve')
    plt.show()