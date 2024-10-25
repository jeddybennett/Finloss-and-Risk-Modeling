import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pymc3 as pm
import warnings
import logging
# Suppress all warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
from p_value import get_p_value


def convert_shorthand(value):
    value = value.strip()
    if value.endswith('K'):
        return float(value[:-1]) * 1_000
    elif value.endswith('M'):
        return float(value[:-1]) * 1_000_000
    else:
        return float(value)

# Load data
def load_questionnaire_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def load_historical_data():
    # Load affected count ranges
    affected_counts_df = pd.read_csv('data/affected_count_range.csv')

    # Load subcategory data (response costs, litigated cases, fines & penalties)
    response_costs_df = pd.read_csv('data/response_costs.csv')
    litigated_cases_df = pd.read_csv('data/litigated_cases.csv')
    fines_penalties_df = pd.read_csv('data/fines_penalties.csv')
    economic_loss_df = pd.read_csv('data/economic_loss.csv')

    return affected_counts_df, response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df

def calculate_lambda(p_value):
    p_value = np.clip(p_value, 0, 1 - 1e-10)
    lambda_value = -np.log(1 - p_value)
    return lambda_value

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

def calculate_lognormal_params(num_samples):
    # Collect all loss amounts
    all_loss_amounts = calculate_expected_loss()
    
    # Perform Bayesian estimation
    trace = bayesian_estimate_mu_sigma(all_loss_amounts, num_samples)
    
    # Extract posterior samples for mu and sigma
    mu_samples = trace.posterior['mu'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()
    
    # Return the posterior samples
    return mu_samples, sigma_samples

def bayesian_estimate_mu_sigma(all_loss_amounts, num_samples):
    # Take the logarithm of loss amounts
    log_loss_amounts = np.log(all_loss_amounts)
    
    # Define the model
    with pm.Model() as model:
        # Priors for mu and sigma
        mu_prior = pm.Normal('mu', mu=np.mean(log_loss_amounts), sigma=10)
        sigma_prior = pm.HalfNormal('sigma', sigma=10)
        
        # Likelihood
        observed_data = pm.Normal('observed_data', mu=mu_prior, sigma=sigma_prior, observed=log_loss_amounts)
        
        # Sample from the posterior
        trace = pm.sample(draws=num_samples, tune=1000, chains=2, cores=1, return_inferencedata=True)
    
    return trace

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


def calculate_fines(regulations, num_records_lost, annual_revenue, weight=1):
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

# Define the determine_company_size function
def determine_company_size(num_employees, current_revenue, prev_year_revenue1, prev_year_revenue2):

    average_revenue = (current_revenue + prev_year_revenue1 + prev_year_revenue2) / 3

    revenue_vals = np.array([prev_year_revenue2, prev_year_revenue1, current_revenue])

    #check if value is a linear regression
    predicted_revenue = np.polyfit(np.arange(3), revenue_vals, 1)[0] * 3 + revenue_vals[-1]

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
    
"""Add weight based off the NAICS code of the company, and historical information of that industry"""

def run_monte_carlo_simulations_pymc3(
    lambda_value, mu_samples, sigma_samples, company_data, range_tuples, probabilities, subcategory_proportions,
    num_sims=10000, num_years=5):
    total_losses = []
    subcategory_losses_list = {category: [] for category in subcategory_proportions.keys()}

    # Ensure that the number of simulations matches the number of mu and sigma samples
    num_samples = len(mu_samples)
    if num_sims > num_samples:
        raise ValueError("num_sims cannot be greater than the number of posterior samples.")
    
    for sim in range(num_sims):
        mu = mu_samples[sim]
        sigma = sigma_samples[sim]
        
        # Simulate number of events per year
        events_per_year = np.random.poisson(lam=lambda_value, size=num_years)
        
        total_loss_simulation = 0
        
        for num_events in events_per_year:
            if num_events > 0:
                # Simulate loss amounts
                loss_amounts = np.random.lognormal(mean=mu, sigma=sigma, size=num_events)
                
                # Simulate number of records lost
                record_counts = []
                for _ in range(num_events):
                    selected_range_idx = np.random.choice(len(range_tuples), p=probabilities)
                    low_rc, high_rc = range_tuples[selected_range_idx]
                    if np.isinf(high_rc):
                        num_records = np.random.exponential(scale=low_rc)
                    else:
                        num_records = np.random.uniform(low_rc, high_rc)
                    record_counts.append(num_records)
                
                # Calculate fines
                fines = np.array([
                    calculate_fines(
                        regulations=company_data['regulations'],
                        num_records_lost=num_records,
                        annual_revenue=company_data['annual_revenue']
                    )
                    for num_records in record_counts
                ])
                
                total_loss_year = np.sum(loss_amounts) + np.sum(fines)
            else:
                total_loss_year = 0
            
            total_loss_simulation += total_loss_year
        
        total_losses.append(total_loss_simulation)
        
        # Allocate losses to subcategories
        for category, proportion in subcategory_proportions.items():
            sub_loss = total_loss_simulation * proportion
            subcategory_losses_list[category].append(sub_loss)
    
    total_losses = np.array(total_losses)
    for category in subcategory_losses_list:
        subcategory_losses_list[category] = np.array(subcategory_losses_list[category])
    
    return total_losses, subcategory_losses_list
    
def plot_loss_exceedance_curve(total_losses):
    sorted_losses = np.sort(total_losses)[::-1]
    exceedance_probabilities = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

    plt.figure(figsize=(10,6))
    plt.plot(sorted_losses, exceedance_probabilities)
    plt.xlabel('Total Loss Over Simulation Period')
    plt.ylabel('Exceedance Probability')
    plt.title('Loss Exceedance Curve')
    plt.show()
  
def model(plot = False):
    # Load data
    questionnaire_df = load_questionnaire_data('questionnaire_data.csv')
    (affected_counts_df, response_costs_df, litigated_cases_df,
     fines_penalties_df, economic_loss_df) = load_historical_data()

    # Prepare historical data
    range_tuples, probabilities = prepare_affected_counts(affected_counts_df)
    subcategory_proportions = calculate_subcategory_proportions(
        response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df
    )

    # Iterate over each company
    for idx, company_row in questionnaire_df.iterrows():
        # Extract company-specific data
        company_data = {
            'company_name': company_row['company_name'],
            'annual_revenue': company_row['annual_revenue'],
            'prev_year_revenue1': company_row['prev_year_revenue1'],
            'prev_year_revenue2': company_row['prev_year_revenue2'],
            'num_employees': company_row['num_employees'],
            'regulations': company_row['regulations'].split(','),
            'E': company_row['E'],  
            'T': company_row['T'], 
            'M': company_row['M'],

            #determine how to assign a weight to the NAICS code
            'NAICS': company_row['NAICS']
            # Additional fields as needed
        }

        # Determine company size
        company_size = determine_company_size(
            num_employees=company_data['num_employees'],
            current_revenue=company_data['annual_revenue'],
            prev_year_revenue1=company_data['prev_year_revenue1'],
            prev_year_revenue2=company_data['prev_year_revenue2']
        )

        # Handle companies larger than 'medium'
        if company_size not in ['micro', 'small', 'medium']:
            company_size = 'medium'  # Adjust as per your model's capability
        company_data['company_size'] = company_size

        # Calculate p_value using the imported function
        p_value = get_p_value(
            company_size=company_data['company_size'],
            T=company_data['T'],
            E=company_data['E'],
            M=company_data['M']
        )

        # Calculate λ
        lambda_value = calculate_lambda(p_value)

         # Run simulations
        num_simulations = 10000
        num_years = 1

        # Calculate μ and σ
        mu, sigma = calculate_lognormal_params(num_samples=num_simulations)
       
        

        total_losses, subcategory_losses_list = run_monte_carlo_simulations_pymc3(
            lambda_value, mu, sigma, company_data, range_tuples, probabilities,
            subcategory_proportions, num_sims=num_simulations, num_years=num_years
        )

        # Analyze results (as before)
        expected_total_loss = np.mean(total_losses)
        # lower_bound_total_loss = np.percentile(total_losses, 5)
        # upper_bound_total_loss = np.percentile(total_losses, 95)

        print(f"\n--- Risk Assessment for {company_data['company_name']} ---")
        print(f"Expected Total Loss: ${expected_total_loss:,.2f}")
        # print(f"Lower Bound on Total Loss: ${lower_bound_total_loss:,.2f}")
        # print(f"Upper Bound on Total Loss: ${upper_bound_total_loss:,.2f}")

        print("\nExpected Subcategory Losses:")
        for category in subcategory_proportions.keys():
            expected_sub_loss = np.mean(subcategory_losses_list[category])
            print(f"{category}: ${expected_sub_loss:,.2f}")

        if plot:
            plot_loss_exceedance_curve(total_losses)


if __name__ == '__main__':
    model()