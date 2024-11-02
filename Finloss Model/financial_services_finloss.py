import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# import pymc3 as pm
import warnings
import logging
import json
from helper_func import *
from V_calc.alt_v import v_calc
from V_calc.parse_security import parse_security_data
# Suppress all warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
from p_value import get_p_value
    
"""Add weight based off the NAICS code of the company, and historical information of that industry"""

def run_monte_carlo_simulations(
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
    

  
def model(plot=False):
    # Load data from JSON
    data = load_questionnaire_data('questionnaire_output.json')

    business_data = data['businessQuestionnaire']
    security_data = data['securityQuestionnaire']

    security_scores = parse_security_data(security_data)

    # Parse and extract company data
    company_data = {
        'name': business_data['name'],
        'phone': business_data['phone'],
        'email': business_data['email'],
        'location': business_data['country'],
        'company_name': business_data['companyName'],
        'primary_industry': business_data['primaryIndustry'],
        'annual_revenue': parse_revenue(business_data['annualRevenue']),
        'prev_year_revenue1': parse_number(business_data.get('revenue2022', '0')),
        'prev_year_revenue2': parse_number(business_data.get('revenue2023', '0')),
        'num_employees': parse_employee_count(business_data['employeeCount']),
        'regulations': [reg.strip() for reg in business_data['regulations']],
        'recent_breach': business_data['recentBreach'],
        'data_exposed': business_data['dataExposed'],
        'records_exposed': business_data['recordsExposed'],
        'E': derive_E(business_data.get('incidentCount', '1')),
        'T': derive_T(business_data.get('dataExposed', [])),
        'M': derive_M(business_data.get('cybersecurityBudget', '0')),
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
        company_size=company_data['company_size'], security_data = security_scores
    )

    # Calculate λ
    lambda_value = max(calculate_lambda(p_value), 0.1)

    # Run simulations
    num_simulations = 10000
    num_years = 1

    # Calculate μ and σ
    mu_samples, sigma_samples = calculate_lognormal_params(num_samples=num_simulations)

    # Prepare historical data
    (affected_counts_df, response_costs_df, litigated_cases_df,
     fines_penalties_df, economic_loss_df) = load_historical_data()

    range_tuples, probabilities = prepare_affected_counts(affected_counts_df)
    subcategory_proportions = calculate_subcategory_proportions(
        response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df
    )

    total_losses, subcategory_losses_list = run_monte_carlo_simulations(
        lambda_value, mu_samples, sigma_samples, company_data, range_tuples, probabilities,
        subcategory_proportions, num_sims=num_simulations, num_years=num_years
    )

    # Analyze results
    expected_total_loss = round(np.mean(total_losses),2)
    lower_bound_total_loss = np.percentile(total_losses, 2.5)
    upper_bound_total_loss = np.percentile(total_losses, 97.5)

    # print(f"\n--- Risk Assessment for {company_data['company_name']} ---")
    # print(f"Expected Total Loss: ${expected_total_loss:,.2f}")
    # # print(f"95% Credible Interval: (${lower_bound_total_loss:,.2f}, ${upper_bound_total_loss:,.2f})")

    # print("\nExpected Subcategory Losses:")
    # for category in subcategory_proportions.keys():
    #     expected_sub_loss = np.mean(subcategory_losses_list[category])
    #     print(f"{category}: ${expected_sub_loss:,.2f}")

    

    results = {
        'company_name': company_data['company_name'],
        'expected_total_loss': expected_total_loss,
        'expected_subcategory_losses': {},
    }

    for category in subcategory_proportions.keys():
        expected_sub_loss = np.mean(subcategory_losses_list[category])
        results['expected_subcategory_losses'][category] = round(expected_sub_loss,2)
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)
    
    with open('risk_assessment_results.json', 'w') as outfile:
        json.dump(serializable_results, outfile, indent=1)
    
    if plot:
        plot_loss_exceedance_curve(total_losses)
    
    return serializable_results

if __name__ == '__main__':
    results = model(True)
    print(results)