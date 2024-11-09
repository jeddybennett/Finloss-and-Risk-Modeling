import json
import os

# Constants for R multipliers and score limits
class Constants:
    R_MULTIPLIERS = {
        'critical': 4,
        'high': 3,
        'medium': 2,
        'low': 1
    }
    MAX_SCORE = 32
    MIN_SCORE = 0

class DataProcessor:
    """Class to handle processing of different types of data files."""
    
    def process_data(file_name):
        """Process PT JSON file to calculate mean of (CV * R_multiplier)."""
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing PT file '{file_name}': {e}")
            return 0

        total, count = 0, 0

        for item in data:
            if isinstance(item, dict):  # Ensure item is a dictionary
                cv_value = max(Constants.MIN_SCORE, min(10, item.get('CV', 0)))  # Cap CV at 10 and not negative
                r_value = item.get('R')

                if r_value and r_value.lower() in Constants.R_MULTIPLIERS:
                    multiplier = Constants.R_MULTIPLIERS[r_value.lower()]
                    total += cv_value * multiplier
                    count += 1

        return total / count if count > 0 else 0

    def process_questionnaire(data):
        """Process questionnaire dictionary to calculate total capped score."""

        total_score = 0

        for key,score in data.items():
            total_score += score
                
        return min(total_score, Constants.MAX_SCORE)

    def process_assessment(file_names):
        """Process assessment JSON files and calculate the mean value."""
        for file_name in file_names:
            if os.path.isfile(file_name):
                try:
                    with open(file_name, 'r') as file:
                        data = json.load(file)
                    total_score = 0
                    count = 0
                    for item in data:
                        if isinstance(item, dict):
                            value = item.get('value')
                            if value in (1, 2, 3, 4):
                                total_score += value
                                count += 1
                            elif value != 'NA':
                                print(f"Warning: Invalid entry '{value}' in {file_name}. Skipping.")

                    return total_score / count if count > 0 else 0
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error processing file '{file_name}': {e}")
        return 0

class ValueCalculator:
    """Class to calculate final value from processed scores."""

    def calculate_value(Q, assessment = None, PT = None):
        """Calculate the combined value based on provided formula."""
        if PT:
            if assessment:
                return 2.31 * PT + 0.20543 * Q - 0.1 * assessment + 1
            else:
                return 2.25 * PT + 0.3125 * Q

        if Q and assessment:
            return 0.75 * assessment + 0.78 * Q


def v_calc(company_size, security_data, assessment_data):
    # File paths
    #cv_file_name = 'V_calc/PT.json'
    
    questionnaire_file_name = security_data

    assessment_name = assessment_data['assessmentType']
    score = assessment_data['assessmentScore']

    Q = DataProcessor.process_questionnaire(questionnaire_file_name)

    # Process data to obtain PT, Q, and assessment values
    
    if assessment_name == 'PT':
        PT = score
        assessment_value = None
    else:
        assessment_value = score
        PT = None

    # Calculate final V value
    V = ValueCalculator.calculate_value(Q, assessment = assessment_value, PT = PT)
    if company_size == 'micro':
        return V / 5
    elif company_size == 'small':
        return 20 + V / 1.25
    else:
        return 50 + V * 9.5
