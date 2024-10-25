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

    def process_questionnaire(file_name):
        """Process questionnaire JSON file to calculate total capped score."""
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing questionnaire file: {e}")
            return 0

        total_score = 0

        for item in data:
            if isinstance(item, dict):  # Ensure item is a dictionary
                score = item.get('score')
                if isinstance(score, (int, float)):
                    total_score += max(Constants.MIN_SCORE, min(Constants.MAX_SCORE, score))

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

    def calculate_value(a, b, Q, c, assessment, d, PT = None):
        """Calculate the combined value based on provided formula."""
        if PT:
            return a * PT + b * Q + c * assessment + d
        else:
            PT = 0
            return a * PT + b * Q + c * assessment + d

def v_calc(company_size):
    # File paths
    cv_file_name = 'V_calc/PT.json'
    questionnaire_file_name = 'V_calc/questionnaire.json'
    assessment_file_names = ['V_calc/NAssess.json', 'cis.json', 'sig.json']

    # Coefficients for the calculation
    a, b, c, d = 2.31, 0.20543, -0.1, 1

    # Process data to obtain PT, Q, and assessment values
    if cv_file_name:
        PT = DataProcessor.process_data(cv_file_name)
    else:
        PT = None
    Q = DataProcessor.process_questionnaire(questionnaire_file_name)
    assessment_value = DataProcessor.process_assessment(assessment_file_names)

    # Calculate final V value
    V = ValueCalculator.calculate_value(a, b, Q, c, assessment_value, d, PT = PT)
    if company_size == 'micro':
        return V / 5
    elif company_size == 'small':
        return 20 + V / 1.25
    else:
        return 50 + V * 9.5
