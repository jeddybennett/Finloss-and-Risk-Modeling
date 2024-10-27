import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mapping_values():
    mappings = {
    "materialBreach": {
        "Yes": 0,
        "No": 3
    },
    "remoteAccess": {
        "Yes": 1,
        "No": 2,
        "Remote access not permitted": 3
    },
    "remoteAccessTypes": {
        "IT Support Staff": 1,
        "Cybersecurity Professionals": 1,
        "Business Continuity / Disaster Recovery Teams": 1,
        "Executive Leadership": 1
    },
    "securityPlanPractice": {
        "Annually": 1,
        "Biannually": 2,
        "Quarterly": 3,
        "Monthly": 4,
        "Never": 0
    },
    "documentedSecurityPlan": {
        "Yes": 2,
        "No": 0
    },
    "securityTraining": {
        "Annually": 1,
        "Biannually": 2,
        "Quarterly": 3,
        "No training offered": 0
    },
    "alternativeTraining": {
        "Yes": 1,
        "No": 0
    },
    "incidentResponseTraining": {
        "Monthly": 3,
        "Quarterly": 2,
        "Annually": 1,
        "Never": 0
    },
    "riskAssessment": {
        "Weekly/Daily": 5,
        "Quarterly": 4,
        "Biannually": 3,
        "Annually": 2,
        "Never": 1
    },
    "necessaryPatches": {
        "Yes": 1,
        "No": 0
    },
    "patchesIn30Days": {
        "Yes": 1,
        "No": 0
    },
    "dataEncryption": {
        "Yes": 1,
        "No": 0
    },
    "dataBacking": {
        "Weekly/Daily": 3,
        "Monthly": 2,
        "Other": 1,
        "Not backed up": 0
    },
    "thirdPartyControls": {
        "Yes": 2,
        "No": 1,
        "No third party": 3
    },
    "securityAlerts": {
        "Real - time": 4,
        "Within hour": 2,
        "Within day": 0.67,
        "Don't monitor": 0,
        "Response is in real-time": 4
    },
    "edrTools": {
        "Yes": 3,
        "No": 1
    },
    "edrToolsUsed": {
        "Any EDR tool": 1
    }
}
    return mappings


def parse_security_data(security_data):
    scores = {}
    mappings = mapping_values()
    # For each question, get the answer
    for question, answer in security_data.items():
        # Handle special cases with conditional mappings
        if question == 'remoteAccessTypes':
            # Sum the scores for each selected type
            total = 0
            for item in answer:
                item_score = mappings.get(question, {}).get(item, 0)
                total += item_score
            scores[question] = total
        elif question == 'edrToolsUsed':
            # Only consider if 'edrTools' is 'Yes'
            if security_data.get('edrTools') == 'Yes':
                item_score = mappings.get(question, {}).get(answer, 0)
                scores[question] = item_score
            else:
                scores[question] = 0
        elif question == 'documentedSecurityPlan':
            # Only consider if 'securityPlanPractice' is 'Never'
            if security_data.get('securityPlanPractice') == 'Never':
                normalized_answer = normalize_answer(answer)
                item_score = mappings.get(question, {}).get(normalized_answer, 0)
                scores[question] = item_score
            else:
                scores[question] = 0
        elif question == 'alternativeTraining':
            # Only consider if 'securityTraining' is 'No training offered'
            if security_data.get('securityTraining') == 'No training offered':
                normalized_answer = normalize_answer(answer)
                item_score = mappings.get(question, {}).get(normalized_answer, 0)
                scores[question] = item_score
            else:
                scores[question] = 0
        elif question == 'patchesIn30Days':
            # Only consider if 'necessaryPatches' is 'No'
            if security_data.get('necessaryPatches') == 'No':
                normalized_answer = normalize_answer(answer)
                item_score = mappings.get(question, {}).get(normalized_answer, 0)
                scores[question] = item_score
            else:
                scores[question] = 0
        else:
            # Get the score for the answer
            normalized_answer = normalize_answer(answer)
            item_score = mappings.get(question, {}).get(normalized_answer, 0)
            scores[question] = item_score
    return scores

def normalize_answer(answer):
    # Normalize answers to match mapping keys
    if isinstance(answer, str):
        answer = answer.strip()
        if answer == "":
            return "No"
        if answer == "Response is in real-time":
            return "Real - time"
        if answer == "Weekly/Daily":
            return "Weekly/Daily"
        if answer == "No training offered":
            return "No training offered"
    return answer