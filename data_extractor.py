import pandas as pd
from io import StringIO

# Data from Advisen

# 1. Cyber Case (with Loss) Count by Case Type/Case Status

data = {
    "Case Type": [
        "Data - Malicious Breach",
        "Privacy - Unauthorized Contact or Disclosure",
        "Data - Physically Lost or Stolen",
        "Data - Unintentional Disclosure",
        "Network/Website Disruption",
        "Identity - Fraudulent Use/Account Access",
        "Privacy - Unauthorized Data Collection",
        "Phishing, Spoofing, Social Engineering",
        "Skimming, Physical Tampering",
        "IT - Processing Errors",
        "Cyber Extortion",
        "IT - Configuration/Implementation Errors",
        "Industrial Controls & Operations",
        "Total"
    ],
    "Case Count (Response Costs)": [
        206, 1, 15, 13, 114, 1, None, 21, 6, 7, 1, 7, 3, 395
    ],
    "Loss Amount (Response Costs)": [
        4046940, 700, 86365, 4902, 2092951, 45, None, 76784, 26274, 907548, 177, 349172, 1187, 7593045
    ],
    "Case Count (Economic Loss)": [
        667, 5, 41, 9, 215, 989, 3, 278, 98, 33, 8675, 20, 6, 11041
    ],
    "Loss Amount (Economic Loss)": [
        15669566, 524120, 2078884, 38713, 4018173, 14409001, 32012, 1132811, 63908, 1493802, 271472, 473471, 188188, 40644124
    ],
    "Case Count (Litigated Cases)": [
        398, 2590, 35, 96, 7, 35, 245, 30, 9, 8, None, 28, 1, 3555
    ],
    "Loss Amount (Litigated Cases)": [
        5029375, 7572005, 180918, 242773, 14219, 26490, 7408302, 742363, 75672, 305749, None, 230203, 7, 18070452
    ],
    "Case Count (Fines & Penalties)": [
        543, 1306, 46, 248, 11, 29, 172, 23, 3, 26, 8676, 43, 3, 2453
    ],
    "Loss Amount (Fines & Penalties)": [
        2674547, 2316381, 26690, 467718, 23462, 379629, 7408302, 1178045, 175, 1704319, None, 430707, 11019, 16620996
    ]
}

df = pd.DataFrame(data)

df_response_costs = df[['Case Type', 'Case Count (Response Costs)', 'Loss Amount (Response Costs)']]
df_economic_loss = df[['Case Type', 'Case Count (Economic Loss)', 'Loss Amount (Economic Loss)']]
df_litigated_cases = df[['Case Type', 'Case Count (Litigated Cases)', 'Loss Amount (Litigated Cases)']]
df_fines_penalties = df[['Case Type', 'Case Count (Fines & Penalties)', 'Loss Amount (Fines & Penalties)']]


print("Response Costs DataFrame:")
print(df_response_costs)
print("\nEconomic Loss DataFrame:")
print(df_economic_loss)
print("\nLitigated Cases DataFrame:")
print(df_litigated_cases)
print("\nFines & Penalties DataFrame:")
print(df_fines_penalties)

df_response_costs.to_csv('response_costs.csv', index=False)
df_economic_loss.to_csv('economic_loss.csv', index=False)
df_litigated_cases.to_csv('litigated_cases.csv', index=False)
df_fines_penalties.to_csv('fines_penalties.csv', index=False)


# 2. Affected Count Range by Case Type


# Data
data = {
    "Affected Count Range": ["(0, 10]", "(10, 100]", "(100, 1K]", "(1K, 10K]", "(10K, 100K]", "(100K, 1M]", "(1M, )", "Total"],
    "Case Count": [58269, 14872, 20437, 14143, 6651, 2660, 1726, 118758]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to calculate average affected count
def average_affected_count(row):
    if row["Affected Count Range"] == "Total":
        return None  # Skip total for average calculation

    # Extract range values
    if row["Affected Count Range"] == "(1M, )":
        lower_bound = 1000000  # 1M
        upper_bound = 2000000  # Assuming an upper limit for calculation
    else:
        lower_bound, upper_bound = map(lambda x: int(x.replace(',', '').replace('K', '000').replace('M', '000000').replace(']', '').replace('(', '')) if 'K' in x or 'M' in x else int(x.replace(',', '').replace(']', '').replace('(', '')), row["Affected Count Range"].split(', '))
    
    # Calculate average
    average = (lower_bound + upper_bound) / 2
    return average

# Add Average Affected Count column
df["Average Affected Count"] = df.apply(average_affected_count, axis=1)

# Display the DataFrame
print(df)

# Save to CSV
df.to_csv('affected_count_range.csv', index=False)


# 3. Data Breach Count by Industry

import pandas as pd

# Data
data = {
    "Affected Count Range": [
        "Industrial Controls & Operations",
        "Undetermined/Other",
        "IT - Processing Errors",
        "Identity - Fraudulent Use/Account Access",
        "IT - Configuration/Implementation Errors",
        "Skimming, Physical Tampering",
        "Privacy - Unauthorized Data Collection",
        "Phishing, Spoofing, Social Engineering",
        "Network/Website Disruption",
        "Data - Physically Lost or Stolen",
        "Data - Unintentional Disclosure",
        "Privacy - Unauthorized Contact or Disclosure",
        "Data - Malicious Breach"
    ],
    "With Affected Count": [
        25, 382, 556, 736, 750, 680, 1125, 3745, 
        1031, 8407, 18260, 37719, 45341
    ],
    "With no Affected Count": [
        120, 151, 474, 345, 665, 854, 581, 1313,
        6923, 1718, 4192, 4621, 16714
    ],
    "Total": [
        145, 533, 1030, 1081, 1415, 1534, 1706, 5058,
        7954, 10125, 22452, 42340, 62055
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame without totals
print(df)

# Save to CSV
df.to_csv('cyber_case_count_by_type.csv', index=False)


# 4. Data Breach Count by Case Type 
# Data
data = {
    "Case Type": [
        "Data - Malicious Breach",
        "Privacy - Unauthorized Contact or Disclosure",
        "Data - Unintentional Disclosure",
        "Data - Physically Lost or Stolen",
        "Network/Website Disruption",
        "Phishing, Spoofing, Social Engineering",
        "Privacy - Unauthorized Data Collection",
        "Skimming, Physical Tampering",
        "IT - Configuration/Implementation Errors",
        "Identity - Fraudulent Use/Account Access",
        "IT - Processing Errors",
        "Undetermined/Other",
        "Industrial Controls & Operations"
    ],
    "Case Count": [
        62055, 42340, 22452, 10125, 7954, 5058,
        1706, 1534, 1415, 1081, 1030, 533, 145
    ]
}

# Create DataFrame
df_cases = pd.DataFrame(data)

# Display the DataFrame without totals
print(df_cases)

# Save to CSV
df_cases.to_csv('cyber_case_count_by_case_type.csv', index=False)


# 5.  Size and count of data breach of Data Breaches

import pandas as pd

# Create the initial DataFrame
data = {
    'Company Size': ['(0, 1M]', '(1M, 10M]', '(10M, 100M]', '(100M, 1B]', 
                     '(1B, 10B]', '(10B, 100B]', '(100B, )', 'Unknown'],
    'PRV': [15902, 19387, 22430, 12727, 7663, 1666, 1283, 15890],
    'PUB': [485, 1436, 2490, 4717, 7206, 10306, 3667, 6664],
    'GOV': [307, 675, 2428, 1498, 1819, 7842, 2067, 3020],
    'NPR': [710, 620, 953, 435, 550, 33, None, 552],
    'Total': [17404, 22118, 28301, 19377, 17238, 19847, 7017, 26126]
}

df = pd.DataFrame(data)

# Define a mapping for new categories
size_mapping = {
    '(0, 1M]': 'Small',
    '(1M, 10M]': 'Medium',
    '(10M, 100M]': 'Large',
    '(100M, 1B]': 'Very Large',
    '(1B, 10B]': 'Giant',
    '(10B, 100B]': 'Mega',
    '(100B, )': 'Colossal',
    'Unknown': 'Unknown'
}

# Apply the mapping
df['Company Size'] = df['Company Size'].map(size_mapping)

# Display the modified DataFrame
print(df)
df.to_csv('count_by_company_size_and_type.csv', index=False)


# 6. Industry and Case Count

import pandas as pd

# Define the data
data = {
    "Industry": ["Finance, Insurance, And Real Estate"],
    "Prior 2008": [1720],
    "2008": [1542],
    "2009": [944],
    "2010": [1075],
    "2011": [1431],
    "2012": [2013],
    "2013": [2578],
    "2014": [2514],
    "2015": [2798],
    "2016": [3144],
    "2017": [2685],
    "2018": [2778],
    "2019": [2249],
    "2020": [2236],
    "2021": [2139],
    "2022": [1637],
    "2023": [359]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Save to CSV
df.to_csv('financial_services_historical_case_count.csv', index=False)



# 7. Data Breach Count by Country

data = {
    "Country": ["USA", "CAN", "GBR", "IND", "AUS", "DEU", "FRA", "ITA", "BRA", "IDN", "Others"],
    "Case Count": [118810, 13799, 4572, 1321, 1212, 1092, 1036, 820, 727, 608, 13431]
}

df = pd.DataFrame(data)
print(df)

# Save to CSV
df.to_csv('data_breach_count_by_country.csv', index=False)


# 8. Data Breach Count by State
import pandas as pd

# Define the data
data = {
    "State": ["CA", "NY", "TX", "MA", "FL", "IL", "PA", "VA", "GA", "NJ", "Others"],
    "Case Count": [15992, 12561, 7819, 7648, 7302, 6061, 4889, 3631, 3475, 3338, 46094]
}

# Create the DataFrame
df = pd.DataFrame(data)
print(df)

# Save to CSV
df.to_csv('data_breach_count_by_state.csv', index=False)