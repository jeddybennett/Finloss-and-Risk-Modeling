import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_V_micro(T, E, M):
    V1 = -0.2937 * E + 0.2467 * T - 0.0010 * T**2 + 0.6192 * M
    V2 = -0.2937 * E + 0.12335 * T + 0.6192 * M
    return np.where(T <= 123.35, V1, V2)

def calculate_V_small(T, E, M):
    V1 = -1.2994 * E + 0.3368 * T - 0.0004 * T**2 + 2.8878 * M
    V2 = -1.2994 * E + 0.0193 * T + 2.8878 * M + 62.7695
    return np.where(T <= 421, V1, V2)

def calculate_V_medium(T, E, M):
    V1 = -16.7507 * E + 0.701 * T - 0.0002 * T**2 + 32.5225 * M
    V2 = -16.7507 * E + 0.31222 * T + 32.5225 * M + 67.08712
    return np.where(T <= 1752.5, V1, V2)

def calculate_p_value_micro(V):
    sig = 1.017 / (1 + np.exp(-0.415 * (V - 10.703)))
    p = 0.2 + 0.77 * sig
    return p / 100  # Convert percentage to decimal

def calculate_p_value_small(V):
    sig = 1.017 / (1 + np.exp(-0.415 * (V/4 - 15.703)))
    p = 0.2 + 0.77 * sig
    return p / 100  # Convert percentage to decimal

def calculate_p_value_medium(V):
    sig = 1.017 / (1 + np.exp(-0.415 * (V/47.5 - 10.703)))
    p = 0.2 + 0.77 * sig
    return p / 100  # Convert percentage to decimal


def get_p_value(company_size, T, E, M):
    if company_size.lower() == 'micro':
        V = calculate_V_micro(T, E, M)
        p_value = calculate_p_value_micro(V)
    elif company_size.lower() == 'small':
        V = calculate_V_small(T, E, M)
        p_value = calculate_p_value_small(V)
    elif company_size.lower() == 'medium':
        V = calculate_V_medium(T, E, M)
        p_value = calculate_p_value_medium(V)
    else:
        raise ValueError("Invalid company size. Must be 'micro', 'small', or 'medium'.")
    
    return p_value
