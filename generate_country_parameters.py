import numpy as np
import pandas as pd
import os
from collections import defaultdict

countries = ["United Kingdom", "Canada", "Brazil", "South Africa", 
    "Kenya", "The Philippines", "Sierra Leone", "Syria"]

d = []
extended_d = []
beta = []
P_IH = []
P_HD = []

S = []
E1 = []
E2 = []
E3 = []
E4 = []
E5 = []
I = []
A = []
R = []
pop = []

contact = []

risk = []

def build_over75_fraction(r, data_path, country):

    # Calculate the over 75 fraction, as data is split into groups within 75+
    frac_pop_over75_regime = []
    IC_df = pd.read_csv(
        os.path.join(data_path, f'{country}/Start_pop_{r}.csv'),
        skiprows=15, header=None, dtype=np.float64
    )

    frac_pop_over75_regime.append((1/np.sum(np.asarray(IC_df))) * np.sum(np.asarray(IC_df),axis=1))

    return frac_pop_over75_regime

def build_risk_data(r, data_path, country):

    frac_pop_over75_regime = build_over75_fraction(r, data_path, country)

    # Load risk factor data for each country
    risk_factor_data = pd.read_csv(
        os.path.join(data_path, f'{country}/Risks_{r}.csv'),
        dtype=np.float64
    )
    extended_d = risk_factor_data['symptom_risk'].tolist()
    extended_beta = risk_factor_data['susceptibility'].tolist()
    extended_P_IH = risk_factor_data['hospitalisation_risk'].tolist()
    extended_P_HD = risk_factor_data['death_risk'].tolist()

    d_regime = np.asarray(extended_d[:15] + [np.sum(np.multiply(extended_d[15:], frac_pop_over75_regime))], dtype=np.float64)
    beta_regime = np.asarray(extended_beta[:15] + [np.sum(np.multiply(extended_beta[15:], frac_pop_over75_regime))], dtype=np.float64)
    P_IH_regime = np.asarray(extended_P_IH[:15] + [np.sum(np.multiply(extended_P_IH[15:], frac_pop_over75_regime))], dtype=np.float64)
    P_HD_regime = np.asarray(extended_P_HD[:15] + [np.sum(np.multiply(extended_P_HD[15:], frac_pop_over75_regime))], dtype=np.float64)

    return d_regime, extended_d, beta_regime, P_IH_regime, P_HD_regime

# reguires nu_symp and extended_d lists
def build_population_IC(r, mu, extended_d, data_path, country):

    # Unpack the mu and d values for this regime and country
    regimes_mu = mu
    extended_regimes_d = extended_d

    # Load population data for each country
    population_file = os.path.join(data_path, f"{country}/Start_pop_{r}.csv")

    # Unpack the susceptible population data
    S_population_data = pd.read_csv(population_file, usecols=range(0, 5), header=None, dtype=np.float64)
    extended_S = np.asarray(S_population_data)
    under_75_S = extended_S[:15, :]
    over_75_S = extended_S[15:, :]
    reduced_S = np.vstack((under_75_S, np.sum(over_75_S, axis=0)))
    S_regime = reduced_S

    # Repeat the same process for E1, E2, E3, E4, E5, I, and R populations
    E1_population_data = pd.read_csv(population_file, usecols=range(5, 10), header=None, dtype=np.float64)
    extended_E1 = np.asarray(E1_population_data)
    under_75_E1 = extended_E1[:15, :]
    over_75_E1 = extended_E1[15:, :]
    reduced_E1 = np.vstack((under_75_E1, np.sum(over_75_E1, axis=0)))
    E1_regime = reduced_E1

    # E2 population data
    E2_population_data = pd.read_csv(population_file, usecols=range(10, 15), header=None, dtype=np.float64)
    extended_E2 = np.asarray(E2_population_data)
    under_75_E2 = extended_E2[:15, :]
    over_75_E2 = extended_E2[15:, :]
    reduced_E2 = np.vstack((under_75_E2, np.sum(over_75_E2, axis=0)))
    E2_regime = reduced_E2

    # E3 population data
    E3_population_data = pd.read_csv(population_file, usecols=range(15, 20), header=None, dtype=np.float64)
    extended_E3 = np.asarray(E3_population_data)
    under_75_E3 = extended_E3[:15, :]
    over_75_E3 = extended_E3[15:, :]
    reduced_E3 = np.vstack((under_75_E3, np.sum(over_75_E3, axis=0)))
    E3_regime = reduced_E3

    # E4 population data
    E4_population_data = pd.read_csv(population_file, usecols=range(20, 25), header=None, dtype=np.float64)
    extended_E4 = np.asarray(E4_population_data)
    under_75_E4 = extended_E4[:15, :]
    over_75_E4 = extended_E4[15:, :]
    reduced_E4 = np.vstack((under_75_E4, np.sum(over_75_E4, axis=0)))
    E4_regime = reduced_E4

    # E5 population data
    E5_population_data = pd.read_csv(population_file, usecols=range(25, 30), header=None, dtype=np.float64)
    extended_E5 = np.asarray(E5_population_data)
    under_75_E5 = extended_E5[:15, :]
    over_75_E5 = extended_E5[15:, :]
    reduced_E5 = np.vstack((under_75_E5, np.sum(over_75_E5, axis=0)))
    E5_regime = reduced_E5

    # I and A population data
    IA_population_data = pd.read_csv(population_file, usecols=range(30, 35), header=None, dtype=np.float64)

    extended_I = np.zeros_like(np.asarray(IA_population_data))
    for j, v in enumerate(("U", "F", "B", "W1", "W2")):
        extended_I[:, j] = np.matmul(np.diag(regimes_mu[v] * np.array(extended_regimes_d)), np.asarray(IA_population_data)[:, j])
        under_75_I = extended_I[:15, :]
        over_75_I = extended_I[15:, :]
        reduced_I = np.vstack((under_75_I, np.sum(over_75_I, axis=0)))
        I_regime = reduced_I

    extended_A = np.zeros_like(np.asarray(IA_population_data))
    for j, v in enumerate(("U", "F", "B", "W1", "W2")):
        extended_A[:, j] = np.matmul(np.diag(1 - (regimes_mu[v] * np.array(extended_regimes_d))), np.asarray(IA_population_data)[:, j])
        under_75_A = extended_A[:15, :]
        over_75_A = extended_A[15:, :]
        reduced_A = np.vstack((under_75_A, np.sum(over_75_A, axis=0)))
        A_regime = reduced_A

    # R population data
    R_population_data = pd.read_csv(population_file, usecols=[35], header=None, dtype=np.float64)
    extended_R = np.asarray(R_population_data)
    under_75_R = extended_R[:15, :]
    over_75_R = extended_R[15:, :]
    reduced_R = np.vstack((under_75_R, np.sum(over_75_R, axis=0)))
    R_regime = reduced_R

    # Total population data
    population_data = pd.read_csv(population_file, header=None, dtype=np.float64)
    extended_population = np.asarray(population_data)
    under_75_population = extended_population[:15, :]
    over_75_population = extended_population[15:, :]
    reduced_population = np.vstack((under_75_population, np.sum(over_75_population, axis=0)))
    pop_regime = np.sum(reduced_population, axis=1)

    S.append(S_regime)
    E1.append(E1_regime)
    E2.append(E2_regime)
    E3.append(E3_regime)
    E4.append(E4_regime)
    E5.append(E5_regime)
    I.append(I_regime)
    A.append(A_regime)
    R.append(R_regime)
    pop.append(pop_regime)

    return (S_regime, E1_regime, E2_regime, E3_regime, E4_regime, E5_regime, I_regime, A_regime, R_regime), pop_regime

def build_contact_data(r, data_path, country):

    # Load contact matrix data for each country
    contact_matrix_file = os.path.join(data_path, f"{country}/Contacts_{r}.csv")
    contact_matrix_data = pd.read_csv(contact_matrix_file, header=None, dtype=np.float64)
    contact_regime = np.asarray(contact_matrix_data)

    contact.append(contact_regime)

    return contact_regime

# Correct the contact matrix to verify detailed balance requires current contact, pop dict
def correct_contact_matrix(contact_matrix, population):

    contact_matrix = np.asarray(contact_matrix, dtype=np.float64)
    population_vector = np.asarray(population, dtype=np.float64)
    corrected_contact_matrix = np.zeros_like(contact_matrix)
    for i in range(contact_matrix.shape[0]):

        corrected_contact_matrix[i, i] = contact_matrix[i, i]

        for j in range(i):
            numerator = (contact_matrix[i, j] * population_vector[i] 
                         + contact_matrix[j, i] * population_vector[j]
            ) / 2
            corrected_contact_matrix[i, j] = numerator / population_vector[i]
            corrected_contact_matrix[j, i] = numerator / population_vector[j]

    return corrected_contact_matrix