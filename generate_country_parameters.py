import os
import numpy as np
import pandas as pd


##### Initialising risk data to be usable #####


# Calculated the fraction of a population who is over 75
##### Would change this so it is a list of proportions, rather than one single proportion.
def build_over75_fraction(r, data_path, country):

    # Calculate the over 75 fraction, as data is split into groups within 75+
    frac_pop_over75_regime = []
    IC_df = pd.read_csv(
        os.path.join(data_path, f'{country}/Start_pop_{r}.csv'),
        skiprows=15, header=None, dtype=np.float64
    )

    frac_pop_over75_regime.append((1/np.sum(np.asarray(IC_df))) * np.sum(np.asarray(IC_df),axis=1))

    # Return float
    return frac_pop_over75_regime


# Build risk data for Syria, for Bouros model
def build_syria_data(r, data_path, country):

    frac_pop_over75_regime = build_over75_fraction(r, data_path, country)

    risk_factor_data = pd.read_csv(
        os.path.join(data_path, f'Syria/Risks_{r}.csv'),
        dtype=np.float64
    )

    # Extended, as over-75 portion still needs collapsed
    extended_P_IH_syria = risk_factor_data['hospitalisation_risk'].tolist()
    extended_P_HD_syria = risk_factor_data['death_risk'].tolist()

    P_IH_syria = np.asarray(extended_P_IH_syria[:15] + [np.sum(np.multiply(extended_P_IH_syria[15:], frac_pop_over75_regime))], dtype=np.float64)
    P_HD_syria = np.asarray(extended_P_HD_syria[:15] + [np.sum(np.multiply(extended_P_HD_syria[15:], frac_pop_over75_regime))], dtype=np.float64)

    # Return 2 16-element arrays
    return P_IH_syria, P_HD_syria


# Build risk data
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

    # Return 4 16-element arrays
    return d_regime, extended_d, beta_regime, P_IH_regime, P_HD_regime


##### #####



##### Create initial conditions from data #####


# Interpret the Start_pop.csv files into a useful, relevant form
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

    # Return (16,5)-element arrays, 16-element arrays
    return (S_regime, E1_regime, E2_regime, E3_regime, E4_regime, E5_regime, I_regime, A_regime, R_regime), pop_regime


##### #####



##### Create contact matricies, and correct to satisfy detailed balance #####


# Create contact matrix
def build_contact_data(r, data_path, country):

    # Load contact matrix data for each country
    contact_matrix_file = os.path.join(data_path, f"{country}/Contacts_{r}.csv")
    contact_matrix_data = pd.read_csv(contact_matrix_file, header=None, dtype=np.float64)
    contact_regime = np.asarray(contact_matrix_data)

    # Return (16,16)-element array
    return contact_regime


# Correct the contact matrix to verify detailed balance requires current contact, pop dict
def correct_contact_matrix(contact_matrix, population):

    contact_matrix = np.asarray(contact_matrix, dtype=np.float64)
    population_vector = np.asarray(population, dtype=np.float64)
    corrected_contact_matrix = np.zeros_like(contact_matrix)

    # Correct the contact matrix to satisfy detailed balance, so total amount of contacts is consistent
    for i in range(contact_matrix.shape[0]):

        corrected_contact_matrix[i, i] = contact_matrix[i, i]

        for j in range(i):
            numerator = (contact_matrix[i, j] * population_vector[i] 
                         + contact_matrix[j, i] * population_vector[j]
            ) / 2
            corrected_contact_matrix[i, j] = numerator / population_vector[i]
            corrected_contact_matrix[j, i] = numerator / population_vector[j]

    # Return (16,16)-element array
    return corrected_contact_matrix


##### #####




##### Calculate the YLL (Years of Life Lost) per death at every age
def calculate_yll(data_path, country):

    # Using age and ex distribution
    age_sex_dist_file = os.path.join(data_path, f"Age_sex_distribution_{country}.csv")
    age_sex_yll_file = os.path.join(data_path, f"Life_expectancy_{country}.csv")

    dist_data = np.asarray(pd.read_csv(age_sex_dist_file))
    yll_data = np.asarray(pd.read_csv(age_sex_yll_file))

    dist_data_grouped = np.zeros((16, 2))
    yll_data_grouped = np.zeros((16, 2))
    avg_yll_by_age = np.zeros(16)

    # Group into 16 age groups, 2 sexes
    for a in range(15):
        dist_data_grouped[a] = np.sum(dist_data[5*a:5*a+5], axis=0)[1:]
    dist_data_grouped[15] = np.sum(dist_data[75:], axis=0)[1:]

    # Group into 16 age groups, 2 sexes
    yll_data_grouped[0] = np.asarray((
        (
            dist_data[0, 1]*yll_data[0, 1] + np.sum(dist_data[1:5, 1], axis=0)*yll_data[1, 1]
        )/ np.sum(dist_data[0:5, 1], axis=0),
        (
            dist_data[0, 2]*yll_data[0, 2] + np.sum(dist_data[1:5, 2], axis=0)*yll_data[1, 2]
        )/ np.sum(dist_data[0:5, 2], axis=0)    
    ))
    for a in range(1, 15):
        yll_data_grouped[a] = yll_data[a+1, 1:]
    yll_data_grouped[15] = np.asarray((
        (
            np.sum(dist_data[75:80, 1], axis=0)*yll_data[16, 1] + np.sum(dist_data[80:85, 1], axis=0)*yll_data[17, 1] + np.sum(dist_data[85:, 1], axis=0)*yll_data[18, 1]
        ) / np.sum(dist_data[75:, 1], axis=0),
        (
            np.sum(dist_data[75:80, 2], axis=0)*yll_data[16, 2] + np.sum(dist_data[80:85, 2], axis=0)*yll_data[17, 2] + np.sum(dist_data[85:, 2], axis=0)*yll_data[18, 2]
        ) / np.sum(dist_data[75:, 2], axis=0)
    ))

    # Using YLL and distribution data for both sexes, collapse into average YLL per age group
    for a in range(16):
        avg_yll_by_age[a] = (
            dist_data_grouped[a, 0] * yll_data_grouped[a, 0] + dist_data_grouped[a, 1] * yll_data_grouped[a, 1]
        ) / sum(dist_data_grouped[a])

    # Return 16-element array
    return avg_yll_by_age


##### #####