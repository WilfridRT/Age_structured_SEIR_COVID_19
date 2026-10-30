import numpy as np
import scipy.integrate as spi
import scipy.stats
import pandas as pd
import generate_country_parameters as cp
import generate_global_parameters as gp
from collections import defaultdict



# Defining the overarching simulation parameters
total_days =  151
age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
    '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+']
countries = ["United Kingdom", "Canada", "Brazil", "South Africa", 
    "Kenya", "Philippines", "Sierra Leone", "Syria"]
vaccine_status = ("U", "F", "B", "W1", "W2")


# Defining the number of simulations
regimes = np.arange(1, 101, 1).tolist()


# Folder path to data
data_path = '/home/teasdale/Downloads/warwick-covid-transmission-main/warwickmodel/data/'



##### Booster Vaccinations #####


# Finding the proportion of each age group that is boostable (S or R) and then gets boosted
def vaccine_strategy(n, stratified_population, population):

    stratified_population = list(stratified_population)
    S_population = np.asarray(stratified_population[0], dtype=float)
    R_population = np.asarray(stratified_population[8], dtype=float).ravel()

    boostable_population= S_population[:, [0, 1, 3, 4]].sum(axis=1) + R_population
    uptake = 0.9 * boostable_population
    
    # Different priority strategies and rankings
    priority_tiers = {
        1: [[i] for i in range(15, -1, -1)],
        2: [[12, 13, 14], [15]] + [[i] for i in range(11, -1, -1)],
        3: [[12, 13, 14], [10, 11], [15]] + [ [i] for i in range(9, -1, -1)],
        4: [[10, 11], [12, 13, 14], [15]] + [ [i] for i in range(9, -1, -1)],
        5: [[4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15]]
            + [[i] for i in range(3, -1, -1)],
        6: [[4, 5, 6, 7, 8, 9], [15], [10, 11, 12, 13, 14]]
            + [[i] for i in range(3, -1, -1)],
    }

    # Computing number of boosted individuals in each group
    supply = 0.1 * np.asarray(population).sum()
    boosted_population = np.zeros(len(population))
    for tier in priority_tiers[n]:
        ids = np.asarray(tier, dtype=int)
        tier_uptake = uptake[ids].sum()

        if supply <= 0:
            break
        if tier_uptake <= 0:
            continue

        fraction = min(1, supply/tier_uptake)
        boosted_population[ids] = fraction * uptake[ids]
        supply -= boosted_population[ids].sum()

    # Return boosted and boostable as 16-element arrays
    return boosted_population, boostable_population


# Apply booster vaccination to susceptible compartment, assuming no effect on infected/removed individuals
def vaccinate_population(S, strategy_num, stratified_population, population):

    # Isolate the susceptible disease status
    S_update = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in S.items()
    }

    unboosted_status = ("U", "F", "W1", "W2")
    boosted_pop, boostable_pop = vaccine_strategy(strategy_num, stratified_population, population)

    # Calculate the proportion of each age group that gets boosted
    proportion = np.divide(
        boosted_pop, boostable_pop, 
        out=np.zeros_like(boosted_pop), 
        where=boostable_pop>0
    )

    # Apply booster
    for status in unboosted_status:
        update = S_update[status] * proportion
        S_update[status] -= update
        S_update["B"] += update

    # Return the updated S as a dict[vaccine_status][16]
    return S_update


##### #####



##### Creating ICs and Parameters


# Initial conditions for each vaccine status
def create_ic(stratified_population):

    y0 = {} 
    S0_data, E01_data, E02_data, E03_data, E04_data, E05_data, I0_data, A0_data, R0_data = stratified_population

    # Initialize y0 as a dict with keys for each vaccine status,
    # and values a concatenated list of each initial age structure for each diseases status
    for i, key in enumerate(vaccine_status):
        S0 = S0_data[:, i].ravel()
        E01 = E01_data[:, i].ravel()
        E02 = E02_data[:, i].ravel()
        E03 = E03_data[:, i].ravel()
        E04 = E04_data[:, i].ravel()
        E05 = E05_data[:, i].ravel()
        I0 = I0_data[:, i].ravel()
        A0 = A0_data[:, i].ravel()
        y0[key] = np.concatenate([S0, E01, E02, E03, E04, E05, I0, A0])
    y0["R"] = R0_data[:, 0].ravel()

    # Convert y into a numeric numpy array for odeint
    y0_array = np.concatenate([y0[key] for key in vaccine_status] + [y0["R"]])

    # Return IC as a dict[vaccine_status/R][128(16*8)/16]
    return y0_array


# Produce the parameters for use within the simulation
def build_sim_parameters(R, data_path, country):

    mu, nu, rho, phi = gp.build_vaccine_efficacy(R, data_path)

    omega, alpha, gamma, tau, w = gp.build_global_parameters(R, data_path)

    d, extended_d, beta, P_IH, P_HD = cp.build_risk_data(R, data_path, country)

    # Sample from the respective gamma distributions
    th_var = 12.1**2
    th_mean = 0.00001
    theta = th_var / th_mean
    k = th_mean / theta
    time_to_hosp = scipy.stats.gamma(k, scale=theta).pdf(np.arange(1, 31))
    delta_IH = (time_to_hosp / np.sum(time_to_hosp)).tolist()              

    th_var = 12.1**2
    th_mean = 10
    theta = th_var / th_mean
    k = th_mean / theta
    time_to_hosp = scipy.stats.gamma(k, scale=theta).pdf(np.arange(1, 31))
    delta_HD = (time_to_hosp / np.sum(time_to_hosp)).tolist()                  

    # See S1_Table for data types
    return (alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD), extended_d


##### #####



##### Building SEIAR Model #####


# Define the age-structured SEIAR model for a given country
def SEIAR_age_country(y, t, parameters, population, contact_matrix):

    alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD = parameters
    a = len(age_groups)

    # Initialize S, E, I, A as defaultdicts
    S, E1, E2, E3, E4, E5, I, A = ({} for i in range(8))
    
    # Unpack the state vector y into the corresponding compartments for each vaccine status
    offset = 0
    for key in vaccine_status:
        status_state = y[offset:offset + 8 * a].reshape(8, a)
        S[key], E1[key], E2[key], E3[key], E4[key], E5[key], I[key], A[key] = status_state
        offset += 8 * a
    R = y[offset:offset + a]

    # Calculate the overall force of infection
    infectious_prevalence = sum(
        (I[key] + tau * A[key]) / population
        for key in vaccine_status
    )
    plambda = beta * omega * (
        contact_matrix @ infectious_prevalence
    )

    # Evaluate all the derivatives for each compartment and vaccine status
    dS_dt, dE1_dt, dE2_dt, dE3_dt, dE4_dt, dE5_dt, dI_dt, dA_dt = ({} for i in range(8))
    dS_dt["F"] = - nu["F"] * plambda * S["F"] - w * S["F"]
    dS_dt["B"] = - nu["B"] * plambda * S["B"] - w * S["B"]
    dS_dt["W1"] = - nu["W1"] * plambda * S["W1"] - w * S["W1"] + w * S["B"] + w * R
    dS_dt["W2"] = - nu["W2"] * plambda * S["W2"] - w * S["W2"] + w * S["W1"] + w * S["F"]
    dS_dt["U"] = - plambda * S["U"] + w * S["W2"]
    for key in vaccine_status:
        dE1_dt[key] = nu[key] * plambda * S[key] - alpha * E1[key]
        dE2_dt[key], dE3_dt[key], dE4_dt[key], dE5_dt[key] = alpha * (E1[key]-E2[key]), alpha * (E2[key]-E3[key]), alpha * (E3[key]-E4[key]), alpha * (E4[key]-E5[key])
        dI_dt[key] = mu[key] * d * alpha * E5[key] - gamma * I[key]
        dA_dt[key] = (1 - mu[key] * d) * alpha * E5[key] - gamma * A[key]
    dR_dt = gamma * (I["U"] + A["U"] + I["F"] + A["F"] + I["B"] + A["B"] + I["W1"] + A["W1"] + I["W2"] + A["W2"]) - w * R

    # Convert back into numpy array for odeint
    dy_dt = []
    for key in vaccine_status:
        dy_dt = np.concatenate([dy_dt, dS_dt[key], dE1_dt[key], dE2_dt[key], dE3_dt[key], dE4_dt[key], dE5_dt[key], dI_dt[key], dA_dt[key]])
    dy_dt = np.concatenate([dy_dt, dR_dt])

    # Return a 128(16*8) + 16 element list
    return dy_dt


##### #####



##### Running the model and plotting results #####


# Calculate solutions of the ODE system
def solve_model(stratified_population, parameters, population, contact_matrix, strategy_number, booster_delay):

    a = len(age_groups)

    # Create initial conditions and parameters
    y0_array = create_ic(stratified_population)

    # Timescale
    t1 = np.linspace(0, booster_delay + 1, booster_delay + 1)
    t2 = np.linspace(booster_delay, total_days, total_days - booster_delay)

    # Solve the ODE system using odeint, up until the booster day
    solution_pre_booster = spi.odeint(
        SEIAR_age_country,
        y0_array,
        t1,
        args = (parameters, population, contact_matrix),
        rtol=1e-10,
        atol=1e-12,
    )

    # Unpack the solution into a dictionary for each vaccine status and the removed compartment
    solutions_dict1 = {}
    for key in vaccine_status:
        status_index = list(vaccine_status).index(key)
        solutions_key1 = solution_pre_booster[:, status_index * 8 * a:(status_index + 1) * 8 * a]
        solutions_dict1[key] = solutions_key1.reshape(len(t1), 8, a)
    solutions_dict1["R"] = solution_pre_booster[:, len(vaccine_status) * 8 * a:].reshape(len(t1), a)

    # Apply booster to the final state
    S_dict = {}
    for key in vaccine_status:
        S_dict[key] = solutions_dict1[key][-1, 0, :]
    S_boost = vaccinate_population(S_dict, strategy_number, stratified_population, population)

    # Intialise a new initial population array based on the boosted previous final state
    y1 = {}
    for key in vaccine_status:
        S1 = S_boost[key]
        E11, E12, E13, E14, E15, I1, A1 = (solutions_dict1[key][-1, i, :] for i in range(1, 8))
        y1[key] = np.concatenate([S1, E11, E12, E13, E14, E15, I1, A1])
    y1["R"] = solutions_dict1["R"][-1, :]
    y1_array = np.concatenate([y1[key] for key in vaccine_status] + [y1["R"]])

    # Solve the ODE system with boosted population
    solution_post_booster = spi.odeint(
        SEIAR_age_country,
        y1_array,
        t2,
        args = (parameters, population, contact_matrix),
        rtol=1e-10,
        atol=1e-12,
    )

    solutions_dict2 = {}
    for key in vaccine_status:
        status_index = list(vaccine_status).index(key)
        solutions_key = solution_post_booster[:, status_index * 8 * a:(status_index + 1) * 8 * a]
        solutions_dict2[key] = solutions_key.reshape(len(t2), 8, a)
    solutions_dict2["R"] = solution_post_booster[:, len(vaccine_status) * 8 * a:].reshape(len(t2), a)

    # Concatenate the pre and post booster solutions
    solutions_dict = {}
    for key in vaccine_status:
        solutions_dict[key] = np.concatenate((solutions_dict1[key], solutions_dict2[key][1:]), axis=0)
    solutions_dict["R"] = np.concatenate((solutions_dict1["R"], solutions_dict2["R"][1:]), axis=0)
    t_plot = np.concatenate((t1[:-1], t2))

    # Return the solutions dict as a dict[vaccine_status/R][151,8,16/151,16], and the timescale as a 150-element array
    return solutions_dict, t_plot


##### #####



##### Calculating hospitalisations and deaths #####


# Calculate hospitalisations and deaths based on the solutions of the SEIAR model
def hosp_death(solutions_dict, parameters):

    # Unpack the parameters for hospitalisation and death
    alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD = parameters

    # Create dictionaries to store time-by-age hospitalisation and death counts
    n_times = len(solutions_dict["U"][:, 0, 0])
    Hdict = {}
    Ddict = {}

    # Define the appropriate convolution
    def delay_convolution(values, kernel):
        result = np.zeros_like(values)
        for a in range(values.shape[1]):
            result[:, a] = np.convolve(values[:, a], kernel, mode='full')[:n_times]
        return result

    # Calculate hospitalisations and deaths for each vaccine status
    for key in vaccine_status:

        E5 = solutions_dict[key][:, 5, :]
        J = mu[key] * d * alpha * E5
        Hdict[key] = rho[key] * P_IH * delay_convolution(J, delta_IH)
        Ddict[key] = phi[key] * P_HD * delay_convolution(Hdict[key], delta_HD)

    # Return 2 dicts[vaccine_status][151,16]
    return Hdict, Ddict


# Calculate the respective total deaths and total yll for each country and strategy
def compare_deaths_yll(total_D_dict, population_dict):

    total_deaths = {
        country: [
            np.sum(
                total_D_dict[country][strategy][key].sum()
                for key in vaccine_status
            ) for strategy in range(6)
        ] for country in countries
    }

    total_yll = {
        country: [
            np.sum(
                np.sum((
                    total_D_dict[country][strategy][key] 
                    for key in vaccine_status
                ), axis=0) * cp.calculate_yll(data_path, country)
            ) for strategy in range(6)
        ] for country in countries
    }

    deaths_per_100k = {
        country: [
            (total_deaths[country][strategy] / np.sum(population_dict[country])) * 100000
        for strategy in range(6)
        ] for country in countries
    }

    yll_per_100K = {
        country: [
            (total_yll[country][strategy] / np.sum(population_dict[country])) * 100000
        for strategy in range(6)
        ] for country in countries
    }

    # Return dicts[country][6]
    return deaths_per_100k, yll_per_100K


##### #####



##### Calculating means across simulations #####


# Simple CI calculator
def calculate_CI(list, confidence):

    upper = np.quantile(list, 0.5 + confidence / 2, axis=0)
    lower = np.quantile(list, 0.5 - confidence / 2, axis=0) 

    return upper, lower


# Calculate the mean population structure across all simulations
def mean_population():

    population_dict = defaultdict()
    
    for country in countries:
    
        country_population = []
    
        for r in regimes:
    
            parameters_regime, extended_d = build_sim_parameters(r, data_path, country) 
            mu_regime = parameters_regime[3]
    
            stratified_population, population = cp.build_population_IC(r, mu_regime, extended_d, data_path, country)
            country_population.append(population)
    
        population_dict[country] = np.mean(np.array(country_population), axis=0)

        # Progress Indicator
        print(f"Mean population for {country} found")

    # Return dict[country][16]
    return population_dict


# Calculate the mean solution, hospitalisations and deaths across all simulations
def mean_CI_solution():

    total_solutions_dict = defaultdict()
    total_H_dict = defaultdict()
    total_D_dict = defaultdict()

    for country in countries:
    
            total_solutions_country = []
            total_H_country = []
            total_D_country = []
    
            for strategy_number in range(1, 7):
    
                total_solutions_strategy = []
                total_H_strategy = []
                total_D_strategy = []
    
                for r in regimes:

                    # utilise the Syrian parameters to follow Bouros model
                    P_IH_syria, P_HD_syria = cp.build_syria_data(r, data_path, country)
    
                    parameters_regime, extended_d = build_sim_parameters(r, data_path, country) 
                    mu_regime = parameters_regime[3]
    
                    parameters_regime = list(parameters_regime)  # Convert to list for mutability
                    parameters_regime[11] = P_IH_syria
                    parameters_regime[13] = P_HD_syria

                    stratified_population, population = cp.build_population_IC(r, mu_regime, extended_d, data_path, country)

                    contact_matrix = cp.build_contact_data(r, data_path, country)

                    # Assuming instantaneous boosters
                    booster_delay=0
    
                    solutions_dict_regime, t_plot = solve_model(stratified_population, parameters_regime, population, contact_matrix, strategy_number, booster_delay)
                    Hdict_regime, Ddict_regime = hosp_death(solutions_dict_regime, parameters_regime)
    
                    total_solutions_strategy.append(solutions_dict_regime)
                    total_H_strategy.append(Hdict_regime)
                    total_D_strategy.append(Ddict_regime)

                # Find the mean and CI for relevant solutions across the simulations
                mean_solutions_dict = {key: np.mean([total_solutions_strategy[r-1][key] for r in regimes], axis=0) for key in vaccine_status} | {"R": np.mean([total_solutions_strategy[r-1]["R"] for r in regimes], axis=0)}
                mean_H_dict = {key: np.mean([total_H_strategy[r-1][key] for r in regimes], axis=0) for key in vaccine_status}
                mean_D_dict = {key: np.mean([total_D_strategy[r-1][key] for r in regimes], axis=0) for key in vaccine_status}
    
                CI_solutions_dict = {key: calculate_CI([total_solutions_strategy[r-1][key] for r in regimes], 0.95) for key in vaccine_status} | {"R": calculate_CI([total_solutions_strategy[r-1]["R"] for r in regimes], 0.95)}
                CI_H_dict = {key: calculate_CI([total_H_strategy[r-1][key] for r in regimes], 0.95) for key in vaccine_status}
                CI_D_dict = {key: calculate_CI([total_D_strategy[r-1][key] for r in regimes], 0.95) for key in vaccine_status}

                # Combine the mean and CI into a single structure for each strategy
                sol_mean_CI_strategy = [mean_solutions_dict, CI_solutions_dict]
                hosp_mean_CI_strategy = [mean_H_dict, CI_H_dict]
                death_mean_CI_strategy = [mean_D_dict, CI_D_dict]

                # Append the mean and CI for the current strategy to the overall list for the country
                total_solutions_country.append(sol_mean_CI_strategy)
                total_H_country.append(hosp_mean_CI_strategy)
                total_D_country.append(death_mean_CI_strategy)

                # Progress indicator
                print(f"Completed simulations for {country} under strategy {strategy_number}")

            # Add the solutions for the country to a dict
            total_solutions_dict[country] = total_solutions_country
            total_H_dict[country] = total_H_country
            total_D_dict[country] = total_D_country

    # Return Solutions dict[country][strategy][mean/CI][vaccine_status/R][151, 8, 16/151, 16]
    # Return H_dict, D_dict[country][strategy][mean/CI][vaccine_status][151, 16]
    return total_solutions_dict, total_H_dict, total_D_dict

##### #####



##### Saving results to CSV #####


# Save the solutions to CSV
def save_sol_to_csv(total_solutions_dict, total_H_dict, total_D_dict):

    for country in countries:
        for strategy_number in range(1, 7):
            for vaccine_status_key in vaccine_status:

                mean_solutions = total_solutions_dict[country][strategy_number - 1][0][vaccine_status_key]
                mean_H = total_H_dict[country][strategy_number - 1][0][vaccine_status_key]
                mean_D = total_D_dict[country][strategy_number - 1][0][vaccine_status_key]

                # Save mean solutions to CSV
                for i, disease_state in enumerate(["S", "E1", "E2", "E3", "E4", "E5", "I", "A"]):
                    mean_df = pd.DataFrame(mean_solutions[:, i, :], columns=age_groups)
                    mean_df.to_csv(f"/home/teasdale/Documents/Age-structured SEIR models/results/{country}/mean_SEIAR_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}_Disease:{disease_state}.csv", index=False)

                # Save mean hospitalisations to CSV
                mean_H_df = pd.DataFrame(mean_H, columns=age_groups)
                mean_H_df.to_csv(f"/home/teasdale/Documents/Age-structured SEIR models/results/{country}/mean_hospitalisations_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}.csv", index=False)

                # Save mean deaths to CSV
                mean_D_df = pd.DataFrame(mean_D, columns=age_groups)
                mean_D_df.to_csv(f"/home/teasdale/Documents/Age-structured SEIR models/results/{country}/mean_deaths_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}.csv", index=False)


# Save the population structure to CSV
def save_pop_to_csv(mean_population_dict):

     for country in countries:
        mean_pop_df = pd.DataFrame(mean_population_dict[country])
        mean_pop_df.to_csv(f"/home/teasdale/Documents/Age-structured SEIR models/results/{country}/mean_population_Country:{country}.csv", index=False)


##### #####