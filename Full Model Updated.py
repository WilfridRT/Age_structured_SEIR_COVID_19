import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.stats
from collections import defaultdict
import math as math
import pandas as pd
import generate_country_parameters as cp
import generate_global_parameters as gp

total_days =  150
age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
    '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+']
countries = ["United Kingdom", "Canada", "Brazil", "South Africa", 
    "Kenya", "Philippines", "Sierra Leone", "Syria"]
vaccine_status = ("U", "F", "B", "W1", "W2")

# Over 100 simulations (add later)
regimes = np.arange(1, 101, 1).tolist()

# Add folder path to data file
data_path = '/home/teasdale/Downloads/warwick-covid-transmission-main/warwickmodel/data/'

##### Country Populations #####

# Calculating vaccinated population for each strategy
def vaccine_strategy(n, stratified_population, population):

    stratified_population = list(stratified_population)
    S_population = np.sum(stratified_population[0][:, (0, 1, 3, 4)], axis=1)
    R_population = stratified_population[8].flatten()
    boostable_population = S_population + R_population
    
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
    population = np.asarray(population, dtype=float)
    uptake = 0.9 * boostable_population
    supply = 0.1 * sum(population)
    boosted_population = np.zeros(len(population))
    for tier in priority_tiers[n]:
        tier_uptake = sum(uptake[tier])
        tier_doses = min(supply, tier_uptake)
        for group in tier:
            boosted_population[group] = (tier_doses * (population[group] / sum(population[tier])))
        supply -= tier_doses
        if supply <= 0:
            break

    return boosted_population, boostable_population

# Plotting country population distributions with boosted proportions
def plot_population_graph(population, stratified_population, country):

    ncols, nrows = 3, 2       
    a = len(age_groups)
    angles = np.linspace(0, 2 * np.pi, a, endpoint=False)        
    bar_width = 2 * np.pi / a * 0.85
    fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw={"projection": "polar"}, squeeze=False
    )

    for ax, strategy_num in zip(axes.flat, range(1, 7)):
        boosted_population, boostable_population = vaccine_strategy(strategy_num, stratified_population, population)
        population_by_age = np.asarray(population)
        boostable_by_age = np.asarray(boostable_population)
        boosted_by_age = np.asarray(boosted_population)
        ax.bar(angles, boostable_by_age, width=bar_width, color = "white", edgecolor="none")
        ax.bar(angles, boosted_by_age, width=bar_width, color="cornflowerblue", edgecolor="none")
        ax.bar(angles, boostable_by_age, width=bar_width, facecolor="none", edgecolor="black", linewidth=0.8,)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(age_groups, fontsize=7)
        ax.set_ylim(0, max(boostable_by_age.max(), 1) * 1.15)
        ax.set_yticks([])       # Remove radial (y-axis) tick labels
        ax.grid(False)
        ax.set_title(f"Strategy {strategy_num}")
    fig.suptitle(f"{country} - Vaccinated populations under different strategies")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country}/population_distribution_{country}.png", dpi=300)
    plt.show()
    plt.close(fig)

##### #####

##### Contact Matricies #####
def plot_contact_heatmap(contact_matrix, country):

    a = len(age_groups)

    # Plot the contact matrix for a given country as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(contact_matrix, cmap='viridis', interpolation='nearest')
    plt.title(f'Contact Matrix for {country}')
    plt.xlabel('Age Group')
    plt.ylabel('Age Group')
    plt.colorbar(label='Contact Rate')
    plt.xticks(ticks=np.arange(a), labels=age_groups, rotation=45)
    plt.yticks(ticks=np.arange(a), labels=age_groups, rotation=45)
    plt.tick_params(axis="both", labelsize=7)
    plt.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country}/contact_matrix_{country}.png", dpi=300)
    plt.show()
    plt.close(plt.gcf())

##### #####

##### Adding booster vaccination strategy #####

# Apply booster vaccination to susceptible compartment, assuming no effect on infected/removed individuals
def vaccinate_population(S, strategy_num, stratified_population, population):

    # Isolate unboosted statuses and boosted population numbers
    unboosted_status = ("F", "W1", "W2", "U")
    boosted_pop, boostable_pop = vaccine_strategy(strategy_num, stratified_population, population)
    # For each age group, apply booster to S, moving a proportion from each status to the boosted status
    for i in range(len(population)):
        booster_proportion = boosted_pop[i] / population[i]
        for status in unboosted_status:
                    S["B"][i] = S["B"][i] + booster_proportion * S[status][i]
                    S[status][i] = (1 - booster_proportion) * S[status][i]

    return S

##### #####

##### Building SEIAR Model #####

# Define the age-structured SEIAR model for a given country
def SEIAR_age_country(y, t, parameters, population, contact_matrix):
    # Unpack the parameters
    alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD = parameters
    a = len(age_groups)

    # Initialize S, E, I, A as defaultdicts of numpy arrays
    S, E1, E2, E3, E4, E5, I, A = ({} for i in range(8))
    
    # Unpack the state vector y into the corresponding compartments for each vaccine status
    offset = 0
    for key in vaccine_status:
        status_state = y[offset:offset + 8 * a].reshape(8, a)
        S[key], E1[key], E2[key], E3[key], E4[key], E5[key], I[key], A[key] = status_state
        offset += 8 * a
    R = y[offset:offset + a]

    # Calculate the overall force of infection
    #plambda = beta * (omega) / population * sum(
    #    contact_matrix @ (I[key] + tau * A[key]) for key in vaccine_status
    #)

    infectious_prevalence = sum(
        (I[key] + tau * A[key]) / population
        for key in vaccine_status
    )

    plambda = beta * omega * (
        contact_matrix @ infectious_prevalence
    )

    # Evaluate all the derivatives for each compartment and vaccine status
    dS_dt, dE1_dt, dE2_dt, dE3_dt, dE4_dt, dE5_dt, dI_dt, dA_dt = ({} for i in range(8))  # Initialize derivatives as defaultdicts of numpy array
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

    return dy_dt

# Initial conditions for each vaccine status
def create_ic(stratified_population):
    y0 = {}  # Initialize y as a dictionary of numpy arrays
    S0_data, E01_data, E02_data, E03_data, E04_data, E05_data, I0_data, A0_data, R0_data = stratified_population
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

    return y0_array

# Produce the parameters for use within the simulation
def build_sim_parameters(R, data_path, country):

    mu, nu, rho, phi = gp.build_vaccine_efficacy(R, data_path)

    omega, alpha, gamma, tau, w = gp.build_global_parameters(R, data_path)

    d, extended_d, beta, P_IH, P_HD = cp.build_risk_data(R, data_path, country)

    th_var = 12.1**2
    th_mean = 0.00001
    theta = th_var / th_mean
    k = th_mean / theta
    time_to_hosp = scipy.stats.gamma(k, scale=theta).pdf(np.arange(1, 31))
    delta_IH = (time_to_hosp / np.sum(time_to_hosp)).tolist()                      # Probability distribution of time from infection to hospitalisation, normalised

    th_var = 12.1**2
    th_mean = 10
    theta = th_var / th_mean
    k = th_mean / theta
    time_to_hosp = scipy.stats.gamma(k, scale=theta).pdf(np.arange(1, 31))
    delta_HD = (time_to_hosp / np.sum(time_to_hosp)).tolist()                      # Probability distribution of time from hospitalisation to death, normalised

    return (alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD), extended_d

##### #####

##### Running the model and plotting results #####

# Calculate solutions of the ODE system

def solve_model(stratified_population, parameters, population, contact_matrix, strategy_number, booster_delay):

    a = len(age_groups)

    # Create initial conditions and parameters
    y0_array = create_ic(stratified_population)

    # Timescale
    t1 = np.linspace(0, booster_delay + 1, (booster_delay + 1))
    t2 = np.linspace(booster_delay, total_days, (total_days - booster_delay + 1))

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

    # Apply booster 
    S_dict = {}
    for key in vaccine_status:
        S_dict[key] = solutions_dict1[key][-1, 0, :]
    S_boost = vaccinate_population(S_dict, strategy_number, stratified_population, population)

    # Intialise a new initial population based on the boosted previous final state
    y1 = {}
    for key in vaccine_status:
        S1 = S_boost[key]
        E11, E12, E13, E14, E15, I1, A1 = (solutions_dict1[key][-1, i, :] for i in range(1, 8))
        y1[key] = np.concatenate([S1, E11, E12, E13, E14, E15, I1, A1])
    y1["R"] = solutions_dict1["R"][-1, :]
    y1_array = np.concatenate([y1[key] for key in vaccine_status] + [y1["R"]])

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

    solutions_dict = {}
    for key in vaccine_status:
        solutions_dict[key] = np.concatenate((solutions_dict1[key], solutions_dict2[key][1:]), axis=0)
    solutions_dict["R"] = np.concatenate((solutions_dict1["R"], solutions_dict2["R"][1:]), axis=0)
    t_plot = np.concatenate((t1[:-1], t2))

    return solutions_dict, t_plot

# Plotting SIR graphs
def plot_SIR_graphs(solutions_dict, t_plot, country):

    a = len(age_groups)

    for key in vaccine_status:
        S, E1, E2, E3, E4, E5, I, A = solutions_dict[key].transpose(1, 0, 2)
        E = E1 + E2 + E3 + E4 + E5
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(a):
            ax[i // 4, i % 4].plot(t_plot, S[:, i], label="Susceptible", color="blue")
            ax[i // 4, i % 4].plot(t_plot, E[:, i], label="Exposed", color="orange")
            ax[i // 4, i % 4].plot(t_plot, I[:, i], label="Infected", color="red")
            ax[i // 4, i % 4].plot(t_plot, A[:, i], label="Asymptomatic", color="purple")
            ax[i // 4, i % 4].set_title(f"Age Group: {age_groups[i]}")
            ax[i // 4, i % 4].set_xlabel("Time (days)")
            ax[i // 4, i % 4].set_ylabel("Proportion of Population")
            ax[i // 4, i % 4].grid(True)
        fig.suptitle(f"SEIAR Model Dynamics in {country} for Vaccine Status: {key}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country}/SEIAR_{country}_{key}.png", dpi=300)
        #plt.show()
        plt.close(fig)

    R = solutions_dict["R"]
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(a):
        ax[i // 4, i % 4].plot(t_plot, R[:, i], label="Removed", color="green")
        ax[i // 4, i % 4].set_title(f'Age Group: {age_groups[i]}')
        ax[i // 4, i % 4].set_xlabel('Time (days)')
        ax[i // 4, i % 4].set_ylabel('Proportion of Population')
        ax[i // 4, i % 4].grid(True)
    fig.suptitle(f"Removed Compartment Dynamics in {country}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country}/SEIAR_{country}_R.png", dpi=300)
    #plt.show()
    plt.close(fig)

##### #####

##### Calculating hospitalisations and deaths #####

# Calculate hospitalisations and deaths based on the solutions of the SEIAR model
def hosp_death(solutions_dict, parameters):

    # Unpack the parameters for hospitalisation and death
    alpha, beta, gamma, mu, nu, phi, rho, tau, omega, d, w, P_IH, delta_IH, P_HD, delta_HD = parameters
    a = len(age_groups)

    # Create dictionaries to store time-by-age hospitalisation and death counts
    n_times = len(solutions_dict["U"][:, 0, 0])
    Jdict = {}
    Hdict = {}
    Ddict = {}

    # Calculate hospitalisations and deaths for each vaccine status
    for key in vaccine_status:
        Jdict[key] = np.zeros((n_times, a))
        Hdict[key] = np.zeros((n_times, a))
        Ddict[key] = np.zeros((n_times, a))

        for time_index in range(n_times):
            E5 = solutions_dict[key][time_index, 5, :]
            Jdict[key][time_index] = mu[key] * d * alpha * E5

            max_delay = min(time_index, len(delta_IH))
            Hdict[key][time_index] = rho[key] * P_IH * sum(
                Jdict[key][time_index - delay] * delta_IH[delay - 1]
                for delay in range(1, max_delay + 1)
            )

            max_delay = min(time_index, len(delta_HD))
            Ddict[key][time_index] = phi[key] * P_HD * sum(
                Hdict[key][time_index - delay] * delta_HD[delay - 1]
                for delay in range(1, max_delay + 1)
            )

    return Hdict, Ddict

# Plotting hospitalisations and deaths
def plot_hosp_death(Hdict, Ddict, population, t, strategy_num, booster_delay, country):

    population_size = sum(population)
    factor = 1 / population_size * 100000  # Scale to per 100,000 population
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(t, factor * sum(Hdict[key].sum(axis=1) for key in Hdict.keys()), label="Hospitalisations")
    ax[1].plot(t, factor * sum(Ddict[key].sum(axis=1) for key in Ddict.keys()), label="Deaths")
    ax[0].set_title(f"Hospitalisations in {country} by Vaccine Status")
    ax[0].set_xlabel("Time (days)")
    ax[0].set_ylabel("Number of Hospitalisations (per 100k)")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title(f"Deaths in {country} by Vaccine Status")
    ax[1].set_xlabel("Time (days)")
    ax[1].set_ylabel("Number of Deaths (per 100k)")
    ax[1].legend()
    ax[1].grid(True)
    fig.suptitle(f"Hospitalisations and Deaths in {country}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country}/hosp_death_{country}_booster_delay_{booster_delay}_strategy_{strategy_num}.png", dpi=300)
    plt.show()
    plt.close(fig)

def compare_deaths_yll(total_D_dict, population_dict):

    total_deaths = {
        country: [
            np.sum(
                total_D_dict[country][strategy][0][key].sum()
                for key in vaccine_status
            ) for strategy in range(6)
        ] for country in countries
    }

    total_yll = {
        country: [
            np.sum(
                np.sum((
                    total_D_dict[country][strategy][0][key] 
                    for key in vaccine_status
                ), axis=0) * cp.calculate_yll(data_path, country)
            ) for strategy in range(6)
        ] for country in countries
    }

    print(total_yll)

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

    n_strategies = 6
    n_countries = len(countries)
    x = np.arange(n_countries)
    bar_width = 0.13

    colours = [
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:pink"
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy_index in range(n_strategies):
        deaths = [
            deaths_per_100k[country][strategy_index]
            for country in countries
        ]
        offset = (strategy_index - (n_strategies - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            deaths,
            width=bar_width,
            color=colours[strategy_index],
            label=f"Strategy {strategy_index + 1}",
        )
    ax.set_xlabel("Country")
    ax.set_ylabel("Total deaths (per 100k)")
    ax.set_title("Total deaths by country and vaccination strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/deaths_by_country_and_strategy.png", dpi=300)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy_index in range(n_strategies):
        yll = [
            yll_per_100K[country][strategy_index]
            for country in countries
        ]
        offset = (strategy_index - (n_strategies - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            yll,
            width=bar_width,
            color=colours[strategy_index],
            label=f"Strategy {strategy_index + 1}",
        )
    ax.set_xlabel("Country")
    ax.set_ylabel("Total YLL (per 100k)")
    ax.set_title("Total YLL by country and vaccination strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/yll_by_country_and_strategy.png", dpi=300)
    plt.show()
    plt.close(fig)

##### #####

##### CI calculator #####

def calculate_CI(list, confidence):

    upper = np.quantile(list, 0.5 + confidence / 2, axis=0)
    lower = np.quantile(list, 0.5 - confidence / 2, axis=0) 

    return upper, lower

##### #####

##### Calculating means #####

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

        print(f"Mean population for {country} found")

    return population_dict

def mean_CI_solution():

    total_solutions_dict = defaultdict()
    total_H_dict = defaultdict()
    total_D_dict = defaultdict()

    for country in countries:
    
            total_solutions_country = []
            total_H_country = []
            total_D_country = []
    
            for strategy_number in range(1, 7):
    
                solutions_dict = []
                H_dict = []
                D_dict = []
    
                for r in regimes:
    
                    parameters_syria, extended_d_syria = build_sim_parameters(r, data_path, "Syria")
                    P_IH_syria = parameters_syria[11]
                    P_HD_syria = parameters_syria[13]
    
                    parameters_regime, extended_d = build_sim_parameters(r, data_path, country) 
                    mu_regime = parameters_regime[3]
    
                    parameters_regime = list(parameters_regime)  # Convert to list for mutability
                    parameters_regime[11] = P_IH_syria
                    parameters_regime[13] = P_HD_syria
    
                    # Build the population structure for the country
                    stratified_population, population = cp.build_population_IC(r, mu_regime, extended_d, data_path, country)
    
                    # Build the contact matrices for the country
                    contact_matrix = cp.build_contact_data(r, data_path, country)
                    corrected_contact_matrix = cp.correct_contact_matrix(contact_matrix, population)
                    contact_matrix = corrected_contact_matrix
    
                    booster_delay=0
    
                    solutions_dict_regime, t_plot = solve_model(stratified_population, parameters_regime, population, contact_matrix, strategy_number, booster_delay)
                    Hdict_regime, Ddict_regime = hosp_death(solutions_dict_regime, parameters_regime)
    
                    solutions_dict.append(solutions_dict_regime)
                    H_dict.append(Hdict_regime)
                    D_dict.append(Ddict_regime)
    
                mean_solutions_dict = {key: np.mean([solutions_dict[r-1][key] for r in regimes], axis=0) for key in vaccine_status} | {"R": np.mean([solutions_dict[r-1]["R"] for r in regimes], axis=0)}
                mean_H_dict = {key: np.mean([H_dict[r-1][key] for r in regimes], axis=0) for key in vaccine_status}
                mean_D_dict = {key: np.mean([D_dict[r-1][key] for r in regimes], axis=0) for key in vaccine_status}
    
                CI_solutions_dict = {key: calculate_CI([solutions_dict[r-1][key] for r in regimes], 0.95) for key in vaccine_status} | {"R": calculate_CI([solutions_dict[r-1]["R"] for r in regimes], 0.95)}
                CI_H_dict = {key: calculate_CI([H_dict[r-1][key] for r in regimes], 0.95) for key in vaccine_status}
                CI_D_dict = {key: calculate_CI([D_dict[r-1][key] for r in regimes], 0.95) for key in vaccine_status}
    
                country_sol_mean_CI = [mean_solutions_dict, CI_solutions_dict]
                country_hosp_mean_CI = [mean_H_dict, CI_H_dict]
                country_death_mean_CI = [mean_D_dict, CI_D_dict]
    
                total_solutions_country.append(country_sol_mean_CI)
                total_H_country.append(country_hosp_mean_CI)
                total_D_country.append(country_death_mean_CI)
    
                print(f"Completed simulations for {country} under strategy {strategy_number}")
    
            total_solutions_dict[country] = total_solutions_country
            total_H_dict[country] = total_H_country
            total_D_dict[country] = total_D_country

    return total_solutions_dict, total_H_dict, total_D_dict

##### #####

if __name__ == "__main__":

    population_dict = mean_population()

    total_solutions_dict, total_H_dict, total_D_dict = mean_CI_solution()

    compare_deaths_yll(total_D_dict, population_dict)
