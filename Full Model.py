import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from collections import defaultdict
import math as math
import pandas as pd



##### Country Populations #####

# Putting countries and populations in a dictionary
def build_countries(population_data):
    countries = defaultdict(lambda: np.zeros(a))

    # Mapping .csv codes onto age groups
    age_group_map = {
        f"SP.POP.{code}": i
        for i, code in enumerate([
            "0004", "0509", "1014", "1519", "2024",
            "2529", "3034", "3539", "4044", "4549",
            "5054", "5559", "6064", "6569", "7074",
        ])
    } 
    age_group_map["SP.POP.7579"] = 15
    age_group_map["SP.POP.80UP"] = 15

    # Compiling population structure in a dictionary
    for _, row in population_data.iterrows():
        country_code = row["Country Code"]
        series_code = row["Series Code"]
        if pd.isna(series_code):
            continue
        age_code =  series_code.rsplit(".", 1)[0]
        age_index = age_group_map.get(age_code)
        if age_index is None:
            continue
        population = pd.to_numeric(row["2023 [YR2023]"], errors="raise")
        countries[country_code][age_index] += population

    return countries

# Calculating vaccinated population for each strategy
def vaccine_strategy(n, countries):

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
    boosted_dict = {}
    for country, population in countries.items():
        population = np.asarray(population, dtype=float)
        uptake = 0.9 * population
        supply = 0.1 * sum(population)
        boosted = np.zeros(a)
        for tier in priority_tiers[n]:
            tier_uptake = sum(uptake[tier])
            tier_doses = min(supply, tier_uptake)
            for group in tier:
                boosted[group] = (tier_doses * (population[group] / sum(population[tier])))
            supply -= tier_doses
            if supply <= 0:
                break
        boosted_dict[country] = boosted

    return boosted_dict

# Plotting country population distributions with boosted proportions
def plot_population_graph(countries, country_code):
    age_labels = [
        "0-4", "5-9", "10-14", "15-19",
        "20-24", "25-29", "30-34", "35-39",
        "40-44", "45-49", "50-54", "55-59",
        "60-64", "65-69", "70-74", "75+",
    ]
    ncols, nrows = 3, 2       
    angles = np.linspace(0, 2 * np.pi, a, endpoint=False)        
    bar_width = 2 * np.pi / a * 0.85
    fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw={"projection": "polar"}, squeeze=False
    )
    for ax, vaccine_strategy_num in zip(axes.flat, range(1, 7)):
        population_by_age = np.asarray(countries[country_code])
        vaccinated_by_age = np.asarray(vaccine_strategy(vaccine_strategy_num, countries)[country_code])
        values = np.asarray(population_by_age, dtype=float)
        ax.bar(angles, values, width=bar_width, color = "white", edgecolor="none")
        ax.bar(angles, vaccinated_by_age, width=bar_width, color="cornflowerblue", edgecolor="none")
        ax.bar(angles, values, width=bar_width, facecolor="none", edgecolor="black", linewidth=0.8,)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(age_labels, fontsize=7)
        ax.set_ylim(0, max(values.max(), 1) * 1.15)
        ax.set_yticks([])       # Remove radial (y-axis) tick labels
        ax.grid(False)
        ax.set_title(f"Strategy {vaccine_strategy_num}")
    fig.suptitle(f"{country_code} - Vaccinated populations under different strategies")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country_code}/population_distribution_{country_code}.png", dpi=300)
    #plt.show()
    plt.close(fig)

##### #####

##### Contact Matricies #####

# Build the contact matricies for each country requied
def contact_matrix(countries, contact_data):
    # Create country-matrix dictionaries for each type of contact matrix
    # Each dictionary will have country codes as keys and 256-element arrays (16x16 matrices) as values
    home_dict, work_dict, school_dict, others_dict, all_dict = (defaultdict(lambda: []) for i in range(5))
    for _, row in contact_data.iterrows():
        country_code = row["iso3c"]
        setting = row["setting"]
        if country_code in countries.keys() and setting == "overall":
            if row["location_contact"] == "home":
                home_dict[country_code].append(row["mean_number_of_contacts"])
            elif row["location_contact"] == "work":
                work_dict[country_code].append(row["mean_number_of_contacts"])
            elif row["location_contact"] == "school":
                school_dict[country_code].append(row["mean_number_of_contacts"])
            elif row["location_contact"] == "others":
                others_dict[country_code].append(row["mean_number_of_contacts"])
            elif row["location_contact"] == "all":
                all_dict[country_code].append(row["mean_number_of_contacts"])
    
    # Convert the lists to numpy matrices for each country
    for d in [home_dict, work_dict, school_dict, others_dict, all_dict]:
        for k in d:
            d[k] = np.array(d[k])
            d[k] = d[k].reshape(a, a)
    
    return home_dict, work_dict, school_dict, others_dict, all_dict

def plot_contact_heatmap(contact_matricies, country_code):
    # Plot the different contact matrices for a given country as subplot heatmaps
    home_matrix, work_matrix, school_matrix, others_matrix, all_matrix = (
        contact_matricies[0][country_code], 
        contact_matricies[1][country_code], 
        contact_matricies[2][country_code], 
        contact_matricies[3][country_code],
        contact_matricies[4][country_code]
    )

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.delaxes(axes[5])

    # Plot each contact matrix
    axes[0].imshow(home_matrix, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Home')
    axes[1].imshow(work_matrix, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Work')
    axes[2].imshow(school_matrix, cmap='viridis', interpolation='nearest')
    axes[2].set_title('School')
    axes[3].imshow(others_matrix, cmap='viridis', interpolation='nearest')
    axes[3].set_title('Others')
    axes[4].imshow(all_matrix, cmap='viridis', interpolation='nearest')
    axes[4].set_title('All')
    for ax in axes:
        ax.set_xticks(ticks=np.arange(a), labels=[f'{5*i} - {5*i + 4}' for i in range(a)], rotation=45)
        ax.set_yticks(ticks=np.arange(a), labels=[f'{5*i} - {5*i + 4}' for i in range(a)], rotation=45)
        ax.tick_params(axis="both", labelsize=7)
    fig.suptitle(f'Contact Matrices for {country_code}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country_code}/contact_matrices_{country_code}.png", dpi=300)
    #plt.show()
    plt.close(fig)

##### #####

##### Adding booster vaccination strategy #####

# Apply booster vaccination to susceptible compartment, assuming no effect on infected/removed individuals
def vaccinate_population(S, strategy_num, countries, country_code):

    # Isolate unboosted statuses and boosted population numbers
    unboosted_status = ("F", "W1", "W2", "U")
    boosted_pop = vaccine_strategy(strategy_num, countries)

    # For each age group, apply booster to S, moving a proportion from each status to the boosted status
    for i in range(len(countries[country_code])):
        booster_proportion = boosted_pop[country_code][i] / countries[country_code][i]
        for status in unboosted_status:
                    S["B"][i] = S["B"][i] + booster_proportion * S[status][i]
                    S[status][i] = (1 - booster_proportion) * S[status][i]

    return S

##### #####

##### Building SEIAR Model #####

# Vaccine status with reduced susceptibility level, and prevalence
vaccine_status = {"F": 0.0, "B": 0.0, "W1": 0.0, "W2": 0.0, "U": 1.0}

# Define the age-structured SEIAR model for a given country
def SEIAR_age_country(y, t, parameters, vaccine_status, countries, all_matrix, country_code):
    # Unpack the parameters
    alpha, beta, gamma, mu, nu, tau, omega, d, w, P_IH, rho, delta_IH, P_HD, phi, delta_HD = parameters

    # Initialize S, E, I, A as defaultdicts of numpy arrays
    S, E1, E2, E3, E4, E5, I, A = ({} for i in range(8))
    
    # Unpack the state vector y into the corresponding compartments for each vaccine status
    offset = 0
    for key in vaccine_status.keys():
        status_state = y[offset:offset + 8 * a].reshape(8, a)
        S[key], E1[key], E2[key], E3[key], E4[key], E5[key], I[key], A[key] = status_state
        offset += 8 * a
    R = y[offset:offset + a]

    # Calculate the overall force of infection
    plambda = beta * omega / countries[country_code] * sum(
        all_matrix @ (I[key] + tau * A[key]) for key in vaccine_status.keys()
    )

    # Evaluate all the derivatives for each compartment and vaccine status
    dS_dt, dE1_dt, dE2_dt, dE3_dt, dE4_dt, dE5_dt, dI_dt, dA_dt = ({} for i in range(8))  # Initialize derivatives as defaultdicts of numpy array
    dS_dt["F"] = - nu["F"] * plambda * S["F"] - w * S["F"]
    dS_dt["B"] = - nu["B"] * plambda * S["B"] - w * S["B"]
    dS_dt["W1"] = - nu["W1"] * plambda * S["W1"] - w * S["W1"] + w * S["B"] + w * R
    dS_dt["W2"] = - nu["W2"] * plambda * S["W2"] - w * S["W2"] + w * S["W1"] + w * S["F"]
    dS_dt["U"] = - plambda * S["U"] + w * S["W2"]
    for key in vaccine_status.keys():
        dE1_dt[key] = nu[key] * plambda * S[key] - alpha * E1[key]
        dE2_dt[key], dE3_dt[key], dE4_dt[key], dE5_dt[key] = alpha * (E1[key]-E2[key]), alpha * (E2[key]-E3[key]), alpha * (E3[key]-E4[key]), alpha * (E4[key]-E5[key])
        dI_dt[key] = mu[key] * d * alpha * E5[key] - gamma * I[key]
        dA_dt[key] = (1 - mu[key] * d) * alpha * E5[key] - gamma * A[key]
    dR_dt = gamma * (I["U"] + A["U"] + I["F"] + A["F"] + I["B"] + A["B"] + I["W1"] + A["W1"] + I["W2"] + A["W2"]) - w * R

    # Convert back into numpy array for odeint
    dy_dt = []
    for key in vaccine_status.keys():
        dy_dt = np.concatenate([dy_dt, dS_dt[key], dE1_dt[key], dE2_dt[key], dE3_dt[key], dE4_dt[key], dE5_dt[key], dI_dt[key], dA_dt[key]])
    dy_dt = np.concatenate([dy_dt, dR_dt])

    return dy_dt

# Initial conditions for each vaccine status
def create_ic(countries, country_code):
    y0 = {}  # Initialize y as a dictionary of numpy arrays
    for key in vaccine_status.keys():
        S0 = vaccine_status[key]*np.full(a, 0.85)*countries[country_code]
        E01 = vaccine_status[key]*np.full(a, 0.01)*countries[country_code]
        E02 = vaccine_status[key]*np.full(a, 0.01)*countries[country_code]
        E03 = vaccine_status[key]*np.full(a, 0.01)*countries[country_code]
        E04 = vaccine_status[key]*np.full(a, 0.01)*countries[country_code]
        E05 = vaccine_status[key]*np.full(a, 0.01)*countries[country_code]
        I0 = vaccine_status[key]*np.full(a, 0.05)*countries[country_code]
        A0 = vaccine_status[key]*np.full(a, 0.05)*countries[country_code]
        y0[key] = np.concatenate([S0, E01, E02, E03, E04, E05, I0, A0])
    y0["R"] = np.zeros(a)  # Initialize R as an array of zeros for each age group

    # Convert y into a numeric numpy array for odeint
    y0_array = np.concatenate([y0[key] for key in vaccine_status.keys()] + [y0["R"]])

    return y0_array

# Produce the parameters for use within the simulation
def build_sim_parameters():
    gamma_pdf = lambda x, shape, scale: (x ** (shape - 1) * np.exp(-x / scale)) / (scale ** shape * math.gamma(shape))

    alpha = 1                   # Rate of progression through exposed states
    beta = np.array(
        [0.611, 0.354, 0.4, 0.432, 0.608, 0.7, 0.715, 0.69, 0.708, 0.766, 0.896, 1.002, 1.046, 1.307, 1.387, 1.416]
    )                           # Age-specific transmission rates
    gamma = 1/9                 # Recovery rate
    mu = {
        "F": 0.857, "B": 0.667, "W1": 0.75, "W2": 0.842, "U": 1.0
    }                           # Immune-dependant risk of symptomatic infection
    nu = {
        "F": 0.7, "B": 0.6, "W1": 0.8, "W2": 0.95, "U": 1.0
    }                           # Immune-dependant risk of infection
    tau = 0.255                 # Relative infectiousness of asymptomatic individuals
    omega = 0.304               # Overall infectiousness - why necessary if lumped in with beta? surely would lead to errors in fitting?
    d = np.array(
        [0.068, 0.015, 0.021, 0.026, 0.067, 0.098, 0.104, 0.094, 0.101, 0.125, 0.193, 0.261, 0.293, 0.539, 0.633, 0.678]
    )                           # Probablility of symptomatic infection for each age group
    w = 1/90                    # Rate of waning immunity
    P_IH = np.array(
        [0.0045, 0.0044, 0.0024, 0.0020, 0.0016, 0.0011, 0.0017, 0.0024, 0.0031, 0.0042, 0.0043, 0.0041, 0.0055, 0.0065, 0.0064,0.0069] 
    )                           # Probablilty an age group is hospitalised
    rho = {
        "F": 0.286, "B": 0.167, "W1": 0.188, "W2": 0.421, "U": 1.0
    }                           # Immunity dependant hospitalisation rates
    shape_IH = (0.0001/ 12.1)**2  
    scale_IH = 12.1**2 / 0.0001 # Parameters for the IH gamma distribution
    delta_IH = [        
        gamma_pdf(q, shape=shape_IH, scale=scale_IH) for q in range(1, 31)
    ] / sum([
        gamma_pdf(q, shape=shape_IH, scale=scale_IH) for q in range(1, 31)
    ])                          # Probability distribution of time from infection to hospitalisation, normalised
    P_HD = np.array(
        [0.0011, 0.00075, 0.0010, 0.0055, 0.0064, 0.013, 0.012, 0.013, 0.020, 0.022, 0.034, 0.061, 0.10, 0.098, 0.18, 0.30]
    )                           # Probability an age group dies after hospitalisation
    phi = {
        "F": 1, "B": 0.5, "W1": 1, "W2": 1, "U": 1
    }                           # Immunity dependant death rates
    shape_HD = (10 / 12.1)**2
    scale_HD = 12.1**2 / 10     # Parameters for the HD gamma distribution
    delta_HD = [
        gamma_pdf(q, shape=shape_HD, scale=scale_HD) for q in range(1, 31)
    ] / sum([
        gamma_pdf(q, shape=shape_HD, scale=scale_HD) for q in range(1, 31)
    ])                          # Probability distribution of time from hospitalisation to death, normalised

    return alpha, beta, gamma, mu, nu, tau, omega, d, w, P_IH, rho, delta_IH, P_HD, phi, delta_HD

##### #####

##### Running the model and plotting results #####

# Calculate solutions of the ODE system

def solve_model(countries, all_matrix, country_code, strategy_number, booster_delay):
    
    # Create initial conditions and parameters
    y0_array = create_ic(countries, country_code)
    parameters = build_sim_parameters()

    # Timescale
    t1 = np.arange(0, booster_delay + 1)
    t2 = np.arange(booster_delay, 200)

    # Solve the ODE system using odeint, up until the booster day
    solution_pre_booster = spi.odeint(
        SEIAR_age_country,
        y0_array,
        t1,
        args = (parameters, vaccine_status, countries, all_matrix, country_code),
        rtol=1e-10,
        atol=1e-12,
    )

    # Unpack the solution into a dictionary for each vaccine status and the removed compartment
    solutions_dict1 = {}
    for key in vaccine_status.keys():
        status_index = list(vaccine_status).index(key)
        solutions_key1 = solution_pre_booster[:, status_index * 8 * a:(status_index + 1) * 8 * a]
        solutions_dict1[key] = solutions_key1.reshape(len(t1), 8, a)
    solutions_dict1["R"] = solution_pre_booster[:, len(vaccine_status) * 8 * a:].reshape(len(t1), a)

    # Apply booster 
    S_dict = {}
    for key in vaccine_status.keys():
        S_dict[key] = solutions_dict1[key][-1, 0, :]
    S_boost = vaccinate_population(S_dict, strategy_number, countries, country_code)

    # Intialise a new initial population based on the boosted previous final state
    y1 = {}
    for key in vaccine_status.keys():
        S1 = S_boost[key]
        E11, E12, E13, E14, E15, I1, A1 = (solutions_dict1[key][-1, i, :] for i in range(1, 8))
        y1[key] = np.concatenate([S1, E11, E12, E13, E14, E15, I1, A1])
    y1["R"] = solutions_dict1["R"][-1, :]
    y1_array = np.concatenate([y1[key] for key in vaccine_status.keys()] + [y1["R"]])

    solution_post_booster = spi.odeint(
        SEIAR_age_country,
        y1_array,
        t2,
        args = (parameters, vaccine_status, countries, all_matrix, country_code),
        rtol=1e-10,
        atol=1e-12,
    )

    solutions_dict2 = {}
    for key in vaccine_status.keys():
        status_index = list(vaccine_status).index(key)
        solutions_key = solution_post_booster[:, status_index * 8 * a:(status_index + 1) * 8 * a]
        solutions_dict2[key] = solutions_key.reshape(len(t2), 8, a)
    solutions_dict2["R"] = solution_post_booster[:, len(vaccine_status) * 8 * a:].reshape(len(t2), a)

    solutions_dict = {}
    for key in vaccine_status.keys():
        solutions_dict[key] = np.concatenate((solutions_dict1[key], solutions_dict2[key][1:]), axis=0)
    solutions_dict["R"] = np.concatenate((solutions_dict1["R"], solutions_dict2["R"][1:]), axis=0)
    t_plot = np.concatenate((t1, t2[1:]))

    # Plot the SIR graphs for each vaccine status and the removed compartment, with subplots for each age group
    plot_SIR_graphs(solutions_dict, country_code, t_plot)

    return solutions_dict, t_plot

# Plotting SIR graphs
def plot_SIR_graphs(solutions_dict, country_code, t):

    for key in vaccine_status.keys():
        S, E1, E2, E3, E4, E5, I, A = solutions_dict[key].transpose(1, 0, 2)
        E = E1 + E2 + E3 + E4 + E5
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(a-1):
            ax[i // 4, i % 4].plot(t, S[:, i], label="Susceptible", color="blue")
            ax[i // 4, i % 4].plot(t, E[:, i], label="Exposed", color="orange")
            ax[i // 4, i % 4].plot(t, I[:, i], label="Infected", color="red")
            ax[i // 4, i % 4].plot(t, A[:, i], label="Asymptomatic", color="purple")
            ax[i // 4, i % 4].set_title(f"Ages {5 * i} - {5 * i + 4}")
            ax[i // 4, i % 4].set_xlabel("Time (days)")
            ax[i // 4, i % 4].set_ylabel("Proportion of Population")
            ax[i // 4, i % 4].grid(True)
        ax[3, 3].plot(t, S[:, 15], label="Susceptible", color="blue")
        ax[3, 3].plot(t, E[:, 15], label="Exposed", color="orange")
        ax[3, 3].plot(t, I[:, 15], label="Infected", color="red")
        ax[3, 3].plot(t, A[:, 15], label="Asymptomatic", color="purple")
        ax[3, 3].set_title("Ages 75+")
        ax[3, 3].set_xlabel("Time (days)")
        ax[3, 3].set_ylabel("Proportion of Population")
        ax[3, 3].grid(True)
        fig.suptitle(f"SEIAR Model Dynamics in {country_code} for Vaccine Status: {key}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country_code}/SEIAR_{country_code}_{key}.png", dpi=300)
        #plt.show()
        plt.close(fig)

    R = solutions_dict["R"]
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(a):
        ax[i // 4, i % 4].plot(t, R[:, i], label="Removed", color="green")
        ax[i // 4, i % 4].set_title(f'Ages {5 * i} - {5 * i + 4}')
        ax[i // 4, i % 4].set_xlabel('Time (days)')
        ax[i // 4, i % 4].set_ylabel('Proportion of Population')
        ax[i // 4, i % 4].grid(True)
    ax[3, 3].plot(t, R[:, 15], label="Removed", color="green")
    ax[3, 3].set_title("Ages 75+")
    ax[3, 3].set_xlabel("Time (days)")
    ax[3, 3].set_ylabel("Proportion of Population")
    ax[3, 3].grid(True)
    fig.suptitle(f"Removed Compartment Dynamics in {country_code}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country_code}/SEIAR_{country_code}_R.png", dpi=300)
    #plt.show()
    plt.close(fig)

##### #####

##### Calculating hospitalisations and deaths #####

# Calculate hospitalisations and deaths based on the solutions of the SEIAR model
def hosp_death(solutions_dict):

    # Unpack the parameters for hospitalisation and death
    alpha, beta, gamma, mu, nu, tau, omega, d, w, P_IH, rho, delta_IH, P_HD, phi, delta_HD = build_sim_parameters()

    # Create dictionaries to store time-by-age hospitalisation and death counts
    n_times = len(solutions_dict["U"][:, 0, 0])
    Jdict = {}
    Hdict = {}
    Ddict = {}

    # Calculate hospitalisations and deaths for each vaccine status
    for key in vaccine_status.keys():
        Jdict[key] = np.zeros((n_times, a))
        Hdict[key] = np.zeros((n_times, a))
        Ddict[key] = np.zeros((n_times, a))

        for time_index in range(n_times):
            E5 = solutions_dict[key][time_index, 4, :]
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
def plot_hosp_death(Hdict, Ddict, country_code, t, strategy_num, booster_delay):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for key in vaccine_status.keys():
        ax[0].plot(t, Hdict[key].sum(axis=1), label=f"Hospitalisations - {key}")
        ax[1].plot(t, Ddict[key].sum(axis=1), label=f"Deaths - {key}")
    ax[0].set_title(f"Hospitalisations in {country_code} by Vaccine Status")
    ax[0].set_xlabel("Time (days)")
    ax[0].set_ylabel("Number of Hospitalisations")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title(f"Deaths in {country_code} by Vaccine Status")
    ax[1].set_xlabel("Time (days)")
    ax[1].set_ylabel("Number of Deaths")
    ax[1].legend()
    ax[1].grid(True)
    fig.suptitle(f"Hospitalisations and Deaths in {country_code} by Vaccine Status")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"/home/teasdale/Documents/Age-structured SEIR models/graphs/{country_code}/hosp_death_{country_code}_booster_delay_{booster_delay}_strategy_{strategy_num}.png", dpi=300)
    # plt.show()
    plt.close(fig)

##### #####

if __name__ == "__main__":

    # Number of age groups
    a = 16

    # Vaccine status with initial prevalence
    vaccine_status = {"F": 0.1, "B": 0.0, "W1": 0.1, "W2": 0.1, "U": 0.7}

    population_data = pd.read_csv("/home/teasdale/Downloads/P_Data_Extract_From_Population_estimates_and_projections/af73915f-ee54-4038-b432-b97e3df0493e_Data.csv")
    contact_data = pd.read_csv("/home/teasdale/Downloads/kieshaprem-synthetic-contact-matrices-6e0eebc/output/syntheticcontactmatrices2020/synthetic_contacts_2020.csv")

    # Build the population structure for the country
    countries = build_countries(population_data)
    

    for country_code in countries.keys():
    
        # Build the contact matrices for the country
        contact_matricies = contact_matrix(countries, contact_data)
        all_matrix = contact_matricies[4][country_code]

        plot_population_graph(countries, country_code)
        plot_contact_heatmap(contact_matricies, country_code)

        for strategy_num in range(1, 7):
            for booster_delay in (5, 10, 20, 50, 100):
                solutions_dict, t_plot = solve_model(countries, all_matrix, country_code, strategy_num, booster_delay)
    
                Hdict, Ddict = hosp_death(solutions_dict)
                plot_hosp_death(Hdict, Ddict, country_code, t_plot, strategy_num, booster_delay)