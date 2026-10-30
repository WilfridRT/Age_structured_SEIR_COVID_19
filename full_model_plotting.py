import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import full_model_simulating as fm
from collections import defaultdict


# Defining the overarching plotting parameters
total_days =  151
age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
    '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+']
countries = ["United Kingdom", "Canada", "Brazil", "South Africa", 
    "Kenya", "Philippines", "Sierra Leone", "Syria"]
vaccine_status = ("U", "F", "B", "W1", "W2")


# Defining the path to the results
results_path = "/home/teasdale/Documents/Age-structured SEIR models/results/"



##### Reading and rebuilding the results files #####


# Reading population CSV
def read_mean_population(results_path, country):

    file_path = os.path.join(
        results_path,
        country,
        f"mean_population_Country:{country}.csv"
    )
    df = pd.read_csv(file_path)
    array = np.asarray(df)

    # Returing a 16-element array
    return array


# Building the mean population dictionary
def rebuild_mean_population_dict(results_path):

    mean_population_dict = defaultdict()
    for country in countries:
        mean_population_dict[country] = read_mean_population(results_path, country)

    # Returning a dict[country][16]
    return mean_population_dict


# Reading SEIAR, hosp and death CSVs
def read_SEIAR_to_dict(results_path, country, strategy_number, vaccine_status_key, disease_state):

    file_path = os.path.join(
        results_path,
        country,
        f"mean_SEIAR_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}_Disease:{disease_state}.csv"
    )
    df = pd.read_csv(file_path)
    array = np.asarray(df)

    # Return a (151,16)-array
    return array


def read_hospitalisations_to_dict(results_path, country, strategy_number, vaccine_status_key):

    file_path = os.path.join(
        results_path,
        country,
        f"mean_hospitalisations_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}.csv"
    )
    df = pd.read_csv(file_path)
    array = np.asarray(df)

    # Return a (151,16)-array
    return array


def read_deaths_to_dict(results_path, country, strategy_number, vaccine_status_key):

    file_path = os.path.join(
        results_path,
        country,
        f"mean_deaths_Country:{country}_Strategy:{strategy_number}_Vaccine:{vaccine_status_key}.csv"
    )
    df = pd.read_csv(file_path)
    array = np.asarray(df)

    # Return a (151,16)-array
    return array


def rebuild_solutions_H_D_dict(results_path):

    total_solutions_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    total_H_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    total_D_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    for country in countries:
        for strategy_number in range(1, 7):
            for vaccine_status_key in vaccine_status:

                # Read the mean values for each disease state
                mean_solutions = np.zeros((total_days, 8, len(age_groups)))
                for i, disease_state in enumerate(["S", "E1", "E2", "E3", "E4", "E5", "I", "A"]):
                    mean_solutions[:, i, :] = read_SEIAR_to_dict(results_path, country, strategy_number, vaccine_status_key, disease_state)

                mean_H = read_hospitalisations_to_dict(results_path, country, strategy_number, vaccine_status_key)
                mean_D = read_deaths_to_dict(results_path, country, strategy_number, vaccine_status_key)

                total_solutions_dict[country][strategy_number - 1][vaccine_status_key] = mean_solutions
                total_H_dict[country][strategy_number - 1][vaccine_status_key] = mean_H
                total_D_dict[country][strategy_number - 1][vaccine_status_key] = mean_D

    # Return solutions_dict, H_dict, D_dict as in full_model.py, but with CI removed
    return total_solutions_dict, total_H_dict, total_D_dict


##### #####



##### Plotting Population #####


# Plotting country population distributions with boosted proportions
def plot_population_graph(population, stratified_population, country):

    ncols, nrows = 3, 2       
    a = len(age_groups)
    angles = np.linspace(0, 2 * np.pi, a, endpoint=False)        
    bar_width = 2 * np.pi / a * 0.85
    fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw={"projection": "polar"}, squeeze=False
    )

    # Plot boosted proportion against boostable and total population
    for ax, strategy_num in zip(axes.flat, range(1, 7)):
        boosted_population, boostable_population = fm.vaccine_strategy(strategy_num, stratified_population, population)
        population_by_age = np.asarray(population)
        boostable_by_age = np.asarray(boostable_population)
        boosted_by_age = np.asarray(boosted_population)
        ax.bar(angles, population_by_age, width=bar_width, color = "white", edgecolor="none")
        ax.bar(angles, boostable_by_age, width=bar_width, color="lightgray", edgecolor="none")
        ax.bar(angles, boosted_by_age, width=bar_width, color="cornflowerblue", edgecolor="none")
        ax.bar(angles, population_by_age, width=bar_width, facecolor="none", edgecolor="black", linewidth=0.8,)
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



##### Plotting Contact Matricies #####


# Plotting heatmaps of the contact matricies
def plot_contact_heatmap(contact_matrix, country):

    a = len(age_groups)

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



##### Plotting results #####


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
        plt.show()
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
    plt.show()
    plt.close(fig)


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


# Plotting total deaths and yll for country and strategy
def plot_deaths_yll(deaths_per_100k, yll_per_100K):
    
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