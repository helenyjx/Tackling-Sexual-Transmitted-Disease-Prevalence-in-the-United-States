"""Diff-in-Diff Analysis of CDC STD Data."""

import numpy as np
import pandas as pd
import altair as alt
import datetime
import re
import statsmodels.formula.api as smf
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def convert_file_to_df(filepath):
    """Convert file into dataframe from filepath."""
    std_df = pd.read_csv(filepath, skiprows=6)
    std_copy = std_df.copy()
    std_copy = std_copy.replace("2020 (COVID-19 Pandemic)", "2020")
    std_copy["Cases"] = std_copy["Cases"].str.replace(",", "")
    std_copy["Population"] = std_copy["Population"].str.replace(",", "")
    std_copy["Year"] = pd.to_datetime(std_copy["Year"], format="%Y")
    std_copy["Year"] = std_copy["Year"].dt.strftime("%Y")
    std_copy = std_copy[std_copy["Year"] < "2020"]
    cols = ["Cases", "Rate per 100000", "Population"]
    std_copy[cols] = std_copy[cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    return std_copy


std_by_overall_analysis = convert_file_to_df(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Separated by Indicator.csv"""
)
std_by_disease = convert_file_to_df(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Separated by Indicator.csv"""
)
std_by_race = convert_file_to_df(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Separated by Race.csv"""
)
std_by_gender = convert_file_to_df(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Separated by Gender.csv"""
)
std_by_age = convert_file_to_df(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Separated by Age.csv"""
)


select_states = [
    "Georgia",
    "Maryland",
    "New York",
    "Delaware",
    "New Jersey",
    "Arizona",
    "Virginia",
    "Ohio",
    "Pennsylvania",
    "Florida",
    "Louisiana",
    "California",
    "Texas",
    "Illinois",
    "North Carolina",
    "Mississippi",
    "Nevada",
    "Tennessee",
    "Arkansas",
    "New Mexico",
    "Missouri",
]
std_by_overall_analysis_select_states = std_by_overall_analysis.loc[
    std_by_overall_analysis["Geography"].isin(select_states)
]
std_by_disease_select_states = std_by_disease.loc[
    std_by_disease["Geography"].isin(select_states)
]
std_by_race_select_states = std_by_race.loc[
    std_by_race["Geography"].isin(select_states)
]
std_by_gender_select_states = std_by_gender.loc[
    std_by_gender["Geography"].isin(select_states)
]
std_by_age_select_states = std_by_age.loc[
    std_by_age["Geography"].isin(select_states)
]

print(std_by_overall_analysis_select_states.isna().sum())
print(std_by_disease_select_states.isna().sum())
print(std_by_race_select_states.isna().sum())
print(std_by_gender_select_states.isna().sum())
print(std_by_age_select_states.isna().sum())

races = [
    "White",
    "Native Hawaiian/Other Pacific Islander",
    "Multiracial",
    "Hispanic/Latino",
    "Black/African American",
    "Asian",
    "American Indian/Alaska Native",
]

# drop rows with missing values in the 'Cases' column for each race
for race in races:
    std_by_overall_analysis_select_states_no_missing = (
        std_by_overall_analysis_select_states.loc[
            (std_by_overall_analysis_select_states["Cases"].notna())
        ]
    )
    std_by_disease_select_states_no_missing = std_by_disease_select_states.loc[
        (std_by_disease_select_states["Cases"].notna())
    ]
    std_by_race_select_states_no_missing = std_by_race_select_states.loc[
        (std_by_race_select_states["Cases"].notna())
    ]
    std_by_gender_select_states_no_missing = std_by_gender_select_states.loc[
        (std_by_gender_select_states["Cases"].notna())
    ]
    std_by_age_select_states_no_missing = std_by_age_select_states.loc[
        (std_by_age_select_states["Cases"].notna())
    ]

std_by_overall_analysis_select_states_no_missing = (
    std_by_overall_analysis_select_states_no_missing.rename(
        columns={
            "Rate per 100000": "Rate_per_100000",
            "Age Group": "Age",
            "Race/Ethnicity": "Race",
        }
    )
)
std_by_disease_select_states_no_missing = (
    std_by_disease_select_states_no_missing.rename(
        columns={
            "Rate per 100000": "Rate_per_100000",
            "Age Group": "Age",
            "Race/Ethnicity": "Race",
        }
    )
)
std_by_race_select_states_no_missing = (
    std_by_race_select_states_no_missing.rename(
        columns={
            "Rate per 100000": "Rate_per_100000",
            "Age Group": "Age",
            "Race/Ethnicity": "Race",
        }
    )
)
std_by_gender_select_states_no_missing = (
    std_by_gender_select_states_no_missing.rename(
        columns={
            "Rate per 100000": "Rate_per_100000",
            "Age Group": "Age",
            "Race/Ethnicity": "Race",
        }
    )
)
std_by_age_select_states_no_missing = (
    std_by_age_select_states_no_missing.rename(
        columns={
            "Rate per 100000": "Rate_per_100000",
            "Age Group": "Age",
            "Race/Ethnicity": "Race",
        }
    )
)

std_by_overall_analysis_select_states_no_missing[
    "Year"
] = std_by_overall_analysis_select_states_no_missing["Year"].astype(int)
std_by_disease_select_states_no_missing[
    "Year"
] = std_by_disease_select_states_no_missing["Year"].astype(int)
std_by_race_select_states_no_missing[
    "Year"
] = std_by_race_select_states_no_missing["Year"].astype(int)
std_by_gender_select_states_no_missing[
    "Year"
] = std_by_gender_select_states_no_missing["Year"].astype(int)
std_by_age_select_states_no_missing[
    "Year"
] = std_by_age_select_states_no_missing["Year"].astype(int)


def partition_data(
    df, state_col, value_col, test_state, control_states, policy_year
):
    """
    Split data into pre-policy and post-policy.
    Args:
        df (pandas DataFrame): the dataframe containing the data.
        state_col: column containing the state names.
        value_col: values to be analyzed.
        test_state : treatment state.
        control_states (list of str):  control states.
        policy_year (int): the year in which the policy change occurred.
    Returns:
        Four pandas DataFrames, representing the pre-policy and
        post-policy data for the test state and control states.
    """
    data_test = df.loc[
        df[state_col] == test_state, [state_col, "Year", value_col]
    ]
    data_test["treat"] = 1
    test_pre = data_test[data_test["Year"] < policy_year]
    test_post = data_test[data_test["Year"] >= policy_year]

    # Split the control state data into pre-policy and post-policy
    data_control = df.loc[
        df[state_col].isin(control_states), [state_col, "Year", value_col]
    ]
    data_control["treat"] = 0
    control_pre = data_control[data_control["Year"] < policy_year]
    control_post = data_control[data_control["Year"] >= policy_year]

    return test_pre, test_post, control_pre, control_post


def reg_fit(data, color, yvar, xvar, legend, ylabel, alpha=0.05):
    """Fit regression model on diff-in-diff graph."""
    colour = color

    # Filter out missing data
    x = data.loc[data[yvar].notnull(), xvar]

    # Calculate the x-axis range and step size
    xmin, xmax = x.min(), x.max()
    step = (xmax - xmin) / 100

    # Generate a grid of x-axis values for plotting
    grid = np.arange(xmin, xmax + step, step)

    # Generate predictions using the linear regression model
    model = smf.ols(f"{yvar} ~ {xvar}", data=data).fit()
    predictions = pd.DataFrame({xvar: grid})
    predictions[yvar] = model.predict(predictions[xvar])
    ci = model.conf_int(alpha=alpha)
    predictions["ci_low"] = model.get_prediction(predictions[xvar]).conf_int(
        alpha=alpha
    )[:, 0]
    predictions["ci_high"] = model.get_prediction(predictions[xvar]).conf_int(
        alpha=alpha
    )[:, 1]

    # Create a chart with the regression line and confidence interval
    predictions["Treat"] = f"{legend}"

    reg = (
        alt.Chart(predictions)
        .mark_line()
        .encode(
            x=alt.X(
                xvar,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format="T", title="Year"),
            ),
            y=alt.Y(yvar, title=ylabel),
            color=alt.value(colour),
            opacity=alt.Opacity("Treat", legend=alt.Legend(title="Legend")),
        )
    )
    ci = (
        alt.Chart(predictions)
        .mark_errorband(opacity=0.3)
        .encode(
            x=alt.X(xvar, title=xvar),
            y=alt.Y("ci_low", title=ylabel, scale=alt.Scale(zero=False)),
            y2="ci_high",
            color=alt.value(colour),
        )
    )
    # Groups
    grouped_means = data.groupby(xvar, as_index=False)[[yvar]].mean()
    scatter = (
        alt.Chart(grouped_means)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=xvar,
            y=alt.Y(yvar, title=ylabel),
            color=alt.value(colour),
            tooltip=[xvar, yvar],
        )
    )
    chart = ci + reg + scatter

    return predictions, chart


def plotting_chart(
    data, xvar, yvar, legend, policy_year, color, ylabel, alpha=0.05
):
    """Plot a chart with the data and a vertical rule at the policy year."""
    years = list(np.arange(data[xvar].min(), data[xvar].max() + 1))
    pol_year = [int(policy_year)]

    # Plot chart
    fit, reg_chart = reg_fit(
        color=color,
        data=data,
        yvar=yvar,
        xvar=xvar,
        legend=legend,
        ylabel=ylabel,
        alpha=alpha,
    )
    policy = pd.DataFrame({"Year": pol_year})

    rule = (
        alt.Chart(policy)
        .mark_rule(strokeDash=[10, 7], color="black", strokeWidth=3)
        .encode(alt.X("Year:Q", title="Year"))
    )
    return (reg_chart + rule).properties(width=800, height=400)


md_states = [
    "Maryland",
    "Delaware",
    "New Jersey",
    "Arizona",
    "Virginia",
    "Ohio",
    "Pennsylvania",
]
std_by_overall_analysis_md = (
    std_by_overall_analysis_select_states_no_missing.loc[
        std_by_overall_analysis_select_states_no_missing["Geography"].isin(
            md_states
        )
    ]
)
std_by_disease_md = std_by_disease_select_states_no_missing.loc[
    std_by_disease_select_states_no_missing["Geography"].isin(md_states)
]
std_by_race_md = std_by_race_select_states_no_missing.loc[
    std_by_race_select_states_no_missing["Geography"].isin(md_states)
]
std_by_gender_md = std_by_gender_select_states_no_missing.loc[
    std_by_gender_select_states_no_missing["Geography"].isin(md_states)
]
std_by_age_md = std_by_age_select_states_no_missing.loc[
    std_by_age_select_states_no_missing["Geography"].isin(md_states)
]


ny_states = [
    "New York",
    "Florida",
    "Louisiana",
    "California",
    "Texas",
    "Illinois",
]
std_by_overall_analysis_ny = (
    std_by_overall_analysis_select_states_no_missing.loc[
        std_by_overall_analysis_select_states_no_missing["Geography"].isin(
            ny_states
        )
    ]
)
std_by_disease_ny = std_by_disease_select_states_no_missing.loc[
    std_by_disease_select_states_no_missing["Geography"].isin(ny_states)
]
std_by_race_ny = std_by_race_select_states_no_missing.loc[
    std_by_race_select_states_no_missing["Geography"].isin(ny_states)
]
std_by_gender_ny = std_by_gender_select_states_no_missing.loc[
    std_by_gender_select_states_no_missing["Geography"].isin(ny_states)
]
std_by_age_ny = std_by_age_select_states_no_missing.loc[
    std_by_age_select_states_no_missing["Geography"].isin(ny_states)
]

ga_states = [
    "Georgia",
    "North Carolina",
    "Mississippi",
    "Nevada",
    "Tennessee",
    "Arkansas",
    "New Mexico",
    "Missouri",
]
std_by_overall_analysis_ga = (
    std_by_overall_analysis_select_states_no_missing.loc[
        std_by_overall_analysis_select_states_no_missing["Geography"].isin(
            ga_states
        )
    ]
)
std_by_disease_ga = std_by_disease_select_states_no_missing.loc[
    std_by_disease_select_states_no_missing["Geography"].isin(ga_states)
]
std_by_race_ga = std_by_race_select_states_no_missing.loc[
    std_by_race_select_states_no_missing["Geography"].isin(ga_states)
]
std_by_gender_ga = std_by_gender_select_states_no_missing.loc[
    std_by_gender_select_states_no_missing["Geography"].isin(ga_states)
]
std_by_age_ga = std_by_age_select_states_no_missing.loc[
    std_by_age_select_states_no_missing["Geography"].isin(ga_states)
]

overall_std = (
    std_by_overall_analysis_select_states_no_missing.groupby(
        ["Geography", "Year"]
    )["Cases", "Rate_per_100000"]
    .sum()
    .reset_index()
)


def std_analysis_dataframe(dataframe, std_column, std_column_value):
    """Create dataframe for state attribute analysis."""
    overall_std_disease = (
        dataframe.groupby(["Geography", "Year", std_column])[
            "Cases", "Rate_per_100000"
        ]
        .sum()
        .reset_index()
    )
    overall_std_analysis_df = overall_std_disease.loc[
        overall_std_disease[std_column] == std_column_value
    ]
    overall_std_analysis_df = overall_std_analysis_df.drop(
        [std_column], axis=1
    )
    return overall_std_analysis_df


overall_std_chlamydia = std_analysis_dataframe(
    std_by_disease_select_states_no_missing, "Indicator", "Chlamydia"
)
overall_std_gonorrhea = std_analysis_dataframe(
    std_by_disease_select_states_no_missing, "Indicator", "Gonorrhea"
)
overall_std_young_adults = std_analysis_dataframe(
    std_by_age_select_states_no_missing, "Age", "13-24"
)
overall_std_race_multiracial = std_analysis_dataframe(
    std_by_race_select_states_no_missing, "Race", "Multiracial"
)
overall_std_race_black = std_analysis_dataframe(
    std_by_race_select_states_no_missing, "Race", "Black/African American"
)
overall_std_gender_male = std_analysis_dataframe(
    std_by_gender_select_states_no_missing, "Sex", "Male"
)
overall_std_gender_female = std_analysis_dataframe(
    std_by_gender_select_states_no_missing, "Sex", "Female"
)

treatment_state_md = "Maryland"
treatment_state_ny = "New York"
treatment_state_ga = "Georgia"
md_control_states = [
    "Delaware",
    "New Jersey",
    "Arizona",
    "Virginia",
    "Ohio",
    "Pennsylvania",
]
ny_control_states = ["Florida", "Louisiana", "California", "Texas", "Illinois"]
ga_control_states = [
    "North Carolina",
    "Mississippi",
    "Nevada",
    "Tennessee",
    "Arkansas",
    "New Mexico",
    "Missouri",
]

# Diff-in-Diff Analysis for MD
overall_std_md = partition_data(
    overall_std,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)
std_chlamydia_md = partition_data(
    overall_std_chlamydia,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)
std_gonorrhea_md = partition_data(
    overall_std_gonorrhea,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)
std_race_black_md = partition_data(
    overall_std_race_black,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)
std_gender_female_md = partition_data(
    overall_std_gender_female,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)
std_age_young_adults_md = partition_data(
    overall_std_young_adults,
    "Geography",
    "Rate_per_100000",
    treatment_state_md,
    md_control_states,
    2015,
)

overall_pre_md = plotting_chart(
    overall_std_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
overall_post_md = plotting_chart(
    overall_std_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
overall_pre_md_control = plotting_chart(
    overall_std_md[2],
    "Year",
    "Rate_per_100000",
    """Control States - Virginia, Arizona, Ohio, """
    """New Jersey, Pennsylvania, Delaware""",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
overall_post_md_control = plotting_chart(
    overall_std_md[3],
    "Year",
    "Rate_per_100000",
    """Control States - Virginia, Arizona, Ohio, """
    """New Jersey, Pennsylvania, Delaware""",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)

final_overall_std_md = (
    overall_pre_md
    + overall_post_md
    + overall_pre_md_control
    + overall_post_md_control
)
final_overall_std_md.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Total STD Rate in Maryland vs Control States"""
)

pre_md_chlamydia = plotting_chart(
    std_chlamydia_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
post_md_chlamydia = plotting_chart(
    std_chlamydia_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
pre_md_control_chlamydia = plotting_chart(
    std_chlamydia_md[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
post_md_control_chlamydia = plotting_chart(
    std_chlamydia_md[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    " #4CBB17",
    "STD Rate per 100,000",
)

final_md_chlamydia = (
    pre_md_chlamydia
    + post_md_chlamydia
    + pre_md_control_chlamydia
    + post_md_control_chlamydia
)
final_md_chlamydia.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Chlamydia STD Rate in Maryland vs Control States"""
)

pre_md_gonorrhea = plotting_chart(
    std_gonorrhea_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
post_md_gonorrhea = plotting_chart(
    std_gonorrhea_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
pre_md_control_gonorrhea = plotting_chart(
    std_gonorrhea_md[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
post_md_control_gonorrhea = plotting_chart(
    std_gonorrhea_md[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    " #4CBB17",
    "STD Rate per 100,000",
)

final_md_gonorrhea = (
    pre_md_gonorrhea
    + post_md_gonorrhea
    + pre_md_control_gonorrhea
    + post_md_control_gonorrhea
)
final_md_gonorrhea.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Gonorrhea STD Rate in Maryland vs Control States"""
)

pre_md_black = plotting_chart(
    std_race_black_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
post_md_black = plotting_chart(
    std_race_black_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
pre_md_control_black = plotting_chart(
    std_race_black_md[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
post_md_control_black = plotting_chart(
    std_race_black_md[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    " #4CBB17",
    "STD Rate per 100,000",
)

final_md_black = (
    pre_md_black + post_md_black + pre_md_control_black + post_md_control_black
)
final_md_black.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Black/African American Residents """
    """in Maryland vs Control States"""
)

pre_md_female = plotting_chart(
    std_gender_female_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
post_md_female = plotting_chart(
    std_gender_female_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
pre_md_control_female = plotting_chart(
    std_gender_female_md[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
post_md_control_female = plotting_chart(
    std_gender_female_md[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    " #4CBB17",
    "STD Rate per 100,000",
)

final_md_female = (
    pre_md_female
    + post_md_female
    + pre_md_control_female
    + post_md_control_female
)
final_md_female.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Females in """
    """Maryland vs Control States"""
)

pre_md_young_adults = plotting_chart(
    std_age_young_adults_md[0],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
post_md_young_adults = plotting_chart(
    std_age_young_adults_md[1],
    "Year",
    "Rate_per_100000",
    "Maryland",
    2015,
    "Green",
    "STD Rate per 100,000",
)
pre_md_control_young_adults = plotting_chart(
    std_age_young_adults_md[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    "#4CBB17",
    "STD Rate per 100,000",
)
post_md_control_young_adults = plotting_chart(
    std_age_young_adults_md[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2015,
    " #4CBB17",
    "STD Rate per 100,000",
)

final_md_young_adults = (
    pre_md_young_adults
    + post_md_young_adults
    + pre_md_control_young_adults
    + post_md_control_young_adults
)
final_md_young_adults.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Young Residents (13-24 year olds) """
    """in Maryland vs Control States"""
)

# Diff-in-Diff Analysis for NY
overall_std_ny = partition_data(
    overall_std,
    "Geography",
    "Rate_per_100000",
    treatment_state_ny,
    ny_control_states,
    2016,
)
std_chlamydia_ny = partition_data(
    overall_std_chlamydia,
    "Geography",
    "Rate_per_100000",
    treatment_state_ny,
    ny_control_states,
    2016,
)
std_race_multiracial_ny = partition_data(
    overall_std_race_multiracial,
    "Geography",
    "Rate_per_100000",
    treatment_state_ny,
    ny_control_states,
    2016,
)
std_gender_male_ny = partition_data(
    overall_std_gender_male,
    "Geography",
    "Rate_per_100000",
    treatment_state_ny,
    ny_control_states,
    2016,
)
std_age_young_adults_ny = partition_data(
    overall_std_young_adults,
    "Geography",
    "Rate_per_100000",
    treatment_state_ny,
    ny_control_states,
    2016,
)

overall_pre_ny = plotting_chart(
    overall_std_ny[0],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
overall_post_ny = plotting_chart(
    overall_std_ny[1],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
overall_pre_ny_control = plotting_chart(
    overall_std_ny[2],
    "Year",
    "Rate_per_100000",
    "Control States - Florida, Louisiana, Texas, California, Illinois",
    2016,
    "#4682B4",
    "STD Rate per 100,000",
)
overall_post_ny_control = plotting_chart(
    overall_std_ny[3],
    "Year",
    "Rate_per_100000",
    "Control States - Florida, Louisiana, Texas, California, Illinois",
    2016,
    " #4682B4",
    "STD Rate per 100,000",
)

final_overall_std_ny = (
    overall_pre_ny
    + overall_post_ny
    + overall_pre_ny_control
    + overall_post_ny_control
)
final_overall_std_ny.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Total STD Rate in New York vs Control States"""
)

pre_ny_chlamydia = plotting_chart(
    std_chlamydia_ny[0],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
post_ny_chlamydia = plotting_chart(
    std_chlamydia_ny[1],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
pre_ny_control_chlamydia = plotting_chart(
    std_chlamydia_ny[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    "#4682B4",
    "STD Rate per 100,000",
)
post_ny_control_chlamydia = plotting_chart(
    std_chlamydia_ny[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    " #4682B4",
    "STD Rate per 100,000",
)

final_ny_chlamydia = (
    pre_ny_chlamydia
    + post_ny_chlamydia
    + pre_ny_control_chlamydia
    + post_ny_control_chlamydia
)
final_ny_chlamydia.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Chlamydia STD Rate in """
    """New York vs Control States"""
)

pre_ny_multiracial = plotting_chart(
    std_race_multiracial_ny[0],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
post_ny_multiracial = plotting_chart(
    std_race_multiracial_ny[1],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
pre_ny_control_multiracial = plotting_chart(
    std_race_multiracial_ny[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    "#4682B4",
    "STD Rate per 100,000",
)
post_ny_control_multiracial = plotting_chart(
    std_race_multiracial_ny[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    " #4682B4",
    "STD Rate per 100,000",
)

final_ny_multiracial = (
    pre_ny_multiracial
    + post_ny_multiracial
    + pre_ny_control_multiracial
    + post_ny_control_multiracial
)
final_ny_multiracial.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Multiracial Residents """
    """in New York vs Control States"""
)

pre_ny_male = plotting_chart(
    std_gender_male_ny[0],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
post_ny_male = plotting_chart(
    std_gender_male_ny[1],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
pre_ny_control_male = plotting_chart(
    std_gender_male_ny[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    "#4682B4",
    "STD Rate per 100,000",
)
post_ny_control_male = plotting_chart(
    std_gender_male_ny[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    " #4682B4",
    "STD Rate per 100,000",
)

final_ny_male = (
    pre_ny_male + post_ny_male + pre_ny_control_male + post_ny_control_male
)
final_ny_male.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Males in """
    """New York vs Control States"""
)

pre_ny_young_adults = plotting_chart(
    std_age_young_adults_ny[0],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
post_ny_young_adults = plotting_chart(
    std_age_young_adults_ny[1],
    "Year",
    "Rate_per_100000",
    "New York",
    2016,
    "Blue",
    "STD Rate per 100,000",
)
pre_ny_control_young_adults = plotting_chart(
    std_age_young_adults_ny[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    "#4682B4",
    "STD Rate per 100,000",
)
post_ny_control_young_adults = plotting_chart(
    std_age_young_adults_ny[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2016,
    " #4682B4",
    "STD Rate per 100,000",
)

final_ny_young_adults = (
    pre_ny_young_adults
    + post_ny_young_adults
    + pre_ny_control_young_adults
    + post_ny_control_young_adults
)
final_ny_young_adults.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Young Residents (13-24 year olds) """
    """in New York vs Control States"""
)

# Diff-in-Diff Analysis of Georgia
overall_std_ga = partition_data(
    overall_std,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)
std_chlamydia_ga = partition_data(
    overall_std_chlamydia,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)
std_gonorrhea_ga = partition_data(
    overall_std_gonorrhea,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)
std_race_black_ga = partition_data(
    overall_std_race_black,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)
std_gender_female_ga = partition_data(
    overall_std_gender_female,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)
std_age_young_adults_ga = partition_data(
    overall_std_young_adults,
    "Geography",
    "Rate_per_100000",
    treatment_state_ga,
    ga_control_states,
    2017,
)

overall_pre_ga = plotting_chart(
    overall_std_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
overall_post_ga = plotting_chart(
    overall_std_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
overall_pre_ga_control = plotting_chart(
    overall_std_ga[2],
    "Year",
    "Rate_per_100000",
    """Control States - Virginia, Arizona, Ohio, """
    """New Jersey, Pennsylvania, Delaware""",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
overall_post_ga_control = plotting_chart(
    overall_std_ga[3],
    "Year",
    "Rate_per_100000",
    """Control States - Virginia, Arizona, Ohio, """
    """New Jersey, Pennsylvania, Delaware""",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_overall_std_ga = (
    overall_pre_ga
    + overall_post_ga
    + overall_pre_ga_control
    + overall_post_ga_control
)
final_overall_std_ga.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Total STD Rate in Georgia vs Control States"""
)

pre_ga_chlamydia = plotting_chart(
    std_chlamydia_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
post_ga_chlamydia = plotting_chart(
    std_chlamydia_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
pre_ga_control_chlamydia = plotting_chart(
    std_chlamydia_ga[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
post_ga_control_chlamydia = plotting_chart(
    std_chlamydia_ga[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_ga_chlamydia = (
    pre_ga_chlamydia
    + post_ga_chlamydia
    + pre_ga_control_chlamydia
    + post_ga_control_chlamydia
)
final_ga_chlamydia.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Chlamydia STD Rate in Georgia vs Control States"""
)

pre_ga_gonorrhea = plotting_chart(
    std_gonorrhea_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
post_ga_gonorrhea = plotting_chart(
    std_gonorrhea_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
pre_ga_control_gonorrhea = plotting_chart(
    std_gonorrhea_ga[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
post_ga_control_gonorrhea = plotting_chart(
    std_gonorrhea_ga[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_ga_gonorrhea = (
    pre_ga_gonorrhea
    + post_ga_gonorrhea
    + pre_ga_control_gonorrhea
    + post_ga_control_gonorrhea
)
final_ga_gonorrhea.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on Gonorrhea STD Rate in Georgia vs Control States"""
)

pre_ga_black = plotting_chart(
    std_race_black_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
post_ga_black = plotting_chart(
    std_race_black_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
pre_ga_control_black = plotting_chart(
    std_race_black_ga[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
post_ga_control_black = plotting_chart(
    std_race_black_ga[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_ga_black = (
    pre_ga_black + post_ga_black + pre_ga_control_black + post_ga_control_black
)
final_ga_black.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Black/African American Residents """
    """in Georgia vs Control States"""
)

pre_ga_female = plotting_chart(
    std_gender_female_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
post_ga_female = plotting_chart(
    std_gender_female_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
pre_ga_control_female = plotting_chart(
    std_gender_female_ga[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
post_ga_control_female = plotting_chart(
    std_gender_female_ga[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_ga_female = (
    pre_ga_female
    + post_ga_female
    + pre_ga_control_female
    + post_ga_control_female
)
final_ga_female.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Females in Georgia vs Control States"""
)

pre_ga_young_adults = plotting_chart(
    std_age_young_adults_ga[0],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
post_ga_young_adults = plotting_chart(
    std_age_young_adults_ga[1],
    "Year",
    "Rate_per_100000",
    "Georgia",
    2017,
    "Red",
    "STD Rate per 100,000",
)
pre_ga_control_young_adults = plotting_chart(
    std_age_young_adults_ga[2],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)
post_ga_control_young_adults = plotting_chart(
    std_age_young_adults_ga[3],
    "Year",
    "Rate_per_100000",
    "Control States",
    2017,
    "#ff6f6a",
    "STD Rate per 100,000",
)

final_ga_young_adults = (
    pre_ga_young_adults
    + post_ga_young_adults
    + pre_ga_control_young_adults
    + post_ga_control_young_adults
)
final_ga_young_adults.properties(
    title="""Difference in Difference Analysis of Policy """
    """Based on STD Rate among Young Residents (13-24 year olds) """
    """in Georgia vs Control States"""
)
