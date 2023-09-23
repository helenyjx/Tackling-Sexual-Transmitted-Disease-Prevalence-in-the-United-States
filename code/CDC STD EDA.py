"""EDA of the CDC Data."""

import numpy as np
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt
import datetime

sexual_disease_data = pd.read_csv(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Collective.csv""",
    skiprows=6,
)
sexual_disease_data_copy = sexual_disease_data.copy()

sexual_disease_data_copy = sexual_disease_data_copy.replace(
    "2020 (COVID-19 Pandemic)", "2020"
)
sexual_disease_data_copy["Cases"] = sexual_disease_data_copy[
    "Cases"
].str.replace(",", "")
sexual_disease_data_copy["Population"] = sexual_disease_data_copy[
    "Population"
].str.replace(",", "")
sexual_disease_data_copy["Year"] = pd.to_datetime(
    sexual_disease_data_copy["Year"], format="%Y"
)
sexual_disease_data_copy["Year"] = sexual_disease_data_copy[
    "Year"
].dt.strftime("%Y")
cols = ["Cases", "Rate per 100000", "Population"]
sexual_disease_data_copy[cols] = sexual_disease_data_copy[cols].apply(
    pd.to_numeric, errors="coerce", axis=1
)

sexual_disease_data_copy.dtypes

for column in sexual_disease_data_copy:
    print(sexual_disease_data_copy[column].unique())

sexual_disease_data_age_group = sexual_disease_data_copy.groupby(
    ["Age Group"]
)["Cases"].sum()
sexual_disease_data_age_group

sexual_disease_data_race = sexual_disease_data_copy.groupby(
    ["Race/Ethnicity"]
)["Cases"].sum()
sexual_disease_data_race

sexual_disease_data_gender = sexual_disease_data_copy.groupby(["Sex"])[
    "Cases"
].sum()
sexual_disease_data_gender

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
sexual_disease_data_select_states = sexual_disease_data_copy.loc[
    sexual_disease_data_copy["Geography"].isin(select_states)
]

print(sexual_disease_data_select_states.isna().sum())

sexual_disease_data_select_states_null_values = (
    sexual_disease_data_select_states[
        sexual_disease_data_select_states.isna().any(axis=1)
    ]
)
sexual_disease_data_select_states_null_values

for column in sexual_disease_data_select_states_null_values:
    print(sexual_disease_data_select_states_null_values[column].unique())

# define the list of races to check for missing values
races = [
    "White",
    "Native Hawaiian/Other Pacific Islander",
    "Multiracial",
    "Hispanic/Latino",
    "Black/African American",
    "Asian",
    "American Indian/Alaska Native",
]

# check for missing values in the 'Cases' column for each race
for race in races:
    missing_count = (
        sexual_disease_data_select_states.loc[
            sexual_disease_data_select_states["Race/Ethnicity"] == race,
            "Cases",
        ]
        .isna()
        .sum()
    )
    print(
        f"""Number of missing values in "{race}" race category"""
        + f"""in "Population" column: {missing_count}"""
    )

# drop rows with missing values in the 'Cases' column for each race
for race in races:
    sexual_disease_data_select_states_no_missing = (
        sexual_disease_data_select_states.loc[
            (sexual_disease_data_select_states["Cases"].notna())
        ]
    )

sexual_disease_data_select_states_no_missing

print(sexual_disease_data_select_states_no_missing.isna().sum())

sexual_disease_data_md = sexual_disease_data_select_states_no_missing.loc[
    sexual_disease_data_select_states_no_missing["Geography"] == "Maryland"
]
sexual_disease_data_ny = sexual_disease_data_select_states_no_missing.loc[
    sexual_disease_data_select_states_no_missing["Geography"] == "New York"
]
sexual_disease_data_ga = sexual_disease_data_select_states_no_missing.loc[
    sexual_disease_data_select_states_no_missing["Geography"] == "Georgia"
]

md_control_states = [
    "Delaware",
    "New Jersey",
    "Arizona",
    "Virginia",
    "Ohio",
    "Pennsylvania",
]
sexual_disease_data_md_control = (
    sexual_disease_data_select_states_no_missing.loc[
        sexual_disease_data_select_states_no_missing["Geography"].isin(
            md_control_states
        )
    ]
)
ny_control_states = ["Florida", "Louisiana", "California", "Texas", "Illinois"]
sexual_disease_data_ny_control = (
    sexual_disease_data_select_states_no_missing.loc[
        sexual_disease_data_select_states_no_missing["Geography"].isin(
            ny_control_states
        )
    ]
)
ga_control_states = [
    "North Carolina",
    "Mississippi",
    "Nevada",
    "Tennessee",
    "Arkansas",
    "New Mexico",
    "Missouri",
]
sexual_disease_data_ga_control = (
    sexual_disease_data_select_states_no_missing.loc[
        sexual_disease_data_select_states_no_missing["Geography"].isin(
            ga_control_states
        )
    ]
)


def std_pie_chart(std_df, std_column, std_column_2, state_colors, state):
    """Create pie charts for EDA analysis."""
    # Group the dataframe by 'Disease' to calculate the total number of cases
    std_df_grouped = std_df.groupby(std_column).sum()

    # Create a dataframe for 'Cases' and 'Rate_per_100000'
    std_cases = pd.DataFrame(std_df_grouped[std_column_2])

    # Create a larger figure
    fig = plt.figure(figsize=(14, 12))

    colors = state_colors

    # Create a pie chart to show the distribution of diseases
    # based on the number of cases
    patches, texts, autotexts = plt.pie(
        std_cases[std_column_2],
        labels=std_cases.index,
        autopct="%1.1f%%",
        colors=colors,
        textprops={"color": "black"},
    )

    plt.title(
        f"Distribution of Diseases Based on {std_column_2} in {state}",
        color="black",
    )

    # Show the plot
    pie_chart = plt.show()
    return pie_chart


ny_colors = [
    "#00FFFF",
    "#89CFF0",
    "#ADD8E6",
    "#6082B6",
    "#6495ED",
    "#40B5AD",
    "#000080",
]
ga_colors = [
    "#FF69B4",
    "#FF1493",
    "#C71585",
    "#DB7093",
    "#FFB6C1",
    "#FF0000",
    "#8B0000",
]
md_colors = [
    "#088F8F",
    "#097969",
    "#90EE90",
    "#008000",
    "#98FB98",
    "#8A9A5B",
    "#023020",
]

ny_std_by_indicator = std_pie_chart(
    sexual_disease_data_ny, "Indicator", "Cases", ny_colors, "New York"
)
ny_controls_std_by_indicator = std_pie_chart(
    sexual_disease_data_ny_control,
    "Indicator",
    "Cases",
    ny_colors,
    "New York Control States",
)
ga_std_by_indicator = std_pie_chart(
    sexual_disease_data_ga, "Indicator", "Cases", ga_colors, "Georgia"
)
ga_controls_std_by_indicator = std_pie_chart(
    sexual_disease_data_ga_control,
    "Indicator",
    "Cases",
    ga_colors,
    "Georgia Control States",
)
md_std_by_indicator = std_pie_chart(
    sexual_disease_data_md, "Indicator", "Cases", md_colors, "Maryland"
)
md_controls_std_by_indicator = std_pie_chart(
    sexual_disease_data_md_control,
    "Indicator",
    "Cases",
    md_colors,
    "Maryland Control States",
)


sexual_disease_data_2 = pd.read_csv(
    """/Users/sukhpreetsahota/Desktop/Duke/Spring 2023/IDS 701.01.SP23/"""
    """Project/Project Work - My Folder/Data Sources/"""
    """CDC STD Disease Data_Collective_HIV and AIDS Classifications.csv""",
    skiprows=6,
)
sexual_disease_data_2_copy = sexual_disease_data_2.copy()

sexual_disease_data_2_copy = sexual_disease_data_2_copy.replace(
    "2020 (COVID-19 Pandemic)", "2020"
)
sexual_disease_data_2_copy["Cases"] = sexual_disease_data_2_copy[
    "Cases"
].str.replace(",", "")
sexual_disease_data_2_copy["Population"] = sexual_disease_data_2_copy[
    "Population"
].str.replace(",", "")
sexual_disease_data_2_copy["Year"] = pd.to_datetime(
    sexual_disease_data_2_copy["Year"], format="%Y"
)
sexual_disease_data_2_copy["Year"] = sexual_disease_data_2_copy[
    "Year"
].dt.strftime("%Y")
cols = ["Cases", "Rate per 100000", "Population"]
sexual_disease_data_2_copy[cols] = sexual_disease_data_2_copy[cols].apply(
    pd.to_numeric, errors="coerce", axis=1
)

sexual_disease_data_2_copy.dtypes

for column in sexual_disease_data_2_copy:
    print(sexual_disease_data_2_copy[column].unique())

sexual_disease_data_age_group_2 = sexual_disease_data_2_copy.groupby(
    ["Age Group"]
)["Cases"].sum()
sexual_disease_data_age_group_2

sexual_disease_data_race_2 = sexual_disease_data_2_copy.groupby(
    ["Race/Ethnicity"]
)["Cases"].sum()
sexual_disease_data_race_2

sexual_disease_data_gender_2 = sexual_disease_data_2_copy.groupby(["Sex"])[
    "Cases"
].sum()
sexual_disease_data_gender_2

sexual_disease_data_2_select_states = sexual_disease_data_2_copy.loc[
    sexual_disease_data_2_copy["Geography"].isin(select_states)
]

print(sexual_disease_data_2_select_states.isna().sum())

# drop rows with missing values in the 'Cases' column for each race
for race in races:
    sexual_disease_data_select_states_no_missing_2 = (
        sexual_disease_data_2_select_states.loc[
            (sexual_disease_data_2_select_states["Cases"].notna())
        ]
    )

sexual_disease_data_select_states_no_missing_2

sexual_disease_data_md_2 = sexual_disease_data_select_states_no_missing_2.loc[
    sexual_disease_data_select_states_no_missing_2["Geography"] == "Maryland"
]
sexual_disease_data_ny_2 = sexual_disease_data_select_states_no_missing_2.loc[
    sexual_disease_data_select_states_no_missing_2["Geography"] == "New York"
]
sexual_disease_data_ga_2 = sexual_disease_data_select_states_no_missing_2.loc[
    sexual_disease_data_select_states_no_missing_2["Geography"] == "Georgia"
]

sexual_disease_data_md_control_2 = (
    sexual_disease_data_select_states_no_missing_2.loc[
        sexual_disease_data_select_states_no_missing_2["Geography"].isin(
            md_control_states
        )
    ]
)
sexual_disease_data_ny_control_2 = (
    sexual_disease_data_select_states_no_missing_2.loc[
        sexual_disease_data_select_states_no_missing_2["Geography"].isin(
            ny_control_states
        )
    ]
)
sexual_disease_data_ga_control_2 = (
    sexual_disease_data_select_states_no_missing_2.loc[
        sexual_disease_data_select_states_no_missing_2["Geography"].isin(
            ga_control_states
        )
    ]
)

ny_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_ny_2, "Indicator", "Cases", ny_colors, "New York"
)
ny_controls_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_ny_control_2,
    "Indicator",
    "Cases",
    ny_colors,
    "New York Control States",
)
ga_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_ga_2, "Indicator", "Cases", ga_colors, "Georgia"
)
ga_controls_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_ga_control_2,
    "Indicator",
    "Cases",
    ga_colors,
    "Georgia Control States",
)
md_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_md_2, "Indicator", "Cases", md_colors, "Maryland"
)
md_controls_std_by_indicator_2 = std_pie_chart(
    sexual_disease_data_md_control_2,
    "Indicator",
    "Cases",
    md_colors,
    "Maryland Control States",
)


def std_bar_chart(std_df, std_column, std_column_2, state_colors, state):
    """Create bar charts for EDA analysis."""
    # Create a larger figure
    fig, ax = plt.subplots(figsize=(20, 10))

    colors = state_colors

    labels = list(std_df[std_column].unique())
    labels_sorted = labels.sort(reverse=True)

    sizes = [
        int(std_df[(std_df[std_column] == label)][std_column_2].mean())
        for label in labels
    ]

    # Add custom colors to the bar chart
    bars = ax.barh(labels, sizes, color=colors)

    # Add formatting to the bar chart
    ax.set_title(
        f"Average STD {std_column_2} by {std_column} in {state}", color="black"
    )
    ax.set_xlabel(f"{std_column_2}", color="black")
    ax.set_ylabel(f"{std_column}", color="black")
    plt.xticks(color="black")
    plt.yticks(color="black")

    # Add values of STD_per_100000 sum for each race in black color
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + 0.4,
            str(round(sizes[i], 1)),
            color="black",
        )

    bar_chart = plt.show()
    return bar_chart


ny_std_by_race = std_bar_chart(
    sexual_disease_data_ny_2,
    "Race/Ethnicity",
    "Rate per 100000",
    ny_colors,
    "New York",
)
ny_controls_std_by_race = std_bar_chart(
    sexual_disease_data_ny_control_2,
    "Race/Ethnicity",
    "Rate per 100000",
    ny_colors,
    "New York Control States",
)
ga_std_by_race = std_bar_chart(
    sexual_disease_data_ga_2,
    "Race/Ethnicity",
    "Rate per 100000",
    ga_colors,
    "Georgia",
)
ga_controls_std_by_race = std_bar_chart(
    sexual_disease_data_ga_control_2,
    "Race/Ethnicity",
    "Rate per 100000",
    ga_colors,
    "Georgia Control States",
)
md_std_by_race = std_bar_chart(
    sexual_disease_data_md_2,
    "Race/Ethnicity",
    "Rate per 100000",
    md_colors,
    "Maryland",
)
md_controls_std_by_race = std_bar_chart(
    sexual_disease_data_md_control_2,
    "Race/Ethnicity",
    "Rate per 100000",
    md_colors,
    "Maryland Control States",
)


def std_doughnut_chart(std_df, std_column, std_column_2, state_colors, state):
    """Create doughtnut charts for EDA analysis."""
    # Create a larger figure
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = state_colors

    labels = list(std_df[std_column].unique())
    labels_sorted = labels.sort(reverse=True)

    sizes = [
        int(std_df[(std_df[std_column] == label)][std_column_2].mean())
        for label in labels
    ]

    # Create a colormap with different shades of green
    colors = state_colors

    # Add custom colors to the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor="white"),
    )

    # Add formatting to the bar chart
    ax.axis("equal")
    ax.set_title(
        f"Average STD {std_column_2} by {std_column} in {state}",
        color="black",
        pad=20,
    )
    plt.xticks(color="black")
    plt.yticks(color="black")

    # Change the text color on the chart
    for text in texts:
        text.set_color("black")
    for autotext in autotexts:
        autotext.set_color("black")

    doughnut_chart = plt.show()
    return doughnut_chart


ny_std_by_age = std_doughnut_chart(
    sexual_disease_data_ny_2,
    "Age Group",
    "Rate per 100000",
    ny_colors,
    "New York",
)
ny_controls_std_by_age = std_doughnut_chart(
    sexual_disease_data_ny_control_2,
    "Age Group",
    "Rate per 100000",
    ny_colors,
    "New York Control States",
)
ga_std_by_age = std_doughnut_chart(
    sexual_disease_data_ga_2,
    "Age Group",
    "Rate per 100000",
    ga_colors,
    "Georgia",
)
ga_controls_std_by_age = std_doughnut_chart(
    sexual_disease_data_ga_control_2,
    "Age Group",
    "Rate per 100000",
    ga_colors,
    "Georgia Control States",
)
md_std_by_age = std_doughnut_chart(
    sexual_disease_data_md_2,
    "Age Group",
    "Rate per 100000",
    md_colors,
    "Maryland",
)
md_controls_std_by_age = std_doughnut_chart(
    sexual_disease_data_md_control_2,
    "Age Group",
    "Rate per 100000",
    md_colors,
    "Maryland Control States",
)

ny_std_by_gender = std_doughnut_chart(
    sexual_disease_data_ny_2, "Sex", "Rate per 100000", ny_colors, "New York"
)
ny_controls_std_by_gender = std_doughnut_chart(
    sexual_disease_data_ny_control_2,
    "Sex",
    "Rate per 100000",
    ny_colors,
    "New York Control States",
)
ga_std_by_gender = std_doughnut_chart(
    sexual_disease_data_ga_2, "Sex", "Rate per 100000", ga_colors, "Georgia"
)
ga_controls_std_by_gender = std_doughnut_chart(
    sexual_disease_data_ga_control_2,
    "Sex",
    "Rate per 100000",
    ga_colors,
    "Georgia Control States",
)
md_std_by_gender = std_doughnut_chart(
    sexual_disease_data_md_2, "Sex", "Rate per 100000", md_colors, "Maryland"
)
md_controls_std_by_gender = std_doughnut_chart(
    sexual_disease_data_md_control_2,
    "Sex",
    "Rate per 100000",
    md_colors,
    "Maryland Control States",
)
