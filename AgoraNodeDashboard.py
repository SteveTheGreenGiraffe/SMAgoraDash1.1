# BRANCH FOR SMAgoraDash250720

# Current path and command:
# cd "C:\Users\StephenMettler\GREEN GIRAFFE Dropbox\Stephen Mettler\PC\Documents\Commercial pursuits (DT)\S4\SG"
# streamlit run AgoraNodeDashboard.py

import streamlit as st
import altair as alt
import pandas as pd
import os
from Modify_AgoraDE_files import expandIntRenInNode, safe_divide, processLineCosts, showGenPriceDeciles
import numpy as np


df = pd.read_csv("https://www.dropbox.com/scl/fi/rj0zjhajipdrn1518zas7/nodal_dem_gen_with_prices_and_categories_summed_2025_07_14_14_19.csv?rlkey=9ukpa5a0nzv45dupg22fi1bo5&st=v1b1yrkm&dl=1")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="raise")
avail_years = list(df["Datetime"].dt.year.unique())
col1, col2 = st.columns([1, 1])
with col1:
    node = st.selectbox("Choose a node", df["Selected hub"].unique())
with col2:
    year = st.selectbox("Choose a year", avail_years, index=avail_years.index(2024))

# TAG: Call for showGenPriceDeciles will go after the expandIntRenNode call

#if "submit_triggered" not in st.session_state:
#    st.session_state["submit_triggered"] = False

# Set displaced capture rates to 0 until passed values
displaced_all_capture = 0
displaced_positive_capture = 0

# Call initial function, without boosts
(
    node_df,
    capture_results,
    return_string,
    gw_dict,
    cost_summary,
    cost_dict,
    hourly_gen_df,
    marginal_impact,
    all_existing_gen_costs,
    cost_pool_after_substituted_gen,
    all_existing_GWh,
    demand_return_string,
    all_demand_GWh,
    all_demand_wholesale_costs,
    ex_im_impact_binned,
    ex_im_impact_binned_display_df,
    summary_by_source_df
) = expandIntRenInNode(df, node, year)

# Call gen-price deciles
(
    final_showGenPriceDeciles_df
) = showGenPriceDeciles(df, year)


# Call supplemental T&D cost analysis function
return_string_TandD_added, total_TD_costs = processLineCosts(node, year)

all_in_existing_cost_stack = all_demand_wholesale_costs + total_TD_costs
all_in_new_cost_stack_prior_to_TD_delta = cost_pool_after_substituted_gen + total_TD_costs
unit_cost_stack = all_in_existing_cost_stack / ( all_demand_GWh * 1_000 )
return_string_addition = (
    f"\nAll-in node unit cost of electricity demand EUR {unit_cost_stack:,.0f}/MWh"
    f"\nExisting cost base {all_in_existing_cost_stack/1_000_000:,.0f} MEUR"
)

# Display result
col1, col2 = st.columns([3, 3])
with col1:
    st.set_page_config(layout="wide")
    st.markdown("**Existing generation**")
    st.text(return_string)
    st.markdown("**Existing demand**")
    st.text(demand_return_string)
with col2:
    st.markdown("**Existing T&D**")
    st.text(return_string_TandD_added)
    st.markdown("**Existing all-in cost pools**")
    st.text(return_string_addition)

if "marginal_impact" not in st.session_state:
    st.session_state["marginal_impact"] = ""

def manual_input_only(label_base: str, type, min_val=0, max_val=500, default_val=50):
    if type == "Boost":
        unit = "(%)"
        key = f"{label_base}_boost"
    else:
        unit = "(kEUR annual)"
        key = f"{label_base}_cost"

    if key not in st.session_state:
        st.session_state[key] = default_val

    return st.number_input(
        f"{label_base} {unit}",
        min_value=min_val,
        max_value=max_val,
        value=st.session_state[key],
        key=key,
    )

# Placeholder initial inputs for all the sliders
default_solar_pct = 0
default_onshore_pct = 0
default_offshore_pct = 0
default_solar_cst = 100
default_onshore_cst = 200
default_offshore_cst = 300

solar_pct = default_solar_pct
onshore_pct = default_onshore_pct
offshore_pct = default_offshore_pct
solar_cst = default_solar_cst
onshore_cst = default_onshore_cst
offshore_cst = default_offshore_cst

with st.form("input_form"):

    col1, col2 = st.columns([3, 3])

    with col1:
        # User adjusts sliders to boost renewables
        solar_pct = manual_input_only("Solar", "Boost", 0, 200, 0)
        onshore_pct = manual_input_only("Onshore", "Boost", 0, 200, 0)
        offshore_pct = manual_input_only("Offshore", "Boost", 0, 200, 0)
        solar_cst = manual_input_only("Solar", "Cost", 0, 500, 100)
        onshore_cst = manual_input_only("Onshore", "Cost", 0, 500, 200)
        offshore_cst = manual_input_only("Offshore", "Cost", 0, 500, 300)

    submitted = st.form_submit_button("Update results")


#if st.session_state["submit_triggered"]:
if submitted:

    # Use updated session state values here:
    solar_pct = st.session_state["Solar_boost"]
    onshore_pct = st.session_state["Onshore_boost"]
    offshore_pct = st.session_state["Offshore_boost"]
    solar_cst = st.session_state["Solar_cost"]
    onshore_cst = st.session_state["Onshore_cost"]
    offshore_cst = st.session_state["Offshore_cost"]

    # Re-call function with boost inputs added
    (
        _,
        _,
        return_string,
        _,
        cost_summary,
        cost_dict,
        hourly_gen_df,
        marginal_impact,
        all_existing_gen_costs,
        cost_pool_after_substituted_gen,
        all_existing_GWh,
        demand_return_string,
        all_demand_GWh,
        all_demand_wholesale_costs,
        ex_im_impact_binned,
        ex_im_impact_binned_display_df,
        summary_by_source_df
    ) = expandIntRenInNode(
        df,
        node,
        year,
        boost_pcts={
            "solar": solar_pct / 100,
            "wind onshore": onshore_pct / 100,
            "wind offshore": offshore_pct / 100
        },
        cost_per_mw={
            "solar": solar_cst * 1_000,
            "wind onshore": onshore_cst * 1_000,
            "wind offshore": offshore_cst * 1_000
        },
        GW_in_node_by_source=gw_dict
    )

    st.session_state["marginal_impact"] = marginal_impact

if "marginal_impact" in st.session_state and st.session_state["marginal_impact"]:
    st.markdown("**New generation**")
    st.text(st.session_state["marginal_impact"])
elif "marginal_impact" not in st.session_state:
    st.info("Press **Update results** to calculate new generation.")

with col2:
    with st.expander("Hourly generation", expanded=True):
        # Prepare long-form DataFrame
        hourly_long = hourly_gen_df.reset_index().melt(
            id_vars="Hour",
            var_name="Source",
            value_name="Generation"
        )

        # Set graph stack order
        stack_order=[
            "Spilled solar", "Spilled wind onshore", "Spilled wind offshore",
            "Extra solar", "Extra wind onshore", "Extra wind offshore",
            "Remaining displacable generation", "Displacable generation",
            "Solar", "Wind onshore", "Wind offshore"
        ]

        # Mapping source -> stack index
        stack_order_mapping = {source: i for i, source in enumerate(stack_order)}

        # Add it to long-form DataFrame
        hourly_long["stack_order"] = hourly_long["Source"].map(stack_order_mapping)

        # Create and render Altair chart
        chart = alt.Chart(hourly_long).mark_bar().encode(
            x=alt.X("Hour:O", title="Hour of Day"),
            y=alt.Y("Generation:Q", title="Generation (MW)"),
            color=alt.Color(
                "Source:N",
                sort=stack_order,
                scale=alt.Scale(domain=stack_order, range=[
                    "#C554088B", "#4F6CCA86", "#0A094697",
                    "#C55408", "#4F6CCA", "#0A0946",
                    "#5F0202", "#5F0202",
                    "#C55408", "#4F6CCA", "#0A0946"
                ]),
                legend=alt.Legend(orient="bottom")
            ),
            order=alt.Order(
                field="stack_order",
                sort="descending"
            )
        ).properties(
            width="container",
            height=300,
            autosize=alt.AutoSizeParams(
                type='fit-x',
                contains='padding'
            )
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            orient='bottom'
        )

        st.altair_chart(chart, use_container_width=True)


st.table(ex_im_impact_binned_display_df)

final_showGenPriceDeciles_display_df = final_showGenPriceDeciles_df.copy()
for col in final_showGenPriceDeciles_display_df.columns:
    if col == "Agora price [EUR/MWh]" or col == "DE price [EUR/MWh]":
        final_showGenPriceDeciles_display_df[col] = final_showGenPriceDeciles_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

st.table(final_showGenPriceDeciles_display_df)

required_categories = {"solar", "onshore", "offshore", "Total", "Displaced gen."}

if cost_dict and required_categories.intersection(cost_dict.keys()):

    # Convert cost_dict to DataFrame
    cost_df = pd.DataFrame(cost_dict).T  # Transpose so sources are rows
    cost_df.reset_index(inplace=True)    # Move the index (e.g., 'Solar', 'Total') into a column
    cost_df.rename(columns={"index": "Category"}, inplace=True)  # Rename that column for clarity

    # Apply formatting to the cost_df DataFrame
    def format_currency(x):
        return f"{x:,.0f}"
    def format_energy(x):
        return f"{x:,.0f}"
    styled_df = cost_df.style.format({
        "Cost (MEUR)": format_currency,
        "Gen. (GWh)": format_energy,
        "Unit cost (EUR/MWh)": format_energy
    })

    st.subheader("Cost Breakdown Table")
    st.dataframe(styled_df, hide_index=True)

    chart1_df = cost_df[cost_df["Category"].isin(["solar", "onshore", "offshore", "Total"])]
    st.subheader("Unit cost by source and total weighted (EUR/MWh)")
    st.bar_chart(chart1_df.set_index("Category")["Unit cost (EUR/MWh)"])

    chart2_df = cost_df[cost_df["Category"].isin(["Total", "Displaced gen."])]
    st.write("Displaced generation cost after accounting for spill")
    st.bar_chart(chart2_df.set_index("Category")["Unit cost (EUR/MWh)"])

