# BRANCH FOR SMAgoraDash250720

# Current path and command:
# cd "C:\Users\StephenMettler\GREEN GIRAFFE Dropbox\Stephen Mettler\PC\Documents\Commercial pursuits (DT)\S4\SG"
# python Modify_AgoraDE_files.py --mode countryPrice
# python Modify_AgoraDE_files.py --mode exportEarnings
# python Modify_AgoraDE_files.py --mode showSumExportsByCountry
# python Modify_AgoraDE_files.py --mode costSumInsideDE
# python Modify_AgoraDE_files.py --mode sumDisplacable
# python Modify_AgoraDE_files.py --mode showAllNodeCFs
# python Modify_AgoraDE_files.py --mode expandIntRenInNode
# python Modify_AgoraDE_files.py --mode standard_initial_series
# python Modify_AgoraDE_files.py --mode assessLineCosts
# python Modify_AgoraDE_files.py --mode []
# python Modify_AgoraDE_files.py --mode []
# python Modify_AgoraDE_files.py --mode []
# python Modify_AgoraDE_files.py --mode []

# Mode list in workflows:
# standard_initial_series - Runs all steps between "Consolidated" Agora files (entry still required) and file ready to run Streamlit dashboard
# Breakout pieces
#   countryPrice - From a consolidated "price" CSV from "Consolidate" step, separate out country prices vs. node-to-node prices
#   Exports workflow
#       exportEarnings - From "national" and "flows only" files from above,  multiply MWh from flows out of Germany times source country's EUR/MWh power price
#       showSumExportsByCountry - "Show" function to display key results from above. This can be used to reflect total value of German power exports abroad, as opposed to cost of serving demand within Germany
#   costSumInsideDE - From "nodal" and "price" CSVs from "Consolidate" step, and "national" prices from "exportEarnings" step (to reflect German wholesale prices), multiply demand in all DE nodes by A) Agora-assessed prices if prices were optimized for grid constraints, and B) Actual wholesale DE power price
#   sumDisplacable - Add to above document a column summing all "displacable" generation (everything except for intermittent renewables)
#   showNodeCF - Calculates generation amounts vs. nameplate capacity, CFs, and capture prices across generation types, helper function for use in others
#   showAllNodeCFs - Runs above function for entire set


import os
import pandas as pd
from datetime import datetime
from collections import defaultdict
import argparse
import re
import ast
import numpy as np


# Set energy destinations outside Germany
TRADING_COUNTRIES = ["Austria", "Belgium", "Switzerland", "Czech Republic", "France", "Luxembourg", "Denmark 1", "Denmark 2", "Norway", "Netherlands", "Poland", "Sweden"]

timestamp = datetime.today().strftime("%Y_%m_%d_%H_%M")

def standard_initial_series():
    NODAL_FILE = "merged_nodal_data_2025_07_07_16_54.csv"
    PRICE_FILE = "merged_price_data_2025_07_08_11_02.csv"

    national_file_name, flows_file_name = countryPrice(PRICE_FILE)
    print(f"✅ Calculated national prices, saved in {national_file_name} and {flows_file_name}")

    export_file_name = exportEarnings(flows_file_name, national_file_name)
    print(f"✅ Calculated DE power export earnings, saved in {export_file_name}")

    grand_total = showSumExportsByCountry(export_file_name)
    print(f"✅ Summed DE power export earnings: €{grand_total:,.0f} M")

    cost_summed_file_name = costSumInsideDE(NODAL_FILE, PRICE_FILE, national_file_name)
    print("✅ Compiled cost-summed file")

    print("✅ Summarized costs:")
    showCostSumInsideDE(cost_summed_file_name)

    displacable_summed_file_name = sumDisplacable(cost_summed_file_name)

def safe_divide(numerator, denominator):
    try:
        if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(denominator, (pd.Series, pd.DataFrame)):
            result = numerator.divide(denominator).replace([np.inf, -np.inf], np.nan)
            return result.fillna(0)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = numerator / denominator
            if np.isinf(result) or pd.isna(result):
                return 0
            return result
    except ZeroDivisionError:
        return 0
    except Exception:
        return 0

def countryPrice(TARGET_FILE):

    # Load target file
    df = pd.read_csv(TARGET_FILE)

    national_df = df[df["From location"] == df["To location"]]

    # Filter rows where From != To (inter-nodal flows)
    flows_df = df[df["From location"] != df["To location"]]

    # Declare file names to return
    national_file_name = (f"merged_national_prices_{timestamp}.csv")
    flows_file_name = (f"merged_price_flows_only_{timestamp}.csv")

    # Save to separate CSVs
    national_df.to_csv(f"{national_file_name}", index=False, encoding="utf-8-sig")
    flows_df.to_csv(f"{flows_file_name}", index=False, encoding="utf-8-sig")

    print(f"✅ Split complete: {len(national_df)} national rows, {len(flows_df)} flow rows.")

    return national_file_name, flows_file_name

def exportEarnings(TARGET_FILE_A, TARGET_FILE_B):

    # Load target files
    #TARGET_FILE_A = "merged_price_flows_only_2025-07-01_15-35.csv"
    #TARGET_FILE_B = "merged_national_prices_2025-07-01_15-35.csv"
    flows_df = pd.read_csv(TARGET_FILE_A)
    national_df = pd.read_csv(TARGET_FILE_B)

    # Ensure timestamps are treated consistently
    flows_df["Datetime"] = pd.to_datetime(flows_df["Datetime"])
    national_df["Datetime"] = pd.to_datetime(national_df["Datetime"])

    # Filter only rows in flows where destination is NOT Germany
    flows_abroad = flows_df[flows_df["To location"].isin(TRADING_COUNTRIES)].copy()

    # Merge to get the destination country's price at that time
    # Assumes "To" in flows corresponds to "From" in national-level price data
    flows_abroad = flows_abroad.merge(
        national_df[["Datetime", "From location", "Power price [EUR/MWh]"]],
        left_on=["Datetime", "To location"], # "To" country in flows should match "From" in national price file
        right_on=["Datetime", "From location"],
        how="left",
        suffixes=('', '_dest')
    )

    # Compute value at destination price
    flows_abroad["Value_at_dest"] = flows_abroad["Power export [MWh/hour]"] * flows_abroad["Power price [EUR/MWh]_dest"]

    # Declare file name to return
    export_file_name = (f"merged_flows_with_export_value{timestamp}.csv")
    
    # Save the result
    flows_abroad.to_csv(f"{export_file_name}", index=False, encoding="utf-8-sig")

    print(f"✅ Flow file updated with value estimates for international flows. Filename: {export_file_name}")

    return export_file_name

def showSumExportsByCountry(TARGET_FILE):

    # Load target file
    #TARGET_FILE = "merged_flows_with_export_value2025_07-07_17_28.csv"
    flows_df = pd.read_csv(TARGET_FILE)

    # Cut all rows that aren't exports, group by export destination and sum value
    exports_abroad = flows_df[flows_df["To location"].isin(TRADING_COUNTRIES)]
    export_totals = exports_abroad.groupby("To location")["Value_at_dest"].sum().sort_values(ascending=False)

    # Print results
    #print("📊 Export value by destination country:")
    grand_total = 0
    for country, value in export_totals.items():
        value_m = value / 1_000_000
        #print(f"{country}: €{value_m:,.0f} M")
        grand_total += value_m
    
    #print(f"\n Total exports: €{grand_total:,.0f} M")

    export_only_file_name = (f"export_value_sum_only{timestamp}.csv")
    exports_abroad.to_csv(f"{export_only_file_name}", index=False, encoding="utf-8-sig")

    return grand_total

def costSumInsideDE(TARGET_FILE_A, TARGET_FILE_B, TARGET_FILE_C):

    # Load target files
    #TARGET_FILE_A = "merged_nodal_data_2025-07-01_15-33.csv"
    #TARGET_FILE_B = "merged_price_data_2025-07-01_15-33.csv"
    #TARGET_FILE_C = "merged_national_prices_2025-07-01_15-35.csv"
    demand_df = pd.read_csv(TARGET_FILE_A)
    prices_df = pd.read_csv(TARGET_FILE_B)
    national_df = pd.read_csv(TARGET_FILE_C)

    # Change price file's hub name header to align with demand file's, and fix broken names
    prices_df = prices_df.rename(columns={"From location": "Selected hub"})

    # Drop duplicates: keep only the first price per node per hour
    prices_deduplicated = prices_df.drop_duplicates(subset=["Datetime", "Selected hub"])

    # Merge price into nodes_df
    merged_df = pd.merge(
        demand_df,
        prices_deduplicated[["Datetime", "Selected hub", "Power price [EUR/MWh]"]],
        on=["Datetime", "Selected hub"],
        how="left"
    )

    merged_df = merged_df.rename(columns={"Power price [EUR/MWh]": "Agora wholesale price [EUR/MWh]"})

    # Merge national-level price (Germany) by Datetime, and drop duplicates
    national_df = national_df.rename(columns={"From location": "Country", "Power price [EUR/MWh]": "DE wholesale price [EUR/MWh]"})
    germany_prices = national_df[national_df["Country"] == "Germany"][["Datetime", "DE wholesale price [EUR/MWh]"]]
    germany_prices = germany_prices.drop_duplicates(subset=["Datetime"], keep="first")
    merged_df = pd.merge(merged_df, germany_prices, on="Datetime", how="left")

    # Multiply demand by price
    merged_df["Total Agora cost (EUR)"] = merged_df["Total electricity demand"] * 1_000 * merged_df["Agora wholesale price [EUR/MWh]"]
    merged_df["Total DE cost (EUR)"] = merged_df["Total electricity demand"] * 1_000 * merged_df["DE wholesale price [EUR/MWh]"]

    # Declare file name to return
    cost_summed_file_name = f"nodal_dem_and_gen_w_prices_added_{timestamp}.csv"

    # Save result
    merged_df.to_csv(cost_summed_file_name, index=False, encoding="utf-8-sig")
    print("✅ Saved merged data.")

    return cost_summed_file_name

def showCostSumInsideDE(TARGET_FILE):

    all_df = pd.read_csv(TARGET_FILE)

    # Sum and print total cost and power demand, by node and in total
    #Agora_cost_by_node = all_df.groupby("Selected hub")["Total Agora cost (EUR)"].sum()
    DE_cost_by_node = all_df.groupby("Selected hub")["Total DE cost (EUR)"].sum()
    demand_by_node = all_df.groupby("Selected hub")["Total electricity demand"].sum()

    sorted_nodes = DE_cost_by_node.sort_values(ascending=False).index

    #print("\n💡 Totals by node:")

    for node in sorted_nodes:
        #Agora_cost_m = Agora_cost_by_node[node] / 1_000_000
        DE_cost_m = DE_cost_by_node[node] / 1_000_000
        demand_GWh = demand_by_node[node]
        #Agora_cost_per_MWh = ( Agora_cost_by_node[node] / 1_000 ) / demand_by_node[node]
        DE_cost_per_MWh = ( DE_cost_by_node[node] / 1_000 ) / demand_by_node[node]
        print(f"{node}: €{DE_cost_m:,.0f} M, Demand: {demand_GWh:,.0f} GWh, DE EUR {DE_cost_per_MWh:,.2f}/MWh")

    # Print overall total
    #Agora_overall_cost = all_df["Total Agora cost (EUR)"].sum() / 1_000_000
    DE_overall_cost = all_df["Total DE cost (EUR)"].sum() / 1_000_000
    overall_GWh = all_df["Total electricity demand"].sum()
    #Agora_overall_rate = ( Agora_overall_cost * 1_000 ) / overall_GWh
    DE_overall_rate = ( DE_overall_cost * 1_000 ) / overall_GWh
    print(f"\n🔢 Total across all nodes: €{DE_overall_cost:,.0f} M, Demand: {overall_GWh:,.0f} GWh, EUR {DE_overall_rate:,.2f}/MWh")

def sumDisplacable(TARGET_FILE):

    df = pd.read_csv(TARGET_FILE)

    df["Displacable generation"] = df[[
        "Conventional",
        "Biomass",
        "Hydro",
        #"Other renewable generation",
        #"Nuclear",
        #"Lignite",
        #"Hard Coal",
        #"Natural Gas",
        #"Pumped storage generation",
        #"Other conventional generation"
    ]].sum(axis=1)

    df["Renewable generation"] = df[[
        "Wind offshore",
        "Wind onshore",
        "Solar",
    ]].sum(axis=1)

    df["Net export / (import) GWh"] = ( ( df["Displacable generation"] + df["Renewable generation"] ) - df["Total electricity demand"] )

    # Declare file name to return and save result
    displacable_summed_file_name = f"nodal_dem_gen_with_prices_and_categories_summed_{timestamp}.csv"
    df.to_csv(displacable_summed_file_name, index=False, encoding="utf-8-sig")
    print(f"✅ Saved merged data including displacable generation at {displacable_summed_file_name}")

    return displacable_summed_file_name

def showNodeCF(df, node, year):

    node_df = df[
        (df["Selected hub"] == node) &
        (df["Datetime"].dt.year == year)
    ].copy()

    # Set %ile for nameplate basis
    quantile_used = .99

    # Set up nested loops for capture rate variations; internal names on left, CSV column names on right
    sources = {
        "solar": "Solar",
        "wind onshore": "Wind onshore",
        "wind offshore": "Wind offshore",
        "displacable": "Displacable generation",
        "demand": "Total electricity demand"
    }

    capture_results = {}

    GW_in_node_by_source = {}

    return_string = "Annual generation: "
    source_return_string = ""
    demand_return_string = ""
    all_existing_GWh = 0
    all_existing_gen_costs = 0
    all_demand_GWh = 0
    all_demand_wholesale_costs = 0

    summary_by_source = []
    
    for source_key, source_col in sources.items():

        existing_GWh = node_df[source_col].sum()
        GW_in_node = node_df[source_col].quantile(quantile_used)
        row_count = node_df[source_col].count()
        capacity_factor = 100 * safe_divide( existing_GWh, ( GW_in_node * row_count ) )
        GW_in_node_by_source[source_key] = GW_in_node

        result_col = f"{source_key}_captured"
        node_df[result_col] = (node_df["DE wholesale price [EUR/MWh]"].clip(lower=0)) * node_df[source_col]
        captured_total = node_df[result_col].sum()
        capture_rate = safe_divide(captured_total, existing_GWh)

        result_col = f"{source_key}_wholesale"
        node_df[result_col] = (node_df["DE wholesale price [EUR/MWh]"]) * node_df[source_col]
        wholesale_total = node_df[result_col].sum()

        if existing_GWh != 0:
            summary_by_source.append({
                "Source": source_key,
                "Total GWh": existing_GWh,
                "Capacity (GW)": GW_in_node,
                "Rows": row_count,
                "Capacity Factor (%)": capacity_factor,
                "Capture rate (€/MWh)": capture_rate,
                "Wholesale total (EUR)": wholesale_total
            })

            if source_key != "demand":
                source_return_string += f"\n{existing_GWh:,.0f} GWh {source_key}, {capacity_factor:,.0f}% CF, EUR {capture_rate:,.1f}/MWh"
                all_existing_GWh += existing_GWh
                all_existing_gen_costs += wholesale_total
            else:
                demand_return_string = f"\n{existing_GWh:,.0f} GWh {source_key}, {capacity_factor:,.0f}% CF, EUR {capture_rate:,.1f}/MWh"
                all_demand_GWh = existing_GWh
                all_demand_wholesale_costs = wholesale_total

    summary_by_source_df = pd.DataFrame(summary_by_source)

    # Fix all_existing costs to EUR, currently kEUR
    all_existing_gen_costs = all_existing_gen_costs * 1_000
    all_demand_wholesale_costs = all_demand_wholesale_costs * 1_000

    wholesale_rate = safe_divide(all_existing_gen_costs, ( all_existing_GWh * 1_000 ))

    return_string += f"{all_existing_GWh:,.0f} GWh, EUR {wholesale_rate:,.1f}/MWh{source_return_string}"

    # Assess impact of Agora-assessed price based on net export / import balance
    node_df["Ex.-(Im.) bin"] = pd.qcut(
        node_df["Net export / (import) GWh"],
        q=10,
        duplicates="drop"
    )
    node_df["Ex.-(Im.) bin label"] = node_df["Ex.-(Im.) bin"].apply(
        lambda interval: f"{interval.left:.2f} to {interval.right:.2f} GW"
    )
    def sumproduct_abs(x):
        return (x["Net export / (import) GWh"].abs() * x["Agora wholesale price [EUR/MWh]"]).sum()
    ex_im_impact_binned = (
        node_df.groupby("Ex.-(Im.) bin label", observed=False)
        .apply(lambda group: pd.Series({
            "Agora wholesale price [EUR/MWh]": group["Agora wholesale price [EUR/MWh]"].mean(),
            "DE wholesale price [EUR/MWh]": group["Agora wholesale price [EUR/MWh]"].mean(),
            "Sum net export / (import) GWh": group["Net export / (import) GWh"].sum(),
            "Sumproduct MEUR cost of net GWh": sumproduct_abs(group) / 1_000
        }))
        .reset_index()
    )
    ex_im_impact_binned_display_df = ex_im_impact_binned.copy()
    for col in ex_im_impact_binned_display_df.columns:
        if col != "Ex.-(Im.) bin label":
            ex_im_impact_binned_display_df[col] = ex_im_impact_binned_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

    return (
        node_df, 
        capture_results, 
        return_string, 
        GW_in_node_by_source, 
        all_existing_GWh, 
        all_existing_gen_costs, 
        demand_return_string, 
        all_demand_GWh, 
        all_demand_wholesale_costs, 
        ex_im_impact_binned, 
        ex_im_impact_binned_display_df, 
        summary_by_source_df
    )

def showGenPriceDeciles(df, year):

    node_list = df["Selected hub"].unique()

    decile_gen_vs_price_sources = {
        "Solar",
        "Wind onshore",
        "Wind offshore",
        "Natural Gas"
    }

    standard_df = df[df["Datetime"].dt.year == year]

    # Compress dataset by unique timestamps for DE specifically
    germany_df = (
        standard_df
        .groupby("Datetime")[list(decile_gen_vs_price_sources)]
        .sum()
        .reset_index()
    )

    #hourly_totals = {}
    hourly_totals_bins = {}
    for source in decile_gen_vs_price_sources:
        shaken = germany_df[source] + np.random.uniform(0, 1e-6, size=len(germany_df))
        _, bins = pd.qcut(shaken, q=10, retbins=True, duplicates="drop")
        hourly_totals_bins[source] = bins


    all_results = []

    def runDeciles(sub_df, node, bins=None, use_hourly_totals=False):
        results = []

        for source in decile_gen_vs_price_sources:
            
            # Skip offshore wind if the node doesn't have any
            if sub_df[source].sum() == 0:
                continue
            
            temp_df = sub_df.copy()
            decile_col = "__temp_decile__"

            # "use_hourly_totals" condition means running all of DE, and need to use compressed dataframe and its specifically calculated bins
            if use_hourly_totals:
                # Map total generation from compressed germany_df by hour to rows
                temp_df['total_gen_all_nodes'] = temp_df['Datetime'].map(
                    germany_df.set_index('Datetime')[source]
                )
                # Assign deciles using provided bins based on total gen across all nodes
                temp_df[decile_col] = pd.cut(
                    temp_df['total_gen_all_nodes'],
                    bins=bins[source],
                    include_lowest=True
                )
            else:
                # Separate zero and positive generation rows
                zero_mask = temp_df[source] == 0
                nonzero = temp_df.loc[~zero_mask].copy()

                if len(nonzero) >= 10:
                    # Add jitter and calculate deciles only on positive generation
                    shaken_not_stirred = nonzero[source] + np.random.uniform(0, 1e-6, size=len(nonzero))
                    nonzero[decile_col] = pd.qcut(
                        shaken_not_stirred,
                        q=10,
                        duplicates="drop"
                    )
                    # Assign decile bins back to temp_df, zeros get their own bin label
                    temp_df[decile_col] = np.nan
                    temp_df[decile_col] = temp_df[decile_col].astype("object")
                    temp_df.loc[zero_mask, decile_col] = "Zero generation"
                    temp_df.loc[~zero_mask, decile_col] = nonzero[decile_col].astype(str)
                else:
                    # If too few positive values, fallback to regular qcut on all rows
                    shaken_not_stirred = temp_df[source] + np.random.uniform(0, 1e-6, size=len(temp_df))
                    temp_df[decile_col] = pd.qcut(
                        shaken_not_stirred,
                        q=10,
                        duplicates="drop"
                    )

            grouped = (
                temp_df
                .groupby(temp_df[decile_col], observed=True, group_keys=False)
                .apply(lambda group: pd.Series({
                    "Agora price [EUR/MWh]": group["Agora wholesale price [EUR/MWh]"].mean(),
                    "DE price [EUR/MWh]": group["DE wholesale price [EUR/MWh]"].mean()
                }))
                .reset_index()
                .rename(columns={decile_col: "decile_gen_vs_price"})
            )

            def sort_key(x):
                if x == "Zero generation":
                    return -1  # Put this first
                elif isinstance(x, pd.Interval):
                    return x.left  # Sort intervals by left bound
                else:
                    return float('inf')  # Any unexpected value goes last

            grouped = grouped.sort_values(by="decile_gen_vs_price", key=lambda col: col.map(sort_key)).reset_index(drop=True)  

            def format_bin(interval):
                if isinstance(interval, pd.Interval):
                    return f"{interval.left:.2f} to {interval.right:.2f} GW"
                else:
                    return str(interval)  # e.g. "Zero generation"
                
            grouped["GW gen."] = grouped["decile_gen_vs_price"].apply(format_bin)
            
            #grouped["GW gen."] = grouped["decile_gen_vs_price"].apply(
            #    lambda interval: f"{interval.left:.2f} to {interval.right:.2f} GW"
            #)
            grouped = grouped.drop(columns=["decile_gen_vs_price"])
            cols = grouped.columns.tolist()
            cols.insert(0, cols.pop(cols.index("GW gen.")))
            grouped = grouped[cols]

            grouped["Source"] = source
            grouped["Node"] = node
            grouped["Year"] = year

            results.append(grouped)

            # Optional final cleanup
            if decile_col in temp_df.columns:
                del temp_df[decile_col]

        #print(f"✅ Completed node: {node} | Total rows: {sum(len(r) for r in results)}")
        return pd.concat(results, ignore_index=True)

    # Run first for all of Germany
    all_results.append(runDeciles(standard_df, "All DE", bins=hourly_totals_bins, use_hourly_totals=True))

    # Run for all nodes
    for node in node_list:
        node_df = standard_df[standard_df["Selected hub"] == node]
        if node_df.empty:
            continue
        all_results.append(runDeciles(node_df, node=node, bins=None, use_hourly_totals=False))

    final_df = pd.concat(all_results, ignore_index=True)

    return final_df

# Note: This isn't part of the main series function, which focuses on running everything for a selected node - but this will run it for every single node and print results, fast way to see entire set
def showAllNodeCFs():

    # Load target file
    TARGET_FILE = ".csv"
    df = pd.read_csv(TARGET_FILE)

    node_list = df["Selected hub"].unique()

    for node in node_list:
        showNodeCF(TARGET_FILE, node)

def expandIntRenInNode(df, node, year, boost_pcts=None, cost_per_mw=None, GW_in_node_by_source=None):

    (
        node_df,
        capture_results,
        return_string,
        GW_in_node_by_source, 
        all_existing_GWh, 
        all_existing_gen_costs, 
        demand_return_string, 
        all_demand_GWh, 
        all_demand_wholesale_costs, 
        ex_im_impact_binned, 
        ex_im_impact_binned_display_df, 
        summary_by_source_df 
    ) = showNodeCF(df, node, year)

    # Set displaced capture rates to 0 until passed values
    displaced_all_capture = 0
    displaced_positive_capture = 0
    
    sources = {
        "wind offshore": "Wind offshore",
        "wind onshore": "Wind onshore",
        "solar": "Solar",
    }

    cost_summary = ""
    marginal_impact = ""
    cost_dict = {}

    # Define initial hourly_gen_df with existing sources, before it gets overwritten by new one once user submits inputs
    hourly_gen_df = pd.DataFrame({
        "Hour": list(range(24)),
        "Solar": [0]*24,
        "Wind onshore": [0]*24,
        "Wind offshore": [0]*24,
        "Displacable generation": [0]*24,
        "Extra solar": [0]*24,
        "Extra wind onshore": [0]*24,
        "Extra wind offshore": [0]*24,
        "Spilled solar": [0]*24,
        "Spilled wind onshore": [0]*24,
        "Spilled wind offshore": [0]*24,
        "Remaining displacable generation": [0]*24
    })
    hourly_gen_df.set_index("Hour", inplace=True)

    # Make an "hours" column that can be used for avg. generation per hour
    node_df["Timestamp"] = pd.to_datetime(node_df["Datetime"])
    node_df["Hour"] = node_df["Timestamp"].dt.hour

    # Calc. avg. generation per hour by generation source
    existing_columns = [
        "Solar", "Wind onshore", "Wind offshore", "Displacable generation"
    ]

    hourly_means = node_df.groupby("Hour")[existing_columns].mean()
    for col in existing_columns:
        hourly_gen_df[col] = hourly_means[col]

    # Initialize this, as only gets set once user inputs submitted
    cost_pool_after_substituted_gen = 0

    if boost_pcts and cost_per_mw and GW_in_node_by_source:

        hourly_gen_df["Displacable generation"] = 0

        # Create a duplicate of "Displacable generation" that will then be taken down by further renewables
        node_df["Remaining displacable generation"] = node_df["Displacable generation"]

        boost_pcts = {k.lower(): v for k, v in boost_pcts.items()}
        cost_per_mw = {k.lower(): v for k, v in cost_per_mw.items()}
        GW_in_node_by_source = {k.lower(): v for k, v in GW_in_node_by_source.items()}

        for source_key, source_col in sources.items():

            boost_pct = boost_pcts[source_key]
            node_df[f"Extra {source_key}"] = node_df[source_col] * boost_pct

            hourly_means = node_df.groupby("Hour")[f"Extra {source_key}"].mean()
            hourly_gen_df[f"Extra {source_key}"] = hourly_means

            # Any new renewable generation that exceeds remaining dispatchable generation is excess
            node_df[f"Spilled {source_key}"] = (
                node_df[f"Extra {source_key}"] - node_df["Remaining displacable generation"]
            ).clip(lower=0)
            hourly_means = node_df.groupby("Hour")[f"Spilled {source_key}"].mean()
            hourly_gen_df[f"Spilled {source_key}"] = hourly_means

            # Remove new renewable generation from "Remaining displacable generation" that it's assumed to offset 
            node_df["Remaining displacable generation"] = (
                node_df["Remaining displacable generation"] - node_df[f"Extra {source_key}"]
            ).clip(lower=0)

        hourly_means = node_df.groupby("Hour")["Remaining displacable generation"].mean()
        hourly_gen_df["Remaining displacable generation"] = hourly_means
        
        total_extra_by_source = {}
        for source_key in sources:
            total_extra_by_source[source_key] = node_df[f"Extra {source_key}"].sum()

        total_extra_GWh = sum(total_extra_by_source.values())

        node_df["Extra renewables total"] = sum(
            node_df[f"Extra {source_key}"] for source_key in sources
        )

        node_df["Displaced"] = node_df[["Extra renewables total", "Displacable generation"]].min(axis=1)
        node_df["Excess"] = node_df["Extra renewables total"] - node_df["Displaced"]

        total_displaced_GWh = node_df["Displaced"].sum()

        # Calculate displaced capture rate - i.e., what the now-displaced generation was earning based on wholesale prices, and theoretically what the new renewable generation is by replacing it
        node_df["Displaced capture"] = (node_df["Displaced"] * node_df["DE wholesale price [EUR/MWh]"])
        sum_displaced_capture = node_df["Displaced capture"].sum()
        displaced_all_capture = safe_divide(sum_displaced_capture, total_displaced_GWh)
        displaced_positive_capture = safe_divide(node_df.loc[node_df["Displaced capture"] > 0, "Displaced capture"].sum(), total_displaced_GWh)

        extra_costs = {}
        unit_costs = {}

        for source_key in sources:
            gw = GW_in_node_by_source[source_key]
            boost = boost_pcts[source_key]
            this_source_cost = cost_per_mw[source_key]
            extra_GWh = total_extra_by_source[source_key]

            extra_costs[source_key] = gw * boost * this_source_cost
            unit_costs[source_key] = safe_divide(extra_costs[source_key], extra_GWh)

        total_extra_cost = sum(extra_costs.values())
        total_new_gen_unit_cost = safe_divide(total_extra_cost, total_extra_GWh)
        displaced_unit_cost = safe_divide(total_extra_cost, total_displaced_GWh)

        cost_dict = {
            source_key: {
                "Gen. (GWh)": total_extra_by_source[source_key],
                "Cost (MEUR)": extra_costs[source_key]/1_000,
                "Unit cost (EUR/MWh)": unit_costs[source_key]
            }
            for source_key in sources
        }

        cost_dict["Total"] = {
            "Gen. (GWh)": sum(total_extra_by_source.values()),
            "Cost (MEUR)": total_extra_cost/1_000,
            "Unit cost (EUR/MWh)": total_new_gen_unit_cost
        }

        cost_dict["Displaced gen."] = {
            "Gen. (GWh)": total_displaced_GWh,
            "Cost (MEUR)": total_extra_cost/1_000,
            "Unit cost (EUR/MWh)": displaced_unit_cost
        }

        # Now process marginal impact of incremental generation

        new_gen_GWh = cost_dict["Total"]["Gen. (GWh)"]
        abated_gen_GWh = cost_dict["Displaced gen."]["Gen. (GWh)"]
        abatement_share = 100 * safe_divide(abated_gen_GWh, new_gen_GWh)
        new_gen_unit_cost = cost_dict["Total"]["Unit cost (EUR/MWh)"]
        unit_cost_per_new_gen_used = safe_divide(new_gen_unit_cost, abatement_share/100)

        node_df["Displaced share of displacable"] = np.where(
            node_df["Displacable generation"] != 0,
            node_df["Displaced"] / node_df["Displacable generation"],
            0
        )

        node_df["Abated tCO2"] = node_df["Displaced share of displacable"] * node_df["Total grid emissions"]

        emissions_abatement = (node_df["Displaced share of displacable"] * node_df["Total grid emissions"]).sum()

        # So now need to calculate the higher whole-system wholesale energy cost, then divide abatement by that...
        cost_pool_minus_displaced_gen = all_existing_gen_costs - sum_displaced_capture
        cost_pool_after_substituted_gen = cost_pool_minus_displaced_gen + total_extra_cost
        cost_pool_delta = cost_pool_after_substituted_gen - all_existing_gen_costs
        abatement_cost = safe_divide(1_000*cost_pool_delta, emissions_abatement)
 
        marginal_impact = (
            f"Added {new_gen_GWh:,.0f} GWh at {new_gen_unit_cost:,.0f}/MWh gen. unit cost, "
            f"of which {abatement_share:,.0f}% was used to abate dispatchable gen., for EUR {unit_cost_per_new_gen_used:,.0f}/MWh new gen. used"
            f"\nAbated {abated_gen_GWh:,.0f} GWh dispatchable gen., earning wt. wholesale {displaced_all_capture:,.0f}/MWh all-inclusive, {displaced_positive_capture:,.0f}/MWh ignoring negatives (i.e., also implied new ren. capture price)"
            f"\nNet generation cost shift from {all_existing_gen_costs/1_000:,.0f} MEUR to {cost_pool_after_substituted_gen/1_000:,.0f} MEUR, change of {cost_pool_delta/1_000:,.0f} MEUR ({100*(cost_pool_delta/all_existing_gen_costs):+,.1f}%)"
            f"\nRough est. CO2 abatement {emissions_abatement:,.0f}t, {abatement_cost:,.0f}/tCO2"
        )

    # Also processed binned results into something fun
    hypothetical_fallback_cost = summary_by_source_df.loc[summary_by_source_df["Source"] == "displacable", "Capture rate (€/MWh)"].values[0]
    ex_im_impact_binned["DE wholesale price [EUR/MWh]"] = pd.to_numeric(
        ex_im_impact_binned["DE wholesale price [EUR/MWh]"], errors="coerce"
    )
    ex_im_impact_binned["Sum net export / (import) GWh"] = pd.to_numeric(
        ex_im_impact_binned["Sum net export / (import) GWh"], errors="coerce"
    )
    ex_im_impact_binned["Delta vs. X cost (MEUR)"] = ex_im_impact_binned.apply(
        lambda row: (hypothetical_fallback_cost - row["DE wholesale price [EUR/MWh]"]) * abs(row["Sum net export / (import) GWh"]) / 1_000,
        axis=1
    )

    ex_im_impact_binned_display_df = ex_im_impact_binned.copy()
    for col in ex_im_impact_binned_display_df.columns:
        if col != "Ex.-(Im.) bin label":
            ex_im_impact_binned_display_df[col] = ex_im_impact_binned_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

    
    return (
        node_df,
        capture_results,
        return_string,
        GW_in_node_by_source,
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
        summary_by_source_df,
    )

def assessLineCosts(TARGET_FILE, year):

    # Load target file
    df = pd.read_csv(TARGET_FILE)

    # Parse date column, extract year, and downselect it
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Year"] = df["Datetime"].dt.year
    print("Initial processed")
    #df = df[df["Year"] == year].copy()

    df["Flow_MWh"] = np.where(
        df["Power export [MWh/hour]"] > 0,
        df["Power export [MWh/hour]"],
        np.where(
            df["Power import [MWh/hour]"] > 0,
            df["Power import [MWh/hour]"],
            0
        )
    )

    df["Mode"] = np.where(
        df["Power export [MWh/hour]"] > 0,
        "aligned",
        np.where(df["Power import [MWh/hour]"] > 0, "flipped", "unknown")
    )

    df["From"] = np.where(
        df["Mode"] == "aligned",
        df["From location"],
        np.where(df["Mode"] == "flipped", df["To location"], df["From location"])
    )

    df["To"] = np.where(
        df["Mode"] == "aligned",
        df["To location"],
        np.where(df["Mode"] == "flipped", df["From location"], df["To location"])
    )

    print("Direction determined")

    df["GWh flow"] = df["Flow_MWh"] / 1_000

    # Drop duplicates across final desired columns
    df = df.drop_duplicates(subset=["From", "To", "Datetime"])

    # Addressing data issue of nameplate capacity varying for a given line across time in some places, by setting to average
    avg_capacity = (
        df.groupby(["From", "To", "Year"])["Capacity [MW]"]
        .mean()
        .reset_index()
        .rename(columns={"Capacity [MW]": "Listed capacity [GW]"})
    )
    avg_capacity["Listed capacity [GW]"] = avg_capacity["Listed capacity [GW]"] / 1_000
    df = df.drop(columns=["Capacity [MW]"], errors="ignore")
    df = pd.merge(df, avg_capacity, on=["From", "To", "Year"], how="left")

    df = df.drop(columns=[
        "From location",
        "To location",
        "Power export [MWh/hour]",
        "Power import [MWh/hour]",
        "Flow_MWh",
        "Mode"
    ])

    total_GWh_flow_by_year = (
        df.groupby(["Year", "From", "To"])["GWh flow"]
        .sum()
        .reset_index()
        .rename(columns={"GWh flow": "Total flow [GWh]"})
    )
    print("GWh by year processed")

    # Set given percentile of flow per year, per pair, as capacity metric (given dataset itself is inconsistent on capacity)
    capacity_proxy = (
        df.groupby(["Year", "From", "To"])["GWh flow"]
        .quantile(0.99)
        .reset_index()
        .rename(columns={"GWh flow": "Max. used capacity [GW]"})
    )

    # Retain only one capacity value per From-To pair per year
    listed_capacity = (
        df[["Year", "From", "To", "Listed capacity [GW]"]]
        .drop_duplicates(subset=["Year", "From", "To", "Listed capacity [GW]"])
    )

    merged_df = pd.merge(
        total_GWh_flow_by_year,
        capacity_proxy,
        on=["Year", "From", "To"],
        how="inner"
    )
    
    merged_df = pd.merge(
        merged_df,
        listed_capacity,
        on=["Year", "From", "To"],
        how="left"
    )

    # Calculate hours in year
    merged_df["Hours in year"] = merged_df["Year"].apply(lambda y: 8784 if y in [2020, 2024] else 8760)

    # Compute utilization
    merged_df["Utilization [%]"] = 100 * merged_df["Total flow [GWh]"] / ( merged_df["Max. used capacity [GW]"] * merged_df["Hours in year"] )

    # Categorize flows as international or domestic
    def is_cross_border(node_pair, TRADING_COUNTRIES):
        if not isinstance(node_pair, (tuple, list)):
            return False
        return any(str(location) in TRADING_COUNTRIES for location in node_pair if pd.notna(location))
    
    merged_df["Cross-border tag"] = merged_df.apply(
        lambda row: "Cross-border" if is_cross_border((row["From"], row["To"]), TRADING_COUNTRIES) else "Internal",
        axis=1
    )

    #merged_df.to_csv("testPreCost.csv", index=False, encoding="utf-8-sig")
    #print(f"Saved PreCost file")

    # ASSIGN COSTS ------------------------------------------

    # Assign transmission and distribution cost pools
    cost_pools = {
        2019: {
            "international_transmission_cost_pool": 225_000_000,
            "internal_transmission_cost_pool": 5_000_000_000,
            "internal_distribution_cost_pool": 24_000_000_000,
            "internal_OW_link_cost_pool": 2_000_000_000,
        },
        2020: {
            "international_transmission_cost_pool": 225_000_000,
            "internal_transmission_cost_pool": 5_500_000_000,
            "internal_distribution_cost_pool": 25_000_000_000,
            "internal_OW_link_cost_pool": 2_000_000_000,
        },
        2021: {
            "international_transmission_cost_pool": 275_000_000,
            "internal_transmission_cost_pool": 6_000_000_000,
            "internal_distribution_cost_pool": 26_000_000_000,
            "internal_OW_link_cost_pool": 1_900_000_000,
        },
        2022: {
            "international_transmission_cost_pool": 275_000_000,
            "internal_transmission_cost_pool": 6_500_000_000,
            "internal_distribution_cost_pool": 27_000_000_000,
            "internal_OW_link_cost_pool": 2_100_000_000,
        },
        2023: {
            "international_transmission_cost_pool": 275_000_000,
            "internal_transmission_cost_pool": 7_000_000_000,
            "internal_distribution_cost_pool": 28_000_000_000,
            "internal_OW_link_cost_pool": 2_900_000_000,
        },
        2024: {
            "international_transmission_cost_pool": 275_000_000,
            "internal_transmission_cost_pool": 7_500_000_000,
            "internal_distribution_cost_pool": 28_500_000_000,
            "internal_OW_link_cost_pool": 3_300_000_000,
        }
    }

    # HERE: Need to rework this thing to run for all years, selecting right cost base for each
    
    #Original, to delete when all working
    #international_transmission_cost_pool = 225_000_000
    #internal_transmission_cost_pool = 6_500_000_000
    internal_distribution_cost_pool = 25_000_000_000
    #internal_OW_link_cost_pool = 2_250_000_000

    # Between the two nodes, in 2024 Nordsee-West appeared to have around 75% the OW generation capacity and NOS 25%, so set based on that. Could divide in any given year by share of generation
    internal_DE_nodes_with_OW_links = {
        (node, yr): share * cost_pools[year]["internal_OW_link_cost_pool"]
        for yr in range(2019, 2025)  # 2025 excluded → ends at 2024
        for node, share in {
            "Nord-Ost-See": .35,
            "Nordsee-West": .65
        }.items()
    }

    ow_nodes = {node for (node, _) in internal_DE_nodes_with_OW_links.keys()}

    # DISTRIBUTE COSTS FOR CROSS-BORDER TRANSMISSION LINES ------------------------------------------

    # Filter a df for cross-border only
    international_df = merged_df[merged_df["Cross-border tag"] == "Cross-border"]
    #international_df = merged_df[
    #    (merged_df["Cross-border tag"] == "Cross-border") & (merged_df["Year"] == year)
    #].copy()

    # Calculate total capacity per year
    total_international_capacity_by_year = international_df.groupby("Year")["Max. used capacity [GW]"].transform("sum")
    # Calculate share of total capacity (by year)
    international_df["Int'l cost share"] = (
        international_df["Max. used capacity [GW]"] / total_international_capacity_by_year
    )
    
    # Look up correct cost pool
    def get_international_cost(year):
        return cost_pools.get(year, {}).get("international_transmission_cost_pool", 0)
    
    international_df["Year_cost_pool"] = international_df["Year"].apply(get_international_cost)

    # Calculate cost per row
    international_df["Int'l cost"] = (
        international_df["Int'l cost share"] * international_df["Year_cost_pool"]
    )

    international_df.drop(columns=["Year_cost_pool"], inplace=True)

    # Merge back into main df
    merged_df = pd.merge(
        merged_df,
        international_df[["Year", "From", "To", "Int'l cost"]],
        on=["Year", "From", "To"],
        how="left"
    )

    # DISTRIBUTE COSTS FOR DOMESTIC TRANSMISSION LINES ------------------------------------------

    # Filter a df for domestic transmission only
    domestic_df = merged_df[merged_df["Cross-border tag"] == "Internal"]
    #domestic_df = merged_df[
    #    (merged_df["Cross-border tag"] == "Internal") & (merged_df["Year"] == year)
    #].copy()

    # Calculate total capacity per year
    total_domestic_capacity_by_year = domestic_df.groupby("Year")["Max. used capacity [GW]"].transform("sum")

    # Calculate share of total capacity
    domestic_df["Domestic transmission cost share"] = (
        domestic_df["Max. used capacity [GW]"] / total_domestic_capacity_by_year
    )

    # Look up correct cost pool
    def get_domestic_transmission_cost(year):
        return cost_pools.get(year, {}).get("internal_transmission_cost_pool", 0)
    
    domestic_df["Year_cost_pool"] = domestic_df["Year"].apply(get_domestic_transmission_cost)

    # Calculate cost
    domestic_df["Domestic transmission cost"] = (
        domestic_df["Domestic transmission cost share"] * domestic_df["Year_cost_pool"]
    )

    # Merge back into main df
    merged_df = pd.merge(
        merged_df,
        domestic_df[["Year", "From", "To", "Domestic transmission cost"]],
        on=["Year", "From", "To"],
        how="left"
    )

    merged_df.to_csv("testDomTransCost.csv", index=False, encoding="utf-8-sig")
    print(f"Saved DomTransCost file")

    # PROCESS DATAFRAME FOR NEXT STEPS ------------------------------------------

    # Define new column about whether power is flowing to German demand (i.e., excluding flows out of DE, which are included in export earnings and otherwise ignored in further analysis)
    merged_df["Serves DE demand"] = merged_df.apply(
        lambda row: (
            row["From"] in TRADING_COUNTRIES and row["To"] not in TRADING_COUNTRIES
            if row["Cross-border tag"] == "Cross-border" else True
        ),
        axis=1
    )

    # Get all columns, then reorder column to be directly after "Cross-border tag"
    cols = merged_df.columns.tolist()
    idx = cols.index("Cross-border tag")
    cols.remove("Serves DE demand")
    cols.insert(idx + 1, "Serves DE demand")

    # Reorder dataframe
    merged_df = merged_df[cols]

    # Create united cost column from international and domestic transmission cost columns
    merged_df["Total transmission cost"] = (
        merged_df["Int'l cost"].fillna(0) + merged_df["Domestic transmission cost"].fillna(0)
    )

    merged_df["Total cost per MWh from transmission lines"] = safe_divide(
        merged_df["Total transmission cost"],
        merged_df["Total flow [GWh]"]
    )

    # BUILD internal_node_df, ALL NODES INSIDE GERMANY, THEIR INTERNAL POWER GEN. AND DISTRIBUTION COSTS, FOR LATER LINKING WITH TRANSMISSION ------------------------------------------

    # Size Germany nodes' peak loads as a rough proxy for DSO cost share
    # TAG:SM - Non-systematic, in regular series should get processed in costSumInsideDE with result returned
    internal_node_df = pd.read_csv("nodal_dem_and_gen_w_prices_added_2025_07_11_16_07.csv")

    # DIVERSION UP HERE: Need to get, for each node, total power actually consumed, in order to figure out distribution cost per MWh delivered

    # Parse date and extract year
    internal_node_df["Datetime"] = pd.to_datetime(internal_node_df["Datetime"], errors="coerce")
    internal_node_df["Year"] = internal_node_df["Datetime"].dt.year
    DE_node_capacities = (
        internal_node_df
        .groupby(["Selected hub", "Year"])["Total electricity demand"]
        .quantile(0.99)
        .reset_index()
        .rename(columns={"Total electricity demand": "99%ile demand [MW]"})
    )
    annual_nodal_demand = (
        internal_node_df
        .groupby(["Selected hub", "Year"])["Total electricity demand"]
        .sum()
        .reset_index()
        .rename(columns={"Total electricity demand": "Annual nodal GWh demand"})
    )
    DE_node_capacity_dict = {
        (row["Selected hub"], row["Year"]): row["99%ile demand [MW]"]
        for _, row in DE_node_capacities.iterrows()
    }

    # Grab OW generation specifically for the nodes with it
    filtered_df = internal_node_df[internal_node_df["Selected hub"].isin(ow_nodes)]

    ow_gen_df = (
        filtered_df
        .groupby(["Selected hub", "Year"])["Wind offshore"]
        .sum()
        .reset_index()
        .rename(columns={
            "Selected hub": "Node",
            "Wind offshore": "Annual OW generation [GWh]"
        })
    )

    # Grab all nodes' generation profiles for later processing annual gen., dem., import and export
    all_nodes = internal_node_df["Selected hub"].dropna().unique()
    all_gen_by_node = []

    GEN_COLUMNS = [
        "Conventional",
        "Biomass",
        "Hydro",
        "Solar",
        "Wind onshore",
        "Wind offshore"
    ]

    # Calculate total generation per node-year
    internal_node_df["Total generation"] = internal_node_df[GEN_COLUMNS].sum(axis=1)

    all_gen_df = (
        internal_node_df
        .groupby(["Selected hub", "Year"])["Total generation"]
        .sum()
        .reset_index()
        .rename(columns={
            "Selected hub": "Node",
            "Total generation": "Annual total generation [GWh]"
        })
    )

    capacity_df = pd.DataFrame.from_dict(DE_node_capacity_dict, orient="index").reset_index()
    capacity_df[["Node", "Year"]] = pd.DataFrame(capacity_df["index"].tolist(), index=capacity_df.index)
    capacity_df = capacity_df.drop(columns="index")
    capacity_df = capacity_df.rename(columns={0: "99%ile demand [MW]"})

    capacity_df["Sum peak DE yearly GW DSO capacity"] = capacity_df.groupby("Year")["99%ile demand [MW]"].transform("sum")
    capacity_df["Demand share"] = capacity_df["99%ile demand [MW]"] / capacity_df["Sum peak DE yearly GW DSO capacity"]
    capacity_df["Domestic distribution cost"] = capacity_df["Demand share"] * internal_distribution_cost_pool
    capacity_df = capacity_df.merge(
        annual_nodal_demand,
        left_on=["Node", "Year"],
        right_on=["Selected hub", "Year"],
        how="left"
    ).drop(columns="Selected hub")


    capacity_df = capacity_df.merge(
        ow_gen_df,
        on=["Node", "Year"],
        how="left"
    )
    capacity_df["Annual OW generation [GWh]"] = capacity_df["Annual OW generation [GWh]"].fillna(0)

    capacity_df = capacity_df.merge(
        all_gen_df,
        on=["Node", "Year"],
        how="left"
    )
    capacity_df["Annual total generation [GWh]"] = capacity_df["Annual total generation [GWh]"].fillna(0)

    #capacity_df = capacity_df[capacity_df["Year"] == year].copy()
    #print("Here?")

    capacity_df["OW cost"] = 0  # Init to zero

    for (node, yr), cost in internal_DE_nodes_with_OW_links.items():
        capacity_df.loc[
            (capacity_df["Node"] == node) & (capacity_df["Year"] == yr),
            "OW cost"
        ] += cost

    #capacity_df.to_csv("capacityTest.csv", index=False, encoding="utf-8-sig")
    #print(f"Saved capacityTest")

    
    # COMBINE internal_node_df, ALL NODES INSIDE GERMANY, WITH merged_df DATA ON TRANSMISSION INTO EACH NODE ------------------------------------------

    # This assumes "To" is the destination node
    incoming_flows = merged_df[
        (merged_df["To"].isin(capacity_df["Node"]))
    ].copy()

    grouped = (
        merged_df[merged_df["To"].isin(capacity_df["Node"])]
        .groupby(["To", "Year"])
        .agg({
            "From": list,
            "Total flow [GWh]": list,
            "Total transmission cost": list,
            "Utilization [%]": list,
            "Total cost per MWh from transmission lines": list
        })
        .reset_index()
        .rename(columns={"To": "Node"})
    )


    capacity_df = capacity_df.merge(grouped, on=["Node", "Year"], how="left")
    #capacity_df = capacity_df[capacity_df["Year"] == year].copy()

    # Sum imported energy and costs of affiliated transmision lines
    capacity_df["Sum imported GWh"] = capacity_df["Total flow [GWh]"].apply(lambda lst: sum(lst) if isinstance(lst, list) else 0)
    capacity_df["Sum transmission cost"] = capacity_df["Total transmission cost"].apply(lambda lst: sum(lst) if isinstance(lst, list) else 0)

    # Sum exported energy for a given node by checking all other nodes' imports from that node
    
    # Initialize export GWh column
    capacity_df["Sum domestically exported GWh"] = 0.0
    # Iterate through each node and compute how much it exported
    for i, row in capacity_df.iterrows():
        this_node = row["Node"]
        this_year = row["Year"]
        exported_sum = 0.0

        for j, other_row in capacity_df.iterrows():
            if i == j or other_row["Year"] != this_year:
                continue

            if "From" in other_row and isinstance(other_row["From"], list):
                for from_node, gwh in zip(other_row["From"], other_row.get("Total flow [GWh]", [])):
                    if from_node == this_node:
                        exported_sum += gwh

        capacity_df.at[i, "Sum domestically exported GWh"] = exported_sum

    # TAG: SM - Normally saves nodal cost file here for passing to processLineCosts, but now setting that one to run directly from the already-run Nodal_cost_file
    #capacity_df.to_csv("Nodal_cost_file.csv", index=False, encoding="utf-8-sig")
    #print(f"Saved nodal cost file")


# HERE. Current nodal cost file has the right years in it, just need to make sure this is appropriately downselecting within it
def processLineCosts(node, year):

    NODAL_FILE = "https://github.com/user-attachments/files/21368252/Nodal_cost_file.csv"
    df = pd.read_csv(NODAL_FILE)
    df = df[df["Year"] == year]
    df.set_index("Node", inplace=True)

    if node not in df.index:
        return f"Node '{node}' not found for year {year}"

    node_row = df.loc[node]

    # If multiple rows match (shouldn't happen here due to year filtering), pick the first but flag
    if isinstance(node_row, pd.DataFrame):
        node_row = node_row.iloc[0]
        print("ALERT: Multiple results hit in processLineCost, need to check")
    
    GWh_DSO = node_row["Annual nodal GWh demand"]
    cost_DSO = node_row["Domestic distribution cost"]
    GWh_OW = node_row.get("Annual OW generation [GWh]", 0)
    cost_OW = node_row.get("OW cost", 0)
    GWh_TSO = node_row.get("Sum imported GWh", 0)
    cost_TSO = node_row.get("Sum transmission cost", 0)

    total_TD_costs = cost_DSO + cost_OW + cost_TSO
    
    DSO_unit = safe_divide ( cost_DSO, ( GWh_DSO * 1_000 ) )
    OW_unit = safe_divide ( cost_OW, ( GWh_OW * 1_000 ) )
    TSO_unit = safe_divide ( cost_TSO, ( GWh_TSO * 1_000 ) )

    OW_unit_all_demand = cost_OW / ( GWh_DSO * 1_000 )
    TSO_unit_all_demand = cost_TSO / ( GWh_DSO * 1_000 )

    return_string_TandD_added = (
        f"\nDSO: {cost_DSO/1_000_000:,.0f} MEUR, {GWh_DSO:,.0f} GWh, EUR {DSO_unit:,.0f}/MWh"
        f"\nTSO: {cost_TSO/1_000_000:,.0f} MEUR, {GWh_TSO:,.0f} GWh, EUR {TSO_unit:,.0f}/MWh (per DSO unit, EUR {TSO_unit_all_demand:,.0f}/MWh)"
    )

    if GWh_OW > 0:
        return_string_TandD_added += (
            f"\nOW: {cost_OW/1_000_000:,.0f} MEUR, {GWh_OW:,.0f} GWh, EUR {OW_unit:,.0f}/MWh (per DSO unit, EUR {OW_unit_all_demand:,.0f}/MWh)"
        )

    return return_string_TandD_added, total_TD_costs

# ArgParse to enable various modes
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run sheet processing")
    parser.add_argument(
        "--mode", choices=["countryPrice", "standard_initial_series", "exportEarnings", "showSumExportsByCountry", "costSumInsideDE", "assessLineCosts", "showCostSumInsideDE", "sumDisplacable", "expandIntRenInNode", "showAllNodeCFs"], required=True,
        help="Which mode to run"
    )
    args = parser.parse_args()

    try:
        if args.mode == "countryPrice":
            countryPrice()
        elif args.mode == "standard_initial_series":
            standard_initial_series()
        elif args.mode == "exportEarnings":
            exportEarnings()
        elif args.mode == "showSumExportsByCountry":
            showSumExportsByCountry()
        elif args.mode == "costSumInsideDE":
            costSumInsideDE()
        elif args.mode == "showCostSumInsideDE":
            showCostSumInsideDE()
        elif args.mode == "sumDisplacable":
            sumDisplacable()
        elif args.mode == "expandIntRenInNode":
            expandIntRenInNode()
        elif args.mode == "assessLineCosts":
            assessLineCosts("merged_price_flows_only_2025_07_11_16_07.csv", 2024)
        elif args.mode == "showAllNodeCFs":
            showAllNodeCFs()

    except Exception as e:
        print(f"❌ Script failure: {e}")