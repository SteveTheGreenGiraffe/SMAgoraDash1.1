# Current path and command:
# cd "C:\Users\StephenMettler\GREEN GIRAFFE Dropbox\Stephen Mettler\PC\Documents\Commercial pursuits (DT)\S4\SG"
# streamlit run AgoraNodeDashboard.py

import streamlit as st
import altair as alt
import pandas as pd
import os
import calendar
from Modify_AgoraDE_files import expandIntRenInNode, safe_divide, processLineCosts, showGenPriceDeciles, estimateSystemValue
import numpy as np
from pandas.tseries.offsets import MonthEnd

# TAG: Here.
# Charts now working. Some weird elements with chart widths varying in a way I don't want them to (requires dealing with standardizing key widths) that I won't deal with for now
# Additional elements you'll want: To right of current chart, guide about what these lines actually mean - bars of the various power use configurations
# Then put hours opportunities on left under the chart, values on the right
# Bottom row would then be screenshot of Agorameter and explaining the concept

# Tag: Later want to add inputs for price percentiles (and min.-max. overrides) below the chart, but leave for now

st.set_page_config(layout="wide")

st.markdown(
    """
# Reactive Demand contracting to co-optimize demand, grid, and generation in Germany

**Reactive Demand (RD):** Electricity use which is designed to ramp up and down based on grid conditions. RD can raise productive utilization of the existing electricity grid and generation, driving system savings.<br>
*Downward RD:* Demand is on by default, then turns down when the grid is strained. *High-capex, high-utilization-required assets*<br>
*Upward RD:* Demand is off by default, then turns up when the grid is under-utilized and excess power is available. *Lower-capex, lower-utilization-tolerant assets*<br>

RD saves money in three ways:<br>
**Direct T&D savings:** Raise energy throughput on existing grid infrastructure, without raising peak demand, which would require grid expansion investment. This spreads fixed transmission and distribution (T&D) costs over more MWhs delivered, driving net savings to the entire system<br>
**Direct utilization savings:** Make productive use of existing energy generation potential currently being wasted for lack of demand, so that fixed costs of generation assets are spread over more MWhs productively utilized<br>
**Indirect planning savings:** Support iterative, “layered” demand commitment required to mitigate back-to-back risk – in which new electricity generation cannot be built without certainty on demand, but new demand assets cannot be built without certainty on generation – enabling de-risked, capital-efficient investment in new generation assets and their supply chains

""",
    unsafe_allow_html=True
)

# --- Page highest-level column hierarchy ---
left_col, right_col = st.columns([3, 3])

price_col = "Agora wholesale price [EUR/MWh]"
data_path = "nodal_dem_gen_with_prices_and_categories_summed_2025_07_14_14_19.csv"

@st.cache_data
def get_date_bounds(path):
    d = pd.read_csv(path, usecols=["Datetime"])
    d["Datetime"] = pd.to_datetime(d["Datetime"], errors="coerce")
    d = d.dropna(subset=["Datetime"])
    years = sorted(d["Datetime"].dt.year.unique().tolist())
    return years

avail_years = get_date_bounds(data_path)
avail_months = list(range(1, 13))

with left_col:

    chart_placeholder = st.empty()

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    node_placeholder = c1.empty()

    with c2:
        start_year  = st.selectbox("Start year", avail_years, index=0, key="start_year")
    with c3:
        start_month = st.selectbox("Start month", avail_months, index=0, format_func=lambda m: calendar.month_abbr[m], key="start_month")
    with c4:
        end_year  = st.selectbox("End year", avail_years, index=len(avail_years)-1, key="end_year")
    with c5:
        end_month = st.selectbox("End month", avail_months, index=11, format_func=lambda m: calendar.month_abbr[m], key="end_month")

    # Now run your heavy compute for the chosen window
    monthly_hub_summary, df_hourly = estimateSystemValue(
        file=data_path,
        start_year=start_year, start_month=start_month,
        end_year=end_year,   end_month=end_month,
    )

    # Inclusive start (first day of start month)
    start_dt = pd.Timestamp(start_year, start_month, 1)

    # Exclusive end: 1 hour past the last day of end month
    # works cleanly for hourly data and avoids off-by-one
    end_dt = pd.Timestamp(end_year, end_month, 1) + MonthEnd(1) + pd.Timedelta(hours=1)

    # Build hubs and render the node selector into the placeholder
    hubs = sorted(df_hourly["Selected hub"].dropna().unique().tolist())
    if not hubs:
        st.warning("No data in the selected period. Adjust the date range.")
        st.stop()

    # Resolve default node stably
    prev = st.session_state.get("node")
    default_node = prev if prev in hubs else hubs[0]

    node = node_placeholder.selectbox(
        "Choose a node",
        hubs,
        index=hubs.index(default_node),
        key="node",
    )

    # --- Build chart dataframe for the selected node ---
    needed_cols = [
        "Datetime", "Selected hub", price_col,
        "all_disp_min", "disp_ex_OCGT_max", "all_disp_max", "RFNBO_threshold"
    ]
    node_df = (
        df_hourly.loc[df_hourly["Selected hub"] == node, needed_cols]
        .sort_values("Datetime")
        .copy()
    )
    node_df[price_col] = pd.to_numeric(node_df[price_col], errors="coerce")
    node_df = node_df.dropna(subset=["Datetime", price_col])

    # Lines first, scatter on top
    th_cols = ["all_disp_min", "disp_ex_OCGT_max", "all_disp_max", "RFNBO_threshold"]
    th_long = node_df.melt(id_vars=["Datetime"], value_vars=th_cols,
                        var_name="Threshold", value_name="EUR_per_MWh")

    alt.data_transformers.disable_max_rows()

    # --- Build a clean plotting frame with safe names ---
    th_cols_src = ["all_disp_min", "disp_ex_OCGT_max", "all_disp_max", "RFNBO_threshold"]

    plot = (
        df_hourly.loc[df_hourly["Selected hub"] == node, ["Datetime", "Agora wholesale price [EUR/MWh]"] + th_cols_src]
        .copy()
    )

    # Safe names
    plot = plot.rename(columns={
        "Datetime": "dt",
        "Agora wholesale price [EUR/MWh]": "price",
        "all_disp_min": "t_min",
        "disp_ex_OCGT_max": "t_exocgt",
        "all_disp_max": "t_max",
        "RFNBO_threshold": "t_rfnbo",
    })

    # Coerce types
    plot["dt"]    = pd.to_datetime(plot["dt"], errors="coerce").dt.tz_localize(None)
    plot["price"] = pd.to_numeric(plot["price"], errors="coerce")
    for c in ["t_min", "t_exocgt", "t_max", "t_rfnbo"]:
        plot[c] = pd.to_numeric(plot[c], errors="coerce")

    # Drop bad rows
    plot = plot.dropna(subset=["dt", "price"])
    if plot.empty:
        st.warning("No data to plot for this node and period.")
        st.stop()

    # Epoch x and domain
    plot["ts_epoch"] = (plot["dt"].astype("int64") // 10**9).astype("int64")
    xmin = int(plot["ts_epoch"].min())
    xmax = int(plot["ts_epoch"].max())

    # Ticks every 3 months within domain
    tick_dt  = pd.date_range(plot["dt"].min().floor("D"),
                            plot["dt"].max().ceil("D"),
                            freq="3MS")
    tick_vals = ((tick_dt.astype("int64") // 10**9).astype(int)).tolist()

    x_axis = alt.X(
        "ts_epoch:Q",
        scale=alt.Scale(domain=[xmin, xmax]),                   # clamps padding
        axis=alt.Axis(values=tick_vals,
                    labelExpr="timeFormat(toDate(datum.value * 1000), '%b %y')",
                    title=None, labelAngle=-45, labelOverlap=True)
    )

    # --- shared Y domain from PRICES ONLY ---
    p_lo = min(-50, np.nanpercentile(plot["price"], 1))
    p_hi = max(400, np.nanpercentile(plot["price"], 99))

    # guard against degenerate or NaN values
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo >= p_hi:
        p_lo = float(np.nanmin(plot["price"]))
        p_hi = float(np.nanmax(plot["price"]))
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo >= p_hi:
            # last resort: expand a tiny band
            center = float(plot["price"].mean())
            p_lo, p_hi = center - 1.0, center + 1.0

    y_shared = alt.Scale(domain=[float(p_lo), float(p_hi)], nice=False, clamp=True)

    y_enc_scatter = alt.Y("price:Q",   scale=y_shared, title="Price / Threshold (€/MWh)")
    y_enc_lines   = alt.Y("thr_val:Q", scale=y_shared, title="Price / Threshold (€/MWh)")

    # Shared Y domain
    thr_long = (
        plot[["ts_epoch", "t_min", "t_exocgt", "t_max", "t_rfnbo"]]
        .dropna()
        .melt(id_vars=["ts_epoch"], var_name="thr_name", value_name="thr_val")
    )
    label_map = {
        "t_max":    "All dispatchable max",
        "t_exocgt": "Non-OCGT max",
        "t_min":    "All dispatchable min",
        "t_rfnbo":  "RFNBO threshold",
    }
    thr_long["thr_label"] = thr_long["thr_name"].map(label_map)

    color_scale = alt.Scale(
        domain=list(label_map.values()),
        range=["#800000", "#800000", "#614DA0", "#614DA0"]  # min, ex-OCGT, max, RFNBO
    )
    dash_scale = alt.Scale(
        domain=list(label_map.values()),
        range=[
            [0],
            [4, 4],
            [4, 4],
            [0],
        ]
    )

    # --- constants for band tops/bottoms from your shared Y scale ---
    y_lo, y_hi = y_shared.domain
    band_frame = plot[["ts_epoch", "t_min", "t_exocgt", "t_max", "t_rfnbo"]].copy()
    band_frame["y_lo"] = float(y_lo)
    band_frame["y_hi"] = float(y_hi)

    # --- shaded bands (areas) ---
    # Scarcity band: between Non-OCGT max and All-disp max
    area_scarcity = (
        alt.Chart(band_frame)
        .mark_area(opacity=0.10, color="#800000")  # red tint
        .encode(
            x=x_axis,
            y=alt.Y("t_max:Q",   scale=y_shared, title=None),
            y2=alt.Y2("t_exocgt:Q")
        )
    )
    # Extreme scarcity: above All-disp max to top of domain
    area_above = (
        alt.Chart(band_frame)
        .mark_area(opacity=0.20, color="#800000")
        .encode(
            x=x_axis,
            y=alt.Y("y_hi:Q",    scale=y_shared, title=None),
            y2=alt.Y2("t_max:Q")
        )
    )
    # Spill-like band: between RFNBO and All-disp min
    area_spill_band = (
        alt.Chart(band_frame)
        .mark_area(opacity=0.18, color="#614DA0")  # green tint
        .encode(
            x=x_axis,
            y=alt.Y("t_min:Q",   scale=y_shared, title=None),
            y2=alt.Y2("t_rfnbo:Q")
        )
    )
    # Deep spill: below RFNBO down to bottom of domain
    area_below = (
        alt.Chart(band_frame)
        .mark_area(opacity=0.20, color="#614DA0")
        .encode(
            x=x_axis,
            y=alt.Y("t_rfnbo:Q", scale=y_shared, title=None),
            y2=alt.Y2("y_lo:Q")
        )
    )

    scatter = (
        alt.Chart(plot)
        .mark_circle(size=6, opacity=0.15, color="#134ac0ff")
        .encode(x=x_axis, y=y_enc_scatter,
                tooltip=[alt.Tooltip("dt:T"), alt.Tooltip("price:Q", format=".1f")])
    )

    lines = (
        alt.Chart(thr_long)
        .mark_line(interpolate="step-after", strokeWidth=2.0)
        .encode(
            x=x_axis,
            y=alt.Y("thr_val:Q", scale=y_shared, title="Price / Threshold (€/MWh)"),
            color=alt.Color("thr_label:N", title="Threshold", scale=color_scale),
            strokeDash=alt.StrokeDash("thr_label:N", scale=dash_scale, title="Line style")
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=1)
        .encode(y="y:Q")
    )

    # --- compose, enforce shared scales, remove gridlines ---
    chart = (
        area_below
        + area_spill_band
        + area_scarcity
        + area_above
        + zero_line
        + scatter
        + lines
    ).resolve_scale(y="shared", x="shared").properties(
        width="container",
        height=360,
        title=f"Price vs thresholds — {node}"
    ).configure_axis(
        grid=False  # kill vertical and horizontal gridlines
    )

    #st.altair_chart(chart, use_container_width=True)
    chart_placeholder.altair_chart(chart, use_container_width=True)

with right_col:
    st.markdown(" ")  # tiny spacer if needed
    st.image("grid_condition_summary.png", use_container_width=True)

# --- Hour and value sum charts ---

# Filter to selected node and period
ms = monthly_hub_summary.copy()
ms["MonthStart"] = pd.to_datetime(dict(year=ms["Year"], month=ms["MonthNum"], day=1))

ms = ms.loc[
    (ms["Selected hub"] == node)
    & (ms["MonthStart"] >= start_dt)
    & (ms["MonthStart"] < end_dt)
].sort_values(["Year", "MonthNum"])

# Compute disjoint components
ms["H_pull_mid"]  = (ms["H_below_all_disp_min"] - ms["H_below_RFNBO_min"]).clip(lower=0)
ms["H_pull_deep"] = ms["H_below_RFNBO_min"]

ms["H_push_mid"]  = ms["H_NOT_above_disp_ex_OCGT_max"]
ms["H_push_peak"] = (ms["H_NOT_above_all_disp_max"] - ms["H_NOT_above_disp_ex_OCGT_max"]).clip(lower=0)

# Values (€/MW) using year-specific grid unit cost already merged
ms["V_pull_mid"]  = ms["H_pull_mid"]  * ms["grid_unit_cost_per_MWh"]
ms["V_pull_deep"] = ms["H_pull_deep"] * ms["grid_unit_cost_per_MWh"]
ms["V_push_mid"]  = ms["H_push_mid"]  * ms["grid_unit_cost_per_MWh"]
ms["V_push_peak"] = ms["H_push_peak"] * ms["grid_unit_cost_per_MWh"]

# Long forms for stacked bars
pull_hours_long = ms.melt(
    id_vars=["MonthStart"],
    value_vars=["H_pull_mid", "H_pull_deep"],
    var_name="Band", value_name="Hours"
)
pull_value_long = ms.melt(
    id_vars=["MonthStart"],
    value_vars=["V_pull_mid", "V_pull_deep"],
    var_name="Band", value_name="Value_EUR_per_MW"
)

push_hours_long = ms.melt(
    id_vars=["MonthStart"],
    value_vars=["H_push_mid", "H_push_peak"],
    var_name="Band", value_name="Hours"
)

push_value_long = ms.melt(
    id_vars=["MonthStart"],
    value_vars=["V_push_mid", "V_push_peak"],
    var_name="Band", value_name="Value_EUR_per_MW"
)

# Friendly labels and colors
label_map = {
    "H_pull_mid":  "Below disp. min.",
    "H_pull_deep": "Below RFNBO",
    "V_pull_mid":  "Below disp. min.",
    "V_pull_deep": "Below RFNBO",
    "H_push_mid":  "Above non-OCGT max.",
    "H_push_peak": "Above all disp. max.",
    "V_push_mid":  "Above non-OCGT max.",
    "V_push_peak": "Above all disp. max.",
}
for df in (pull_hours_long, pull_value_long, push_hours_long, push_value_long):
    df["BandLabel"] = df["Band"].map(label_map)

# Color domain must match labels exactly
#push_domain  = ["Base: ≤ ex-OCGT max", "Add: between ex-OCGT & all-disp max"]
#push_colors  = ["#9ecae1", "#3182bd"]  # light, then darker

color_scale_pull = alt.Scale(domain=["Below disp. min.","Below RFNBO"],
                            range=["#8fd19e", "#2e7d32"])
color_scale_push = alt.Scale(domain=["Above non-OCGT max.","Above all disp. max."],
                            range=["#f4a261", "#7c2610"])

# X axis formatting
x_enc = alt.X("MonthStart:T",
            axis=alt.Axis(format="%b %y", labelAngle=-45, title=None))

# --- Upward regulation opportunity ---
#st.subheader("Upward regulation opportunity")

chart_pull_hours = (
    alt.Chart(pull_hours_long)
    .mark_bar()
    .encode(
        x=x_enc,
        y=alt.Y("Hours:Q", title="Hours"),
        color=alt.Color("BandLabel:N", title="Band", scale=color_scale_pull),
        order=alt.Order("BandLabel", sort="ascending"),
        tooltip=[alt.Tooltip("MonthStart:T", format="%b %Y"),
                "BandLabel:N",
                alt.Tooltip("Hours:Q", format=",")]
    )
    .properties(height=220)
)

chart_pull_value = (
    alt.Chart(pull_value_long)
    .mark_bar()
    .encode(
        x=x_enc,
        y=alt.Y("Value_EUR_per_MW:Q", title="Value (€/MW)"),
        color=alt.Color("BandLabel:N", title="Band", scale=color_scale_pull),
        order=alt.Order("BandLabel", sort="ascending"),
        tooltip=[alt.Tooltip("MonthStart:T", format="%b %Y"),
                "BandLabel:N",
                alt.Tooltip("Value_EUR_per_MW:Q", format=",.0f")]
    )
    .properties(height=220)
)

#st.altair_chart(chart_pull_hours.configure_axis(grid=False), use_container_width=True)
#st.altair_chart(chart_pull_value.configure_axis(grid=False), use_container_width=True)

# --- Downward regulation opportunity ---
#st.subheader("Downward regulation opportunity")

chart_push_hours = (
    alt.Chart(push_hours_long)
    .mark_bar()
    .encode(
        x=x_enc,
        y=alt.Y("Hours:Q", title="Hours"),
        color=alt.Color("BandLabel:N", title="Band", scale=color_scale_push),
        order=alt.Order("BandLabel", sort="ascending"),
        tooltip=[alt.Tooltip("MonthStart:T", format="%b %Y"),
                "BandLabel:N",
                alt.Tooltip("Hours:Q", format=",")]
    )
    .properties(height=220)
)

chart_push_value = (
    alt.Chart(push_value_long)
    .mark_bar()
    .encode(
        x=x_enc,
        y=alt.Y("Value_EUR_per_MW:Q", title="Value (€/MW)"),
        color=alt.Color("BandLabel:N", title="Band", scale=color_scale_push),
        order=alt.Order("BandLabel", sort="ascending"),
        tooltip=[alt.Tooltip("MonthStart:T", format="%b %Y"),
                "BandLabel:N",
                alt.Tooltip("Value_EUR_per_MW:Q", format=",.0f")]
    )
    .properties(height=220)
)

#st.altair_chart(chart_push_hours.configure_axis(grid=False), use_container_width=True)
#st.altair_chart(chart_push_value.configure_axis(grid=False), use_container_width=True)

#common_width = 700
#common_height = 220

#chart_pull_hours  = chart_pull_hours.properties(width=common_width, height=common_height)
#chart_pull_value  = chart_pull_value.properties(width=common_width, height=common_height)
#chart_push_hours  = chart_push_hours.properties(width=common_width, height=common_height)
#chart_push_value  = chart_push_value.properties(width=common_width, height=common_height)

st.subheader("Upward regulation opportunity")
up_left, up_right = st.columns([1, 1])

with up_left:
    st.altair_chart(
        chart_pull_hours.configure_axis(grid=False),
        use_container_width=True,
    )

with up_right:
    st.altair_chart(
        chart_pull_value.configure_axis(grid=False),
        use_container_width=True,
    )

st.subheader("Downward regulation opportunity")
down_left, down_right = st.columns([1, 1])

with down_left:
    st.altair_chart(
        chart_push_hours.configure_axis(grid=False),
        use_container_width=True,
    )

with down_right:
    st.altair_chart(
        chart_push_value.configure_axis(grid=False),
        use_container_width=True,
    )