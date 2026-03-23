import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict, Optional


# --- 1. DOMAIN MODELS & CONSTANTS ---
CHART_SIZE = 500

BLUE_PALETTE = [
    [0.0, "#E0F2FF"],
    [0.25, "#9CCCF7"],
    [0.5, "#4A90E2"],
    [0.75, "#1F78D1"],
    [1.0, "#0D47A1"]
]
LIGHT_BLUE = BLUE_PALETTE[0][1]
DARK_BLUE  = BLUE_PALETTE[-1][-1]

AMBER_PALETTE = [
    [0.0, '#FFF8E1'],
    [0.2, '#FFE082'],
    [0.4, '#FFC107'],
    [0.6, '#FF9800'],
    [0.8, '#F57C00'],
    [1.0, '#E65100']
]
LIGHT_AMBER = AMBER_PALETTE[0][1]
DARK_AMBER  = AMBER_PALETTE[-1][-1]

# One distinct color per expenditure category — avoids the all-blue problem.
CATEGORY_COLORS = {
    "FOOD_MONTHLY":          "#2196F3",  # Blue
    "CLOTH_MONTHLY":         "#4CAF50",  # Green
    "HOUSING_WATER_MONTHLY": "#FF9800",  # Orange
    "HEALTH_MONTHLY":        "#E91E63",  # Pink / Red
    "EDUCATION_MONTHLY":     "#9C27B0",  # Purple
}


class Expenditure(Enum):
    FOOD          = "FOOD_MONTHLY"
    CLOTH         = "CLOTH_MONTHLY"
    HOUSING_WATER = "HOUSING_WATER_MONTHLY"
    HEALTH        = "HEALTH_MONTHLY"
    EDUCATION     = "EDUCATION_MONTHLY"

    def get_display_label(self) -> str:
        label_map = {
            "FOOD":          "Food",
            "CLOTH":         "Cloth",
            "HOUSING_WATER": "Housing & Water",
            "HEALTH":        "Health",
            "EDUCATION":     "Education",
        }
        return label_map.get(self.name, self.name.replace("_", " ").title())


class RegionMapping(Enum):
    REGION_1  = ("Region I - Ilocos Region",                        "Region I (Ilocos Region)")
    REGION_2  = ("Region II - Cagayan Valley",                      "Region II (Cagayan Valley)")
    REGION_3  = ("Region III - Central Luzon",                      "Region III (Central Luzon)")
    REGION_4A = ("Region IVA - CALABARZON",                         "Region IV-A (CALABARZON)")
    MIMAROPA  = ("Region IVB - MIMAROPA",                           "MIMAROPA Region")
    REGION_5  = ("Region V - Bicol",                                "Region V (Bicol Region)")
    REGION_6  = ("Region VI - Western Visayas",                     "Region VI (Western Visayas)")
    REGION_7  = ("Region VII - Central Visayas",                    "Region VII (Central Visayas)")
    REGION_8  = ("Region VIII - Eastern Visayas",                   "Region VIII (Eastern Visayas)")
    REGION_9  = ("Region IX - Zamboanga Peninsula",                 "Region IX (Zamboanga Peninsula)")
    REGION_10 = ("Region X - Northern Mindanao",                    "Region X (Northern Mindanao)")
    REGION_11 = ("Region XI - Davao",                               "Region XI (Davao Region)")
    REGION_12 = ("Region XII - SOCCSKSARGEN",                       "Region XII (SOCCSKSARGEN)")
    REGION_13 = ("Region XIII - Caraga",                            "Region XIII (Caraga)")
    NCR       = ("National Capital Region",                         "National Capital Region (NCR)")
    CAR       = ("Cordillera Administrative Region",                "Cordillera Administrative Region (CAR)")
    BARMM     = ("Bangsamoro Autonomous Region in Muslim Mindanao", "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)")

    @classmethod
    def get_map_dict(cls) -> Dict[str, str]:
        return {item.fies_name: item.shp_name for item in cls}

    @property
    def fies_name(self) -> str:
        return self.value[0]

    @property
    def shp_name(self) -> str:
        return self.value[1]


# --- 2. DATA SERVICES ---
@st.cache_data
def fetch_and_preprocess_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame, List[str], pd.DataFrame]:
    df      = pd.read_csv('./datasets/clean/fies_2023.csv')
    gdf     = gpd.read_file("./datasets/raw/Regions.shp.shp")
    risk_df = pd.read_csv('./datasets/clean/disaster_risk_index.csv')

    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    gdf = gdf.to_crs(epsg=4326)

    mapping_dict = RegionMapping.get_map_dict()
    nat_avg_df   = df[df['REGION'] == "All Regions (National Avg)"].copy()
    regional_df  = df[df['REGION'] != "All Regions (National Avg)"].copy()
    regional_df['SHP_NAME'] = regional_df['REGION'].map(mapping_dict)

    map_gdf = gdf.merge(regional_df, left_on='name', right_on='SHP_NAME')
    map_gdf = map_gdf.merge(
        risk_df[['PH Region', 'Disaster Risk Score']],
        left_on='REGION', right_on='PH Region', how='left'
    )

    options = ["All Regions (National Avg)"] + sorted(map_gdf['REGION'].unique().tolist())
    return nat_avg_df, map_gdf, options, risk_df


# --- 3. UI STYLING & HELPERS ---

def inject_custom_css():
    st.set_page_config(
        layout="wide",
        page_title="W.A.I.S. Relocation Hub: PH Regional Cost & Resilience Navigator",
        page_icon="🇵🇭"
    )
    st.markdown("""
        <style>
        button.viewerBadge_link__1S137, .main header, a.header-anchor { display: none !important; }
        .block-container {
            padding-top: 2rem; padding-bottom: 0rem;
            max-width: 100% !important;
            padding-left: 2rem; padding-right: 2rem;
        }
        h1 { margin-top: -30px; padding-bottom: 10px; font-size: 2rem !important; }
        .stMetric {
            background-color: #ffffff; padding: 15px; border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1); border: 1px solid #eee; text-align: center;
        }
        div[data-testid="stCheckbox"] { margin-bottom: -15px; }
        .stMetric [data-testid="stMetricLabel"],
        .stMetric [data-testid="stMetricLabel"] p,
        .stMetric [data-testid="stMetricLabel"] span { color: #555555 !important; font-size: 0.9rem !important; }
        .stMetric [data-testid="stMetricValue"],
        .stMetric [data-testid="stMetricValue"] span  { color: #1a1a1a !important; }
        .stMetric [data-testid="stMetricDelta"],
        .stMetric [data-testid="stMetricDelta"] span  { color: #555555 !important; }
        </style>
    """, unsafe_allow_html=True)


def _get_category_display_label(category_value: str) -> str:
    label_map = {
        "FOOD_MONTHLY":          "Food",
        "CLOTH_MONTHLY":         "Cloth",
        "HOUSING_WATER_MONTHLY": "Housing & Water",
        "HEALTH_MONTHLY":        "Health",
        "EDUCATION_MONTHLY":     "Education",
    }
    return label_map.get(category_value, category_value.replace('_MONTHLY', '').title())


def _shorten_region_name(region: str) -> str:
    """Abbreviates very long region names so chart axes never overflow."""
    overrides = {
        "Bangsamoro Autonomous Region in Muslim Mindanao":         "BARMM",
        "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)": "BARMM",
        "Cordillera Administrative Region":                        "CAR",
        "Cordillera Administrative Region (CAR)":                  "CAR",
        "National Capital Region":                                 "NCR",
        "National Capital Region (NCR)":                           "NCR",
    }
    return overrides.get(region, region)


def initialize_sidebar_controls(region_options: List[str]) -> Tuple[str, Optional[str], List[str]]:
    st.sidebar.title("🔍 Navigator Controls")

    selected_region = st.sidebar.selectbox("Focus Region", options=region_options, key="primary_region")

    st.sidebar.markdown("### Compare with Another Region")
    enable_compare = st.sidebar.toggle("Enable Region Comparison", value=False)

    compare_region = None
    if enable_compare:
        compare_options = [r for r in region_options if r != selected_region]
        compare_region  = st.sidebar.selectbox("Compare Region", options=compare_options, key="compare_region")

    st.sidebar.markdown("### Filter by Essential Needs")

    def bulk_set_category_state(is_enabled: bool):
        for category in Expenditure:
            st.session_state[category.name] = is_enabled

    c1, c2 = st.sidebar.columns(2)
    c1.button("Select All", on_click=bulk_set_category_state, args=(True,),  use_container_width=True)
    c2.button("Clear",      on_click=bulk_set_category_state, args=(False,), use_container_width=True)

    active_categories = []
    for category in Expenditure:
        if category.name not in st.session_state:
            st.session_state[category.name] = True
        if st.sidebar.checkbox(category.get_display_label(), key=category.name):
            active_categories.append(category.value)

    return selected_region, compare_region, active_categories


# --- 4. VISUALIZATION ENGINE ---

def build_regional_choropleth(map_gdf: gpd.GeoDataFrame, highlight_indices: List[int]) -> go.Figure:
    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()),
        locations=map_gdf.index,
        z=map_gdf['DYNAMIC_Z'],
        colorscale=BLUE_PALETTE,
        marker_opacity=0.7,
        marker_line_width=0.5,
        marker_line_color="black",
        selectedpoints=highlight_indices,
        selected={'marker':   {'opacity': 1.0}},
        unselected={'marker': {'opacity': 0.15}},
        hovertemplate="<b>%{customdata[0]}</b><br>Total: ₱%{z:,.2f}<extra></extra>",
        customdata=map_gdf[['REGION']],
        colorbar=dict(title="Monthly Spending (PHP)"),
    ))
    fig.update_layout(
        mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=CHART_SIZE,
    )
    return fig


def build_expenditure_bar_chart(
    data_row:      pd.Series,
    categories:    List[str],
    y_max:         float,
    compare_row:   Optional[pd.Series] = None,
    compare_label: Optional[str]       = None,
    primary_label: Optional[str]       = None,
) -> go.Figure:
    """Vertical stacked bar chart for Viz 2a (single region) and 2b (comparison).

    Label strategy
    --------------
    Per-segment on-bar text is intentionally omitted: thin slices (e.g. Cloth,
    Education) produce illegible labels that overlap adjacent segments. Instead:
      • Each segment shows its full value in the hover tooltip.
      • The grand total is annotated clearly above every bar.
      • The Y-axis carries peso-formatted tick labels so scale is always readable.
    """
    fig = go.Figure()

    if compare_row is not None:
        # ── 2b: side-by-side stacked bars ────────────────────────────────────
        primary_short = _shorten_region_name(primary_label or "Region A")
        compare_short = _shorten_region_name(compare_label or "Region B")
        x_labels      = [primary_short, compare_short]

        for cat in categories:
            cat_label   = _get_category_display_label(cat)
            color       = CATEGORY_COLORS.get(cat, "#607D8B")
            pv          = float(data_row[cat])
            cv          = float(compare_row[cat])
            fig.add_trace(go.Bar(
                name=cat_label,
                x=x_labels,
                y=[pv, cv],
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" + f"{cat_label}: ₱%{{y:,.2f}}<extra></extra>",
            ))

        # Annotate totals above each bar
        pt = float(data_row[categories].sum())
        ct = float(compare_row[categories].sum())
        for xp, total in [(primary_short, pt), (compare_short, ct)]:
            fig.add_annotation(x=xp, y=total, text=f"<b>₱{total:,.0f}</b>",
                               showarrow=False, yshift=10, font=dict(size=12, color="#111111"))
        stack_max = max(pt, ct)
        fig.update_layout(barmode='stack', xaxis=dict(title="Region"))

    else:
        # ── 2a: single stacked bar, one column per category ──────────────────
        region_short = _shorten_region_name(primary_label or "Region")

        for cat in categories:
            cat_label = _get_category_display_label(cat)
            color     = CATEGORY_COLORS.get(cat, "#607D8B")
            val       = float(data_row[cat])
            fig.add_trace(go.Bar(
                name=cat_label,
                x=[region_short],
                y=[val],
                marker_color=color,
                hovertemplate=(f"<b>{region_short}</b><br>"
                               f"{cat_label}: ₱%{{y:,.2f}}<extra></extra>"),
            ))

        total     = float(data_row[categories].sum())
        stack_max = total
        fig.add_annotation(x=region_short, y=total,
                           text=f"<b>Total: ₱{total:,.0f}</b>",
                           showarrow=False, yshift=12,
                           font=dict(size=12, color="#111111"))
        fig.update_layout(barmode='stack', xaxis=dict(title=""))

    fig.update_layout(
        template="plotly_white",
        margin={"t": 40, "b": 10, "l": 110, "r": 10},
        yaxis=dict(
            title="Amount in Philippine Pesos (PHP)",
            range=[0, stack_max * 1.3],
            showticklabels=True,
            showgrid=True,
            tickprefix="₱",
            tickformat=",.0f",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title_text="Category"),
        height=CHART_SIZE,
    )
    return fig


def build_regional_distribution_bar(map_gdf: gpd.GeoDataFrame, category: str) -> go.Figure:
    """Horizontal bar — single category across all 17 regions (Viz 2c).

    Long region names (BARMM, CAR, NCR) are shortened on the y-axis so bars
    and labels never get pushed off screen.
    """
    bar_df = map_gdf[['REGION', category]].copy()
    bar_df = bar_df.sort_values(by=category, ascending=True)
    bar_df['REGION_SHORT'] = bar_df['REGION'].apply(_shorten_region_name)

    n      = len(bar_df)
    colors = ['#90CAF9'] * n
    colors[-1] = '#0D47A1'
    colors[-2] = '#0D47A1'
    colors[-3] = '#0D47A1'

    max_val = bar_df[category].max()

    fig = go.Figure(go.Bar(
        x=bar_df[category].values,
        y=bar_df['REGION_SHORT'].values,
        orientation='h',
        marker_color=colors,
        text=bar_df[category].values,
        texttemplate='₱%{text:,.2f}',
        textposition='outside',
        cliponaxis=False,
        hovertemplate='<b>%{y}</b><br>₱%{x:,.2f}<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, max_val * 1.45]),
        yaxis=dict(title="Philippine Regions", tickfont=dict(size=10)),
        margin=dict(t=10, b=20, l=10, r=120),
        height=CHART_SIZE,
    )
    return fig


def get_display_row(region: str, nat_avg_df: pd.DataFrame, map_gdf: gpd.GeoDataFrame) -> pd.Series:
    if region == "All Regions (National Avg)":
        return nat_avg_df.iloc[0]
    return map_gdf[map_gdf['REGION'] == region].iloc[0]


def build_risk_heatmap(risk_df: pd.DataFrame, highlighted_region: Optional[str] = None) -> go.Figure:
    heatmap_df = risk_df[[
        'PH Region',
        'Disaster Frequency (Normalized)',
        'Human Impact (Normalized)',
        'Economic Impact (Normalized)',
        'Disaster Risk Score'
    ]].copy()
    heatmap_df.columns = ['Region', 'Frequency', 'Human Impact', 'Economic Impact', 'Disaster Risk Score']
    heatmap_df = heatmap_df.set_index('Region').sort_values('Disaster Risk Score', ascending=True)

    hover_array = []
    for region in heatmap_df.index:
        rd   = risk_df[risk_df['PH Region'] == region].iloc[0]
        freq = heatmap_df.loc[region, 'Frequency']
        human= heatmap_df.loc[region, 'Human Impact']
        econ = heatmap_df.loc[region, 'Economic Impact']
        risk = heatmap_df.loc[region, 'Disaster Risk Score']
        hover = (
            f"<b>{region}</b><br>"
            f"<b>━━━━━━━━━━━━━━━━━━</b><br>"
            f"<b>📊 RISK ANALYSIS:</b><br>"
            f"  • <b>Risk Score: {risk:.2f}</b><br>"
            f"  • Total Disasters: {int(rd['Disaster Count'])}<br>"
            f"<b>━━━━━━━━━━━━━━━━━━</b><br>"
            f"<b>📈 COMPONENT SCORES:</b><br>"
            f"  • Frequency: {freq:.2f}<br>"
            f"  • Human Impact: {human:.2f}<br>"
            f"  • Economic Impact: {econ:.2f}<br>"
        )
        hover_array.append([hover] * len(heatmap_df.columns))

    y_labels = [f"<b>{r}</b>" if r == highlighted_region else r for r in heatmap_df.index]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale=AMBER_PALETTE,
        customdata=hover_array,
        hovertemplate='%{customdata}<extra></extra>',
        colorbar=dict(
            title="Risk Severity (0-100)", title_side='right',
            tickmode='linear', tick0=0, dtick=10,
            len=0.75, thickness=15,
            tickfont=dict(color='black'), title_font=dict(color='black')
        ),
        xgap=2, ygap=2
    ))
    fig.update_layout(
        title={'text': '<b>Disaster Risk Profile Diagnostic</b>', 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 18, 'color': 'black'}},
        xaxis=dict(title='Risk Component', side='bottom',
                   tickfont=dict(size=11, color='black'), tickangle=-45, showgrid=False),
        yaxis=dict(title='Regions', tickmode='array',
                   tickvals=list(heatmap_df.index), ticktext=y_labels,
                   tickfont=dict(size=11, color='black'), showgrid=False),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=250, r=150, t=100, b=120),
        height=700
    )
    return fig


def build_expenditure_risk_line_chart(map_gdf: gpd.GeoDataFrame, selected_cats: List[str],
                                      selected_region: str) -> go.Figure:
    chart_df = map_gdf.copy()
    chart_df['Dynamic_Exp'] = chart_df[selected_cats].sum(axis=1) if selected_cats else 0
    chart_df = chart_df.sort_values('Dynamic_Exp', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df['REGION'], y=chart_df['Dynamic_Exp'],
        name="Selected Expenditures", mode='markers',
        line=dict(color=DARK_BLUE, width=3), hovertemplate="₱%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=chart_df['REGION'], y=chart_df['Disaster Risk Score'],
        name="Disaster Risk Index", mode='markers',
        line=dict(color=DARK_AMBER, width=3, dash='dot'), yaxis="y2",
        hovertemplate="Risk: %{y:.2f}<extra></extra>"))
    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=chart_df['REGION'],
                   ticktext=[f"<b>{r}</b>" if r == selected_region else r for r in chart_df['REGION']],
                   tickfont=dict(size=11, color='black')),
        yaxis=dict(title=dict(text="Selected Expenditure (PHP)", font=dict(color=DARK_BLUE)),
                   tickfont=dict(color=DARK_BLUE)),
        yaxis2=dict(title=dict(text="Risk Index", font=dict(color=DARK_AMBER)),
                    tickfont=dict(color=DARK_AMBER), overlaying="y", side="right", range=[0, 100])
    )
    if selected_region != "All Regions (National Avg)":
        fig.add_vrect(x0=selected_region, x1=selected_region,
                      fillcolor=DARK_BLUE, opacity=0.1, layer="below", line_width=0)
    return fig


# --- 5. MAIN APPLICATION ---

def main():
    inject_custom_css()

    nat_avg_df, map_gdf, options_list, risk_df = fetch_and_preprocess_data()
    selected_region, compare_region, selected_cats = initialize_sidebar_controls(options_list)

    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0
    display_data_row = get_display_row(selected_region, nat_avg_df, map_gdf)

    if selected_region == "All Regions (National Avg)":
        indices_to_highlight = list(range(len(map_gdf)))
    else:
        indices_to_highlight = map_gdf.index[map_gdf['REGION'] == selected_region].tolist()

    compare_data_row = None
    if compare_region:
        compare_data_row = get_display_row(compare_region, nat_avg_df, map_gdf)
        if compare_region == "All Regions (National Avg)":
            indices_to_highlight = list(range(len(map_gdf)))
        else:
            compare_indices      = map_gdf.index[map_gdf['REGION'] == compare_region].tolist()
            indices_to_highlight = list(set(indices_to_highlight + compare_indices))

    st.markdown(
        "<h1 style='text-align: left;'>🇵🇭 W.A.I.S. Relocation Hub: PH Regional Cost & Resilience Navigator</h1>",
        unsafe_allow_html=True
    )

    # KPI
    current_scope_total = display_data_row[selected_cats].sum() if selected_cats else 0

    if compare_region and compare_data_row is not None:
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(label=f"Monthly Spending — {selected_region}", value=f"₱{current_scope_total:,.2f}")
        with kpi_col2:
            compare_total = compare_data_row[selected_cats].sum() if selected_cats else 0
            delta         = compare_total - current_scope_total
            delta_str     = (f"₱{delta:,.2f} vs {selected_region}"
                             if delta >= 0 else f"-₱{abs(delta):,.2f} vs {selected_region}")
            st.metric(label=f"Monthly Spending — {compare_region}",
                      value=f"₱{compare_total:,.2f}", delta=delta_str)
    else:
        st.metric(label=f"Monthly Spending for {selected_region}", value=f"₱{current_scope_total:,.2f}")

    # Main Visuals Row
    col_map, col_bar = st.columns([1.7, 1.3])

    with col_map:
        st.markdown("##### Regional Cost of Living Overview")
        st.caption("Hover over a region for detailed affordability ratios. "
                   "Darker blue indicates higher regional costs for selected categories.")
        st.plotly_chart(build_regional_choropleth(map_gdf, indices_to_highlight),
                        use_container_width=True, config={'displayModeBar': False})

    with col_bar:
        if selected_cats:
            if len(selected_cats) == 1 and selected_region == "All Regions (National Avg)":
                # Viz 2c
                category_label = _get_category_display_label(selected_cats[0])
                st.markdown(f"##### Regional Monthly {category_label} Expenditure")
                st.caption(f"Currently viewing: {category_label} across all 17 regions")
                st.plotly_chart(
                    build_regional_distribution_bar(map_gdf=map_gdf, category=selected_cats[0]),
                    use_container_width=True, config={'displayModeBar': False}
                )
            else:
                # Viz 2a / 2b
                if compare_data_row is not None:
                    st.markdown("##### Comparison of Monthly Household Expenditure Breakdown")
                    st.caption(f"Comparing {selected_region} vs. {compare_region} — "
                               "hover each segment for per-category values")
                else:
                    st.markdown("##### Monthly Household Expenditure Breakdown")
                    st.caption(f"Expenditure category breakdown for {selected_region} — "
                               "hover each segment for details")
                global_max_val = map_gdf[selected_cats].max().max() if not map_gdf.empty else 10000
                sorted_cats    = display_data_row[selected_cats].sort_values(ascending=False).index.tolist()
                st.plotly_chart(
                    build_expenditure_bar_chart(
                        data_row=display_data_row, categories=sorted_cats, y_max=global_max_val,
                        compare_row=compare_data_row, compare_label=compare_region,
                        primary_label=selected_region,
                    ),
                    use_container_width=True, config={'displayModeBar': False}
                )
        else:
            st.markdown("##### Monthly Household Expenditure Breakdown")
            st.info("Please select at least one category in the sidebar to see the breakdown.")

    # Disaster Risk Heatmap
    st.markdown("---")
    st.subheader("🔥 Regional Vulnerability Matrix")
    with st.container():
        heatmap_highlight = selected_region if selected_region != "All Regions (National Avg)" else None
        st.plotly_chart(build_risk_heatmap(risk_df, highlighted_region=heatmap_highlight),
                        use_container_width=True, config={'displayModeBar': False})

    # # Expenditure - Risk line chart (commented out)
    # st.markdown("---")
    # st.subheader("📈 Expenditure vs. Disaster Risk Correlation")
    # with st.container():
    #     if selected_cats:
    #         st.plotly_chart(build_expenditure_risk_line_chart(map_gdf, selected_cats, selected_region),
    #                         use_container_width=True, config={'displayModeBar': False})
    #     else:
    #         st.info("Please select at least one expenditure category to view the correlation.")


if __name__ == "__main__":
    main()