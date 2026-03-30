import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict, Optional

# --- 1. DOMAIN MODELS & CONSTANTS ---
CHART_SIZE = 500
BAR_HEIGHT_PER_REGION = 45

BLUE_PALETTE = [[0.0, "#E0F2FF"], [0.25, "#9CCCF7"], [0.5, "#4A90E2"], [0.75, "#1F78D1"], [1.0, "#0D47A1"]]
DARK_BLUE = BLUE_PALETTE[-1][-1]
AMBER_PALETTE = [[0.0, '#FFF8E1'], [0.2, '#FFE082'], [0.4, '#FFC107'], [0.6, '#FF9800'], [0.8, '#F57C00'],
                 [1.0, '#E65100']]
DARK_AMBER = AMBER_PALETTE[-1][-1]

CATEGORY_COLORS = {
    "FOOD_MONTHLY": "#2196F3", "CLOTH_MONTHLY": "#4CAF50",
    "HOUSING_WATER_MONTHLY": "#FF9800", "HEALTH_MONTHLY": "#E91E63", "EDUCATION_MONTHLY": "#9C27B0",
}

# --- NEW: Explicit Regional Order as requested ---
OFFICIAL_ORDER = [
    "National Capital Region", "Cordillera Administrative Region", "Region I - Ilocos Region",
    "Region II - Cagayan Valley", "Region III - Central Luzon", "Region IVA - CALABARZON",
    "Region IVB - MIMAROPA", "Region V - Bicol", "Region VI - Western Visayas",
    "Region VII - Central Visayas", "Region VIII - Eastern Visayas",
    "Region IX - Zamboanga Peninsula", "Region X - Northern Mindanao",
    "Region XI - Davao", "Region XII - SOCCSKSARGEN", "Region XIII - Caraga",
    "Bangsamoro Autonomous Region in Muslim Mindanao"
]


class Expenditure(Enum):
    FOOD = "FOOD_MONTHLY"
    CLOTH = "CLOTH_MONTHLY"
    HOUSING_WATER = "HOUSING_WATER_MONTHLY"
    HEALTH = "HEALTH_MONTHLY"
    EDUCATION = "EDUCATION_MONTHLY"

    def get_display_label(self) -> str:
        label_map = {"FOOD": "Food", "CLOTH": "Cloth", "HOUSING_WATER": "Housing & Water", "HEALTH": "Health",
                     "EDUCATION": "Education"}
        return label_map.get(self.name, self.name.replace("_", " ").title())


class RegionMapping(Enum):
    REGION_1 = ("Region I - Ilocos Region", "Region I (Ilocos Region)")
    REGION_2 = ("Region II - Cagayan Valley", "Region II (Cagayan Valley)")
    REGION_3 = ("Region III - Central Luzon", "Region III (Central Luzon)")
    REGION_4A = ("Region IVA - CALABARZON", "Region IV-A (CALABARZON)")
    MIMAROPA = ("Region IVB - MIMAROPA", "MIMAROPA Region")
    REGION_5 = ("Region V - Bicol", "Region V (Bicol Region)")
    REGION_6 = ("Region VI - Western Visayas", "Region VI (Western Visayas)")
    REGION_7 = ("Region VII - Central Visayas", "Region VII (Central Visayas)")
    REGION_8 = ("Region VIII - Eastern Visayas", "Region VIII (Eastern Visayas)")
    REGION_9 = ("Region IX - Zamboanga Peninsula", "Region IX (Zambo Peninsula)")
    REGION_10 = ("Region X - Northern Mindanao", "Region X (Northern Mindanao)")
    REGION_11 = ("Region XI - Davao", "Region XI (Davao Region)")
    REGION_12 = ("Region XII - SOCCSKSARGEN", "Region XII (SOCCSKSARGEN)")
    REGION_13 = ("Region XIII - Caraga", "Region XIII (Caraga)")
    NCR = ("National Capital Region", "National Capital Region (NCR)")
    CAR = ("Cordillera Administrative Region", "Cordillera Administrative Region (CAR)")
    BARMM = ("Bangsamoro Autonomous Region in Muslim Mindanao", "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)")

    @classmethod
    def get_map_dict(cls) -> Dict[str, str]: return {item.fies_name: item.shp_name for item in cls}

    @property
    def fies_name(self) -> str: return self.value[0]

    @property
    def shp_name(self) -> str: return self.value[1]


# --- 2. DATA SERVICES ---
@st.cache_data
def fetch_and_preprocess_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame, List[str], pd.DataFrame]:
    df = pd.read_csv('./datasets/clean/fies_2023.csv')
    gdf = gpd.read_file("./datasets/raw/Regions.shp.shp")
    risk_df = pd.read_csv('./datasets/clean/disaster_risk_index.csv')

    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    gdf = gdf.to_crs(epsg=4326)

    # Ensure independent cities match their regional parents
    gdf['name'] = gdf['name'].replace({'Davao City': 'Region XI (Davao Region)'})

    mapping_dict = RegionMapping.get_map_dict()
    nat_avg_df = df[df['REGION'] == "All Regions (National Avg)"].copy()
    regional_df = df[df['REGION'] != "All Regions (National Avg)"].copy()
    regional_df['SHP_NAME'] = regional_df['REGION'].map(mapping_dict)

    # Use a Left Join to ensure BARMM isn't dropped if naming is slightly off during debug
    map_gdf = gdf.merge(regional_df, left_on='name', right_on='SHP_NAME', how='inner')
    map_gdf = map_gdf.merge(risk_df[['PH Region', 'Disaster Risk Score']], left_on='REGION', right_on='PH Region',
                            how='left')

    return nat_avg_df, map_gdf, OFFICIAL_ORDER, risk_df


# --- 3. UI HELPERS ---
def inject_custom_css():
    st.set_page_config(layout="wide", page_title="PH Regional Navigator", page_icon="🇵🇭")
    st.markdown("""
        <style>
        button.viewerBadge_link__1S137, .main header, a.header-anchor { display: none !important; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); border: 1px solid #eee; text-align: center; height: 120px; }
        </style>
    """, unsafe_allow_html=True)


def _shorten_region_name(region: str) -> str:
    overrides = {"Bangsamoro Autonomous Region in Muslim Mindanao": "BARMM", "Cordillera Administrative Region": "CAR",
                 "National Capital Region": "NCR"}
    return overrides.get(region, region)


def initialize_sidebar_controls(region_options: List[str]) -> Tuple[List[str], List[str]]:
    st.sidebar.title("🔍 Navigator Controls")

    st.sidebar.markdown("### 🗺️ Select Regions")
    select_all = st.sidebar.toggle("Select All Regions", value=True)

    if select_all:
        selected_regions = region_options
        st.sidebar.info(f"All {len(region_options)} regions selected.")
    else:
        with st.sidebar.expander("Choose Specific Regions", expanded=True):
            selected_regions = []
            for reg in region_options:
                if st.checkbox(reg, value=False, key=f"check_{reg}"):
                    selected_regions.append(reg)
            st.sidebar.info(f"{len(selected_regions)} of {len(region_options)} regions selected.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Expenditure Categories")

    def bulk_set_category_state(is_enabled: bool):
        for category in Expenditure: st.session_state[category.name] = is_enabled

    c1, c2 = st.sidebar.columns(2)
    c1.button("Select All", on_click=bulk_set_category_state, args=(True,), use_container_width=True)
    c2.button("Clear", on_click=bulk_set_category_state, args=(False,), use_container_width=True)

    active_categories = []
    for category in Expenditure:
        if category.name not in st.session_state: st.session_state[category.name] = True
        if st.sidebar.checkbox(category.get_display_label(), key=category.name):
            active_categories.append(category.value)

    return selected_regions, active_categories


# --- 4. VISUALIZATION ENGINE ---
def build_regional_choropleth(map_gdf: gpd.GeoDataFrame, highlight_indices: List[int]) -> go.Figure:
    # Add ranking stats for hover
    map_gdf['Exp_Rank'] = map_gdf['DYNAMIC_Z'].rank(ascending=True, method='min')

    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()), locations=map_gdf.index,
        z=map_gdf['DYNAMIC_Z'], colorscale=BLUE_PALETTE, marker_opacity=0.7,
        selectedpoints=highlight_indices, selected={'marker': {'opacity': 1.0}},
        unselected={'marker': {'opacity': 0.15}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "━━━━━━━━━━━━━━━━━━<br>"
            "💰 Monthly Cost: ₱%{z:,.2f}<br>"
            "🏆 Affordability Rank: #%{customdata[1]:.0f} of 17<br>"
            "<extra></extra>"
        ),
        customdata=map_gdf[['REGION', 'Exp_Rank']],
    ))
    fig.update_layout(mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
                      margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=CHART_SIZE)
    return fig


def build_horizontal_stacked_bar(map_gdf: gpd.GeoDataFrame, selected_regions: List[str], categories: List[str],
                                 sort_order: str) -> go.Figure:
    plot_df = map_gdf[map_gdf['REGION'].isin(selected_regions)].copy()
    plot_df['TOTAL'] = plot_df[categories].sum(axis=1)

    if sort_order == "Descending Value":
        plot_df = plot_df.sort_values(by='TOTAL', ascending=True)
    else:
        # Use a mapping dictionary to enforce the Official Regional Order
        order_map = {name: i for i, name in enumerate(OFFICIAL_ORDER)}
        plot_df['sort_idx'] = plot_df['REGION'].map(order_map)
        plot_df = plot_df.sort_values(by='sort_idx', ascending=False) # Ascending=False for H-bar layout

    fig = go.Figure()
    for cat in categories:
        fig.add_trace(go.Bar(
            name=cat.replace('_MONTHLY', '').title(),
            y=plot_df['REGION'].apply(_shorten_region_name),
            x=plot_df[cat],
            orientation='h', marker_color=CATEGORY_COLORS.get(cat),
            hovertemplate="<b>%{y}</b><br>" + f"{cat.replace('_MONTHLY', '').title()}: ₱%{{x:,.2f}}<extra></extra>"
        ))

    # Add explicit total labels to end of bars
    for i, row in plot_df.iterrows():
        fig.add_annotation(x=row['TOTAL'], y=_shorten_region_name(row['REGION']),
                           text=f" ₱{row['TOTAL']:,.0f}", showarrow=False,
                           xanchor='left', font=dict(size=11, color='black'))

    dynamic_height = max(300, len(selected_regions) * BAR_HEIGHT_PER_REGION)
    fig.update_layout(barmode='stack', template="plotly_white", height=dynamic_height,
                      margin={"t": 30, "b": 40, "l": 150, "r": 80},
                      xaxis=dict(title="Monthly Expenditure (PHP)", tickformat=",.0f", tickprefix="₱", showgrid=True),
                      yaxis=dict(title=None, tickfont=dict(size=12, color='black')),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def build_risk_heatmap(risk_df: pd.DataFrame, highlighted_region: Optional[str] = None) -> go.Figure:
    heatmap_df = risk_df[
        ['PH Region', 'Disaster Frequency (Normalized)', 'Human Impact (Normalized)', 'Economic Impact (Normalized)',
         'Disaster Risk Score']].copy()
    heatmap_df.columns = ['Region', 'Frequency', 'Human Impact', 'Economic Impact', 'Risk Score']
    heatmap_df = heatmap_df.set_index('Region').sort_values('Risk Score', ascending=True)

    y_labels = [f"<b>{r}</b>" if r == highlighted_region else r for r in heatmap_df.index]
    fig = go.Figure(
        go.Heatmap(z=heatmap_df.values, x=heatmap_df.columns, y=heatmap_df.index, colorscale=AMBER_PALETTE, xgap=2,
                   ygap=2))
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=list(heatmap_df.index), ticktext=y_labels,
                                 tickfont=dict(size=11, color='black')),
                      margin=dict(l=250, r=150, t=50, b=100), height=700)
    return fig


# --- 5. MAIN ---
def main():
    inject_custom_css()
    nat_avg_df, map_gdf, options_list, risk_df = fetch_and_preprocess_data()
    selected_regions, selected_cats = initialize_sidebar_controls(options_list)

    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0
    indices_to_highlight = map_gdf.index[map_gdf['REGION'].isin(selected_regions)].tolist()

    st.markdown("<h1>🇵🇭 W.A.I.S. Relocation Hub: PH Regional Cost & Resilience Navigator</h1>", unsafe_allow_html=True)

    # --- KPI Section (Selection vs National) ---
    if selected_regions:
        sel_avg = map_gdf[map_gdf['REGION'].isin(selected_regions)][selected_cats].sum(axis=1).mean()
        nat_avg = nat_avg_df[selected_cats].sum(axis=1).iloc[0]

        # Use 2 balanced columns
        k1, k2 = st.columns(2)
        with k1:
            st.metric("National Average Monthly Expenditure", f"₱{nat_avg:,.2f}")
        with k2:
            st.metric(f"Selection Average ({len(selected_regions)} Regions)", f"₱{sel_avg:,.2f}",
                      delta=f"₱{sel_avg - nat_avg:,.2f} vs National Average")

    st.plotly_chart(build_regional_choropleth(map_gdf, indices_to_highlight), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Regional Expenditure Comparison")

    # Localized Sorting Toggle
    sort_order = st.radio("Chart Sort Order:", ["Descending Value", "Official Regional Order"], horizontal=True)

    if selected_cats and selected_regions:
        st.plotly_chart(build_horizontal_stacked_bar(map_gdf, selected_regions, selected_cats, sort_order),
                        use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Regional Vulnerability Matrix")

    # Determine bolding: Only bold if exactly ONE region is selected.
    # If 0 are selected, or all 17 are selected, heatmap_ref becomes None.
    if len(selected_regions) == 1:
        heatmap_ref = selected_regions[0]
    else:
        heatmap_ref = None

    # This call is outside any 'if' blocks, so the heatmap NEVER disappears.
    st.plotly_chart(build_risk_heatmap(risk_df, highlighted_region=heatmap_ref), use_container_width=True)


if __name__ == "__main__":
    main()