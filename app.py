import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import json
from enum import Enum
from typing import List, Tuple, Dict, Optional

# --- 1. DOMAIN MODELS & CONSTANTS ---
# Configuration for chart scaling and visual aesthetics
CHART_SIZE = 500
BAR_LENGTH_PER_REGION = 45

# Color Scales: Blue for affordability/expenditure, Amber for disaster risk
BLUE_PALETTE = [
    [0.0, "#E0F2FF"],
    [0.25, "#9CCCF7"],
    [0.5, "#4A90E2"],
    [0.75, "#1F78D1"],
    [1.0, "#0D47A1"]
]
LIGHT_BLUE = BLUE_PALETTE[0][1]
DARK_BLUE = BLUE_PALETTE[-1][-1]

AMBER_PALETTE = [
    [0.0, '#FFF8E1'],
    [0.2, '#FFE082'],
    [0.4, '#FFC107'],
    [0.6, '#FF9800'],
    [0.8, '#F57C00'],
    [1.0, '#E65100']
]
LIGHT_AMBER = AMBER_PALETTE[0][1]
DARK_AMBER = AMBER_PALETTE[-1][-1]

GREY_PALETTE = [
    [0.0, "#F0F0F0"],
    [1.0, "#BDBDBD"]
]
GREEN = "#4CAF50"
YELLOW = "#FFD300"
RED = "#F44336"

# Mapping expenditure types to distinct colors for the stacked bar charts
CATEGORY_COLORS = {
    "FOOD_MONTHLY":          "#2196F3",  # Blue
    "CLOTH_MONTHLY":         "#4CAF50",  # Green
    "HOUSING_WATER_MONTHLY": "#FF9800",  # Orange
    "HEALTH_MONTHLY":        "#E91E63",  # Pink / Red
    "EDUCATION_MONTHLY":     "#9C27B0",  # Purple
}

# Standard sorting order used across the dashboard (Luzon -> Visayas -> Mindanao)
OFFICIAL_ORDER = [
    "National Capital Region",
    "Cordillera Administrative Region",
    "Region I - Ilocos Region",
    "Region II - Cagayan Valley",
    "Region III - Central Luzon",
    "Region IVA - CALABARZON",
    "Region IVB - MIMAROPA",
    "Region V - Bicol",
    "Region VI - Western Visayas",
    "Region VII - Central Visayas",
    "Region VIII - Eastern Visayas",
    "Region IX - Zamboanga Peninsula",
    "Region X - Northern Mindanao",
    "Region XI - Davao", "Region XII - SOCCSKSARGEN",
    "Region XIII - Caraga",
    "Bangsamoro Autonomous Region in Muslim Mindanao"
]

def get_affordability_status(ratio: float) -> Tuple[str, str]:
    """Helper function to return the affordability status given the ratio"""
    # Using a 1% buffer for 'Break-even' (0.99 to 1.01)
    if ratio < 0.99:
        return "Affordable", GREEN
    if 0.99 <= ratio <= 1.01:
        return "Break-even", YELLOW
    return "Unaffordable", RED

class Expenditure(Enum):
    """Enum to handle expenditure category strings and UI labels"""
    FOOD = "FOOD_MONTHLY"
    CLOTH = "CLOTH_MONTHLY"
    HOUSING_WATER = "HOUSING_WATER_MONTHLY"
    HEALTH = "HEALTH_MONTHLY"
    EDUCATION = "EDUCATION_MONTHLY"

    def get_display_label(self) -> str:
        label_map = {
            "FOOD": "Food",
            "CLOTH": "Cloth",
            "HOUSING_WATER": "Housing & Water",
            "HEALTH": "Health",
            "EDUCATION": "Education"}
        return label_map.get(self.name, self.name.replace("_", " ").title())

class RegionMapping(Enum):
    """Enum to bridge the gap between CSV (FIES) region names and Shapefile (GIS) region names"""
    REGION_1 = ("Region I - Ilocos Region", "Region I (Ilocos Region)")
    REGION_2 = ("Region II - Cagayan Valley", "Region II (Cagayan Valley)")
    REGION_3 = ("Region III - Central Luzon", "Region III (Central Luzon)")
    REGION_4A = ("Region IVA - CALABARZON", "Region IV-A (CALABARZON)")
    MIMAROPA = ("Region IVB - MIMAROPA", "MIMAROPA Region")
    REGION_5 = ("Region V - Bicol", "Region V (Bicol Region)")
    REGION_6 = ("Region VI - Western Visayas", "Region VI (Western Visayas)")
    REGION_7 = ("Region VII - Central Visayas", "Region VII (Central Visayas)")
    REGION_8 = ("Region VIII - Eastern Visayas", "Region VIII (Eastern Visayas)")
    REGION_9 = ("Region IX - Zamboanga Peninsula", "Region IX (Zamboanga Peninsula)")
    REGION_10 = ("Region X - Northern Mindanao", "Region X (Northern Mindanao)")
    REGION_11 = ("Region XI - Davao", "Region XI (Davao Region)")
    REGION_12 = ("Region XII - SOCCSKSARGEN", "Region XII (SOCCSKSARGEN)")
    REGION_13 = ("Region XIII - Caraga", "Region XIII (Caraga)")
    NCR = ("National Capital Region", "National Capital Region (NCR)")
    CAR = ("Cordillera Administrative Region", "Cordillera Administrative Region (CAR)")
    BARMM = ("Bangsamoro Autonomous Region in Muslim Mindanao", "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)")

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
    """Loads and cleans CSV/Shapefile data, simplifying geometries for faster web rendering"""
    df = pd.read_csv('./datasets/clean/fies_2023.csv')
    gdf = gpd.read_file("./datasets/raw/Regions.shp.shp")
    risk_df = pd.read_csv('./datasets/clean/disaster_risk_index.csv')

    # Reduce geometric complexity to improve browser performance
    gdf['geometry'] = gdf['geometry'].simplify(0.01)
    gdf = gdf.to_crs(epsg=4326)

    # Manual fix for Davao City placement within its parent region
    gdf['name'] = gdf['name'].replace({'Davao City': 'Region XI (Davao Region)'})

    # Filter national vs regional data and merge GIS geometry with FIES statistics
    mapping_dict = RegionMapping.get_map_dict()
    nat_avg_df = df[df['REGION'] == "All Regions (National Avg)"].copy()
    regional_df = df[df['REGION'] != "All Regions (National Avg)"].copy()
    regional_df['SHP_NAME'] = regional_df['REGION'].map(mapping_dict)

    # Combine spatial data with expenditure and disaster risk scores
    map_gdf = gdf.merge(regional_df, left_on='name', right_on='SHP_NAME', how='inner')
    map_gdf = map_gdf.merge(
        risk_df[['PH Region', 'Disaster Risk Score']],
        left_on='REGION', right_on='PH Region', how='left')

    return nat_avg_df, map_gdf, OFFICIAL_ORDER, risk_df


# --- 3. UI STYLING & HELPERS ---

def inject_custom_css():
    st.set_page_config(
        layout="wide",
        page_title="NAVI Hub",
        page_icon="🇵🇭",
        initial_sidebar_state="collapsed"  # 👈 mobile-friendly default
    )

    st.markdown("""
    <style>
    button.viewerBadge_link__1S137, .main header, a.header-anchor { display: none !important; }

    .stMetric { 
        padding: 15px; 
        border-radius: 8px; 
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); 
        border: 1px solid #eee; 
        text-align: center; 
        height: 120px; 
    }

    .stMetric > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricDelta"] {
        text-align: center;
        width: 100%;
        justify-content: center;
    }

    /* Prevent horizontal scrolling */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }

    /* 📱 MOBILE RESPONSIVENESS */
    @media (max-width: 768px) {

        /* Stack KPI cards */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Reduce metric size */
        .stMetric {
            height: auto !important;
            padding: 10px;
        }

        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }

        .stMetric [data-testid="stMetricDelta"] {
            font-size: 0.85rem !important;
        }

        /* Header scaling */
        h1 {
            font-size: 1.6rem !important;
        }

        p {
            font-size: 0.9rem !important;
        }

        /* Sidebar becomes overlay-friendly */
        section[data-testid="stSidebar"] {
            width: 80% !important;
        }
    }

    .project-footer {
        text-align: center;
        color: gray;
        font-size: 0.85rem;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


def _shorten_region_name(region: str) -> str:
    """Abbreviates long regional names for better axis label fitting."""
    overrides = {
        "Bangsamoro Autonomous Region in Muslim Mindanao": "BARMM",
        "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)": "BARMM",
        "Cordillera Administrative Region": "CAR",
        "Cordillera Administrative Region (CAR)": "CAR",
        "National Capital Region": "NCR",
        "National Capital Region (NCR)": "NCR",
    }
    return overrides.get(region, region)


def initialize_sidebar_controls(region_options: List[str]) -> Tuple[List[str], List[str]]:
    """Handles the sidebar logic for selecting specific regions and expenditure categories"""
    st.sidebar.title("🔍 Navigator Controls")

    # Region Multi-selector toggle
    st.sidebar.markdown("### 🗺️ Select Regions",
                        help="Choose which PH regions to include in the average calculations and charts.")

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
    st.sidebar.markdown("### 💰 Expenditure Categories",
                        help="Toggle specific spending types to see how they impact regional affordability.")

    # Bulk action buttons for expenditure checkboxes
    def bulk_set_category_state(is_enabled: bool):
        for category in Expenditure: st.session_state[category.name] = is_enabled

    c1, c2 = st.sidebar.columns(2)
    c1.button("Select All", on_click=bulk_set_category_state, args=(True,), use_container_width=True)
    c2.button("Clear", on_click=bulk_set_category_state, args=(False,), use_container_width=True)

    active_categories = []
    for category in Expenditure:
        if category.name not in st.session_state:
            st.session_state[category.name] = True

        label = category.get_display_label()

        if st.sidebar.checkbox(label, key=category.name):
            active_categories.append(category.value)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💳 Personal Finance",
                        help="Input your monthly income to see your personalized Affordability Ratio for each region.")

    user_salary = st.sidebar.number_input("Monthly Salary (PHP)", min_value=0, value=18000, step=1000)

    return selected_regions, active_categories, user_salary


# --- 4. VISUALIZATION ENGINE ---

def build_regional_choropleth(map_gdf: gpd.GeoDataFrame, highlight_indices: List[int], user_salary: float) -> go.Figure:
    """Generates the interactive PH map with affordability ranking on hover"""
    # Calculate affordability rank dynamically based on currently selected categories
    map_gdf['Exp_Rank'] = map_gdf['DYNAMIC_Z'].rank(ascending=True, method='min')

    fig = go.Figure(go.Choroplethmapbox(
        geojson=json.loads(map_gdf.to_json()), locations=map_gdf.index,
        z=map_gdf['DYNAMIC_Z'], colorscale=BLUE_PALETTE, marker_opacity=0.7,
        selectedpoints=highlight_indices, selected={'marker': {'opacity': 1.0}},
        unselected={'marker': {'opacity': 0.15}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "━━━━━━━━━━━━━━━━━━<br>"
            "💰 Monthly Cost: ₱%{z:,.0f}<br>"
            "🏆 Affordability Rank: #%{customdata[1]:.0f} of 17<br>"
            "📊 Status: <span style='color:%{customdata[3]}'><b>%{customdata[2]}</b></span><br>" 
            "<extra></extra>"
        ),
        customdata=map_gdf[['REGION', 'Exp_Rank', 'Aff_Status', 'Aff_Color']],
    ))
    fig.update_layout(mapbox=dict(style="carto-positron", center={"lat": 12.8797, "lon": 121.7740}, zoom=4.4),
                      margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=None)
    return fig


def build_horizontal_stacked_bar(map_gdf: gpd.GeoDataFrame, selected_regions: List[str], categories: List[str],
                                 sort_order: str, user_salary: float) -> go.Figure:
    """Generates the bar chart comparing total regional costs, sorted by value or official order"""
    plot_df = map_gdf[map_gdf['REGION'].isin(selected_regions)].copy()
    plot_df['TOTAL'] = plot_df[categories].sum(axis=1)

    if sort_order == "Descending Value":
        plot_df = plot_df.sort_values(by='TOTAL', ascending=True)
    else:
        order_map = {name: i for i, name in enumerate(OFFICIAL_ORDER)}
        plot_df['sort_idx'] = plot_df['REGION'].map(order_map)
        plot_df = plot_df.sort_values(by='sort_idx', ascending=False)

    LEGEND_LABELS = {
        "FOOD_MONTHLY": "Food",
        "CLOTH_MONTHLY": "Cloth",
        "HOUSING_WATER_MONTHLY": "Housing & Water",
        "HEALTH_MONTHLY": "Health",
        "EDUCATION_MONTHLY": "Education",
    }

    fig = go.Figure()
    for cat in categories:
        fig.add_trace(go.Bar(
            name=LEGEND_LABELS.get(cat, cat.replace('_MONTHLY', '').title()),
            y=plot_df['REGION'].apply(_shorten_region_name),
            x=plot_df[cat],
            orientation='h', marker_color=CATEGORY_COLORS.get(cat),
            hovertemplate="<b>%{y}</b><br>" + f"{LEGEND_LABELS.get(cat, cat.replace('_MONTHLY', '').title())}: ₱%{{x:,.2f}}<extra></extra>"
        ))

    for i, row in plot_df.iterrows():
        ratio = row['TOTAL'] / user_salary if user_salary > 0 else 999
        _, status_color = get_affordability_status(ratio)

        fig.add_annotation(x=row['TOTAL'], y=_shorten_region_name(row['REGION']),
                           text=f" ₱{row['TOTAL']:,.0f}",
                           showarrow=False,
                           xanchor='left', font=dict(size=11, color=status_color))

    # Attach legend to x-axis label
    combined_title = (
        "Monthly Expenditure (PHP)<br><br>"
        f"<span style='font-size:12px;'>"
        f"<span style='color:{GREEN}'>● Affordable</span> &nbsp;&nbsp; "
        f"<span style='color:{YELLOW}'>● Break-even</span> &nbsp;&nbsp; "
        f"<span style='color:{RED}'>● Unaffordable</span>"
        f"</span>"
    )

    dynamic_height = min(600, max(300, len(selected_regions) * 35))
    fig.update_layout(
        barmode='stack',
        template="plotly_white",
        height=dynamic_height,
        margin={"t": 30, "b": 100, "l": 150, "r": 80},
        xaxis=dict(
            title=combined_title,
            tickformat=",.0f",
            tickprefix="₱",
            showgrid=True
        ),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            traceorder='normal',
            itemclick=False,
            itemdoubleclick=False
        )
    )
    return fig


def build_risk_heatmap(risk_df: pd.DataFrame, sort_order: str = "Selected Value/s", selected_regions: List[str] = None, filter_to_selected: bool = False) -> go.Figure:
    """Generates the heatmap displaying frequency and impact of disasters by region"""
    heatmap_df = risk_df[[
        'PH Region',
        'Disaster Frequency (Normalized)',
        'Human Impact (Normalized)',
        'Economic Impact (Normalized)',
        'Disaster Risk Score'
    ]].copy()
    heatmap_df.columns = ['Region', 'Frequency', 'Human Impact', 'Economic Impact', 'Disaster Risk Score']
    heatmap_df = heatmap_df.set_index('Region')

    # Sort by official order first, then split and reorder if needed
    order_map = {name: i for i, name in enumerate(OFFICIAL_ORDER)}
    heatmap_df['sort_idx'] = heatmap_df.index.map(order_map)
    heatmap_df = heatmap_df.sort_values('sort_idx', ascending=False).drop(columns='sort_idx')

    if sort_order == "By Risk Score":
        heatmap_df = heatmap_df.sort_values('Disaster Risk Score', ascending=True)  # ascending=True puts highest at top
    elif sort_order == "Selected Value/s" and selected_regions:
        # Selected regions float to the top; rest stay in official order below
        selected_set = set(selected_regions)
        rest = heatmap_df[~heatmap_df.index.isin(selected_set)]
        selected = heatmap_df[heatmap_df.index.isin(selected_set)]
        heatmap_df = pd.concat([rest, selected])

    # Build multi-line hover strings for diagnostic detail in the heatmap
    hover_array = []
    for region in heatmap_df.index:
        rd = risk_df[risk_df['PH Region'] == region].iloc[0]
        freq = heatmap_df.loc[region, 'Frequency']
        human = heatmap_df.loc[region, 'Human Impact']
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

    # Bold label for all selected regions; plain text for the rest
    selected_set = set(selected_regions) if selected_regions else set()
    y_labels = [
        f"<b>►&nbsp; {r}</b>" if r in selected_set else r
        for r in heatmap_df.index
    ]
    colorbar_cfg = dict(
        title="Risk Severity (0-100)", title_side='right',
        tickmode='linear', tick0=0, dtick=10,
        len=0.75, thickness=15
    )

    fig = go.Figure()

    if filter_to_selected and selected_regions:
        selected_set = set(selected_regions)
        all_y = list(heatmap_df.index)
        all_x = heatmap_df.columns.tolist()
        ncols = len(all_x)

        grey_z, grey_hover, amber_z, amber_hover = [], [], [], []
        for i, region in enumerate(all_y):
            row_vals = heatmap_df.iloc[i].values.tolist()
            if region in selected_set:
                grey_z.append([None] * ncols)
                grey_hover.append(hover_array[i])
                amber_z.append(row_vals)
                amber_hover.append(hover_array[i])
            else:
                grey_z.append(row_vals)
                grey_hover.append(hover_array[i])
                amber_z.append([None] * ncols)
                amber_hover.append(hover_array[i])

        fig.add_trace(go.Heatmap(
            z=grey_z, x=all_x, y=all_y,
            colorscale=GREY_PALETTE,
            customdata=grey_hover,
            hovertemplate='%{customdata}<extra></extra>',
            showscale=False, zmin=0, zmax=100,
            xgap=2, ygap=2
        ))
        fig.add_trace(go.Heatmap(
            z=amber_z, x=all_x, y=all_y,
            colorscale=AMBER_PALETTE,
            customdata=amber_hover,
            hovertemplate='%{customdata}<extra></extra>',
            showscale=True, zmin=0, zmax=100,
            colorbar=colorbar_cfg,
            xgap=2, ygap=2
        ))
    else:
        # Standard single-trace rendering
        fig.add_trace(go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale=AMBER_PALETTE,
            customdata=hover_array,
            hovertemplate='%{customdata}<extra></extra>',
            colorbar=colorbar_cfg,
            xgap=2, ygap=2
        ))
    fig.update_layout(
        title={'text': '<b>Disaster Risk Profile Diagnostic</b>', 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 18}},
        xaxis=dict(title='Risk Component', side='bottom',
                   tickfont=dict(size=12), tickangle=-45, showgrid=False),
        yaxis=dict(title=None, tickmode='array',
                   tickvals=list(heatmap_df.index), ticktext=y_labels,
                   tickfont=dict(size=12), showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=250, r=150, t=100, b=120),
        height=550
    )

    return fig


# --- 5. MAIN APPLICATION ---

def main():
    # Load styling and fetch data
    inject_custom_css()
    nat_avg_df, map_gdf, options_list, risk_df = fetch_and_preprocess_data()
    selected_regions, selected_cats, user_salary = initialize_sidebar_controls(options_list)

    # Dynamic column for choropleth mapping based on selected categories
    map_gdf['DYNAMIC_Z'] = map_gdf[selected_cats].sum(axis=1) if selected_cats else 0

    map_gdf['Aff_Ratio'] = map_gdf['DYNAMIC_Z'].replace(0, 1) / user_salary
    # Create the Status and Color columns using your variables
    status_data = map_gdf['Aff_Ratio'].apply(get_affordability_status)
    map_gdf['Aff_Status'] = status_data.apply(lambda x: x[0])
    map_gdf['Aff_Color'] = status_data.apply(lambda x: x[1])

    indices_to_highlight = map_gdf.index[map_gdf['REGION'].isin(selected_regions)].tolist()

    # Dashboard Header Section
    st.markdown("<h1 style='margin-bottom:0;'>🇵🇭 NAVI Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.2rem; font-weight:bold; margin-top:0; margin-bottom:0;'>National Affordability & Vulnerability Index</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:1rem; color:gray; margin-top:0;'>A PH Regional Cost & Resilience Navigator</p>",
                unsafe_allow_html=True)

    # KPI Row: Compares selection average against national baseline
    if selected_regions:
        sel_avg = map_gdf[map_gdf['REGION'].isin(selected_regions)][selected_cats].sum(axis=1).mean()
        nat_avg = nat_avg_df[selected_cats].sum(axis=1).iloc[0]

        k1, k2, k3 = st.columns([1,1,1], gap="small")
        with k1:
            st.metric("National Average Monthly Expenditure", f"₱{nat_avg:,.2f}",
                      help="The baseline monthly cost for a typical individual across all PH regions")
        with k2:
            st.metric(f"Selected Average ({len(selected_regions)} Regions)", f"₱{sel_avg:,.2f}",
                      delta=f"₱{sel_avg - nat_avg:,.2f} vs National Average",
                      help="The average cost of living for your currently selected PH regions")
        with k3:
            # Ratio: Cost / Income (Lower is better)
            sel_ratio = sel_avg / user_salary if user_salary > 0 else 0
            # Define the help text with clear breakdowns
            ratio_help = """
            **Affordability Ratio Calculation:**
            `Average Monthly Expenditure / Personal Monthly Income`
            **What the values mean:**
            * 🟢 **< 0.99 (Affordable):** Your income comfortably covers the regional cost of living.
            * 🟡 **0.99 - 1.01 (Break-even):** Your income exactly matches the basic regional costs.
            * 🔴 **> 1.01 (Unaffordable):** Regional costs exceed your monthly income.
            """
            st.metric(
                "Your Affordability Ratio",
                f"{sel_ratio:.2f}",
                delta=get_affordability_status(sel_ratio)[0],
                delta_color="normal" if sel_ratio < 1.01 else "inverse",
                help=ratio_help
            )

    # Main Geospatial Navigator
    st.plotly_chart(
        build_regional_choropleth(map_gdf, indices_to_highlight, user_salary),
        use_container_width=True,
        config={"responsive": True}
    )

    st.markdown("---")
    st.subheader("📊 Regional Expenditure Comparison", help="Compare how different spending categories stack up across your selected regions.")

    # Interactive sorting control for the bar chart
    sort_order = st.radio("Chart Sort Order:", ["Descending Value", "Official Regional Order"], horizontal=True)

    if selected_cats and selected_regions:
        st.plotly_chart(
            build_horizontal_stacked_bar(map_gdf, selected_regions, selected_cats, sort_order, user_salary),
            use_container_width=True,
            config={"responsive": True}
        )

    st.markdown("---")
    st.subheader("🔥 Regional Vulnerability Matrix", help="A diagnostic view of climate and disaster risks. Higher scores indicate greater exposure to frequency, human, or economic impacts.")

    # Interactive sorting control for the heatmap
    heatmap_sort_order = st.radio("Chart Sort Order:", ["Selected Value/s", "Official Regional Order", "By Risk Score"], horizontal=True, key="heatmap_sort")

    # Show filter toggle only when a subset of regions is selected
    all_selected = len(selected_regions) == len(options_list)
    if not all_selected:
        filter_to_selected = st.checkbox("Highlight selected regions only", value=False, key="heatmap_filter")
    else:
        filter_to_selected = False

    # When all regions are selected, suppress highlights (nothing stands out if everything is selected)
    heatmap_regions = None if all_selected else selected_regions

    # Only bold a region in the heatmap if exactly ONE region is selected.
    # Prevents NCR bolding on "Select All" default.
    if len(selected_regions) == 1:
        heatmap_ref = selected_regions[0]
    else:
        heatmap_ref = None

    # Final Risk Matrix Visualization
    st.plotly_chart(
        build_risk_heatmap(risk_df, sort_order=heatmap_sort_order, selected_regions=heatmap_regions, filter_to_selected=filter_to_selected),
        use_container_width=True,
        config={"responsive": True}
    )

    # Footer Reference Section
    st.markdown("---")

    with st.container():
        st.markdown(f"""
                <div class="project-footer">
                    <p><b>Data Sources & References:</b><br>
                    Expenditure: FIES Dataset (PSA 2023) • Boundaries: PH PSGC Shapefiles (PSA) • 
                    Demographics: Population & GDP (PSA 2015-2024) • Risk: EM-DAT (2015-2024)
                    </p>
                    <p style="font-style: italic; opacity: 0.7;">
                        Developed as a final requirement for DAT101M: Data Visualization at De La Salle University
                    </p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()