import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os 
import altair as alt
import plotly.colors as pc
import plotly.express as px
st.set_page_config(
    page_title="World Green House Gases Analysis Dashboard",
    page_icon=":world:",
    layout="wide",
)

st.title("World Green House Gases Analysis Dashboard")

# Load your data
work_dir=os.getcwd()+'/'
#df_=pd.read_csv(work_dir+'global_owid-co2-data.csv')
df_=pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv')
df_metadata=pd.read_csv(work_dir+'global_owid-co2-codebook.csv')
df_metadata=pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-codebook.csv')
df=df_[df_['year']>1900]
df['GDP/Capita']=df['gdp']/df['population']
df_addmeta=pd.DataFrame({
    'column':['GDP/Capita'],
    'description':['Gross domestic product (GDP) divided by midyear population - GDP per capita'],
    'source':['Manual Calculation'],
    'unit':['international-$ in 2011 prices']
})
df_metadata=pd.concat([df_metadata, df_addmeta], ignore_index=True)
#df = pd.read_csv('your_data.csv')  # Change to your actual data source
cols = st.columns([1, 3])

#VARIABLES = [col for col in df.columns if col.lower() != 'date']
# ...existing code...

# Get min and max year for slider
#min_year = int(df['year'].min())
#max_year = int(df['year'].max())

#with cols[1]:
#    year_range = st.slider(
#        "Select year range:",
#        min_value=min_year,
#        max_value=max_year,
#        value=(min_year, max_year),
#        step=1)

# Filter dataframe by selected year range
#df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
# Update all references from df to df_filtered below
years = sorted(df['year'].unique())
with cols[0]:
    year_range = st.select_slider("Select year range:", options=years, value=(years[0], years[-1]))

df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

COUNTRIES = df['country'].unique().tolist()
country_resources=['GDP/Capita','GDP','population','trade_co2','gdp_per_unit_energy','gdp','population']
# ...existing code for country and variable selection, but use df_filtered instead of df...
with cols[0]:
    selected_countries = st.multiselect(
        "Choose countries to compare:",
        options=COUNTRIES,
        default=['Indonesia','Malaysia','Australia','Singapore'] if len(COUNTRIES) >= 3 else COUNTRIES,
        placeholder="Select countries from your data"
    )
if not selected_countries:
    st.info("Pick some countries to compare", icon="ℹ️")
    st.stop()

NUMERIC_VARIABLES = [col for col in df.select_dtypes(include='number').columns if col.lower() not in ['year']]

with cols[0]:
    selected_var = st.selectbox(
        "Choose variable to compare:",
        options=NUMERIC_VARIABLES,
        index=NUMERIC_VARIABLES.index('co2') if 'co2' in NUMERIC_VARIABLES else 0,
        placeholder="Select a numeric variable (column) from your data"
    )


if not selected_var:
    st.info("Pick some variables to compare", icon="ℹ️")
    st.stop()
bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)
unit = df_metadata[df_metadata['column'] == selected_var]['unit'].values[0]
# Calculate change for each country
country_changes = {}
for country in selected_countries:
    country_data = df_filtered[df_filtered['country'] == country][selected_var].dropna()
    if not country_data.empty:
        change = country_data.iloc[-1] - country_data.iloc[0]
        country_changes[country] = change

if country_changes:
    max_change_country = max(country_changes.items(), key=lambda x: x[1])
    min_change_country = min(country_changes.items(), key=lambda x: x[1])

    with bottom_left_cell:
        if selected_var in country_resources:
            dedelcol="normal" 
        else:
            dedelcol="inverse"
        cols_metric = st.columns(2)
        cols_metric[0].metric(
            f"Highest \n({unit})",
            df[df['country']==max_change_country[0]]['iso_code'].values[0],
            delta=f"{max_change_country[1]:,.2f}",delta_color=dedelcol,
            width="content",
        )
        cols_metric[1].metric(
            f"Lowest \n({unit})",
            df[df['country']==min_change_country[0]]['iso_code'].values[0],
            delta=f"{min_change_country[1]:,.1f}",delta_color=dedelcol,
            width="content",
        )
else:
    with bottom_left_cell:
        st.info("No data available for selected countries and variable.", icon="ℹ️")
#st.markdown(max_change_country)
# Main plot: raw values for each selected variable
with cols[1]:
    fig = go.Figure()
    for country in selected_countries:
        country_data = df_filtered[df_filtered['country'] == country]
        fig.add_trace(go.Scatter(x=country_data['year'], y=country_data[selected_var], mode='lines', name=f"{country}",marker=dict(size=6)))
    desc_title = df_metadata[df_metadata['column'] == selected_var]['description'].values[0].split(' - ')#[0]
    fig.update_layout(title=f"{desc_title[0].title()}", xaxis_title="Year", yaxis_title=f"{unit}",annotations=[
        dict(
            text=f"{desc_title[1].title()}",
            xref="paper", yref="paper",
            x=0, y=1.1, showarrow=False,
            font=dict(size=14),
         )])
    st.plotly_chart(fig, use_container_width=True)

# ...existing code for peer comparison, but use df_filtered instead of df...
st.markdown(f"## Individual Country vs Peer Average ({year_range[0]} - {year_range[-1]})")
if len(selected_countries) < 2:
    st.warning("Pick 2 or more countries to compare them")
    st.stop()

NUM_COLS = 4
peer_cols = st.columns(NUM_COLS)

for i, country in enumerate(selected_countries):
    country_data = df_filtered[df_filtered['country'] == country]
    peer_countries = [c for c in selected_countries if c != country]
    peer_data = df_filtered[df_filtered['country'].isin(peer_countries)]

    # ---- Chart 1: Country vs Peer Average ----
    peer_avg = peer_data.groupby('year')[selected_var].mean().reset_index()
    country_data = country_data[['year', selected_var]].reset_index(drop=True)
    country_data['Peer average'] = peer_avg[selected_var]

    plot_data = country_data.melt(
        id_vars=['year'],
        value_vars=[selected_var, 'Peer average'],
        var_name='Series',
        value_name='Value'
    )

    chart1 = alt.Chart(plot_data).mark_line().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('Value:Q', title=selected_var.replace('_', ' ').title()),
        color=alt.Color('Series:N',
                        scale=alt.Scale(domain=[selected_var, 'Peer average'],
                                        range=['red', 'gray']),
                        legend=alt.Legend(orient="bottom", labelFontSize=11, titleFontSize=11)),
        tooltip=['year', 'Series', 'Value']
    ).properties(
        title=f"{country} vs Peer Average",
        height=300
    )

    cell = peer_cols[(i * 4 + 0) % NUM_COLS].container(border=True)
    cell.altair_chart(chart1, use_container_width=True)

    # ---- Chart 2: Delta ----
    country_data['Delta'] = country_data[selected_var] - country_data['Peer average']
    chart2 = alt.Chart(country_data).mark_area().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('Delta:Q', title='Delta'),
        tooltip=['year', 'Delta']
    ).properties(
        title=f"{country} minus Peer Average",
        height=300
    )

    cell = peer_cols[(i * 4 + 1) % NUM_COLS].container(border=True)
    cell.altair_chart(chart2, use_container_width=True)

    # ---- Chart 3: GDP Comparison ----
    from scipy.stats import ttest_ind
    import numpy as np

    # ---- Chart 3: Statistical Test (Country vs Each Peer Country) ----
    var_for_test = selected_var  # variable to test differences on

    results = []
    for peer in peer_countries:
        country_vals = df_filtered[df_filtered['country'] == country][var_for_test].dropna()
        peer_vals = df_filtered[df_filtered['country'] == peer][var_for_test].dropna()

        if len(country_vals) > 1 and len(peer_vals) > 1:
            t_stat, p_val = ttest_ind(country_vals, peer_vals, equal_var=False)
            mean_diff = country_vals.mean() - peer_vals.mean()
        else:
            t_stat, p_val, mean_diff = np.nan, np.nan, np.nan

        results.append({
            'Peer Country': peer,
            'Mean Difference': mean_diff,
            't-statistic': t_stat,
            'p-value': p_val
        })

    df_test = pd.DataFrame(results)

    # Determine color based on whether the country is higher or lower than peer
    df_test['color'] = np.where(df_test['Mean Difference'] >= 0, 'red', 'steelblue')

    # Build the bar chart
    chart3 = alt.Chart(df_test).mark_bar(size=35).encode(
        x=alt.X('Peer Country:N', title='Peer Country', sort='-y'),
        y=alt.Y('Mean Difference:Q', title=f"{selected_var.replace('_',' ').title()} Mean Difference"),
        color=alt.Color('color:N', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('Peer Country', title='Peer'),
            alt.Tooltip('Mean Difference', format='.3f'),
            alt.Tooltip('t-statistic', format='.2f'),
            alt.Tooltip('p-value', format='.3f')
        ]
    ).properties(
        title=f"{country} vs Peer — Statistical Comparison",
        height=300
    )

    # Add text labels (p-values above bars)
    text_labels = chart3.mark_text(
        align='center', baseline='bottom', dy=-5, fontSize=11, color='white'
    ).encode(
        x='Peer Country:N',
        y='Mean Difference:Q',
        text=alt.Text('p-value:Q', format='.3f')
    )

    # Combine chart + labels
    combined_chart = chart3 + text_labels

    # Display in grid cell
    cell = peer_cols[(i * 4 + 2) % NUM_COLS].container(border=True)
    cell.altair_chart(combined_chart, use_container_width=True)

    # ---- Chart 4: Rank Table ----
    # Compute mean for ranking
    s = df_filtered.groupby('country')[selected_var].mean().sort_values(ascending=False)

# Ensure country exists in data
    if country in s.index:
        rank_idx = s.index.get_loc(country)
        window = 3
        df_neighbors = s.reset_index()
        df_neighbors.columns = ['country', f'mean_{selected_var}']
        df_neighbors['rank'] = df_neighbors.index + 1

        # Extract neighbors around the country
        df_neighbors[f'mean_{selected_var}'] = df_neighbors[f'mean_{selected_var}'].map('{:,.1f}'.format)
        neighbors = df_neighbors.iloc[max(rank_idx - window, 0): rank_idx + window + 1].copy()
        neighbors = neighbors.reset_index(drop=True)
        neighbors.rename(columns={f'mean_{selected_var}': f'Average {selected_var.replace("_", " ").title()}'}, inplace=True)

        # ---- Highlight selected country row ----
        def highlight_row(row):
            if row['country'] == country:
                # Return style for every column in this row
                return [
                    "background-color: #30336b; color: white; font-weight: bold; font-size: 16px;"
                ] * len(row)
            else:
                return [""] * len(row)
        neighbors.set_index('rank', inplace=True)
        styled_neighbors = neighbors.style.apply(highlight_row, axis=1)
        #styled_neighbors=styled_neighbors.set_index('country')
        # ---- Display table aligned with other graphs ----
        cell = peer_cols[(i * 4 + 3) % NUM_COLS].container(border=True)
        #cell.markdown(f"### Rank around **{country}**")
        cell.dataframe(styled_neighbors, use_container_width=True, height=300)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
st.markdown(f"### Global Map & Top 5 Countries by {desc_title[0].title()} ({year_range[0]} - {year_range[-1]})")
import plotly.express as px
import plotly.colors as pc
import streamlit as st
import pandas as pd

# --- Prepare dataset ---
# ✅ Make a copy to avoid modifying the original DataFrame
dfrankprep = df_filtered.copy()

# ✅ Drop rows with missing ISO codes (removes regions like World, Asia, etc.)
dfrankprep = dfrankprep.dropna(subset=['iso_code'])

# ✅ Group and rank countries
df_rank_var = (
    dfrankprep.groupby(['country'])[selected_var]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={selected_var: f'Average {selected_var.replace("_", " ").title()}'})
)

# ✅ Add rank index
df_rank_var['Rank'] = df_rank_var.index + 1
df_rank_var = df_rank_var.set_index('Rank')

# --- Plotly Choropleth Map ---
color_col = f'Average {selected_var.replace("_", " ").title()}'
color_scale = pc.sequential.OrRd  # You can also try 'Cividis', 'Plasma', 'Turbo'

fig = px.choropleth(
    df_rank_var,
    locations="country",
    locationmode="country names",  # Important for region strings
    color=color_col,
    hover_name="country",
    color_continuous_scale=color_scale,
    projection="natural earth",
)

# --- Styling: OWID-like dark theme ---
fig.update_layout(
    geo=dict(
        showframe=True,
        showcoastlines=True,
        coastlinecolor="white",
#        showland=True,
#        landcolor="#0f0f0f",
        bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_colorbar=dict(
        title=f"Average {selected_var.replace('_', ' ').title()}",
        ticks="outside",
        tickcolor="white",
        tickfont=dict(color="white"),
    ),
)

# --- Add clean borders ---
#fig.update_geos(
    #showcountries=True,
    #countrycolor="white",
    #showcoastlines=True,
    #coastlinecolor="white",
#)

# --- Display in Streamlit ---
st.plotly_chart(fig, use_container_width=True)



def highlight_row2(row):
    if row['country'] in selected_countries:
        # Return style for every column in this row
        return [
            "background-color: #30336b; color: white; font-weight: bold; font-size: 16px;"
        ] * len(row)
    else:
        return [""] * len(row)
    
df_rank_var[f'Average {selected_var.replace("_", " ").title()}']=df_rank_var[f'Average {selected_var.replace("_", " ").title()}'].map('{:,.1f}'.format)
styled_neighbors = df_rank_var.style.apply(highlight_row2, axis=1)
st.dataframe(styled_neighbors, use_container_width=True, height=200)

st.markdown("---")
st.markdown("## Meta Data")
#st.dataframe(df)
def highlight_row3(row):
    if row['column']==selected_var:
        # Return style for every column in this row
        return [
            "background-color: #30336b; color: white; font-weight: bold; font-size: 16px;"
        ] * len(row)
    else:
        return [""] * len(row)
styled_metadata=df_metadata.style.apply(highlight_row3,axis=1)
st.dataframe(styled_metadata)


