import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from branca.colormap import linear


@st.cache_data
def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Convert to datetime
    df["date_occ"] = pd.to_datetime(df["date_occ"])
    
    # Extract features
    df["year"] = df["date_occ"].dt.year
    df["month_name"] = df["date_occ"].dt.month_name()
    df["day_of_week"] = df["date_occ"].dt.day_name()
    df["hour"] = df["date_occ"].dt.hour

    # Optional: define orders for display or plotting
    DAY_ORDER = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    MONTH_ORDER = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    return df, DAY_ORDER, MONTH_ORDER

# Load and preprocess the data once
crime_lapd, DAY_ORDER, MONTH_ORDER = load_and_preprocess_data("crime_20_24_clean.csv")

st.title("LADP Crime Dataset")
st.markdown("""
### Introduction
This dataset contains detailed records of crime incidents in Los Angeles from **2020 to 2025**, 
collected through the LAPD’s legacy reporting system. It includes key information such as the 
time and location of each incident, the type of crime reported, and general victim information 
that has been anonymized to protect privacy.

Since the data was originally transcribed from manual paper reports, it contained various errors 
and missing values. So this dataset was carefully cleaned and restricted to records through 
February 2024. This decision was made to prevent inconsistencies and potential bias that could 
skew the results because of the LAPD transitioned to a new reporting system on March 7, 2024. 
""")
st.space()



st.markdown("""
### Crime Throughout the Years
While monthly crime patterns remain fairly consistent from 2020 to 2024, the overall number of 
incidents has increased over time. Crime activity was highest in 2022 and 2023, while 2020 
recorded the lowest levels. A noticeable dip occurs each February, followed by a gradual rise 
in incidents from November through December.
""")
# Crime by Years
crime_trend = (crime_lapd.groupby(["year", "month_name"]).size().reset_index(name="count"))
crime_trend["month_name"] = pd.Categorical(
    crime_trend["month_name"],
    categories=MONTH_ORDER,
    ordered=True
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=crime_trend, x="month_name", y="count", hue="year", marker="o", palette="tab10")
ax.set_ylim(10000, 25000)
ax.legend(title="Year")
ax.set_title("Monthly Crime Count by Year")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
st.pyplot(fig)
st.space()



st.markdown("""
### Crime Throughout the Days
The heatmap clearly reveals that most crimes happen around noon, highlighted by the dark red at 
12 PM. Following this midday peak, crime levels remain high throughout the afternoon and evening, 
especially on Fridays. In contrast, the early morning hours, roughly between 4:00 AM and 5:00 AM, 
appear to be the quietest period, with fewer incidents reported.
""")
# Crime by hours and days
HOUR_ORDER = list(range(24))
hourly = (crime_lapd
        .pivot_table(
            index="day_of_week",
            columns="hour",
            values="dr_no",
            aggfunc="count"
        ).reindex(index=DAY_ORDER, columns=HOUR_ORDER))
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(hourly, cmap="Reds")
plt.title("Crime by Day of Week and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
st.pyplot(fig)
st.space()



st.markdown("""
### Type of Crimes
There are 139 different types of crimes classified in this dataset. The top 10 most common are 
led by vehicle theft, followed by identity theft, battery, and burglary. Many of these crimes 
tend to occur in locations such as streets, single-family dwellings, or parking lots. The stacked 
bar below suggests that the physical environment plays an important role in the type of criminal 
activity reported.
""")
# Stacked bar of top 10 crime types and premises
top_crimes = crime_lapd['crm_cd_desc'].value_counts().head(10).index
top_premises = crime_lapd['premis_desc'].value_counts().head(5).index
filtered = crime_lapd[crime_lapd['crm_cd_desc'].isin(top_crimes) & crime_lapd['premis_desc'].isin(top_premises)]
stacked_data = filtered.pivot_table(
    index='crm_cd_desc',
    columns='premis_desc',
    values='dr_no',
    aggfunc='count',
    fill_value=0
)
stacked_data = stacked_data.loc[stacked_data.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(10,6))
stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap='nipy_spectral')
ax.set_title("Crime Types by Premises")
ax.set_xlabel("Crime Type")
ax.set_ylabel("Number of Incidents")
plt.xticks(rotation=45, ha='right')
ax.legend(title='Premises')
ax.set_xticklabels([c[:15] + "..." if len(c) > 15 else c for c in stacked_data.index])
st.pyplot(fig)
st.space()




st.markdown("""
### Crime and Gender
The comparison of female and male victims indicates that most crimes affect adults in their mid-20s 
to mid-30s, with the 25-29 age group having the highest number of victims for both genders. 
There's also a clear pattern as people get older. Younger women (15-34) are more often victims 
than men, while in the 35-74 age range, men are more frequently targeted. After age 75, crime 
rates drop sharply for both genders and become roughly equal.
""")
# Bar chart of victim age and sex
filtered = crime_lapd[crime_lapd['vict_sex'].isin(['M', 'F'])].copy()
age_bins = list(range(0, 100, 5))
filtered['age_bin'] = pd.cut(filtered['vict_age'], bins=age_bins, right=False)
age_sex_counts = (filtered.groupby(['age_bin', 'vict_sex'], observed=False).size().unstack(fill_value=0))
age_sex_counts.index = [f"{int(interval.left)}-{int(interval.right-1)}" 
                        for interval in age_sex_counts.index]
age_sex_counts.plot(kind='bar', figsize=(10,6), color=['lightcoral', 'skyblue'], width=0.8)
plt.xlabel("Age Group")
plt.ylabel("Number of Crimes")
plt.title("Crime Counts by Victim Age and Sex")
plt.xticks(rotation=45)
plt.legend(title='Sex')
st.pyplot(plt.gcf())
st.space()



st.markdown("""
### Crime Area
The map shows crime across Los Angeles based on the 21 Community Police Station areas, with 
light yellow for lower-risk zones and deep red for higher-risk zones. Central area has the 
highest crime levels, while Foothill is the safest. 
""")
# Map of danger level by area name 
area_summary = (
    crime_lapd
    .groupby('area_name')
    .agg(
        total_crimes=('dr_no', 'count'),
        part1_crimes=('part_1-2', lambda x: (x == 1).sum()),
        lat=('lat', 'mean'),
        lon=('lon', 'mean')
    )
    .reset_index()
)
area_summary['severity_index'] = (
    area_summary['part1_crimes'] / area_summary['total_crimes']
)
area_summary['crime_norm'] = (
    area_summary['total_crimes'] / area_summary['total_crimes'].max()
)

area_summary['danger_index'] = (
    0.5 * area_summary['crime_norm'] +
    0.5 * area_summary['severity_index']
)

area_summary['risk_level'] = pd.qcut(
    area_summary['danger_index'],
    q=3,
    labels=["Low", "Medium", "High"]
)

colormap = linear.YlOrRd_09.scale(
    area_summary['danger_index'].min(),
    area_summary['danger_index'].max()
)


m = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles="CartoDB positron")
for _, row in area_summary.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=row['danger_index'] * 35,
        color=colormap(row['danger_index']),
        fill=True,
        fill_color=colormap(row['danger_index']),
        fill_opacity=0.7,
        popup=folium.Popup(
        html=(
            f"<b style='font-size:16px'>{row['area_name']}</b><br><br>"
            f"<b>Total Crimes:</b> {row['total_crimes']}<br>"
            f"<b>Risk Level:</b> {row['risk_level']}<br>"
            f"<b>Danger Index:</b> {row['danger_index']:.2f}"
        ),
        max_width=300
        )).add_to(m)

colormap.caption = "Crime Danger Index"
colormap.add_to(m)
st.components.v1.html(m._repr_html_(), height=400)



























