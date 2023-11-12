# To load the dataset and its visualisation
import subprocess
subprocess.call(["pip", "install", "-r", "requirements.txt"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import plotly_express as px
import pycountry_convert as pc
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.features import PCA as PCAVisualizer

# For data modeling
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV

# Model performance
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

#warnings
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

df = pd.read_csv('/content/unsupervised-learning-on-country-data/Country-data.csv')
# This chart shows the top ten countries in each feature.
def pie_chart_highest(df, nam_feature):
    df2 = df.sort_values(by=nam_feature, ascending=False)
    fig = px.pie(df2[:10], values=nam_feature, names='country',title='Top 10 Countries with the Highest '+nam_feature, color_discrete_sequence=px.colors.sequential.Purp_r)
    fig.show()

# This chart shows the ten worst countries in each feature.
def pie_chart_lowest(df, nam_feature):
    df3 = df.sort_values(by=nam_feature, ascending=True)
    fig = px.pie(df3[:10], values=nam_feature, names='country',title='Top 10 Countries with the Lowest '+nam_feature, color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.show()
# Main Streamlit app
def main(): 
    st.subheader("Correlation Heatmap")
    plt.figure(figsize = (10, 5))
    sns.heatmap(df.drop(columns=['country'], axis=1).corr(), annot=True, cmap='Blues', mask=np.triu(np.ones_like(df.drop(columns=['country'], axis=1).corr())))
    plt.show()

    st.subheader("Scatter plots showing linear relations")
    fig = px.scatter(df, x='gdpp', y='income', trendline="ols")
    fig.update_layout(template = 'plotly_dark')
    fig.show()

    fig = px.scatter(df, x='total_fer', y='child_mort', trendline="ols")
    fig.update_layout(template = 'plotly_dark')
    fig.show()

    fig = px.scatter(df, x='imports', y='exports', trendline="ols")
    fig.update_layout(template = 'plotly_dark')
    fig.show()

    fig = px.scatter(df, x='imports', y='exports', trendline="ols")
    fig.update_layout(template = 'plotly_dark')
    fig.show()

    st.subheader("Histograms")

    for column in df.drop(columns=['country']).columns:
         fig = px.histogram(df, x=df[column])
         fig.update_layout(template = 'plotly_dark')
         fig.show()
    
    for column in df.drop(columns=['country']).columns:
        pie_chart_highest(df, column)
        pie_chart_lowest(df, column)
     
    # For numerical columns

    numColumns = ['child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp']

    for column in numColumns:
        fig = px.box(df, x=df[column])
        fig.update_layout(template = 'plotly_dark')
        fig.show()


    for column in df.drop(columns = ['country', 'gdpp']).columns:
        fig = px.scatter(df, x=df[column], y='gdpp')
        fig.update_layout(template = 'plotly_dark')
        fig.show()
        

if __name__ == "__main__":
    main()