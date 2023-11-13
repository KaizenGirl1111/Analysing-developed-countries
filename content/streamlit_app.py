import os
import pandas as pd
import numpy as np
import seaborn as sns
import pandas_profiling
import plotly.express as px
import pycountry_convert as pc
import streamlit as st
import matplotlib.pyplot as plt
#warnings
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming 'Country-data.csv' is in the 'unsupervised-learning-on-country-data' directory
file_path = os.path.join(script_dir, 'unsupervised-learning-on-country-data/Country-data.csv')

df = pd.read_csv(file_path)

# This chart shows the top ten countries in each feature.
def pie_chart_highest(df, nam_feature):
    df2 = df.sort_values(by=nam_feature, ascending=False)
    fig = px.pie(df2[:10], values=nam_feature, names='country', title='Top 10 Countries with the Highest ' + nam_feature,
                 color_discrete_sequence=px.colors.sequential.Purp_r)
    st.plotly_chart(fig)

# This chart shows the ten worst countries in each feature.
def pie_chart_lowest(df, nam_feature):
    df3 = df.sort_values(by=nam_feature, ascending=True)
    fig = px.pie(df3[:10], values=nam_feature, names='country', title='Top 10 Countries with the Lowest ' + nam_feature,
                 color_discrete_sequence=px.colors.sequential.Purples_r)
    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.drop(columns=['country'], axis=1).corr(), annot=True, cmap='Blues',
                mask=np.triu(np.ones_like(df.drop(columns=['country'], axis=1).corr())))
    st.pyplot(fig)

    st.subheader("Scatter plots showing linear relations")
    fig = px.scatter(df, x='gdpp', y='income', trendline="ols")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    fig = px.scatter(df, x='total_fer', y='child_mort', trendline="ols")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    fig = px.scatter(df, x='imports', y='exports', trendline="ols")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    fig = px.scatter(df, x='imports', y='exports', trendline="ols")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader("Histograms")
    for column in df.drop(columns=['country']).columns:
        fig = px.histogram(df, x=df[column])
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    for column in df.drop(columns=['country']).columns:
        pie_chart_highest(df, column)
        pie_chart_lowest(df, column)

    # For numerical columns
    numColumns = ['child_mort', 'exports', 'health', 'imports', 'income',
                  'inflation', 'life_expec', 'total_fer', 'gdpp']

    for column in numColumns:
        fig = px.box(df, x=df[column])
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    for column in df.drop(columns=['country', 'gdpp']).columns:
        fig = px.scatter(df, x=df[column], y='gdpp')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
