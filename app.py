from scipy import stats
from matplotlib.backends.backend_agg import RendererAgg
from make_plots import matplotlib_plot

import sys
import matplotlib
matplotlib.use('Agg')
_lock = RendererAgg.lock

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

plot_types = (
    "Scatter",
)

st.set_page_config(layout="wide")
st.title('Full Health Dataset')

full_health_data = pd.read_csv("data.csv", header=0, sep=",")

show_data = st.checkbox("Show data?", False)
if show_data:
    st.write('Full Health Dataset')
    st.dataframe(full_health_data)

col1, col2 = st.columns([3,1])
with col1:
    st.subheader('Descriptive Statistics')
    st.write(full_health_data.describe())
    st.subheader('Correlation Matrix')
    st.write(round(full_health_data.corr(),2))

with col2:
    st.subheader('Standard Deviation')
    st.write(np.std(full_health_data))
    st.subheader('Coefficient of Variation')
    st.write(np.std(full_health_data) / np.mean(full_health_data))
    st.subheader('Variance')
    st.write(np.var(full_health_data))

chart_type = st.selectbox('Choose your chart type', plot_types)
col3, col4 = st.columns(2)

with col3:
    st.subheader('Linear Regression Calorie_Burnage ~ Average_Pulse')
    plot = matplotlib_plot(chart_type, full_health_data)
    st.pyplot(plot)

    st.subheader('Regression Table Calorie_Burnage ~ Average_Pulse')
    model = smf.ols('Calorie_Burnage ~ Average_Pulse', data = full_health_data)
    results = model.fit()
    st.write(results.summary())

    string = 'Calorie_Burnage = ' + str(results.params['Average_Pulse']) + ' x Average_Pulse + ' + str(results.params['Intercept'])
    st.write('The linear regression function can be written mathematically as:')
    st.markdown('**' + string + '**')
    st.write("""
                _Summary - Predicting Calorie_Burnage with Average_Pulse_\n
                    - Coefficient of 0.3296, which means that Average_Pulse has a very small effect on Calorie_Burnage.\n
                    - High P-value (0.824), which means that we cannot conclude a relationship between Average_Pulse and Calorie_Burnage.\n
                    - R-Squared value of 0, which means that the linear regression function line does not fit the data well.\n
            """)

with col4:
    st.subheader('Linear Regression Calorie_Burnage ~ Average_Pulse + Duration')

    fig, ax = plt.subplots()

    x = full_health_data["Duration"] + full_health_data["Average_Pulse"]
    y = full_health_data ["Calorie_Burnage"]

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x))

    plt.scatter(x, y)
    plt.plot(x, mymodel)
    plt.ylim(ymin=0, ymax=2000)
    plt.xlim(xmin=0, xmax=200)
    plt.xlabel("Duration")
    plt.ylabel ("Calorie_Burnage")

    st.pyplot(fig)

    st.subheader('Regression Table Calorie_Burnage ~ Average_Pulse + Duration')
    new_model = smf.ols('Calorie_Burnage ~ Average_Pulse + Duration', data = full_health_data)
    new_results = new_model.fit()
    st.write(new_results.summary())

    new_string = 'Calorie_Burnage = ' + str(new_results.params['Average_Pulse']) + ' x Average_Pulse + Duration x ' + str(new_results.params['Duration']) + ' ' + str(new_results.params['Intercept'])
    st.write('The linear regression function can be written mathematically as:')
    st.markdown('**' + new_string + '**')
    st.write("""
                _Summary - Predicting Calorie_Burnage with Average_Pulse + Duration_\n
                    - Coefficients have a significant effect on Calorie_Burnage.\n
                        1. Calorie_Burnage increases with 3.17 if Average_Pulse increases by one.\n
                        2. Calorie_Burnage increases with 5.84 if Duration increases by one.\n
                    - Average_Pulse and Duration has a relationship with Calorie_Burnage.\n
                        1. P-value is 0.00 for Average_Pulse, Duration and the Intercept.
                        2. The P-value is statistically significant for all of the variables, as it is less than 0.05.
                    - The Adjusted R-squared is 0.814.\n
                
                ** THIS MODEL FITS THE DATA POINT WELL !!! **
            """)
