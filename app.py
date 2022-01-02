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

col3, col4 = st.columns(2)
with col3:
    st.subheader('Linear Regression')
    chart_type = st.selectbox('Choose your chart type', plot_types)
    plot = matplotlib_plot(chart_type, full_health_data)
    st.pyplot(plot)
    
with col4:    
    st.subheader('Regression Table')
    model = smf.ols('Calorie_Burnage ~ Average_Pulse', data = full_health_data)
    results = model.fit()
    st.write(results.summary())

with col3:
    string = 'Calorie_Burnage = ' + str(results.params['Average_Pulse']) + ' x Average_Pulse + ' + str(results.params['Intercept'])
    st.write('The linear regression function can be written mathematically as:')
    st.markdown('**' + string + '**')




# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)

# print(full_health_data.describe())

# # .......... # .......... # .......... # .......... # .......... # .......... #

# Max_Pulse = full_health_data["Max_Pulse"]
# percentile10 = np.percentile(Max_Pulse, 10)
# print(" ")
# print(percentile10)

# # .......... # .......... # .......... # .......... # .......... # .......... #

# std = np.std(full_health_data)
# print(" ")
# print(std)
# cv = np.std(full_health_data) / np.mean(full_health_data)
# print(" ")
# print(cv)
# var = np.var(full_health_data)
# print(" ")
# print(var)
# Corr_Matrix = round(full_health_data.corr(),2)
# print(Corr_Matrix)

# # .......... # .......... # .......... # .......... # .......... # .......... #

# x = full_health_data["Average_Pulse"]
# y = full_health_data["Calorie_Burnage"]

# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#     return slope * x + intercept

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.ylim(ymin=0, ymax=2000)
# plt.xlim(xmin=0, xmax=200)
# plt.xlabel("Average_Pulse")
# plt.ylabel ("Calorie_Burnage")
# plt.savefig("regression.png")

# # .......... # .......... # .......... # .......... # .......... # .......... #

# model = smf.ols('Calorie_Burnage ~ Average_Pulse', data = full_health_data)
# results = model.fit()
# print(results.summary())

# df = pd.DataFrame(
#     np.random.randn(10, 5),
#     columns=('col %d' % i for i in range(5)))

# st.table(full_health_data)




