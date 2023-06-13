import streamlit as st
import pandas as pd
from statistics import NormalDist
from statsmodels.stats.weightstats import ztest
from scipy.stats import shapiro

st.title('ZTEST')

with st.expander('View data'):
    df = pd.read_csv('https://raw.githubusercontent.com/ethanweed/pythonbook/main/Data/zeppo.csv')
    st.dataframe(df.transpose())

with st.expander('View statistic'):
    st.dataframe(df.describe().transpose())

st.write('### Constructing hypothesis')
st.latex('H_{0..} : \mu = \mu_{0}')
st.latex('H_{1} : \mu = \mu_{1}')

alpha = st.number_input('Masukkan nilai alpha', step=0.0000001, min_value=0.01, max_value=0.99)
null_mean = st.number_input('Masukkan nilai \mu_{0}', step=0.001)

clicked = st.button('Do the z test')

if clicked:
    alpha_z = NormalDist().inv_cdf(p=1-alpha/2)
    z_score, p_value = ztest(df['grades'], value=null_mean, alternative='two-sided') # type: ignore

    if abs(z_score) > alpha_z:
        st.latex('REJECT H_{0}')
    else:
        st.latex('CAN NOT REJECT H_{0}')
    st.write(f'titik kritis = {alpha_z}, hitung z = {z_score}, p value = {p_value}')

st.write('## Check normality')

clicked_2 = st.button('Do the shapiro test')

if clicked_2:
    result = shapiro(df['grades'])
    st.write(result)
    st.bar_chart(df['grades'])