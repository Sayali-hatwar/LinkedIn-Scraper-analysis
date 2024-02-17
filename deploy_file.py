import pickle
import streamlit as st
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

# File setup
st.title('Recommendation System ')
df = pd.read_csv('linkedin_data 2023-03-31_final.csv')
df1=df.drop('Date', axis=1)
df1.rename(columns={'Loaction': 'Location', 'Invovlement': 'Involvement'},inplace= True)
df2=df1.drop_duplicates()

def extract_location(lo):
    l = lo.split(',')
    if len(l) == 3:
        region = l[0]
        state = l[1]
    elif len(l) == 2:
        region = l[0]
        state = l[0]
    else:
        region = " Not Mention (remote)"
        state = " Not mention (Remote)"
    return pd.Series([region, state], index=['Region', 'State'])


df2[['Region', 'State']] = df2["Location"].apply(extract_location)

col1, col2, col3 = st.columns(3)

# Show plots
with col1:
    fig1, ax1 = plt.subplots(figsize=(12,12))
    sns.histplot(df['Job_type'], color='red', bins=10, ax=ax1)
    ax1.set_title('Job Type Distribution', fontsize=20)
    ax1.set_xlabel('Job Type')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

with col2:
    plt.figure(figsize=(12,12))
    plt.title('Top 10 Companies',fontsize=20)
    d3=df2['Company'].value_counts()
    subset = d3[:10]
    subset.plot(kind="pie",autopct="%1f%%")
    st.pyplot(plt)

with col3:
    plt.figure(figsize=(12,12))
    plt.title('Top 10 Industry',fontsize=20)
    d7=df2['Industry'].value_counts()
    subset = d7[:10]
    subset.plot(kind="pie",autopct="%1.2f%%")
    st.pyplot(plt)
  
  
# Applying One hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df2[['Job_Name', 'Company', 'Job_type','Industry', 'Involvement', 'Region', 'State']])
cosine_sim_matrix = cosine_similarity(encoded_data)

    
def recommend_companies_regionwise(region_name, cosine_sim_matrix, df2):
    region_index = df2[df2['Region'] == region_name].index[0]
    cosine_scores_region = list(enumerate(cosine_sim_matrix[region_index]))
    cosine_scores_region = sorted(cosine_scores_region, key=lambda x: x[1], reverse=True)
    cosine_scores_region = cosine_scores_region[1:11] # top 10 similar companies
    region_indices = [i[0] for i in cosine_scores_region]
    return (df2.iloc[region_indices])


def recommend_companies_statewise(state_name, cosine_sim_matrix, df2):
    state_index = df2[df2['State'] == state_name].index[0]
    cosine_scores_state = list(enumerate(cosine_sim_matrix[state_index]))
    cosine_scores_state = sorted(cosine_scores_state, key=lambda x: x[1], reverse=True)
    cosine_scores_state = cosine_scores_state[1:11] # top 10 similar companies
    state_indices = [j[0] for j in cosine_scores_state]
    return (df2.iloc[state_indices])


# Job oportunities according to job title
def recommend_companies_jobname(jobname1, cosine_sim_matrix, df2):
    job_index = df2[df2['Job_Name'] == jobname1].index[0]
    cosine_scores_job = list(enumerate(cosine_sim_matrix[job_index]))
    cosine_scores_job = sorted(cosine_scores_job, key=lambda x:x[1], reverse=True)
    cosine_scores_job = cosine_scores_job[1:11] 
    job_indices = [j[0] for j in cosine_scores_job]
    return (df2.iloc[job_indices])

def app():
    #st.set_page_config(fullscreen=True)



    # Show recommendations
#st.header('Recommendations')
    region_name = st.selectbox('Select a region', df2['Region'].unique(),index=0)
    similar_companies = recommend_companies_regionwise(region_name, cosine_sim_matrix, df2)
    st.subheader(f'Top 10 similar companies to {region_name}:')
    st.write(similar_companies)
    
    state_name = st.selectbox('Select a state', df2['State'].unique())
    similar_companies_state = recommend_companies_statewise(state_name, cosine_sim_matrix, df2)
    st.subheader(f'Top 10 similar companies to {state_name}:')
    st.write(similar_companies_state)
 

    jobname1 = st.selectbox('Select a Job Name', df2['Job_Name'].unique())
    similar_companies_jobname=recommend_companies_jobname(jobname1, cosine_sim_matrix, df2)
    st.subheader(f'Top 10 similar to {jobname1}:')
    st.write(similar_companies_jobname)

    col4, col5, col6 = st.columns(3)

    with col4:
        plt.figure(figsize=(12,12))
        plt.title(f'No. of Applicants in {region_name}',fontsize=20)
        dd4 = similar_companies['Applicant'].value_counts()
        dd4.plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(plt)


    with col5:
        plt.figure(figsize=(12,12))
        plt.title(f'No. of Applicants in {state_name}',fontsize=20 )
        dd5=similar_companies_state['Applicant'].value_counts()
        dd5.plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(plt)


    with col6:
        plt.figure(figsize=(12,12))
        plt.title(f'No. of Applicants for {jobname1}',fontsize=20)
        dd6=similar_companies_jobname['Applicant'].value_counts()
        dd6.plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(plt)





# Run app
if __name__ == '__main__':
    app()
