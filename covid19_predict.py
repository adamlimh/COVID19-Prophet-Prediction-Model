import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Links to CSSEGISandData
confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

df_confirmed = pd.read_csv(confirmed_url)
df_deaths = pd.read_csv(deaths_url)
df_recovered = pd.read_csv(recovered_url)

# Concertrate into one dataframe
df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df=pd.melt(df_confirmed, id_vars=['Province/State','Country','Lat','Long'], var_name='Date').rename(columns={'value':'Confirmed'})
df_merge1=pd.melt(df_deaths, id_vars=['Province/State','Country','Lat','Long'], var_name='Date').rename(columns={'value':'Deaths'})
df_merge2=pd.melt(df_recovered, id_vars=['Province/State','Country','Lat','Long'], var_name='Date').rename(columns={'value':'Recovered'})
df=df.merge(df_merge1).merge(df_merge2)
df['Date'] = df['Date'].astype('datetime64[ns]')

def country_list():
    print(df['Country'].unique())
    
def province_list():
    print(df['Province/State'].dropna().unique())

# group = ['Global','Country','Province']
# types = ['Confirmed','Deaths','Recovered']
def predict_covid(group='Global', area='Global', types='Confirmed',predict_day=30):

    if types not in ['Confirmed','Deaths','Recovered']:
        raise ValueError('Incorrect Types')
        
    if predict_day>0:
        
        # Group by    
        if group == 'Global':
            df2 = df.groupby("Date").sum()[types].reset_index()
        elif group == 'Country':
            df2 = df[df['Country']==area].groupby("Date").sum()[types].reset_index()
        elif group == 'Province':
            df2 = df[df['Province/State']==area].groupby("Date").sum()[types].reset_index()
        else:
            raise ValueError('Incorrect Group')

        if len(df2)==0:
            raise ValueError(f'Incorrect {group}')

        df2.columns = ['ds','y']

        #Train in Prophet
        #Default 80% prediction interval with no tweaking of seasonality-related parameters and additional regressors
        #by increasing changepoint_range, will increase the number of possible places where the rate can change
        #by increasing changepoint_prior_scale, will increase the forecast uncertainty

        m = Prophet(changepoint_range=0.9,changepoint_prior_scale=0.5)

        #Feed the confirmed data
        m.fit(df2)

        # Make the prediction of the following days
        future = m.make_future_dataframe(periods=predict_day)
        forecast = m.predict(future)


        forecast['yhat']=forecast['yhat'].astype(int)
        forecast['yhat_lower']=forecast['yhat_lower'].astype(int)
        forecast['yhat_upper']=forecast['yhat_upper'].astype(int)

        last_record_date = df2['ds'].tail(1).values[0].astype(str)[:10]
        last_record_value = df2['y'].tail(1).values[0]
        print(f'Overall {types} record of {last_record_date}: {last_record_value}')
        print(forecast[['ds', 'yhat']].rename(columns={'ds':'Date','yhat':'Predictive Overall {}'.format(types),'yhat_lower':'Lower range','yhat_upper':'Upper range'}).reset_index(drop=True).tail(1))

        # Visualizing
        forecast_plot = m.plot(forecast)
    
    else:
        raise ValueError('Incorrect Days')
        
if __name__ == '__main__':
  predict_covid(group='Global', area='Global',types='Confirmed',predict_day=30)
