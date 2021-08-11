import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Type = ['Confirmed','Deaths','Recovered']
# predict_day not less than 7 days
def predict_covid(types='Confirmed',predict_day=7):

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

    # Group by Country
    df2 = df.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

    df3 = df.groupby('Date').sum()[types].reset_index()
    df3.columns = ['ds','y']

    #Train in Prophet
    #Default 80% prediction interval with no tweaking of seasonality-related parameters and additional regressors
    #by increasing changepoint_range, will increase the number of possible places where the rate can change
    #by increasing changepoint_prior_scale, will increase the forecast uncertainty
    
    m = Prophet(changepoint_range=0.95,changepoint_prior_scale=0.5)

    #Feed the confirmed data
    m.fit(df3)

    # Make the prediction of the following days
    future = m.make_future_dataframe(periods=predict_day)
    forecast = m.predict(future)


    forecast['yhat']=forecast['yhat'].astype(int)
    forecast['yhat_lower']=forecast['yhat_lower'].astype(int)
    forecast['yhat_upper']=forecast['yhat_upper'].astype(int)

    last_record_date = df3['ds'].tail(1).values[0].astype(str)[:10]
    last_record_value = df3['y'].tail(1).values[0]
    print(f'Last Overall {types} record of {last_record_date}: {last_record_value}')
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds':'Date','yhat':'Predictive Overall {}'.format(types),'yhat_lower':'Lower range','yhat_upper':'Upper range'}).reset_index(drop=True).tail(7))

    # Visualizing
    recovered_forecast_plot = m.plot(forecast)
if __name__ == '__main__':
  predict_covid(types='Confirmed',predict_day=7)
