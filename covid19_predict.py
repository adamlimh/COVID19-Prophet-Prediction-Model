import pandas as pd
from fbprophet import Prophet

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

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()

#Train in Prophet
#Default 80% prediction interval with no tweaking of seasonality-related parameters and additional regressors
#by increasing changepoint_range, will increase the number of possible places where the rate can change
#by increasing changepoint_prior_scale, will increase the forecast uncertainty

recovered.columns = ['ds','y']
m = Prophet(changepoint_range=0.95,changepoint_prior_scale=0.5)

#Feed the recovered data
m.fit(recovered)

# Make the prediction of the following 7 days
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

#Provide the output on Apr. 12, Apr. 13, and Apr. 14
period=pd.date_range(start="2020-04-12",end="2020-04-14")

forecast['yhat']=forecast['yhat'].astype(int)
forecast['yhat_lower']=forecast['yhat_lower'].astype(int)
forecast['yhat_upper']=forecast['yhat_upper'].astype(int)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][forecast['ds'].isin(period)].rename(columns={'ds':'Date','yhat':'Predictive Overall recovered','yhat_lower':'Lower range','yhat_upper':'Upper range'}).reset_index(drop=True))

# Visualizing
recovered_forecast_plot = m.plot(forecast)
