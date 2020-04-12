# COVID19-Prophet-Prediction-Model
Use Prophet, that follows the sklearn model API, to call its fit and predict the number of total recovered on certain date.

# Installation/ Requirements/ Documentation
* Needs Python 3.x installed<br>
* Packages requirement: pandas, matplotlib, fbprophet, pystan<br>
* The easiest way to install Prophet is in Anaconda.<br>
# Nature of the data
* Real world data automatically updated from Johns Hopkins CSSE (https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)<br>
* Province/State: China - province name; US/Canada/Australia/ - city name, state/province name; Others - name of the event (e.g., "Diamond Princess" cruise ship); other countries - blank.<br>
* Country/Region: country/region name conforming to WHO (will be updated).
* Last Update: MM/DD/YYYY HH:mm (24 hour format, in UTC).<br>
* Confirmed: the number of confirmed cases. For Hubei Province: from Feb 13(GMT +8), we report both clinically diagnosed and lab-confirmed cases. For Italy, diagnosis standard might be changed since Feb 27 to "slow the growth of new case numbers."<br>
* Deaths: the number of deaths.<br>
* Recovered: the number of recovered cases.<br>
* Update frequency: once a day around 23:59 (UTC).<br>
  
# Features
Check out screenshots below<br>
![code](https://raw.githubusercontent.com/adamlimh/COVID19-Prophet-Prediction-Model/master/screenshots/code.png)<br>
# Output
![model run](https://raw.githubusercontent.com/adamlimh/COVID19-Prophet-Prediction-Model/master/screenshots/model%20run.png)
![visualizating](https://raw.githubusercontent.com/adamlimh/COVID19-Prophet-Prediction-Model/master/screenshots/visualizating.png)
