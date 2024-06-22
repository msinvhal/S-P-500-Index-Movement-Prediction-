# S&P 500 Index Movement Prediction
Predicting the Movement of the S&P 500 Using Time Series Modeling


**Predicting the S&P 500 Index**

**The Problem:**

Our project focuses on forecasting the S&P 500 index by using machine learning models. By leveraging historical values and incorporating a diverse array of both economic and non-economic predictors, we aim to unravel the complexities of the stock market and identify the key factors that significantly influence price movements. This endeavor will also shed light on how the market's dynamics have evolved over the past four decades, providing insights into its current nature and potential future trends.

**Motivation:**

The motivation behind this topic is our group’s interest in financial markets and desire to explore the dynamics of the S&P 500 to make informed future investment decisions. While the S&P 500 is a proven long-term investment, we recognize the need to understand various factors, both apparent and subtle, that could impact our portfolio. Leveraging our machine learning skills, we sought to create an accurate predictive model for positive investment returns in the S&P 500. Concurrently, we aimed to grasp the methodologies used by financial firms for forecasting stock prices, furthering our knowledge in financial market analysis and prediction.

**Datasets:**

A decent amount of our model's feature data was primarily sourced from the extensive archives of the Federal Reserve Economic Data (FRED), managed by the Federal Reserve Bank of St. Louis. FRED's database, encompassing over 800,000 economic time series, provided us with the flexibility to select datasets that aligned with our specific requirements in terms of granularity and time frame. We focused on integrating a diverse array of economic indicators, including interest rates, inflation rates, and GDP. Additionally, we broadened our dataset by incorporating non-traditional, yet potentially influential factors, such as humidity levels and cloud coverage in New York City that have been studied to potentially impact the mood of investors in the stock market. Furthermore, we considered the exchange rates of the US between countries like Australia and the UK. This data aggregation strategy ensured a robust and multi-dimensional analysis, enhancing the model's ability to capture complex market dynamics. 
The features we finally chose to use in our model were US unemployment rates, US interest rates, US gas prices, US consumer spending, US to Australia exchange rate, US federal surplus, US federal employees amount, US consumer price index, US interest rate spread, US personal savings rate, NYC humidity, NYC cloud cover, and US to UK exchange rates.

**Cleaning the Data:**

Our data collection process focused on acquiring information with daily granularity, extending as far back as the earliest available records, often predating 1980. To align this data with our model's requirements, we implemented a filtration step, ensuring all data fell within our specified date range. Although we had data from the 1980s, we chose to keep our training set to range from January 2010 to August 2020. The testing set ranged from September 2020 to August 2023. This was necessary to ensure that recent predictions weren’t influenced by larger past macroeconomic trends. We wanted to capture much of the 2010s’ post-recession boom in the S&P 500’s prices. The granularity was adjusted to include monthly data (eg, averages for weather metrics over a  month).
A critical step involved the consolidation of various features into a single, comprehensive dataframe. We achieved this via various pandas operations and dataframe manipulations. It is important to note that the S&P 500 data is exclusively available for market-open days. Saturday trading has been possible since the 1980s. Consequently, when extracting monthly data on the first day of each month, we encountered instances where these dates fell on market days that were closed. In such cases, we adjusted these data points to still reflect the current ‘price’ of the index (often, the closing price of the previous day) to maintain consistency with the market's operational schedule, ensuring the integrity and relevance of our dataset for the predictive model.

**Feature Engineering:**

We performed feature selection by a couple different tactics, one being the metric of variance inflation factor (VIF) to determine which features had the most significant impact on the model’s performance. When reviewing VIF scores, features above VIF 10 significantly impact the model. To prevent overfitting, we started removing the highest-scoring features due to their elevated VIFs. We sequentially removed the features “Open, High, and Low” because they had the highest VIFs. Next, we saw that “Consumer Spending” and “CPI” both had very high VIFs. A key visualization which we used showed a representation of the correlations between the features – a heatmap of collinearity (Appendix B). We observed from the correlation matrix that “CPI” and “Consumer Spending” had a correlation of 0.99. We believed it was a good idea to remove one of them because of their high correlation and high VIFs. We then sequentially removed the remaining features with a VIF above 10 (Appendix C). This resulted in enhanced model interpretability and reduced multicollinearity, making the features in the model more independent and conducive to accurate and reliable predictions.

**Regression Techniques:**
 
Ordinary Least Squares Regression:
Overview: The OLS regression aims to provide a simple linear relationship that minimizes the sum of squared differences between predicted and actual S&P 500 prices, offering insights into the potential influence of economic indicators, financial variables, and meteorological factors on the stock market's performance.
Analysis: The training R2 equals around 0.986 in our linear regression model which indicates a reasonably strong fit, signifying that a significant portion of the variation in the S&P price is accounted for by the linear relationship with the selected features.

CART: 
Overview: We performed a decision tree regression to handle non-linearity and irrelevant features by prioritizing informative features for splitting when predicting S&P 500 price.
Analysis: The CART model performed cross-validation to find the optimal ccp_alpha value in order to optimize the performance of the model. The CART model produced an R2 of 0.9746 which indicates a high predictive score but there is likely overfitting due to OSR2 being a negative value indicating poor performance on unseen data. 

Random Forest Model:
Overview: As an ensemble model, random forests build many decision trees during training to prevent overfitting. We understood that it was crucial for us to prevent overfitting when dealing with many different variables. 
Analysis: In the RF model, we attempted to adjust the hyperparameters by modifying random_state, n_estimators, etc. However, the model experienced high runtime without justification, despite achieving a high training R2 of around 0.99. A negative OSR2 indicated overfitting. We decided to leave the model at its default hyperparameters. 

Gradient Boosting Model: 
Overview: Gradient boosting identifies complex patterns and non-linear relationships. It combines multiple decision trees for accuracy and robustness, using gradient descent to minimize errors and prevent overfitting. 
Analysis: In the gradient boosting, we did not perform cross-validation nor adjust the hyperparameters from its default due to the same reasons for the random forest model. This also kept the hyperparameters consistent across RF and gradient boosting models which gave a better measure of which model better predicts S&P 500 prices at its default. The model produced a high R2 of around 0.99 and a negative OSR2 – similar to its peers.

Time Series Modelling:
We eventually realized that a time series model would be the most suitable model for predicting an entity like the S&P 500 owing to the mean reverting characteristics, sequential nature of the data, and overall volatility of the index. Professor Grigas also advised us to follow a rolling horizon style of forecasting for financial patterns. For this part of the project, we chose to keep our training set to range from January 1985 to August 2020. The testing set ranged from September 2020 to August 2023, same as before!

We considered two time series models – an “autoregressive integrated moving average” model (ARIMA) and the “long short-term memory” model (LSTM):

ARIMA:
Overview: Crucially, ARIMA is univariate, and uses differentiation to normalize patterns that are non-stationary for interpretation. S&P 500, by nature, fulfills that criteria. Moreover, it captures a moving average by learning from its past errors when predicting future patterns. We implemented it in a manner which allows for a rolling horizon implementation – which iterates through the testing portion and adds features to update the model’s forecasting. Thus, the model predicts patterns a lot more accurately. 
Analysis: We affirmed that a differentiating factor was needed through the Augmented Dickey-Fuller (ADF) test which generated a statistic which failed to reject the null hypothesis that the data is non-stationary. Moreover, we generated autocorrelation and partial autocorrelation graphs which further instructed our hyperparameter selection for the ‘order’ of the model – represented by (p,d,q). The time series graph, resulting from the predictions, and calculated errors confirmed our hypothesis as a group that a time series model was far superior for financial data. This ultimately served as a turning point in our analysis! A few limitations we recognized were the univariate nature of the model as well as its susceptibility to volatility. However, the S&P 500 does a fair representation of an ‘average’ level of volatility, so we were not exceedingly worried. Our RMSE for this model was around 206.25.


LSTM: 
Overview: The multivariate LSTM was the next model we considered. LSTM represents a complex form of deep learning as it utilizes many hyperparameters, neural networks in the form of ‘gates’. It worked similarly to ARIMA. However, a challenge we encountered was deciding hyperparameters alongside deciding which variables to include, if any, for the LSTM model. We decided to include all of the features we had anyway for linear regression, to serve as a fair reflection between two types of time series. Moreover, to prepare it for analysis, we had to further scale all our features and data to make it digestible for the LSTM model. We understood that the dropout layer hyperparameter acted as a form of ‘validation’ by randomly dropping that proportion of units from the LSTM neural network.
Analysis: Deciphering and understanding hyperparameters of the LSTM function was proved to be a challenging, yet extremely interesting task. Hyperparameters included LSTM layers and units, dropout rate, epochs, batch size, optimizer, and the loss function (MSE selected). We selected fairly standard and ‘stock’ suggestions for hyperparameters while tuning and experimenting accordingly. There were marginal differences between various versions of the code but we saw that the RMSE hovered around 225, with our model’s final value being approximately 225.05.


**Evaluating The Models:**

To evaluate the models we used R2 for training performance and OSR2, RMSE, and MAE for test set performance. While R2 and OSR2 assess how well the model explains the variance of the data, RMSE and MAE provide insight into the average error magnitude. Relying on a single metric could be misleading. A combination of these metrics ensures a balanced view of the model's performance, as shown in the table below.



The models with the highest training R2 were the gradient boosting regressor and random forest regressor, implying that these models fit the sample data exceptionally well. However, the data is most definitely overfitting due to an extremely low OSR2. For instance, an MAE of around 1106 in the case of random forests suggests that, on average, predictions are off by approximately 1106 units. 

As a group, we thought that the reason for this large difference between training and testing results has to do with the dynamic changes in the S&P 500 over time. The larger MAE and RMSE is attributed to the intricate nature of forecasting S&P 500 prices across more than a decade. The dynamic recent evolution of the S&P 500, influenced by factors like inflation and overall market price increases, creates a challenge for predictive models. A key recent event the model could not predict around was the COVID-19 pandemic. Even though the market was quick to return to pre-pandemic levels, the model could not successfully predict that sudden drop. Consequently, the inherent volatility and non-linearities in the financial markets contribute to the observed higher MAE and RMSE emphasizing the need for sophisticated modeling techniques that can adapt to the evolving nature of the data. 

While all models have high in-sample R2 values, suggesting good fits to the training data, the negative OSR2 values indicate potential overfitting and poor generalization to out-of-sample data. Based on our analysis, it became evident that there is a necessity to implement a time series model. The time series models we used, ARIMA and LSTM, generated predictions that were thorough, captured the larger upward trend with mean reversion, and interpreted all temporal deficiencies. Moreover, we also learned after the project that for tasks involving erratic, volatile, and temporal data, it is best to inculcate time series. The time series was trained on all data since 1985 but still had a lesser RMSE and MAE, despite all economic trends occurring on the same testing set – September 2020 to August 2023. It was also a good intellectual challenge to understand the intricacies of time series forecasting. 




**Overall Impact of Project:**

Through this project, we learned the difficulties that go into making financial predictions and forecasting the performance of the financial markets, and the different factors that cause these difficulties to arise. Initially, our models demonstrated promising results, indicated by high R2 values, which suggested a strong fit to the dataset. However, further evaluation revealed certain limitations. The presence of high MAE and RMSE values raised concerns about the precision of these models. We learned that the consequences of potential overfitting on models, especially in all the regression models, could be disastrous when exploring unseen data. We tried to reduce bias, but received a large variance. We were not confident in regression.
A factor for the poor performance in the linear regression models was the overarching effects of inflation over the period of study, and the unprecedented economic impact of the COVID-19 pandemic. These elements collectively led to us understanding that the dynamic and often unpredictable nature of the S&P 500 is a great challenge in supervised learning.  
The most effective models for predicting S&P 500 performance were the time series models, especially the ARIMA model, marginally superior to LSTM. Their superior adaptability to out-of-sample data and resilience to outliers and major disruptive events, such as the COVID-19 pandemic, marked it as the most accurate among all the models we tested. The ARIMA and LSTM models’ robustness and reliability in capturing the temporal dynamics of the S&P 500 was interesting for us to capture. It stands out as the most appropriate model for navigating the complexities and uncertainties of the stock market, thereby facilitating informed and prudent financial decisions. We got a lot more confident in our models after we implemented time series. We underestimated the value of adding multiple new modeling techniques to our project. We were proven wrong! 

**Negative Impact?**
In our group’s opinion, the possibility of negative impacts is not as high as we originally thought. One of the merits of an open and free market is the accessibility to public information. As long as advanced time series’ models are not used on sensitive and insider information to achieve ‘illegal’ forecasts, our project seems to be ethically sound, as all the information is public!

**Expanding Scope? **
We would love to explore this topic by investigating and implementing even more time series models to better understand this topic. Iterating through various hyperparameter values, doing cross validation, and other tried-and-tested supervised learning techniques to build an efficient S&P predictor would be an extremely interesting project. We learned a lot through this project anyway and would love to delve further into the world of time series.







**Appendix A:**

Data links:
S&P 500 Historical Data: https://www.investing.com/indices/us-spx-500-historical-data 
Unemployment Rate: Unemployment rate 
CPI: CPI Index
GDP: https://fred.stlouisfed.org/series/US
Consumer Expenditures: https://fred.stlouisfed.org/series/PCE 
Japan to US exchange rate: https://fred.stlouisfed.org/series/EXJPUS 
China to US exchange rate: https://fred.stlouisfed.org/series/EXCHUS 
Australia to US exchange rate: https://fred.stlouisfed.org/series/DEXUSAL 
UK to US exchange rate: https://fred.stlouisfed.org/series/DEXUSUK 
Interest Rates: https://fred.stlouisfed.org/series/DFF 
Gas Prices: https://fred.stlouisfed.org/series/APU000074714 
Federal Surplus: https://fred.stlouisfed.org/series/MTSDS133FMS 
Federal Employees: https://fred.stlouisfed.org/series/CES9091000001 
Mortgage Rate: https://fred.stlouisfed.org/series/MORTGAGE30US 
Personal Saving Rate: https://fred.stlouisfed.org/series/PSAVERT 
NYC humidity: https://www.visualcrossing.com/weather-history/New%20York%20city
NYC cloud cover: https://www.visualcrossing.com/weather-history/NewYork
US Hospital Revenue: https://fred.stlouisfed.org/series/REV622ALLEST144QSA
US Restaurant Revenue: https://fred.stlouisfed.org/series/MRTSSM7225USN



