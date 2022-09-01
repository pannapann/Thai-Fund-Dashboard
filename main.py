import streamlit as st
import pandas as pd
import plotly.express as px
import finnomena_api as fin
from prophet import Prophet
try:
    from fbprophet import Prophet
except:
    pass
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Api References
# obj = fin.finnomenaAPI()
# df = obj.get_fund_list()
# obj.get_fund_info('ABG')
# obj.get_fund_price('ABG', time_range = '1M')


obj = fin.finnomenaAPI()
df = obj.get_fund_list()

fund_list = df['short_code']
period_list = ['1D', '7D', '1M', '3M', '6M', '1Y', '3Y', '5Y', '10Y', 'MAX']
page_type_list = ['Analyze', 'Forecast']

page_type = st.sidebar.selectbox("Select page type:",page_type_list)
fund = st.sidebar.selectbox("Select a fund:",fund_list)
period = st.sidebar.selectbox("Select time period:",period_list)

if page_type == 'Analyze':
    fund_info = obj.get_fund_info(fund)
    fund_price = obj.get_fund_price(fund, time_range = period)

    st.header(f"{fund} price over {period} period")
    st.subheader(f"Current price : {fund_info['current_price']}")
    fig = px.line(fund_price,
        x = "date", y = "price", title = fund)
    st.plotly_chart(fig)


    col1, col2, col3 = st.columns(3)


    with col1:
        st.write(f"security_name : {fund_info['security_name']}")
        st.write(f"morningstar_id : {fund_info['morningstar_id']}")
        st.write(f"feeder_fund : {fund_info['feeder_fund']}")
        st.write(f"nav_date : {fund_info['nav_date']}")


    with col2:
        st.write(f"total_amount : {fund_info['total_amount']}")
        st.write(f"d_change : {fund_info['d_change']}")
        st.write(f"purchase_fee : {fund_info['purchase_fee']}")
        st.write(f"management_fee : {fund_info['management_fee']}")


    with col3:
        st.write(f"redemption_fee : {fund_info['redemption_fee']}")
        st.write(f"switchIn_fee : {fund_info['switchIn_fee']}")
        st.write(f"switchOut_fee : {fund_info['switchOut_fee']}")
        st.write(f"total_expense_ratio : {fund_info['total_expense_ratio']}")
else:
    st.header(f"{fund} Price Forecasting")
    fund_price = obj.get_fund_price(fund, time_range='MAX')
    st.header('Select train : test ratio')
    traintest = st.slider('train:test:', min_value=0, max_value=100, step=5, value=90)
    train_ratio = traintest / 100
    test_ratio = (100 - traintest) / 100
    train_day = int(len(fund_price)*train_ratio)
    st.write(f'Training day : {train_day}')
    fund_price['date'] = pd.to_datetime(fund_price['date'])
    fund_price['ds'] =  fund_price['date']
    fund_price['y'] =  fund_price['price']
    train = fund_price[:train_day]
    test = fund_price[train_day:]

    train_samples = go.Scatter(x=train['ds'],
                               y=train['y'],
                               mode='lines',
                               name='Train')

    test_samples = go.Scatter(x=test['ds'],
                              y=test['y'],
                              mode='lines',
                              name='Test')

    layout = go.Layout(title={'text': 'Train/test split',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                       xaxis=dict(title='Year'),
                       yaxis=dict(title='AEP_MW'),
                       template='plotly_dark')

    data = [train_samples, test_samples]
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)


    prophet_model = Prophet()
    prophet_model.fit(train)

    # prediction
    test_forecast = prophet_model.predict(test)
    st.write(test_forecast.head())

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig = prophet_model.plot(test_forecast, ax=ax)
    st.pyplot(fig)

    fig2 = prophet_model.plot_components(test_forecast)
    st.pyplot(fig2)

    f2, ax2 = plt.subplots(1)
    f2.set_figheight(5)
    f2.set_figwidth(15)
    test = test.reset_index(drop=True)
    ax2.scatter(test['ds'], test['y'], color='r')
    fig3 = prophet_model.plot(test_forecast, ax=ax2)
    st.pyplot(fig3)

    # forecasting
    future = prophet_model.make_future_dataframe(periods=1000, freq='D')
    forecast_future = prophet_model.predict(future)
    st.write(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    fig4 = prophet_model.plot_components(forecast_future)
    st.pyplot(fig4)

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(20)
    fig5 = prophet_model.plot(forecast_future, ax=ax)
    st.pyplot(fig5)

    trace_open = go.Scatter(
        x=forecast_future["ds"],
        y=forecast_future["yhat"],
        mode='lines',
        name="Forecast"
    )

    trace_high = go.Scatter(
        x=forecast_future["ds"],
        y=forecast_future["yhat_upper"],
        mode='lines',
        fill="tonexty",
        line={"color": "#57b8ff"},
        name="Higher uncertainty interval"
    )

    trace_low = go.Scatter(
        x=forecast_future["ds"],
        y=forecast_future["yhat_lower"],
        mode='lines',
        fill="tonexty",
        line={"color": "#57b8ff"},
        name="Lower uncertainty interval"
    )

    trace_close = go.Scatter(
        x=fund_price["ds"],
        y=fund_price["y"],
        name="Data values"
    )

    data = [trace_open, trace_high, trace_low, trace_close]

    layout = go.Layout(title=f"{fund} Price Forecast", xaxis_rangeslider_visible=True)

    fig6 = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig6)




