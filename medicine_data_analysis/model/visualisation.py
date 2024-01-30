import pandas as pd
import numpy as np
import calendar
import pandas as pd
import numpy as np
import chart_studio.plotly as py
from datetime import date
import plotly.graph_objs as go
import plotly.express as px
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt


import plotly.io as pio
pio.renderers.default='browser'


from plotly.offline import iplot
# For offline use
cf.go_offline()

df = pd.read_csv('data/medicine_data.csv')

print(df.head())

scatter = px.scatter(df, x="MRP_gvmt", y="MRP_market", color='Drug Code')
scatter.update_xaxes(type = 'category')
plot(scatter, auto_open=True)


scatter2 = px.scatter(df, x="Drug Code", y="difference", color='Drug Code')
scatter2.update_yaxes(categoryorder='category ascending')
plot(scatter2, auto_open=True)
