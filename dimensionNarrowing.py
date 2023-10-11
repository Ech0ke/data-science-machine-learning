import pandas as pd
import plotly.express as px
import numpy as np
import locale
import matplotlib.pyplot as plt


# Set the locale to use thousands separators
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Load the CSV data into object
data = pd.read_csv('global-data-on-sustainable-energy.csv')

print(data)
