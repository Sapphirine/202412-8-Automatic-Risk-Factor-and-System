# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:39:36 2023

@author: Wenbo Liu
"""


import os
import pandas as pd
import numpy as np
import warnings
from openpyxl import load_workbook
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set the font to SimHei for displaying Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Fix the issue where minus signs are displayed as blocks


'''
1. Read Excel Data----------------------------------------------------------------------------------------------------------------------
'''

'''
2.1 (Parameters to be modified 3) File Path Configuration: Extract data from the input Excel according to the template format. Copy the file path to the parameters below.
The `openpyxl` function in pandas reads the first worksheet by default.
'''
# Read the original Excel file 
write_file_path = r"F:\A 学习文件（重要资料）\A 研究生学习 - CU (哥伦比亚大学) - 统计\C2 - 研究生第二学期\EECS6893\Final\Risk_Factor_Model_and_Related_System\Risk_Factor_Model_Data\risk_factor_parity_model_output.xlsx"
read_file_path = r"F:\A 学习文件（重要资料）\A 研究生学习 - CU (哥伦比亚大学) - 统计\C2 - 研究生第二学期\EECS6893\Final\Risk_Factor_Model_and_Related_System\Risk_Factor_Model_Data\risk_factor_parity_model_input.xlsx"
# Get the names of all worksheets in the Excel file
sheet_names = pd.ExcelFile(read_file_path).sheet_names
# Create an empty dictionary to store DataFrames of different worksheets
dataframes = {}
# Read each worksheet and store it in the dictionary
for sheet_name in sheet_names:
    # Use pandas' read_excel function to read worksheet data
    df = pd.read_excel(read_file_path, sheet_name=sheet_name)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    # Store the DataFrame in the dictionary with the worksheet name as the key
    dataframes[sheet_name] = df

'''
2.2 Output Display and Content Formatting
'''
# Disable scientific notation and retain two decimal places
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Display all rows and columns without limits (default is 20 if not set)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Avoid warnings
warnings.filterwarnings('ignore')

# Check write permission
def check_write_permission(file_path):
    # Check if the directory has write permission
    directory = os.path.dirname(file_path)
    if os.access(directory, os.W_OK):
        print(f"Directory {directory} has write permission.")
    else:
        print(f"Directory {directory} does not have write permission.")

    # Check if the file exists and has write permission
    if os.path.exists(file_path):
        if os.access(file_path, os.W_OK):
            print(f"File {file_path} has write permission.")
        else:
            print(f"File {file_path} does not have write permission.")
    else:
        print(f"File {file_path} does not exist.")

# Import stored data
def set_index(close_price_ord):
    close_price_ord['Date'] = pd.to_datetime(close_price_ord['Date'], format="%Y%m%d")
    close_price_ord.set_index('Date', inplace=True)

'''
2.3 Result Writing Module
'''
# Use the ExcelWriter class from the pandas library to create an object `writer` for writing to an Excel file.
# Use the parameter engine='openpyxl' to specify using openpyxl as the underlying engine for Excel writing operations.

# Write the result
writer = pd.ExcelWriter(write_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace')
check_write_permission(write_file_path)
# Load the existing Excel file
try:
    book = load_workbook(write_file_path)
except FileNotFoundError:
    # If the file does not exist, an exception will be raised. However, no worries, ExcelWriter will automatically create a new file.
    book = None






'''
2. Macro Factor Model Equations---------------------------------------------------------------------------------------------------------
'''
# Equation 1: Calculate the variance of risk contributions for each factor
def risk_budget_objective(weights, cov, beta, gama):
    weights = np.array(weights)  # weights is a one-dimensional array
    # (1) Calculate intermediate values
    beta_new = np.dot(weights, beta)  # Intermediate value for beta * weight
    gama_new = np.dot(weights, gama)  # Intermediate value for gama * weight
    result = np.concatenate((beta_new, gama_new))  # New matrix (to replace original weight values)
    # (2) Calculate factor risk contributions
    sigma = np.sqrt(np.dot(result, np.dot(cov, result)))  # Portfolio standard deviation
    MRC = np.dot(cov, result) / sigma  # Marginal Risk Contribution (MRC)
    FRC = result * MRC  # Factor Risk Contribution (FRC) for each factor
    # (3) Derive the optimization condition
    delta_FRC = [sum((i - FRC)**2) for i in FRC]  # Variance of risk contributions between factors
    return sum(delta_FRC)

# Equation 2: Factors cannot be shorted, and the sum of weights equals 1
def total_weight_constraint(x):
    return np.sum(x) - 1.0

# Equation 3: Convert decimals to percentages
def percent_formatter(x, pos):
    return f'{x*100:.0f}%'





'''
3. Macro Factor Exposure and Simple Factor Risk Parity Calculation Module--------------------------------------------------------------
'''
def risk_factor_exposure(df1, df2, sheet_name):
    
    '''
    3.1 Convert asset net value data to daily return data
    '''
    # (0) Remove the first row
    df1 = df1.iloc[1:]
    # (1) Get all columns except the first column (Date column), which contain stock data
    stock_columns = df1.columns[1:]
    # (2) Calculate daily returns for each stock and store them in a new DataFrame
    ret = pd.DataFrame()
    # (3) Copy the Date column to the new DataFrame
    ret['Date'] = df1['Date']  
    # (4) Calculate daily returns for each stock and add them to the returns DataFrame
    for stock_column in stock_columns:
        col_name = f'{stock_column}'
        ret[col_name] = df1[stock_column].pct_change()  # Use pct_change() to calculate daily returns
    # (5) Drop rows containing NaN values (first day's data), resulting in daily return data
    ret = ret.dropna()
    # print(ret.head(5))
    # (6) Calculate correlations between assets
    ret_numeric = ret.drop(columns=['Date'], errors='ignore').select_dtypes(include=[float, int])
    ret_corr = ret_numeric.corr()
    # Uncomment the following lines to save correlation results to an Excel file
    # correlation_row = ret_corr.stack().to_frame().T
    # print(correlation_row)
    # correlation_row.to_excel(writer, sheet_name=str(sheet_name), index=True)

    
    
    
    '''
    3.2 Clean factor return data and merge asset return data with factor return data
    '''
    # (0) Remove the first row
    df2 = df2.iloc[1:]
    # (1) Drop rows containing NaN values
    factor = df2.dropna()
    # (2) Merge the two datasets
    merged_data = pd.merge(ret, factor, on='Date', how='outer')
    merged_data = merged_data.dropna()
    # (3) Set the "Date" column as the index
    merged_data.set_index('Date', inplace=True)
    # (4) Exclude data prior to 2023
    cutoff_date = pd.to_datetime('2023-01-01')
    merged_data = merged_data[merged_data.index.to_numpy() >= cutoff_date]
    
    '''
    3.3 Calculate the factor exposure matrix
    '''
    # (1) Extract factor and asset columns
    factor_columns = merged_data.columns[merged_data.columns.str.contains('Factor')]
    asset_columns = merged_data.columns[merged_data.columns.str.contains('Asset')]
    # (2) Initialize the factor exposure matrix
    factor_exposure_matrix = pd.DataFrame(index=asset_columns, columns=factor_columns)
    # (3) Initialize the residual matrix
    residual_matrix = pd.DataFrame(index=merged_data.index, columns=asset_columns)
    # (4) Calculate the factor exposure matrix
    for asset in asset_columns:
        X = merged_data[factor_columns]
        y = merged_data[asset]
        # (4.1) Use linear regression to fit the relationship between asset returns and factors
        model = LinearRegression()
        model.fit(X, y)
        # (4.2) Extract regression coefficients as factor exposures
        factor_exposure_matrix.loc[asset] = model.coef_
        # (4.3) Calculate residuals and store them
        residual_matrix[asset] = y - model.predict(X)
    # (5) Write the asset correlation table and factor exposure matrix to Excel
    factor_exposure_matrix = factor_exposure_matrix.astype(float)  # Factor exposure matrix
    factor_exposure_matrix.to_excel(writer, sheet_name=f'{sheet_name}_FactorExposure', index=True)
    ret_corr.to_excel(writer, sheet_name=f'{sheet_name}_AssetCorrelation', index=True)  # Asset correlation table
    
    '''
    3.5 Use seaborn heatmaps to create visualizations
    '''
    # (1) Asset correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(ret_corr, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Asset Correlation Changes (Calculated with Full Data)')
    plt.show()
    # (2) Factor exposure matrix heatmap (calculated with full data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(factor_exposure_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Factor Exposure Matrix (Calculated with Full Data)')
    plt.show()






'''
5. Implementation------------------------------------------------------------------------------------------------------------------------
Note: The last number represents `frequency`, which indicates how far back in time to calculate the covariance matrix.
'''
risk_factor_exposure(dataframes["Asset"],dataframes["Factor"], "Table")

writer.close()