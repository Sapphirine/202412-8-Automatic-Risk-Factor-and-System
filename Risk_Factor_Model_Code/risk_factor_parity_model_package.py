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
3. Factor Risk Parity Strategy Calculation Module---------------------------------------------------------------------------------------
'''
def factor_risk_parity_model(df1, df2, sheet_name, frequency):
    
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
    # (6) Calculate correlations between assets
    ret_numeric = ret.drop(columns=['Date'], errors='ignore').select_dtypes(include=[float, int])
    ret_corr = ret_numeric.corr()

    '''
    3.2 Clean factor return data and merge asset return data with factor return data
    '''
    # (0) Remove the first row
    df2 = df2.iloc[1:]
    # (1) Drop rows containing NaN values
    factor = df2.dropna()
    factor = pd.DataFrame(factor, columns=df2.columns)
    # (2) Merge the two datasets
    merged_data = pd.merge(ret, factor, on='Date', how='outer')
    merged_data = merged_data.dropna()
    # (3) Set the "Date" column as the index
    merged_data.set_index('Date', inplace=True)
    
    
    '''
    3. Initialize data arrays
    '''
    # Initialize an array for monthly returns (y-axis)
    ret_sum = []
    # Initialize an array for months (x-axis)
    month_x = []
    # Determine the number of major asset classes
    k = ret.shape[1] - 1
    # Initialize 2D arrays for each major asset class
    asset_nv = [[] for _ in range(k)]
    # Initialize 2D arrays for asset allocation ratios
    percen_alloc = [[] for _ in range(k)]



    '''
    3.3 Macro Factor Risk Parity Strategy - Calculate asset weights
    '''
    # Convert the Date column to datetime type
    merged_data.reset_index(inplace=True)
    # Group data by month using `groupby`
    monthly_groups = merged_data.groupby(merged_data['Date'].dt.to_period('M'))
    
    for month, data in monthly_groups:
        # Display data for each month for debugging
        # print(f"Month: {month.to_timestamp().strftime('%Y-%m')}")

        # (1) Retrieve data for the previous n months
        previous_data = pd.DataFrame()  # Create an empty DataFrame to store data for the previous n months
        for i in range(frequency - 1, -1, -1):
            previous_month = month - i - 1
            # Skip if data for the previous month does not exist
            if previous_month not in monthly_groups.groups:
                break
            previous_data = pd.concat(
                [previous_data, monthly_groups.get_group(previous_month)])
        
        # Skip calculation if data for the previous n months is empty
        if previous_data.empty:
            continue
        else:
            '''
            3.3.1 Calculate the factor exposure matrix
            '''
            # (1) Extract factor and asset columns
            factor_columns = previous_data.columns[previous_data.columns.str.contains('Factor')]
            asset_columns = previous_data.columns[previous_data.columns.str.contains('Asset')]
            
            # (2) Initialize the factor exposure matrix
            factor_exposure_matrix = pd.DataFrame(index=asset_columns, columns=factor_columns)
            # (3) Initialize the residual matrix
            residual_matrix = pd.DataFrame(index=previous_data.index, columns=asset_columns)
            
            # (4) Calculate the factor exposure matrix
            for asset in asset_columns:
                X = previous_data[factor_columns]
                y = previous_data[asset]
                # (4.1) Use linear regression to fit the relationship between asset returns and factors
                model = LinearRegression()
                model.fit(X, y)
                # (4.2) Extract regression coefficients as factor exposures
                factor_exposure_matrix.loc[asset] = model.coef_
                # (4.3) Calculate residuals and store them
                residual_matrix[asset] = y - model.predict(X)
            factor_exposure_matrix = factor_exposure_matrix.astype(float)
            
            # (5) Retrieve data for factors and assets over the past N months
            factor_data = previous_data.filter(like='Factor')
            factor_data = pd.concat([previous_data['Date'], factor_data], axis=1)
            asset_data = previous_data.filter(like='Asset')
            asset_data = pd.concat([previous_data['Date'], asset_data], axis=1)
            asset_data_original = data.filter(like='Asset')
            asset_data_original = pd.concat([previous_data['Date'], asset_data_original], axis=1)
            #print(asset_data)
            
            '''
            3.3.2 Calculate proportions for each asset in the Macro Factor Risk Parity model
            '''  
            # (1) Calculate the covariance matrix of assets
            cov_asset = asset_data[asset_columns].cov()
            
            # (2) Calculate the covariance matrix of factors
            factor_data['Date'] = pd.to_datetime(factor_data['Date'])
            factor_data.set_index('Date', inplace=True)
            factor_data = factor_data[factor_columns].astype(float)
            factor_corr = factor_data.corr() 
            cov_factor = factor_data.cov()
            
            # (3) Calculate the residuals matrix
            residual_sum_per_asset = residual_matrix.sum(axis=0)
            residual_exposure_matrix = residual_sum_per_asset.to_frame(name='Residual_Sum')
            residual_exposure_matrix = np.diag(residual_exposure_matrix['Residual_Sum'])  # Obtain the diagonal matrix of residuals
            
            # (4) Solve for risk parity proportions
            # (4.0) Compute the new sigma matrix
            cov_mon = np.array(cov_factor)
            K = cov_mon.shape[1]
            N = residual_exposure_matrix.shape[1]
            combined_matrix = np.zeros((K + N, K + N))
            combined_matrix[:K, :K] = cov_mon
            combined_matrix[K:, K:] = residual_exposure_matrix
            
            # (4.1) Define the initial guess for weights, with the sum of weights equal to 1
            x0 = np.ones(cov_asset.shape[0]) / cov_asset.shape[0]

            # (4.2) Define boundary conditions
            bnds = tuple((0, None) for _ in x0)
            # (4.3) Define constraints where the return value equals 0
            cons = ({'type': 'eq', 'fun': total_weight_constraint})
            # (4.4) Perform multiple iterations to find the optimal solution
            options = {'disp': False, 'maxiter': 10000, 'ftol': 1e-20} 
            # (4.5) Solve the optimization problem to minimize variance and obtain the weights
            solution = minimize(risk_budget_objective, x0, args=(combined_matrix, factor_exposure_matrix, residual_exposure_matrix), 
                                bounds=bnds, constraints=cons, options=options)

            '''
            3.3.3 Calculate returns and allocation ratios for the Macro Factor Risk Parity model
            '''  
            # (1) Calculate returns for each asset over these months
            # (1.1) Select the asset columns, assuming daily return data starts from the second column
            asset_returns = asset_data_original.iloc[:, 1:]

            # (1.2) Calculate daily returns for each asset and cumulative returns for the month
            # Ensure consistent cumulative net value calculation
            cumulative_returns = (1 + asset_returns).cumprod()  # Calculate cumulative net value
            cumulative_returns_month = cumulative_returns.iloc[-1] - 1  # Monthly return

            # Replace cumulative returns in portfolio and assets
            cumuret = cumulative_returns_month.values.reshape(1, -1)[0]
            # (1.3) Calculate portfolio return for the month
            retmonth = np.dot(solution.x, cumuret)
            # (1.4) Append the portfolio return for the month
            ret_sum.append(retmonth)
            # (1.5) Append returns for each asset for the month
            for i in range(k):
                asset_nv[i].append(cumuret[i])                
            # (2) Append the current month to the x-axis array
            mon = month.to_timestamp().strftime('%Y-%m')
            month_x.append(mon)
            # (3) Append allocation ratios for the month
            for i in range(k):
                percen_alloc[i].append(solution.x[i])
       
    # 3. Calculate cumulative portfolio returns for each month and store them in ret_sum
    for i in range(0, len(ret_sum)):
        ret_sum[i] = ret_sum[i] + 1
    for i in range(1, len(ret_sum)):
        ret_sum[i] = ret_sum[i - 1] * ret_sum[i]
    
    # 4. Calculate cumulative returns for each asset for each month and store them in asset_nv
    for i in range(k):
        for j in range(0, len(month_x)):
            asset_nv[i][j] = asset_nv[i][j] + 1
        for j in range(1, len(month_x)):
            asset_nv[i][j] = asset_nv[i][j - 1] * asset_nv[i][j]
    asset_nv = [[nv / asset[0] for nv in asset] for asset in asset_nv] 
    
    # 5. Create a DataFrame for asset allocation proportions
    df_percen_alloc = pd.DataFrame(percen_alloc).T
    df_month_x = pd.DataFrame(month_x)
    
    # 6. Merge monthly dates with corresponding data
    merged_df_percen_alloc = pd.concat([df_month_x, df_percen_alloc], axis=1)
    merged_df_percen_alloc.rename(columns=dict(
        zip(merged_df_percen_alloc.columns, ret.columns)), inplace=True)
    merged_df_percen_alloc.columns.values[0] = 'Date'
    
    # 7. Set the Date column as the index
    merged_df_percen_alloc['Date'] = pd.to_datetime(
        merged_df_percen_alloc['Date'])
    merged_df_percen_alloc.set_index('Date', inplace=True)
    
    # 8. Use the resample method to resample daily (D), forward-filling each month's values
    merged_df_percen_alloc_resampled = merged_df_percen_alloc.resample('D').ffill()
    
    # 9. Write the DataFrame to Excel
    merged_df_percen_alloc_resampled.to_excel(writer, sheet_name=str(sheet_name), index=True)

    # 11. Plot the Factor-Risk-Parity Portfolio and asset return curves
    # (1) Set the figure size
     # Adjust the width and height as needed

    # (2) Plot the asset and factor portfolio return curves
    asset_columns = ret.columns[1:] 
    for i, col_name in enumerate(asset_columns):
        plt.plot(month_x, asset_nv[i], label=col_name)
    plt.plot(month_x, ret_sum, label='Factor-Risk-Parity Portfolio')

    # (3) Set the x-axis and y-axis labels
    plt.xlabel('Month')
    plt.ylabel('Net Value of Assets and Portfolio')
    plt.title('Net Value of Assets and Portfolio Over Time')

    # (4) Rotate x-axis labels for better clarity
    plt.xticks(rotation=45)

    # (5) Add a legend
    plt.legend(loc='upper left', fontsize='large')

    # (6) Display the plot
    plt.show()




'''
4. Implementation------------------------------------------------------------------------------------------------------------------------
Note: The last number represents `frequency`, which indicates how far back in time to calculate the covariance matrix.
'''
factor_risk_parity_model(dataframes["Asset"],dataframes["Factor"], "Factor_Risk_Parity", 1)

writer.close()