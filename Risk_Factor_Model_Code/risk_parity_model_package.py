# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:21:14 2023

@author: Wenbo Liu

"""

import os
import pandas as pd
import numpy as np
import warnings
from openpyxl import load_workbook
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
    close_price_ord['Date'] = pd.to_datetime(
        close_price_ord['Date'], format="%Y%m%d")
    close_price_ord.set_index('Date', inplace=True)

'''
2.3 Result Writing Module
'''
# Use the ExcelWriter class from the pandas library to create an object `writer` for writing to an Excel file.
# Use the parameter engine='openpyxl' to specify using openpyxl as the underlying engine for Excel writing operations.

# Write the result
writer = pd.ExcelWriter(write_file_path, engine='openpyxl',
                        mode='a', if_sheet_exists='replace')
check_write_permission(write_file_path)
# Load the existing Excel file
try:
    book = load_workbook(write_file_path)
except FileNotFoundError:
    # If the file does not exist, an exception will be raised. However, no worries, ExcelWriter will automatically create a new file.
    book = None



'''
2. Risk Parity Strategy Equations-------------------------------------------------------------------------------------------------------
'''
# Equation 1: Calculate the variance of risk contributions for each asset


def risk_budget_objective(weights, cov):
    weights = np.array(weights)  # weights is a one-dimensional array
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))  # Calculate portfolio standard deviation
    # sigma = np.sqrt(weights@cov@weights)
    MRC = np.dot(cov, weights)/sigma  # Calculate Marginal Risk Contribution (MRC)
    TRC = weights * MRC  # Calculate the Total Risk Contribution (TRC) of each asset to the portfolio
    delta_TRC = [sum((i - TRC)**2) for i in TRC]  # Calculate the variance of risk contributions between assets
    return sum(delta_TRC)

# Equation 2: Assets cannot be shorted, and the sum of weights equals 1


def total_weight_constraint(x):
    return np.sum(x) - 1.0

# Equation 3: Convert decimals to percentages


def percent_formatter(x, pos):
    return f'{x*100:.0f}%'


'''
3. Risk Parity Strategy Calculation Module------------------------------------------------------------------------------------------------
'''


def Asset_Allocation_risk_parity(df, sheet_name, frequency):
    '''
    3.1 Convert net value data to daily return data
    '''
    # (1) Get all columns except the first column (Date column), which contain stock data
    # Remove the first row of data
    df = df.iloc[1:]

    # print(df.head(5))
    stock_columns = df.columns[1:]
    # (2) Calculate daily returns for each stock and store them in a new DataFrame
    ret = pd.DataFrame()
    # (3) Copy the Date column to the new DataFrame
    ret['Date'] = df['Date']
    # (4) Calculate daily returns for each stock and add them to the new DataFrame
    for stock_column in stock_columns:
        col_name = f'{stock_column}'
        ret[col_name] = df[stock_column].pct_change()  # Use pct_change() to calculate daily returns
    # (5) Drop rows with NaN values (first day's data), resulting in daily return data
    ret = ret.dropna()
    # (6) Calculate correlations between assets
    ret_numeric = ret.drop(columns=['Date'], errors='ignore').select_dtypes(include=[float, int])
    ret_corr = ret_numeric.corr()

    '''
    3.2 Initialize various data arrays
    '''
    # (1) Initialize an array for monthly returns
    ret_sum = []
    # (2) Initialize an array for months (x-axis)
    month_x = []
    # (3) Determine the number of major asset classes
    k = ret.shape[1] - 1
    # (4) Initialize 2D arrays for each major asset class
    asset_nv = [[] for _ in range(k)]
    # (5) Initialize 2D arrays for asset allocation ratios
    percen_alloc = [[] for _ in range(k)]

    '''
    3.3 Calculate asset weights using risk parity strategy
    '''
    # 1. Convert the Date column to datetime type and group data by month
    ret['Date'] = pd.to_datetime(ret['Date'])
    monthly_groups = ret.groupby(ret['Date'].dt.to_period('M'))
    # 2. Calculate asset weights based on data from the previous n months
    for month, data in monthly_groups:

        # Display data for each month for debugging
        #print(f"Month: {month.to_timestamp().strftime('%Y-%m')}")

        # (1) Retrieve data from the previous n months
        previous_data = pd.DataFrame()  # Create an empty DataFrame to store data from the previous n months
        for i in range(frequency - 1, -1, -1):
            previous_month = month - i - 1
            # If data from the previous n months does not exist, skip to the current month's data
            if previous_month not in monthly_groups.groups:
                break
            previous_data = pd.concat(
                [previous_data, monthly_groups.get_group(previous_month)])

        # (1.1) Skip calculation if data from the previous n months is empty
        if previous_data.empty:
            continue
        # (1.2) Proceed with calculation if data is available
        else:
            # (2) Calculate covariance matrix using data from the previous n months
            previous_data = data.iloc[:, 1:] 
            R_cov = previous_data.cov()
            cov_mon = np.array(R_cov)

            # (3) Calculate asset weights for the current month
            # (3.1) Define initial guess with weights summing to 1
            x0 = np.ones(cov_mon.shape[0]) / cov_mon.shape[0]
            # (3.2) Define boundary conditions
            bnds = tuple((0, None) for x in x0)
            # (3.3) Define constraints where the return value equals 0
            cons = ({'type': 'eq', 'fun': total_weight_constraint})
            # (3.4) Perform multiple iterations to find the optimal solution (Newton's method may require more iterations)
            options = {'disp': False, 'maxiter': 10000, 'ftol': 1e-20}
            # (3.5) Solve the optimization problem to find weights minimizing variance
            solution = minimize(risk_budget_objective, x0, args=(cov_mon), bounds=bnds, constraints=cons, options=options)

            # (4) Calculate monthly returns for each asset
            asset_returns = data.iloc[:, 1:] 

            # Ensure consistent cumulative net value calculation
            cumulative_returns = (1 + asset_returns).cumprod()  # Calculate cumulative net value
            cumulative_returns_month = cumulative_returns.iloc[-1] - 1  # Monthly return

            # Replace cumulative returns in portfolio and assets
            cumuret = cumulative_returns_month.values.reshape(1, -1)[0]

            # (5) Calculate portfolio return for the month
            retmonth = np.dot(solution.x, cumuret)

            # (6) Append portfolio return for the month
            ret_sum.append(retmonth)

            # (7) Append monthly returns for each asset
            for i in range(k):
                asset_nv[i].append(cumuret[i])

            # (8) Append month to x-axis array
            mon = month.to_timestamp().strftime('%Y-%m')
            month_x.append(mon)

            # (9) Append asset allocation ratios for the month
            for i in range(k):
                percen_alloc[i].append(solution.x[i])

    # 3. Calculate cumulative portfolio returns for each month
    for i in range(0, len(ret_sum)):
        ret_sum[i] = ret_sum[i] + 1
    for i in range(1, len(ret_sum)):
        ret_sum[i] = ret_sum[i-1] * ret_sum[i]

    # 4. Calculate cumulative returns for each asset and store them in asset_nv
    for i in range(k):
        for j in range(0, len(month_x)):
            asset_nv[i][j] = asset_nv[i][j] + 1
        for j in range(1, len(month_x)):
            asset_nv[i][j] = asset_nv[i][j-1] * asset_nv[i][j]
    asset_nv = [[nv / asset[0] for nv in asset] for asset in asset_nv] 

    # 5. Create a DataFrame for asset allocation ratios
    df_percen_alloc = pd.DataFrame(percen_alloc).T
    df_month_x = pd.DataFrame(month_x)

    # 6. Merge date and corresponding data
    merged_df_percen_alloc = pd.concat([df_month_x, df_percen_alloc], axis=1)
    merged_df_percen_alloc.rename(columns=dict(
        zip(merged_df_percen_alloc.columns, ret.columns)), inplace=True)
    merged_df_percen_alloc.columns.values[0] = 'Date'

    # 7. Set the Date column as the index
    merged_df_percen_alloc['Date'] = pd.to_datetime(
        merged_df_percen_alloc['Date'])
    merged_df_percen_alloc.set_index('Date', inplace=True)

    # 8. Resample data by day (D) and forward-fill monthly values
    merged_df_percen_alloc_resampled = merged_df_percen_alloc.resample(
        'D').ffill()

    # 9. Write the DataFrame to Excel
    merged_df_percen_alloc_resampled.to_excel(
        writer, sheet_name=str(sheet_name), index=True)

    # 11. Plot the return curves for the Risk-Parity Portfolio and assets
    

    # Plot asset and portfolio return curves
    asset_columns = ret.columns[1:] 
    for i, col_name in enumerate(asset_columns):
        plt.plot(month_x, asset_nv[i], label=col_name)
    plt.plot(month_x, ret_sum, label='Risk-Parity Portfolio')

    # Set axes labels
    plt.xlabel('Month')
    plt.ylabel('Net Value of Assets and Portfolio')
    plt.title('Net Value of Assets and Portfolio Over Time')

    # Make x-axis labels clearer
    plt.xticks(rotation=45)

    # Add legend
    plt.legend(loc='upper left', fontsize='large')

    # Show the plot
    plt.show()



'''
4. Implementation------------------------------------------------------------------------------------------------------------------------
Note: The last number represents `frequency`, which indicates how far back in time to calculate the covariance matrix.
'''
Asset_Allocation_risk_parity(dataframes["Asset"], "Risk_Parity", 1)

writer.close()
