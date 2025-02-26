"""
The main funciton to analyze the unit values of HS-coded commodity
"""
import time
from uv_preparation import extract_trade, clean_trade, detect_outliers
from uv_analysis import dip_test, fit_normal, fit_skewnormal, bootstrap_skewnormal_ci, find_gmm, fit_gmm, fit_gmm2,fit_gmm3
from uv_visualization import plot_histogram, plot_gmm_bic, plot_skn, plot_n

def print_section_header(title):
    print(f"\n{'-'*50}")
    print(f"** {title} **")
    print(f"{'-'*50}\n")

def print_time_info(step_name, start_time):
    elapsed_time = round(time.time() - start_time, 2)  # Calculate elapsed time in seconds
    print(f"{step_name} completed in {elapsed_time} seconds.")
    
def cmltrade_uv(subscription_key, code, year, direction):
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")
    #subscription_key = "b3ca7de8704f4cb4a1ae3cec68730595"
    #code = '740311'
    #year = '2018'
    #direction = 'm'
    
    # Step 1: Extract trade
    print_section_header("Data Extraction")
    start_time  = time.time()
    df = extract_trade(subscription_key, code, year, direction)
    print_time_info("Data extraction", start_time)
    
    # Step 2: clean trade
    print_section_header("Data Cleaning")
    start_time  = time.time()
    df1 = clean_trade(df)
    print_time_info("Data cleaning", start_time)
    
    # Step 3: Detect outliers
    print_section_header("Outlier Detection")
    start_time  = time.time()
    df2 = detect_outliers(df1, ['ln_uv','ln_netWgt'])
    print_time_info("Outlier detection", start_time)
    
    # Step 4: Histogram
    print_section_header("Histogram Plotting")
    start_time  = time.time()
    plot_histogram(df2['ln_uv'], code, year, direction, save_path="740311_m_2018.pdf", ax=None)
    print_time_info("Histogram plotting", start_time)
        
    # Step 5: Perform dip test
    print_section_header("Perform dip test")
    start_time  = time.time()
    is_unimodal, p_value = dip_test(df2['ln_uv'])
    print(f"- Dip test completed. The distribution is {'unimodal' if is_unimodal else 'multimodal'}.")
    print_time_info("Dip test", start_time)
    

    if is_unimodal:
        # Step 6a: Test for Skew-Normal or Normal distribution
        print_section_header("Test for Skew-Normal or Normal distribution")
        start_time  = time.time()
        aic_skn, bic_skn = fit_skewnormal(df2['ln_uv'])['aic_skewnorm'], fit_skewnormal(df2['ln_uv'])['bic_skewnorm']
        aic_n, bic_n = fit_normal(df2['ln_uv'])['aic_norm'], fit_normal(df2['ln_uv'])['bic_norm']
        
        if aic_skn < aic_n and bic_skn < bic_n:
            
            print("- Skew-Normal distribution fits better\n"
                  "based on AIC and BIC values.")
            print_time_info("Test for skew-normal or\n"
                            "normal distribution", start_time)
            
         
            print_section_header("Report statistics")
            start_time  = time.time()
            _= fit_skewnormal(df2['ln_uv'],pt= True)
            print_time_info("Report statistics", start_time)
            
           
            print_section_header("Bootstraping CI for median and mode values")
            start_time  = time.time()
            print("Bootstraping in 1000 times ...")
            ci_median, ci_mode = bootstrap_skewnormal_ci(df2['ln_uv'], n_bootstraps=1000)
            print_time_info("Bootstraping CI", start_time)
        
            print_section_header("Plotting")
            start_time  = time.time()
            plot_skn(df2['ln_uv'], code, year, direction, ci_median=ci_median, 
                     ci_mode=ci_mode, save_path='740400_m_2023_skn.pdf', ax=None)
            print_time_info("Plot", start_time)
            
        else:
            print("- Normal distribution fits better\n"
                  "based on AIC and BIC values.")
            print_time_info("Test for skew-normal or\n"
                            "normal distribution", start_time)
            
            
            print_section_header("Report statistics")
            start_time  = time.time()
            _ = fit_normal(df2,pt= True)
            print_time_info("Report statistics", start_time)
            
           
            print_section_header("Plotting")
            start_time  = time.time()
            plot_n(df2, code, year, direction, save_path=None, ax=None)
            print_time_info("Plot", start_time)
            
            
        
    else:
        # Step 6b: Fit GMM
        print_section_header("Fitting a 1D Gaussian Mixture Model (GMM) on unit values")
        start_time  = time.time()
       
        components, bic_values = find_gmm(df2[['ln_uv']], max_components=50, 
                            convergence_threshold=5, threshold=0.2)
        plot_gmm_bic(bic_values, max_components=50, save_path=None, ax=None)
        fit_gmm(df2, ['ln_uv'], components, code, year, direction, plot = True, save_path='391590_m_2023_gmm.pdf', ax=None)
        print_time_info("Fitting a 1D GMM on unit values" , start_time)
        
        print_section_header("Fitting a 2D GMM on unit values and trade volume")
        start_time  = time.time()
        components, bic_values = find_gmm(df2[['ln_uv','ln_netWgt']], max_components=50, convergence_threshold=5)
        plot_gmm_bic(bic_values, max_components=50, save_path=None, ax=None)
        fit_gmm2(df2[['ln_uv','ln_netWgt']], components, code, year, direction, plot = '2D', save_path=None, ax=None)
        print_time_info("Fitting a 2D GMM on unit values and trade volume" , start_time)
        
        print_section_header("Fitting a 3D GMM on unit values, trade volume, and GDP per capita")
        components, bic_values = find_gmm(df2[['ln_uv','ln_netWgt','ln_gdp']], max_components=50, convergence_threshold=5)
        plot_gmm_bic(bic_values, max_components=50, save_path=None, ax=None)
        fit_gmm3(df2[['ln_uv','ln_netWgt','ln_gdp']], components, code, year, direction, save_path=None)
        print_time_info("Fitting a 3D GMM on unit values, trade volume, and GDP per capita" , start_time)
      
        
    total_time = time.time() - zero_time
    print(f"Analysis complete. Total time: {total_time:.2f} seconds.")

