# UVPicker

## Introduction
**UVPicker** is a Python package designed to analyze the **unit value distribution** of **HS-coded commodities** in the international trade.  
The core function, **`main.py`**, is supported by three submodules:
- **`uv_preparation`** – Data extraction and preprocessing  
- **`uv_analysis`** – Statistical analysis and modeling  
- **`uv_visualization`** – Graphical representation of unit value distributions  

### **Why UVPicker?**
When analyzing the unit values of a traded product across countries, several key questions arise:
- Should we simply collect all unit values and calculate the **mean**?  
- How can we determine whether unit values are **normally distributed** before relying on the mean?  
- What if the dataset includes multiple **sub-products**, each with a distinct unit value mean?  

**UVPicker** is designed to answer these questions by employing statistical techniques to identify **unimodal vs. multimodal distributions** and provide **robust trade value insights** (see illustration below).  

![Figure Description](readme.svg)

---

## **Main Steps**
UVPicker follows a structured workflow to ensure accurate unit value analysis:

1. **Data Extraction**  
   - Retrieves **bilateral trade data** using the [UN Comtrade API](https://github.com/uncomtrade/comtradeapicall)  
   - Requires inputs: **HS code, year, and trade direction** (imports or exports)
     
2. **Data cleaning**  
   - Dropping the sum-up trade in the original 'plus' breakdown mode (eg. trade of country A to the world)
     
3. **Outlier Detection**  
   - Applies the **[modified z-score method](https://books.google.com/books?hl=en&lr=&id=FuuiEAAAQBAJ&oi=fnd&pg=PP1&dq=modified+z+score+MAD&ots=SFP_S9VOSl&sig=KJf70cPJ5eE7Ojn9I5smb7BpqgI)** to filter extreme values  

4. **Unimodality Test**  
   - Uses **[Hartigan's Dip Test](https://projecteuclid.org/journals/annals-of-statistics/volume-13/issue-1/The-Dip-Test-of-Unimodality/10.1214/aos/1176346577.full)** to check if the unit value distribution is unimodal  

5. **Distribution Fitting**  
   - Fits a **unimodal distribution** (Normal, Skew-Normal, etc.) or a **multimodal distribution** (Gaussian Mixture Model, GMM)  

6. **Statistical Reporting & Visualization**  
   - Computes key statistics:  
     - **Mean, median, mode**  
     - **Proportions of each peak**  
     - **95% confidence intervals**  
   - Generates **plots** to illustrate unit value distributions  

---
