# Flight Price Prediction – Exploratory Data Analysis (EDA)

## Project Overview

This project focuses on performing Exploratory Data Analysis (EDA) and feature engineering on a flight price dataset to understand key factors affecting ticket prices and to prepare the data for machine learning models that can predict flight fares. The dataset is similar to public “flight price prediction” datasets collected from online travel portals and contains common booking-related features such as airline, journey date, route, timings, stops, and price.

---

## Dataset Overview

The dataset used is `Flight_Price_Prediction_EDA.ipynb`, which contains details about flight bookings such as airline, journey date, source, destination, timings, total stops, additional information, and the final ticket price (target variable). Public versions of this dataset typically contain 10k+ rows and 10–12 columns, which is sufficient for both EDA and baseline predictive modeling.

**Initial Dataset Dimensions**

- Number of Rows: 10,683  
- Number of Columns: 11  

**Key Columns**

- `Airline`: Name of the airline operating the flight.  
- `Date_of_Journey`: Date of the flight journey.  
- `Source`: City of departure.  
- `Destination`: City of arrival.  
- `Route`: Complete route taken by the flight.  
- `Dep_Time`: Scheduled departure time.  
- `Arrival_Time`: Scheduled arrival time (sometimes with date).  
- `Duration`: Total duration of the journey.  
- `Total_Stops`: Number of stops during the journey.  
- `Additional_Info`: Miscellaneous information about the flight.  
- `Price`: Final ticket price (target variable).  

---

## Data Loading and Initial Inspection

1. **Data Loading**  
   - Loaded the dataset using `pandas` with `pd.read_excel('flight_price.xlsx')` into a DataFrame for further analysis and preprocessing.

2. **Quick Data Preview**  
   - Used `df.head()` and `df.tail()` to inspect the first and last few records to validate that the file was read correctly and that all columns were present.  

3. **Structural Information**  
   - Called `df.info()` to check data types, non-null counts, and memory usage.  
   - Identified that columns such as `Date_of_Journey`, `Dep_Time`, `Arrival_Time`, `Duration`, and `Total_Stops` were stored as `object` (string) type and needed transformation for numerical analysis and model training.[web:5]  

4. **Descriptive Statistics**  
   - Used `df.describe()` to obtain summary statistics for numerical variables, especially `Price`, including mean, standard deviation, minimum, maximum, and quartiles.  
   - These statistics helped to get an initial understanding of price distribution and potential outliers.

---

## Feature Engineering

The original dataset mainly contains raw string-based features that are not directly suitable for most classical machine learning algorithms. Feature engineering steps were performed to extract more informative and numerical features from date/time strings and categorical attributes.

### 1. Date Features from `Date_of_Journey`

- Split `Date_of_Journey` into three new columns:
  - `Day`  
  - `Month`  
  - `Year`  
- Converted these new columns from `object` to `int` type.  
- Dropped the original `Date_of_Journey` column after extraction.  
- **Rationale:** Flight prices often show strong patterns across days of month, months (seasonality, holidays, festivals), and even years; using separate integer features lets models learn these temporal effects more easily.

### 2. Arrival Time Features from `Arrival_Time`

- Cleaned `Arrival_Time` so that only the time portion remained (e.g., extracted `"01:10"` from entries like `"01:10 22 Mar"`).  
- Split the cleaned time into:
  - `Arrival_Hour`  
  - `Arrival_Min`  
- Converted both columns to `int` type and dropped the original `Arrival_Time` column.  
- **Rationale:** Arrival time can influence prices (for example, red-eye flights vs. peak-hour arrivals), and separating hour and minute enables better capturing of time-of-day patterns.[web:7][web:15]  

### 3. Departure Time Features from `Dep_Time`

- Split `Dep_Time` into:
  - `Dep_Hour`  
  - `Dep_Min`  
- Converted both new columns to `int` type and dropped the original `Dep_Time` column.  
- **Rationale:** Departure time is a strong driver of ticket prices; flights leaving at peak hours or highly demanded time windows are often more expensive.[web:7][web:15]  

### 4. Handling `Total_Stops`

- The `Total_Stops` column originally contained categorical strings like:
  - `"non-stop"`, `"1 stop"`, `"2 stops"`, etc.  
- Mapped these string categories to integers, for example:
  - `"non-stop"` → `0`  
  - `"1 stop"` → `1`  
  - `"2 stops"` → `2`  
  - and so on.  
- An initial mapping also handled `NaN` values (e.g., temporarily treating them as `1`), with the note that this decision may need to be revisited after analyzing the distribution and context of missing values.  
- **Rationale:** Machine learning algorithms typically require numerical encodings; mapping stops to a count makes intuitive sense and preserves the ordinal nature of the feature where more stops generally imply longer and sometimes cheaper or more expensive flights depending on the market. 

### 5. Dropping `Route`

- Dropped the `Route` column from the DataFrame.  
- **Rationale:** The `Route` column is often a string representation combining `Source`, `Destination`, and intermediate stops; much of its information is either redundant or can lead to a very high number of unique categories, which can cause unnecessary complexity without clear performance gains in the initial modeling phase.

### 6. Processing `Duration` (In Progress)

- The `Duration` column contains strings like `"2h 50m"`, `"5h"`, `"30m"`, etc.  
- Initial work involved splitting this field into hours and minutes by:
  - Parsing the duration text.  
  - Separating hour and minute components even when one of them is missing.  
- The next step is to either:
  - Create separate columns `Duration_Hour` and `Duration_Min`, or  
  - Create a single `Total_Duration_Minutes` feature by converting everything into minutes.  
- **Rationale:** Duration is one of the most important predictors of price, and converting it into a consistent numeric format significantly improves model interpretability and performance.

---

## Categorical Encoding Plan

After initial feature engineering, several columns remain categorical and will be encoded for modeling:

- `Airline`  
- `Source`  
- `Destination`  
- `Additional_Info`  

For these variables, **One-Hot Encoding** (also called dummy encoding) will be used to convert categories into binary indicator columns without imposing any artificial ordering.[web:5][web:16] This approach is widely used in flight price prediction projects and works well with tree-based and linear models.

---

## Model Preparation and EDA Tasks

With the engineered features in place, the project moves towards deeper EDA and model preparation:

- Visualize distributions of `Price`, `Total_Stops`, `Duration`, and time-based features to understand skewness and outliers.  
- Analyze relationships such as:
  - Price vs. Airline  
  - Price vs. Number of Stops  
  - Price vs. Journey_Month (seasonality effects)  
  - Price vs. Departure/Arrival hour  
- Split the dataset into training and testing sets (for example, 80/20) for unbiased model evaluation.
- Build baseline machine learning models such as:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting / XGBoost  
- Evaluate models using metrics like R², MAE, and RMSE to compare performance.

---

## Why Data Cleaning and Feature Engineering Are Necessary

Flight price data from real-world sources often contains missing values, inconsistent formats (for dates, times, and durations), duplicate rows, and noisy text-based categories. Without cleaning and transformation, even advanced models can underperform because they cannot interpret raw strings or handle irregular formats effectively.

In this project:

- Converting dates, times, durations, and stops into structured numeric features allows models to learn meaningful relationships, such as how prices depend on weekdays vs. weekends, night vs. day flights, and direct vs. multi-stop journeys.
- Removing redundant and high-cardinality columns (like raw routes) and encoding categorical features reduces overfitting risk and simplifies interpretation of feature importance.

---

## Future Scope

This project can be extended in several directions to build a more robust and production-ready flight price prediction system:

- **Complete Duration Engineering**  
  - Finalize parsing for all `Duration` patterns and create a single `Total_Duration_Minutes` feature.  
  - Analyze how duration interacts with number of stops and airlines in influencing price.  

- **Advanced Encoding and Feature Engineering**  
  - Apply One-Hot Encoding (OHE) to `Airline`, `Source`, `Destination`, and `Additional_Info`, and experiment with Target Encoding or Frequency Encoding for high-cardinality features if needed. 
  - Create new features such as:
    - `Journey_DayOfWeek` (weekday vs. weekend)  
    - `Is_Redeye_Flight` (late night or early morning departures)  
    - `Is_Peak_Month` (holiday or festival seasons)  

- **Machine Learning Modeling**  
  - Use the cleaned and encoded dataset to train and compare multiple regression models, including:
    - Random Forest, XGBoost, and other ensemble methods commonly used in similar flight fare projects.
  - Perform hyperparameter tuning using Grid Search or Randomized Search to optimize model performance.  

- **Model Interpretation and Deployment**  
  - Analyze feature importance to understand which factors most strongly affect flight prices (e.g., airline, number of stops, duration, or booking time). 
  - Package the best-performing model into an API or simple web interface that takes flight details as input and returns a predicted fare.  

**Before Cleaning and Feature Engineering**  
<img width="1439" height="241" alt="{2D7A5A09-9B4E-4B37-A96E-7F00076FFA1A}" src="https://github.com/user-attachments/assets/40e715bf-ea63-403d-aeb1-5306d6752088" />
 

**After Cleaning and Feature Engineering**  
<img width="1627" height="283" alt="{8D61CD9D-FCC7-47C7-892D-538965163143}" src="https://github.com/user-attachments/assets/40a6fd19-e038-4e68-8469-d51ea3e123ed" />



---

## Short Summary

- Raw flight price data is messy, with mixed types, inconsistent formats, and highly textual columns that cannot be used directly in machine learning models.
- Through systematic cleaning and feature engineering (date/time decomposition, duration parsing, encoding of stops and categories), the dataset becomes structured, numeric, and model-ready, allowing more accurate and reliable flight price prediction.

