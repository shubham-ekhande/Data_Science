💰 Smart Pricing Optimization System

🚀 Project Overview



This project implements a machine learning–based pricing optimization engine that predicts product demand and simulates revenue across multiple price points to identify a revenue-maximizing price.



The system combines demand forecasting with price simulation to demonstrate how pricing decisions can be optimized using predictive modeling.



📊 Problem Statement



Retail businesses need to determine the optimal price that maximizes revenue while considering:



Customer demand



Freight (shipping) cost



Competitor pricing



Seasonality



Product characteristics



This project builds a data-driven system that predicts demand and evaluates revenue under different pricing scenarios.



🧠 Solution Approach

1️⃣ Data Preprocessing



Removed leakage features (e.g., total\_price).



Applied one-hot encoding to product categories.



Split dataset into training and testing sets.



2️⃣ Demand Prediction Model



Model Used: RandomForestRegressor



Evaluation Metric: R² ≈ 0.63



Target Variable: Quantity Sold (qty)



Features Included:



Unit price



Freight price



Competitor prices



Customer count



Seasonality indicators



Product attributes



3️⃣ Revenue Simulation Engine



For a given product scenario:



Simulate multiple price points



Predict demand for each price



Compute revenue = price × predicted demand



Select price that maximizes revenue



⚠️ Important Insight on Price Elasticity



During exploratory analysis, the dataset did not exhibit a strong downward trend between price and quantity (i.e., weak price elasticity).



As a result:



The model does not learn a strong negative relationship between price and demand.



Revenue optimization often pushes price upward.



This highlights a critical real-world insight:



Optimization systems are only as effective as the underlying economic relationships present in the data.



In dynamic pricing systems, true demand-price elasticity is essential for realistic optimization outcomes.



This project demonstrates both:



The implementation of a pricing optimization engine



The importance of validating economic assumptions in data-driven models



🖥 Streamlit Application



An interactive Streamlit app allows users to:



Adjust price, freight cost, customer count, and seasonality



Simulate demand predictions



Visualize revenue vs. price curve



Identify the optimal revenue-maximizing price



🛠 Tech Stack



Python



Pandas



NumPy



Scikit-learn



Matplotlib



Streamlit



Joblib



▶ How to Run Locally

pip install -r requirements.txt

streamlit run app.py



📌 Future Improvements



Model demand per product category



Introduce elasticity constraints in optimization



Experiment with gradient boosting (XGBoost)



Deploy to Streamlit Cloud



Incorporate time-series modeling



📈 Key Takeaway



This project demonstrates that:



Machine learning can support pricing decisions.



Revenue optimization requires accurate demand modeling.



Data quality and economic relationships are more important than model complexity.



This README makes you look like someone who understands:



ML



Business logic



Economic reasoning



Model limitations



That’s what interviewers care about.

