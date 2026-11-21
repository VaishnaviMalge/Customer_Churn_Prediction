
## **Customer Churn Prediction: End-to-End ML Project with Streamlit App**

**Overview:**

This is an end-to-end Machine Learning project for predicting customer churn. It includes EDA, model building, and creating a UI for user interaction.

**Objective:**

The application predicts whether customers are likely to churn and helps business owners take immediate action.

**Key Responsibilities:**

- Performed data preprocessing and missing value treatment.
- Conducted EDA to understand and derive insights from data.
- Created visualizations to identify trends and patterns.
- Applied SMOTE technique to address imbalanced data.
- Experimented with different algorithms to identify the best-performing model.
- Performed hyperparameter tuning using RandomizedSearchCV.
- Developed a Gradient Boosting Classifier to predict customer churn.
- Evaluated model performance on unseen data.
- Saved the trained model using Pickle.
- Built an interactive Streamlit application for user input and churn prediction.

**Technologies:**

- Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn)
- Streamlit
- Pickle
- Jupyter Notebook

**Key Features:**

1\. **EDA and Preprocessing**

- Missing value treatment
- Data cleaning
- Feature engineering (categorical-to-numerical conversion)

2\. **Model Building**

- Experimented with different algorithms.
- Selected Gradient Boosting as the final model.
- Achieved 0.86 accuracy.

3\. **User Interface**

- Created an interactive application to improve user experience.
- Accepted user inputs and generated churn predictions.

**How to Run:**

1\. **Jupyter Notebook (Customer Churn.ipynb)**

- Contains the EDA and model-building code.
- Run all notebook cells to generate the Pickle file in the working directory.

2\. **app.py**

- Contains the code for the user interface.
- Load the file in VS Code.
- Install Streamlit if not already installed.
- Navigate to the folder containing app.py in the terminal.
- Run `streamlit run app.py`.
- The application opens locally in a web browser.

**Application Inputs:**

- Payment Method
- Monthly Charges
- Total Charges
- Tenure Bins

**Output:** Predicts whether the customer is likely to stay or churn.

*Source: Your uploaded PDF, pages 1–2.*&#x20;

README.pdf
