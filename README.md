# Car Resale Value Predictor API

A containerized, production-ready Machine Learning inference service built to predict the estimated selling price of used cars based on historical data. The application abstracts a complete data preprocessing and regression engine behind a fast, asynchronous FastAPI microservice framework, fully isolated using Docker.

---

## Project Architecture & Pipeline

The system is split into two isolated, functional directories to mirror professional machine learning engineering standards:

```text
Linear_Regression_Model_Project/
├── models/
│   ├── categorical_encoder.pkl   # One-Hot Encoder weights for string categorical values
│   ├── feature_scaler.pkl        # StandardScaler mapping numeric fields
│   └── linear_regression_model.pkl # Trained Scikit-Learn Linear Regression model matrix
├── src/
│   └── app.py                    # Production FastAPI server & real-time transformation script
├── Dockerfile                    # Multi-layer Docker automation script
├── requirements.txt              # Standardized library dependencies framework
└── .gitignore                    # Prevents local artifacts/datasets from leaking to cloud
```

### The Real-Time Feature Pipeline
When a request hits the public API endpoint, it automatically executes the exact data transformation lifecycle used during model training:
1. **Mathematical Transformation:** Translates the raw asset `Present_Price` using log-scaling (`np.log1p`) to mitigate skew.
2. **One-Hot Encoding:** Passes non-numeric metrics (`Fuel_Type`, `Seller_Type`, `Transmission`) through the pre-trained categorical encoder.
3. **Feature Scaling:** Standardizes numeric metrics (`Kms_Driven`, `Car_Age`, `log_present_price`) via the saved feature scaler.
4. **Target Restoration:** Feeds the formatted array into the Regression engine and restores the output into currency denomination using an inverse log transformation (`np.expm1`).

---

## Model Engineering & Production Learnings

Building a production-grade linear regression pipeline highlighted critical design patterns required to transition mathematical models into stable software systems:

* **Logarithmic Skew Mitigation (`np.log1p` & `np.expm1`):** Financial fields such as vehicle prices are highly right-skewed, which violates the normality assumptions of Ordinary Least Squares (OLS). Applying log transformations during training stabilized variance, while the corresponding inverse exponential mapping ensured runtime output units matched real currency denominations exactly.
* **Avoiding Training Data Leakage:** Scaling (`StandardScaler`) and Categorical Encoding configurations were calculated strictly on the training set splits. Serializing these configurations as independent artifacts (`.pkl` weights) ensures validation parameters remain completely isolated from real-time production features during inference.
* **State Synchronization:** The inference input matrices are structured to programmatically mirror the exact feature column sequence used during training. This strict spatial sequence enforcement prevents array mismatches when serving multi-dimensional inputs to the Scikit-Learn prediction engine.

---

## Key Milestones Completed

### Phase 1: Microservice Engineering
Developed a lightweight **FastAPI web application** (`src/app.py`) to serve model predictions dynamically. 
* Implemented strong input validation using **Pydantic** data structures (`CarFeatures`).
* Wrote structural path parameters (`../models/`) allowing the webserver to interact cleanly with trained serialization structures independently.

### Phase 2: Docker Containerization
Configured and compiled an immutable environment footprint using a custom **Dockerfile**:
* Based the project image footprint on `python:3.11-slim` to maintain minimal compute consumption.
* Enforced layer-caching structures to keep build runtimes low.
* Successfully resolved deep-system WSL environment variables to establish local runtime execution parity inside Windows architectures.

### Phase 3: Version Control Integration
Created robust configuration logic boundaries with strict `.gitignore` configurations, protecting data security while successfully validating, staging, and pushing the repository core onto **GitHub** source control.

### Phase 4: Cloud Deployment
Linked the code repository to serverless cloud infrastructure (**Render**), executing an automated CI/CD engine that reads the project’s custom `Dockerfile`, installs dependencies, and deploys it live to a secure public endpoint.

---