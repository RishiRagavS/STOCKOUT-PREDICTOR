# Stockout Predictor

A machine learning project to predict stockouts in retail using historical sales data. This system processes sales data, trains predictive models, and provides a FastAPI backend with MongoDB for real-time predictions.

## Features

- Data processing pipeline for retail sales data
- Machine learning models for stockout prediction (XGBoost, LightGBM)
- FastAPI backend with async MongoDB integration
- Inventory snapshot management
- RESTful API for predictions and data retrieval

## Data Requirements

The project requires the following raw datasets from the [M5 Forecasting - Accuracy competition](https://www.kaggle.com/c/m5-forecasting-accuracy) on Kaggle:

### Required Raw Datasets

1. **sales_train_evaluation.csv**
   - Contains historical sales data for 30,490 products across 10 stores
   - Time series data with daily sales quantities
   - Product hierarchy: item_id, dept_id, cat_id, store_id, state_id

2. **calendar.csv**
   - Calendar information including dates, weekdays, holidays, and special events
   - Maps day columns (d_1, d_2, etc.) to actual dates
   - Includes promotional events and SNAP purchase days

### Data Placement

Place these files in the `data/raw/` directory:
```
data/
  raw/
    calendar.csv
    sales_train_evaluation.csv
```

### Processed Data

The processing scripts will generate:
- `data/processed/sales_long.csv` - Melted sales data in long format
- `data/processed/features_final.csv` - Engineered features for modeling
- `data/processed/inventory_features.csv` - Inventory-related features

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stockout-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB_NAME=stockout_predictor
   ```

4. Start MongoDB (if running locally):
   ```bash
   mongod
   ```

## Usage

### 1. Process Raw Data

Run the data processing pipeline:
```bash
python scripts/process_data.py
```

This creates the processed datasets in `data/processed/`.

### 2. Generate Inventory Data

Create synthetic inventory snapshots:
```bash
python scripts/generate_inventory.py
```

### 3. Engineer Features

Extract features for machine learning:
```bash
python scripts/feature_engineering.py
```

### 4. Train Model

Train the predictive model:
```bash
python scripts/train_model.py
```

Trained models are saved in `saved_models/`.

### 5. Seed Database

Populate MongoDB with processed data:
```bash
python scripts/seed_mongo.py
```

### 6. Start Backend

Launch the FastAPI server:
```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 7. Explore Data (Optional)

Use the Jupyter notebook for data exploration:
```bash
jupyter notebook notebooks/01_exploration.py
```

## API Endpoints

- `GET /products` - List all products
- `GET /inventory/{sku_id}` - Get inventory snapshots for a product
- `GET /predictions/{sku_id}` - Get stockout predictions
- `POST /predictions` - Generate new predictions

## Project Structure

```
stockout-predictor/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── database.py     # MongoDB connection and setup
│   │   ├── main.py         # FastAPI app entry point
│   │   ├── models/         # Pydantic models
│   │   ├── routes/         # API endpoints
│   │   └── services/       # Business logic
├── data/
│   ├── raw/                # Raw datasets (download from Kaggle)
│   └── processed/          # Processed datasets
├── frontend/               # Frontend application
├── notebooks/              # Jupyter notebooks for exploration
├── saved_models/           # Trained ML models
├── scripts/                # Data processing and training scripts
├── requirements.txt        # Python dependencies
└── README.md
```

## Technologies Used

- **Backend**: FastAPI, Motor (async MongoDB driver)
- **Database**: MongoDB with time series collections
- **ML**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License."# STOCKOUT-PREDICTOR" 
