import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging
import os
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.info("python-dotenv not installed, skipping .env file loading")

# Check for environment variable
dagshub_token = os.getenv("CASPTONE_TEST_DAGSHUB")
if dagshub_token:
    # Import MLflow only if token is available
    try:
        import mlflow
        from mlflow.models import infer_signature
        from mlflow.types.schema import Schema, ColSpec
        logging.info("Using DagsHub token for authentication")
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri('https://dagshub.com/sc/capstone_repo.mlflow')
    except ImportError:
        logging.warning("MLflow not available, continuing without tracking")
        dagshub_token = None
else:
    logging.warning("CASPTONE_TEST_DAGSHUB environment variable not set")

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        # Convert DataFrame to numpy array to avoid feature names warning
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train_array, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load and prepare data
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1]  # Keep as DataFrame
        y_train = train_data.iloc[:, -1].values
        
        # Train model
        clf = train_model(X_train, y_train)
        
        # Save model locally
        model_path = os.path.join('models', 'model.pkl')
        save_model(clf, model_path)
        
        # Create experiment_info.json regardless of MLflow success
        os.makedirs('reports', exist_ok=True)
        model_info = {'run_id': 'local-run', 'model_path': model_path}
        with open('reports/experiment_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        
        # Skip MLflow if token is not available
        if not dagshub_token:
            print("Model trained and saved locally without MLflow tracking")
            return
        
        # MLflow operations - only run if token is available
        try:
            # Create explicit schema with float64 types for all columns
            input_schema = Schema([
                ColSpec(type="double", name=str(i)) for i in range(X_train.shape[1])
            ])
            
            # Generate model signature with explicit schema
            signature = infer_signature(X_train, clf.predict(X_train), input_schema=input_schema)
            
            # Start MLflow run
            mlflow.set_experiment("sentiment_classifier_experiment")
            with mlflow.start_run(run_name="sentiment_classifier_training") as run:
                # Log model parameters
                mlflow.log_param("C", 1)
                mlflow.log_param("solver", "liblinear")
                mlflow.log_param("penalty", "l2")
                
                # Log the model with name instead of artifact_path
                mlflow.sklearn.log_model(
                    clf, 
                    name="model",
                    signature=signature
                )
                
                # Try model registration, but don't fail if not supported
                try:
                    # Register the model in Model Registry
                    model_uri = f"runs:/{run.info.run_id}/model"
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name="sentiment_classifier"
                    )
                    
                    logging.info(f"Model registered with name: {registered_model.name}, version: {registered_model.version}")
                    print(f"Model registered with name: {registered_model.name}")
                    print(f"Model version: {registered_model.version}")
                except Exception as e:
                    logging.warning(f"Model registration failed (likely unsupported): {e}")
                    print(f"Model registration failed, but MLflow tracking succeeded: {e}")
                
                # Update model_info with run ID and save again
                model_info['run_id'] = run.info.run_id
                with open('reports/experiment_info.json', 'w') as f:
                    json.dump(model_info, f, indent=4)
                
                print(f"MLflow run ID: {run.info.run_id}")
                print(f"View run at: https://dagshub.com/sc/capstone_repo.mlflow/#/experiments/0/runs/{run.info.run_id}")
                
        except Exception as e:
            logging.warning(f"MLflow operations failed, but model is saved locally: {e}")
            print(f"MLflow operations failed, but model is saved locally: {e}")
    
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        
        # Ensure experiment_info.json exists even on error
        os.makedirs('reports', exist_ok=True)
        if not os.path.exists('reports/experiment_info.json'):
            with open('reports/experiment_info.json', 'w') as f:
                json.dump({'run_id': 'error-run', 'model_path': 'models/model.pkl'}, f, indent=4)

if __name__ == '__main__':
    main()