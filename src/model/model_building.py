import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging
import os
import mlflow
from mlflow.models import infer_signature

# Import dagshub conditionally to avoid errors in GitHub Actions
try:
    import dagshub
except ImportError:
    pass

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

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
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
        # Configure MLflow
        dagshub_token = os.getenv("CAPSTONE_TEST")
        
        if dagshub_token:
            # If running in a GitHub Action or environment with token
            print("Using DagsHub token for authentication")
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri('https://dagshub.com/sc/capstone.mlflow')
            
            # Load and prepare data
            train_data = load_data('./data/processed/train_bow.csv')
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            
            # Create X_train as DataFrame for signature inference
            X_train_df = train_data.iloc[:, :-1]

            # Train model
            clf = train_model(X_train, y_train)
            
            # Save model locally
            save_model(clf, os.path.join('models', 'model.pkl'))
            
            # Generate model signature for better documentation
            signature = infer_signature(X_train_df, clf.predict(X_train_df))
            
            # Start MLflow run
            with mlflow.start_run(run_name="sentiment_classifier_training") as run:
                # Log model parameters
                mlflow.log_param("C", 1)
                mlflow.log_param("solver", "liblinear")
                mlflow.log_param("penalty", "l2")
                
                # Log the model
                mlflow.sklearn.log_model(
                    clf, 
                    "model",
                    signature=signature
                )
                
                # Register the model in Model Registry
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="sentiment_classifier"
                )
                
                logging.info(f"Model registered with name: {registered_model.name}, version: {registered_model.version}")
                print(f"Model registered with name: {registered_model.name}")
                print(f"Model version: {registered_model.version}")
                print(f"View in Model Registry: https://dagshub.com/sc/capstone_repo.mlflow/#/models/{registered_model.name}")
        else:
            # If token is not available, just train and save the model locally
            print("DagsHub token not found - skipping MLflow tracking")
            train_data = load_data('./data/processed/train_bow.csv')
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            
            # Train model
            clf = train_model(X_train, y_train)
            
            # Save model locally
            save_model(clf, os.path.join('models', 'model.pkl'))
            print("Model trained and saved locally without MLflow tracking")

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()