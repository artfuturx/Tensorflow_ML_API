from data.database import DatabaseManager
from src.feature_engineering import FeatureEngineer
from src.model import ReturnRiskModel
from sklearn.model_selection import train_test_split
from src.config import MODEL_CONFIG


def main():
    db_manager = None

    try:
        db_manager = DatabaseManager()
        features_engineering = FeatureEngineer()
        model = ReturnRiskModel()

        print("Loading data...")
        df=db_manager.get_order_data()

        print("Feature engineering...")
        df_processed=features_engineering.create_features(df)

        print("Preparing data...")
        X, y = features_engineering.prepare_data(df_processed)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=MODEL_CONFIG["test_size"], random_state=MODEL_CONFIG["random_state"])

        model.build_model(X.shape[1])
        model.train_model(X_train, y_train)

        loss, accuracy = model.evaluate_model(X_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

        model.save_model("models/return_risk_model.h5")
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error connecting to database: {e}")
        return e
    finally:
        if db_manager is not None:
            db_manager.disconnect()


if __name__ == "__main__":
    main()
