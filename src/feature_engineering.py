from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from src.config import FEATURE_CONFIG, MODEL_CONFIG

class FeatureEngineer:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.customer_features = None
        
    def create_features(self, df):
        df["total_amount"] = df["unit_price"]*df["quantity"]*(1-df["discount"])
        df["discount_amount"] = df["unit_price"]*df["quantity"]*df["discount"]
       
        self.customer_features = df.groupby("customer_id").agg({
            "total_amount": ["mean", "std", "sum"],
            "discount": ["mean", "max", "min"],
            "quantity": ["mean", "sum"],
        }).reset_index()
        
        self.customer_features.columns = ["customer_id", "total_amount_mean", "total_amount_std", "total_amount_sum", "discount_mean", "discount_max", "discount_min", "quantity_mean", "quantity_sum"]
        df = df.merge(self.customer_features, on="customer_id", how="left")


        high_discount = df["discount"] > FEATURE_CONFIG["high_discount_treshold"]
        low_amount = df["total_amount"] < FEATURE_CONFIG["low_amount_treshold"]

        df["return_risk"] = (high_discount&low_amount).astype(int)

        return df
    
    def prepare_data(self,df):
        features_columns = ["unit_price", "quantity", "discount", "total_amount", "discount_amount", 
                            "total_amount_mean", "total_amount_std", "total_amount_sum", "discount_mean", "discount_max", "discount_min", "quantity_mean", "quantity_sum"]
        
        X = df[features_columns]
        y = df["return_risk"]

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y
