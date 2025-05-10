import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.config import MODEL_CONFIG



class ReturnRiskModel:
    def __init__(self):
        self.model = None
        
        
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(64, activation="relu", input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    """
    def split_data(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=MODEL_CONFIG["test_size"], random_state=MODEL_CONFIG["random_state"])
        return X_train, X_test, y_train, y_test
    """


    def train_model(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=MODEL_CONFIG["test_size"], random_state=MODEL_CONFIG["random_state"])
       
        # callbacks = [
        #     EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        #     ModelCheckpoint(filepath="models/best_model.keras", monitor="val_loss", save_best_only=True)]
        
        self.history = self.model.fit(
            X_train, y_train, 
            epochs=MODEL_CONFIG["epochs"],
            validation_data=(X_test, y_test), 
            # callbacks=callbacks,
            verbose=MODEL_CONFIG["verbose"]
        )
        
        return self.history
    

    
    def evaluate_model(self, X, y):
        loss, accuracy = self.model.evaluate(X, y, verbose=MODEL_CONFIG["verbose"])
        return loss, accuracy
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, path):
        # Ensure the path ends with .keras
        if not path.endswith('.keras'):
            path = path.replace('.h5', '.keras')
        self.model.save(path)
        print(f"Model saved successfully to {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        
