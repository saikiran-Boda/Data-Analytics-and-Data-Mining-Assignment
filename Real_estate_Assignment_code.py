import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
from matplotlib.ticker import FuncFormatter


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

class OptimizedXGBoostRealEstate:
    
    
    def __init__(self, data_path=None, random_state=42, output_dir="model_output"):
        self.data_path = r"E:\realtor-data.zip.csv" if data_path is None else data_path
        self.random_state = random_state
        self.output_dir = output_dir
        self.model = None
        self.feature_names = []
        self.encoders = {}
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(random_state)
    
    def run_pipeline(self):
        
        print("Starting pipeline...")
        
        try:
            
            print(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
            print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            
            print("\nSample data:")
            print(df.head(3))
            
            
            print("\nData types:")
            print(df.dtypes)
            
            
            print("\nConverting mixed type columns to strings...")
            
            for col in ['brokered_by', 'status', 'city', 'state', 'street', 'zip_code']:
                if col in df.columns:
                    print(f"Converting {col} to string")
                    df[col] = df[col].astype(str)
            
            
            df = df.replace('nan', 'Unknown').replace('None', 'Unknown')
            
            
            df = self.preprocess_data(df)
            if df is None:
                return None
                
            
            X, y = self.select_features(df)
            
            
            sample_size = min(100000, len(X))
            print(f"Using a sample of {sample_size} records for training")
            indices = np.random.choice(X.index, size=sample_size, replace=False)
            X_sample = X.loc[indices]
            y_sample = y.loc[indices]
            
            
            results = self.train_model(X_sample, y_sample)
            
            
            self.save_key_plots()
            
            print(f"Pipeline complete. Results and visualizations saved to {self.output_dir}")
            return results
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_data(self, df):
       
        print("Preprocessing data...")
        
        try:
            
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[(df['price'].notna()) & (df['price'] > 0)]
            df = df[df['price'] <= df['price'].quantile(0.999)]  
            
            for col in ['bed', 'bath', 'acre_lot', 'house_size']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median()).clip(0)
            
            
            if 'prev_sold_date' in df.columns:
                df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
                now = datetime.now()
                
                valid_dates = df['prev_sold_date'].notna()
                df['days_since_prev_sale'] = np.nan
                if valid_dates.any():
                    df.loc[valid_dates, 'days_since_prev_sale'] = (now - df.loc[valid_dates, 'prev_sold_date']).dt.days
                df['days_since_prev_sale'] = df['days_since_prev_sale'].fillna(-1)
            
            
            df = self.engineer_features(df)
            
            
            self.processed_df = df
            
            
            self.plot_price_distribution(df['price'])
            
            print(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def engineer_features(self, df):
        
        print("Engineering features...")
        
        
        if all(col in df.columns for col in ['price', 'house_size']):
            
            df['price_per_sqft'] = 0.0 
            valid_size = (df['house_size'] > 0)
            if valid_size.any():  
               
                df.loc[valid_size, 'price_per_sqft'] = (
                    df.loc[valid_size, 'price'] / df.loc[valid_size, 'house_size']
                )
            
            df['price_per_sqft'] = df['price_per_sqft'].clip(0, df['price_per_sqft'].quantile(0.99))
            
            df['log_price_per_sqft'] = np.log1p(df['price_per_sqft'])
        
        
        if all(col in df.columns for col in ['bed', 'bath']):
            
            df['bed_bath_ratio'] = 0.0
            valid_bath = (df['bath'] > 0)
            if valid_bath.any():
                df.loc[valid_bath, 'bed_bath_ratio'] = (
                    df.loc[valid_bath, 'bed'] / df.loc[valid_bath, 'bath']
                )
            df['bed_bath_ratio'] = df['bed_bath_ratio'].clip(0, 5)
            df['total_rooms'] = df['bed'] + df['bath']
        
        
        if all(col in df.columns for col in ['acre_lot', 'house_size']):
            df['lot_size_sqft'] = df['acre_lot'] * 43560
           
            df['lot_to_house_ratio'] = 0.0
            valid_house = (df['house_size'] > 0)
            if valid_house.any():
                df.loc[valid_house, 'lot_to_house_ratio'] = (
                    df.loc[valid_house, 'lot_size_sqft'] / df.loc[valid_house, 'house_size']
                )
            df['lot_to_house_ratio'] = df['lot_to_house_ratio'].clip(0, 100)
        
        
        for col in ['price', 'house_size', 'acre_lot']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])
        
        
        if 'house_size' in df.columns:
            size_bins = [0, 750, 1500, 2500, 4000, float('inf')]
            size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
            df['size_category'] = pd.cut(df['house_size'], bins=size_bins, labels=size_labels)
        
        
        if all(col in df.columns for col in ['house_size', 'bed', 'bath']):
            df['inferred_property_type'] = 'Standard'
            luxury_mask = (df['house_size'] > 4000) | ((df['house_size'] > 3000) & (df['bath'] >= 4))
            studio_mask = (df['house_size'] < 650) & (df['bed'] <= 1)
            family_mask = (df['bed'] >= 3) & (df['house_size'] > 1500) & (df['house_size'] <= 4000)
            condo_mask = (df['bed'] <= 2) & (df['house_size'] >= 650) & (df['house_size'] <= 1500)
            
            df.loc[luxury_mask, 'inferred_property_type'] = 'Luxury'
            df.loc[studio_mask, 'inferred_property_type'] = 'Studio'
            df.loc[family_mask, 'inferred_property_type'] = 'Family'
            df.loc[condo_mask, 'inferred_property_type'] = 'Condo'
        
        
        df = self.create_location_features(df)
        
        
        df = self.process_broker_status(df)
        
        print(f"Created new features successfully")
        return df
    
    def create_location_features(self, df):
       
        print("Creating location features...")
        location_cols = [col for col in ['state', 'city', 'zip_code'] if col in df.columns]
        
        if not location_cols:
            return df
            
        
        if 'state' in location_cols:
            
            df['state'] = df['state'].fillna('Unknown')
            
            state_encoder = LabelEncoder()
            df['state_encoded'] = state_encoder.fit_transform(df['state'])
            self.encoders['state'] = state_encoder
            
            
            state_price = df.groupby('state')['price'].agg(['mean', 'median', 'count']).reset_index()
            state_price.columns = ['state', 'state_avg_price', 'state_median_price', 'state_count']
            state_price['state_price_index'] = state_price['state_median_price'] / df['price'].median()
            df = pd.merge(df, state_price, on='state', how='left')
        
        
        if all(col in location_cols for col in ['state', 'city']):
            
            df['city'] = df['city'].fillna('Unknown')
            
            df['city_state'] = df['city'] + ', ' + df['state']
            city_encoder = LabelEncoder()
            df['city_state_encoded'] = city_encoder.fit_transform(df['city_state'])
            self.encoders['city_state'] = city_encoder
            
            
            city_price = df.groupby('city_state')['price'].agg(['mean', 'median', 'count']).reset_index()
            city_price.columns = ['city_state', 'city_avg_price', 'city_median_price', 'city_count']
            df = pd.merge(df, city_price, on='city_state', how='left')
            
            
            if 'state_median_price' in df.columns:
                df['city_to_state_ratio'] = df['city_median_price'] / df['state_median_price']
        
        
        if 'zip_code' in location_cols:
            
            df['zip_code'] = df['zip_code'].str.replace(r'\.0+$', '', regex=True)
            df['zip_code'] = df['zip_code'].fillna('Unknown')
            
            zip_encoder = LabelEncoder()
            df['zip_encoded'] = zip_encoder.fit_transform(df['zip_code'])
            self.encoders['zip_code'] = zip_encoder
            
            
            zip_price = df.groupby('zip_code')['price'].agg(['mean', 'median', 'count']).reset_index()
            zip_price.columns = ['zip_code', 'zip_avg_price', 'zip_median_price', 'zip_count']
            df = pd.merge(df, zip_price, on='zip_code', how='left')
            
            
            if all(col in df.columns for col in ['city_state', 'city_median_price']):
                df['zip_to_city_ratio'] = df['zip_median_price'] / df['city_median_price']
        
        
        if 'city_to_state_ratio' in df.columns:
            df['premium_city'] = (df['city_to_state_ratio'] > 1.1).astype(int)
            df['discount_city'] = (df['city_to_state_ratio'] < 0.9).astype(int)
            
            if 'zip_to_city_ratio' in df.columns:
                df['premium_zip'] = (df['zip_to_city_ratio'] > 1.1).astype(int)
                df['discount_zip'] = (df['zip_to_city_ratio'] < 0.9).astype(int)
        
        print(f"Created location features for {len(location_cols)} location levels")
        return df
    
    def process_broker_status(self, df):
        
        print("Processing broker and status features...")
        
        
        if 'brokered_by' in df.columns:
            df['brokered_by'] = df['brokered_by'].fillna('Unknown')
            
            
            df['brokered_by'] = df['brokered_by'].astype(str)
            df['brokered_by'] = df['brokered_by'].replace('nan', 'Unknown')
            
            broker_encoder = LabelEncoder()
            df['broker_encoded'] = broker_encoder.fit_transform(df['brokered_by'])
            self.encoders['brokered_by'] = broker_encoder
            
            
            broker_stats = df.groupby('brokered_by').agg(
                broker_listings=('price', 'count'),
                broker_avg_price=('price', 'mean')
            ).reset_index()
            
            broker_stats['broker_market_share'] = broker_stats['broker_listings'] / len(df)
            broker_stats['broker_price_tier'] = broker_stats['broker_avg_price'] / df['price'].mean()
            
            
            df = pd.merge(df, broker_stats, on='brokered_by', how='left')
        
        
        if 'status' in df.columns:
            df['status'] = df['status'].fillna('Unknown')
            
            
            df['status'] = df['status'].astype(str)
            df['status'] = df['status'].replace('nan', 'Unknown')
            
            status_encoder = LabelEncoder()
            df['status_encoded'] = status_encoder.fit_transform(df['status'])
            self.encoders['status'] = status_encoder
            
            
            status_stats = df.groupby('status').agg(
                status_count=('price', 'count'),
                status_avg_price=('price', 'mean')
            ).reset_index()
            
            status_stats['status_price_factor'] = status_stats['status_avg_price'] / df['price'].mean()
            
            
            df = pd.merge(df, status_stats, on='status', how='left')
        
        return df
    
    def select_features(self, df):
       
        print("Selecting features...")
        
        exclude_columns = ['price', 'brokered_by', 'state', 'city', 'zip_code', 
                          'street', 'city_state', 'prev_sold_date']
        
        y = df['price']
        
        feature_cols = [col for col in df.columns if col not in exclude_columns and col != 'price']
        
        categorical_cols = []
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
        
        X = pd.get_dummies(df[feature_cols], columns=categorical_cols, drop_first=True)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Selected {len(self.feature_names)} features for modeling")
        
        q1, q3 = y.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
        if outliers.sum() > 0:
            print(f"Removing {outliers.sum()} outliers from target variable")
            X = X[~outliers].copy()
            y = y[~outliers].copy()
        
        return X, y
    
    def train_model(self, X, y, optimize_hyperparams=True, n_iter=10):
        
        print(f"Training model on {X.shape[0]} samples with {X.shape[1]} features...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state)
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 200,
            'objective': 'reg:squarederror',
            'random_state': self.random_state
        }
        
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X_train, y_train, n_iter=n_iter)
            params.update(best_params)
        
        print("Training final model...")
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False 
        )
        
        self.model = model
        
        results = self.evaluate_model()
        
        self.save_model()
        
        return results
    
    def optimize_hyperparameters(self, X_train, y_train, n_iter=10, cv=3):
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'n_estimators': [100, 200]
        }
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state
        )
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best CV score: {-random_search.best_score_:.4f} RMSE")
        print(f"Best parameters: {random_search.best_params_}")
        
        return random_search.best_params_
    
    def evaluate_model(self):
        
        if not hasattr(self, 'model') or self.model is None:
            print("No model to evaluate")
            return None
            
        print("Evaluating model performance...")
        
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        def mape(y_true, y_pred):
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        train_mape = mape(self.y_train, y_pred_train)
        test_mape = mape(self.y_test, y_pred_test)
        
        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape
        }
        
        print(f"Training RMSE: ${train_rmse:.2f}, Test RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
        print(f"Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Training MAPE: {train_mape:.2f}%, Test MAPE: {test_mape:.2f}%")
        
        self.plot_actual_vs_predicted(self.y_test, y_pred_test)
        
        return results
    
    def save_key_plots(self):
        
        if not hasattr(self, 'model') or self.model is None:
            return
        
        self.plot_feature_importance()
        
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            y_pred = self.model.predict(self.X_test)
            self.plot_error_distribution(self.y_test, y_pred)
    
    def plot_price_distribution(self, prices):
       
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(prices, kde=True, bins=50, color='darkblue')
            
            formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
            plt.gca().xaxis.set_major_formatter(formatter)
            
            plt.title('Property Price Distribution', fontsize=14)
            plt.xlabel('Price ($)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/price_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.histplot(np.log1p(prices), kde=True, bins=50, color='darkblue')
            plt.title('Log Property Price Distribution', fontsize=14)
            plt.xlabel('Log(Price+1)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/log_price_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting price distribution: {e}")
    
    def plot_actual_vs_predicted(self, y_true, y_pred):
        
        try:
            plt.figure(figsize=(10, 8))
            
            hb = plt.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', alpha=0.7, mincnt=1)
            plt.colorbar(hb, label='Count')
            
            lims = [
                min(min(y_true), min(y_pred)),
                max(max(y_true), max(y_pred))
            ]
            plt.plot(lims, lims, 'r--', lw=2)
            
            formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            plt.annotate(
                f'R² = {r2:.4f}\nMAE = ${mae:,.0f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=12, ha='left', va='top'
            )
            
            plt.title('Actual vs Predicted Property Prices', fontsize=14)
            plt.xlabel('Actual Price ($)', fontsize=12)
            plt.ylabel('Predicted Price ($)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting actual vs predicted: {e}")
    
    def plot_feature_importance(self, top_n=20):
        
        try:
            if not hasattr(self, 'model') or self.model is None:
                return
                
            importance = self.model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(top_n)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            feature_importance.to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
            
            return feature_importance
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
    
    def plot_price_distribution(self, prices):
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(prices, kde=True, bins=50, color='darkblue')
            formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
            plt.gca().xaxis.set_major_formatter(formatter)
            
            plt.title('Property Price Distribution', fontsize=14)
            plt.xlabel('Price ($)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/price_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.histplot(np.log1p(prices), kde=True, bins=50, color='darkblue')
            plt.title('Log Property Price Distribution', fontsize=14)
            plt.xlabel('Log(Price+1)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/log_price_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting price distribution: {e}")
    
    def plot_actual_vs_predicted(self, y_true, y_pred):
        try:
            plt.figure(figsize=(10, 8))
            
            hb = plt.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', alpha=0.7, mincnt=1)
            plt.colorbar(hb, label='Count')
            
            lims = [
                min(min(y_true), min(y_pred)),
                max(max(y_true), max(y_pred))
            ]
            plt.plot(lims, lims, 'r--', lw=2)
            
            formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            plt.annotate(
                f'R² = {r2:.4f}\nMAE = ${mae:,.0f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=12, ha='left', va='top'
            )
            
            plt.title('Actual vs Predicted Property Prices', fontsize=14)
            plt.xlabel('Actual Price ($)', fontsize=12)
            plt.ylabel('Predicted Price ($)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting actual vs predicted: {e}")
    
    def plot_feature_importance(self, top_n=20):
        try:
            if not hasattr(self, 'model') or self.model is None:
                return
                
            importance = self.model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(top_n)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            feature_importance.to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
            
            return feature_importance
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
    
    def plot_error_distribution(self, y_true, y_pred):
        try:
            errors = y_true - y_pred
            pct_errors = (errors / y_true) * 100
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            sns.histplot(errors, kde=True, color='darkblue', ax=ax1)
            ax1.axvline(x=0, color='r', linestyle='--')
            ax1.set_title('Error Distribution', fontsize=14)
            ax1.set_xlabel('Error ($)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            
            formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
            ax1.xaxis.set_major_formatter(formatter)
            
            sns.histplot(pct_errors, kde=True, color='darkgreen', ax=ax2)
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_title('Percentage Error Distribution', fontsize=14)
            ax2.set_xlabel('Percentage Error (%)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Error distribution plot saved successfully")
        except Exception as e:
            print(f"Error plotting error distribution: {e}")
    
    def save_model(self):
        if self.model is not None:
            with open(f"{self.output_dir}/xgboost_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.output_dir}/xgboost_model.pkl")

if __name__ == "__main__":
    model = OptimizedXGBoostRealEstate(
        data_path=r"C:\Users\Sai Kiran\Downloads\archive (8)\realtor-data.zip.csv", 
        output_dir="model_results"
    )
    
    results = model.run_pipeline()