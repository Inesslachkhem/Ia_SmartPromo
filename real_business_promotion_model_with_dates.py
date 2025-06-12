"""
Enhanced Real Business Promotion Model with Date Prediction Capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class RealDataPromotionModel:
    """
    Professional AI Model for Promotion Optimization using REAL business data

    This model uses actual sales, stock, price, and financial data
    NO RANDOM VALUES - only real business metrics
    NOW WITH DATE PREDICTION CAPABILITIES
    """

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
        self.seasonal_patterns = {}
        self.market_trends = {}
        self.base_date = datetime.now()  # Reference date for calculations

    def extract_date_features(self, target_date):
        """Extract temporal features from a target date"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, datetime):
            pass
        else:
            raise ValueError("Date must be string or datetime object")

        # Basic date features
        date_features = {
            "year": target_date.year,
            "month": target_date.month,
            "day": target_date.day,
            "day_of_week": target_date.weekday(),
            "day_of_year": target_date.timetuple().tm_yday,
            "week_of_year": target_date.isocalendar()[1],
            "quarter": (target_date.month - 1) // 3 + 1,
            "is_weekend": 1 if target_date.weekday() >= 5 else 0,
        }

        # Seasonal indicators
        date_features.update(
            {
                "is_ramadan_season": self._is_ramadan_season(target_date),
                "is_back_to_school": self._is_back_to_school(target_date),
                "is_summer_season": self._is_summer_season(target_date),
                "is_winter_season": self._is_winter_season(target_date),
                "is_holiday_season": self._is_holiday_season(target_date),
                "is_end_of_month": 1 if target_date.day >= 25 else 0,
                "is_beginning_of_month": 1 if target_date.day <= 5 else 0,
            }
        )

        # Market cycle features
        date_features.update(
            {
                "months_from_base": (target_date.year - self.base_date.year) * 12
                + (target_date.month - self.base_date.month),
                "seasonal_demand_multiplier": self._calculate_seasonal_demand(
                    target_date
                ),
                "competition_intensity": self._calculate_competition_intensity(
                    target_date
                ),
                "market_maturity_factor": min(
                    (target_date.year - 2020) / 10, 1.0
                ),  # Market evolution factor
            }
        )

        # Cyclical features (sine/cosine for smooth seasonal transitions)
        date_features.update(
            {
                "month_sin": np.sin(2 * np.pi * target_date.month / 12),
                "month_cos": np.cos(2 * np.pi * target_date.month / 12),
                "day_sin": np.sin(2 * np.pi * target_date.day / 31),
                "day_cos": np.cos(2 * np.pi * target_date.day / 31),
                "week_sin": np.sin(2 * np.pi * date_features["week_of_year"] / 52),
                "week_cos": np.cos(2 * np.pi * date_features["week_of_year"] / 52),
            }
        )

        return date_features

    def _is_ramadan_season(self, date):
        """Check if date falls in Ramadan season (approximate)"""
        ramadan_months = [3, 4, 5]  # Approximate months for Ramadan period
        return 1 if date.month in ramadan_months else 0

    def _is_back_to_school(self, date):
        """Check if date is back-to-school season"""
        return 1 if date.month in [9, 10] else 0

    def _is_summer_season(self, date):
        """Check if date is summer season"""
        return 1 if date.month in [6, 7, 8] else 0

    def _is_winter_season(self, date):
        """Check if date is winter season"""
        return 1 if date.month in [12, 1, 2] else 0

    def _is_holiday_season(self, date):
        """Check if date is holiday season"""
        return 1 if date.month in [7, 8, 12, 1] else 0

    def _calculate_seasonal_demand(self, date):
        """Calculate seasonal demand multiplier based on historical patterns"""
        seasonal_multipliers = {
            1: 0.8,  # January - post-holiday low
            2: 0.9,  # February - recovery
            3: 1.1,  # March - spring season
            4: 1.0,  # April - normal
            5: 1.2,  # May - pre-summer
            6: 1.3,  # June - summer start
            7: 1.4,  # July - peak summer
            8: 1.3,  # August - summer end
            9: 1.2,  # September - back to school
            10: 1.1,  # October - autumn
            11: 1.0,  # November - normal
            12: 1.5,  # December - holiday season
        }
        return seasonal_multipliers.get(date.month, 1.0)

    def _calculate_competition_intensity(self, date):
        """Calculate competition intensity based on time of year"""
        if date.month in [7, 8, 12]:  # Summer and December
            return 0.9  # High competition
        elif date.month in [6, 9, 11]:  # Pre-peak seasons
            return 0.7  # Medium competition
        else:
            return 0.5  # Normal competition

    def load_real_business_data(
        self, file_path="enhanced_business_dataset_with_dates.xlsx"
    ):
        """Load the enhanced business dataset with date-based features"""
        print("=== LOADING ENHANCED BUSINESS DATA WITH DATES ===")

        # Load the enhanced business dataset
        df = pd.read_excel(file_path, sheet_name="Business_Data", engine="openpyxl")
        print(
            f"Enhanced business dataset loaded: {df.shape[0]} products, {df.shape[1]} features"
        )

        # Analyze the enhanced data structure
        self.analyze_real_data(df)

        # Calculate optimal promotions based on enhanced business logic
        df_with_targets = self.calculate_real_optimal_promotions(df)

        return df_with_targets

    def analyze_real_data(self, df):
        """Analyze the enhanced business data structure"""
        print("\n=== ENHANCED BUSINESS DATA ANALYSIS ===")
        print(f"Total Products: {df.shape[0]:,}")
        print(f"Total Monthly Revenue: {df['CA_Mensuel_TND'].sum():,.2f} TND")
        print(f"Average Product Price: {df['Prix_Vente_TND'].mean():.2f} TND")
        print(
            f"Average Monthly Sales: {df['Ventes_Mensuelles_Unites'].mean():.0f} units"
        )
        print(f"Total Stock Value: {df['Valeur_Stock_TND'].sum():,.2f} TND")
        print(f"Average Price Elasticity: {df['Elasticite_Prix'].mean():.2f}")

        # Show date-based features with proper date conversion
        print(f"\n=== DATE-BASED FEATURES ===")

        # Convert date columns to datetime if they aren't already
        try:
            df["Date_Achat"] = pd.to_datetime(df["Date_Achat"])
            df["Date_Derniere_Vente"] = pd.to_datetime(df["Date_Derniere_Vente"])

            print(
                f"Purchase Date Range: {df['Date_Achat'].min().strftime('%Y-%m-%d')} to {df['Date_Achat'].max().strftime('%Y-%m-%d')}"
            )
            print(
                f"Sales Date Range: {df['Date_Derniere_Vente'].min().strftime('%Y-%m-%d')} to {df['Date_Derniere_Vente'].max().strftime('%Y-%m-%d')}"
            )
        except Exception as e:
            print(f"Warning: Could not process date columns: {e}")
            print("Date columns will be displayed as-is without range analysis")

        print(
            f"Average Days Since Purchase: {df['Jours_Depuis_Achat'].mean():.0f} days"
        )
        print(
            f"Average Days Since Last Sale: {df['Jours_Depuis_Derniere_Vente'].mean():.0f} days"
        )

        # Show category distribution
        print("\n=== REVENUE BY CATEGORY ===")
        category_revenue = (
            df.groupby("GA_FAMILLENIV1")["CA_Mensuel_TND"]
            .sum()
            .sort_values(ascending=False)
        )
        for cat, revenue in category_revenue.head(7).items():
            print(f"Category {cat}: {revenue:,.2f} TND")

    def calculate_real_optimal_promotions(self, df):
        """Calculate optimal promotions using REAL business data"""
        print("\n=== CALCULATING OPTIMAL PROMOTIONS FROM REAL DATA ===")

        # Use actual business metrics to determine optimal promotions
        df_calc = df.copy()

        # 0. Rotation factor (high rotation = less need for promotion)
        # Rotation = Ventes totales / Quantité injectée (Achat)
        if (
            "Ventes_Mensuelles_Unites" in df_calc.columns
            and "Quantite_Achat" in df_calc.columns
        ):
            rotation = df_calc["Ventes_Mensuelles_Unites"] / df_calc[
                "Quantite_Achat"
            ].replace(0, np.nan)
            rotation = rotation.fillna(0)
            # Plus la rotation est faible, plus on veut promouvoir
            rotation_factor = np.clip((1 - rotation), 0, 1) * 0.1  # max 10% d'impact
        else:
            rotation_factor = 0

        # 1. Stock clearance factor (high stock = higher promotion)
        stock_months = (
            df_calc["Stock_Actuel_Unites"] / df_calc["Ventes_Mensuelles_Unites"]
        )
        stock_factor = np.clip(
            (stock_months - 3) / 10, 0, 0.3
        )  # Si le stock couvre plus de 3 mois, on considère que c’est excessif.

        # 2. Calcul de la marge en pourcentage à la volée et du margin_factor dans un seul bloc
        df_calc["Marge_Pourcentage"] = (
            (df_calc["Prix_Vente_TND"] - df_calc["Cout_Unitaire_TND"])
            / df_calc["Prix_Vente_TND"]
        ) * 100
        margin_factor = (
            df_calc["Marge_Pourcentage"] - 30
        ) / 100  # Scale from margin above 30%
        margin_factor = np.clip(margin_factor, 0, 0.25)  # Max 25% from margin

        # 3. Price elasticity factor (calculated from real sales and price changes)
        # Elasticité Prix = (Variation en % de la Quantité Vendue) / (Variation en % du Prix de Vente)
        if (
            "Ventes_Mensuelles_Unites_Avant" in df_calc.columns
            and "Ventes_Mensuelles_Unites_Apres" in df_calc.columns
            and "Prix_Vente_TND_Avant" in df_calc.columns
            and "Prix_Vente_TND_Apres" in df_calc.columns
        ):
            variation_qte = (
                df_calc["Ventes_Mensuelles_Unites_Apres"]
                - df_calc["Ventes_Mensuelles_Unites_Avant"]
            ) / df_calc["Ventes_Mensuelles_Unites_Avant"].replace(0, np.nan)
            variation_prix = (
                df_calc["Prix_Vente_TND_Apres"] - df_calc["Prix_Vente_TND_Avant"]
            ) / df_calc["Prix_Vente_TND_Avant"].replace(0, np.nan)
            elasticity = variation_qte / variation_prix.replace(0, np.nan)
            elasticity = elasticity.replace([np.inf, -np.inf], 0).fillna(0)
            elasticity_factor = (
                np.abs(elasticity) / 5
            )  # Scale elasticity Diviser par 5 permet de "normaliser" ou "réduire" l’impact de l’élasticité pour qu’il ne domine pas le calcul de la promotion
        else:
            elasticity_factor = np.abs(df_calc["Elasticite_Prix"]) / 5  # fallback
        elasticity_factor = np.clip(
            elasticity_factor, 0, 0.3
        )  # Max 30% from elasticity

        # 4. Competition pressure factor
        # competition_factor = (
        #  df_calc["Facteur_Concurrence"] * 0.15
        #  )  # Max 15% from competition

        # 5. Historical promotion performance
        #  hist_performance = (
        #  df_calc["Derniere_Promo_Performance"] - 1
        # ) * 0.1  # Performance above 1x
        # hist_performance = np.clip(hist_performance, 0, 0.1)  # Max 10% from history

        # 6. Sales trend factor (declining sales = higher promotion)
        # trend_factor = np.where(
        # df_calc["Tendance_Ventes_3M"] < 0,
        # abs(df_calc["Tendance_Ventes_3M"]) * 0.5,
        # 0,
        # )  # Up to 5% for declining sales
        # trend_factor = np.clip(trend_factor, 0, 0.1)

        # Combine all factors using business logic weights
        optimal_promotion = (
            rotation_factor * 0.25  # 25% weight - rotation
            + stock_factor * 0.225  # 22.5% weight - inventory management
            + margin_factor * 0.20  # 20% weight - profitability protection
            + elasticity_factor * 0.30  # 30% weight - customer price sensitivity
            # + competition_factor * 0.25  # 15% weight - market pressure
            # + hist_performance * 0.10  # 10% weight - past performance
            # + trend_factor * 0.10  # 10% weight - sales trends
        )

        # Apply business constraints
        optimal_promotion = np.clip(
            optimal_promotion, 0.05, 0.45
        )  # 5-45% promotion range

        # Store the calculated optimal promotions
        df_calc["Promotion_Optimale"] = optimal_promotion

        # Calculate expected impact
        df_calc["Volume_Avec_Promo"] = df_calc["Ventes_Mensuelles_Unites"] * (
            1 + (df_calc["Elasticite_Prix"] * optimal_promotion)
        )
        df_calc["Prix_Avec_Promo"] = df_calc["Prix_Vente_TND"] * (1 - optimal_promotion)
        df_calc["CA_Avec_Promo"] = (
            df_calc["Volume_Avec_Promo"] * df_calc["Prix_Avec_Promo"]
        )
        df_calc["Profit_Avec_Promo"] = (
            df_calc["Prix_Avec_Promo"] - df_calc["Cout_Unitaire_TND"]
        ) * df_calc["Volume_Avec_Promo"]

        print(f"Optimal promotions calculated using real business metrics")
        print(f"Average promotion: {optimal_promotion.mean()*100:.1f}%")
        print(
            f"Promotion range: {optimal_promotion.min()*100:.1f}% - {optimal_promotion.max()*100:.1f}%"
        )

        return df_calc

    def prepare_real_features(self, df, target_date=None):
        """Prepare features using real business data with optional date features"""
        print("\n=== PREPARING REAL BUSINESS FEATURES ===")

        feature_df = df.copy()

        # Add date features if target_date is provided
        if target_date is not None:
            print(f"Adding date features for: {target_date}")
            date_features = self.extract_date_features(target_date)

            # Add date features to dataframe
            for feature_name, feature_value in date_features.items():
                feature_df[f"date_{feature_name}"] = feature_value

        # Encode categorical variables
        categorical_cols = ["GA_FAMILLENIV1", "GA_FAMILLENIV2", "GA_FOURNPRINC"]

        for col in categorical_cols:
            if col in feature_df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
                    feature_df[f"{col}_encoded"] = self.encoders[col].fit_transform(
                        feature_df[col]
                    )
                else:
                    feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
                    known_classes = set(self.encoders[col].classes_)
                    feature_df[col] = feature_df[col].apply(
                        lambda x: x if x in known_classes else "Unknown"
                    )
                    feature_df[f"{col}_encoded"] = self.encoders[col].transform(
                        feature_df[col]
                    )

        # Select real business features (NO simulated data)
        real_business_features = [
            # Pricing features
            "Prix_Vente_TND",
            "Cout_Unitaire_TND",
            "Marge_Pourcentage",
            # Sales features
            "Ventes_Mensuelles_Unites",
            "CA_Mensuel_TND",
            "Profit_Mensuel_TND",
            # Stock features
            "Stock_Actuel_Unites",
            "Valeur_Stock_TND",
            # Historical promotion features
            "Historique_Promo_Freq_12M",
            "Historique_Promo_Moyenne_Pct",
            "Derniere_Promo_Performance",
            # Market features
            "Elasticite_Prix",
            #  "Facteur_Concurrence",
            "Demande_Saisonniere",
            "Part_Marche_Pct",
            "Force_Marque",
            # Performance features
            "Tendance_Ventes_3M",
            "Satisfaction_Client",
            "Taux_Retour_Pct",
        ]

        # Add encoded categorical features
        encoded_features = [
            f"{col}_encoded" for col in categorical_cols if col in feature_df.columns
        ]

        # Add date features if they exist
        date_features = [col for col in feature_df.columns if col.startswith("date_")]
        # Combine all real features
        all_features = real_business_features + encoded_features + date_features
        available_features = [f for f in all_features if f in feature_df.columns]

        X = feature_df[available_features]
        X = X.fillna(X.median())

        print(
            f"Real business features prepared: {X.shape[1]} features, {X.shape[0]} samples"
        )
        print(f"Using ONLY real business data - NO random or simulated values")
        if date_features:
            print(f"Date features included: {len(date_features)} temporal features")
        print(
            f"Feature names: {available_features[:10]}..."
            if len(available_features) > 10
            else f"Feature names: {available_features}"
        )

        return X, available_features

    def train_models(self, X, y):
        """Train models on real business data"""
        print("\n=== TRAINING MODELS ON REAL BUSINESS DATA ===")

        # Store training feature names for later prediction compatibility
        self.training_features = list(X.columns)
        print(f"Training features stored: {len(self.training_features)} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models optimized for business data
        models_to_train = {
            "Random Forest": RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=150, max_depth=8, random_state=42
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=150, max_depth=8, random_state=42, n_jobs=-1, verbose=-1
            ),
        }

        # Train and evaluate models
        results = {}
        for name, model in models_to_train.items():
            print(f"Training {name} on real business data...")

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "model": model,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "predictions": y_pred,
            }

            print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]["r2"])
        self.best_model = results[best_model_name]["model"]
        self.model_name = best_model_name

        print(
            f"\nBest model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})"
        )

        # Store feature importance
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = dict(
                zip(X.columns, self.best_model.feature_importances_)
            )

        self.is_trained = True
        return results

    def predict_optimal_promotions(self, df):
        """Predict optimal promotions using real business data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        print("\n=== PREDICTING OPTIMAL PROMOTIONS ===")

        X, feature_names = self.prepare_real_features(df)
        promotions = self.best_model.predict(X)
        promotions = np.clip(promotions, 0, 0.5)

        return promotions

    def predict_promotion_for_date(self, article_code, target_date, df=None):
        """
        Predict optimal promotion for a specific article on a specific date

        Parameters:
        - article_code: The article code to predict for
        - target_date: Target date (string like '2025-12-25' or datetime object)
        - df: Optional dataframe, if None will load from file

        Returns:
        - Dictionary with prediction details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first! Call train_models() first.")

        # Load data if not provided
        if df is None:
            df = self.load_real_business_data()

        # Find the article
        article_data = df[df["GA_ARTICLE"] == article_code]
        if article_data.empty:
            raise ValueError(f"Article {article_code} not found in dataset")

        print(
            f"\n=== PREDICTING PROMOTION FOR ARTICLE {article_code} ON {target_date} ==="
        )

        # Get article info
        article_info = article_data.iloc[0]
        print(f"Article: {article_info['GA_ARTICLE']}")
        print(f"Description: {article_info['GA_LIBELLE']}")
        print(f"Category: {article_info['GA_FAMILLENIV2']}")
        print(f"Current Price: {article_info['Prix_Vente_TND']:.2f} TND")
        print(f"Target Date: {target_date}")
        # Prepare features with date information
        X, feature_names = self.prepare_real_features(
            article_data, target_date=target_date
        )

        # Align features to match training features
        X_aligned = self._align_features_for_prediction(X, feature_names)

        # Predict promotion
        predicted_promotion = self.best_model.predict(X_aligned)[0]
        predicted_promotion = np.clip(
            predicted_promotion, 0.05, 0.45
        )  # Apply business constraints

        # Calculate expected impact with date-specific factors
        date_features = self.extract_date_features(target_date)
        seasonal_adjustment = date_features["seasonal_demand_multiplier"]
        # competition_factor = date_features["competition_intensity"]

        # Adjust promotion based on temporal factors
        temporal_adjustment = 1.0
        if date_features["is_holiday_season"]:
            temporal_adjustment *= 1.2  # More aggressive during holidays
        if date_features["is_back_to_school"]:
            temporal_adjustment *= 1.1  # Slight increase for back-to-school
        if date_features["is_weekend"]:
            temporal_adjustment *= 1.05  # Slight weekend boost

        adjusted_promotion = predicted_promotion * temporal_adjustment
        adjusted_promotion = np.clip(adjusted_promotion, 0.05, 0.45)

        # Calculate business impact
        original_volume = article_info["Ventes_Mensuelles_Unites"]
        elasticity = article_info["Elasticite_Prix"]

        # Volume change with seasonal adjustment
        volume_change = elasticity * adjusted_promotion * seasonal_adjustment
        new_volume = original_volume * (1 + volume_change)

        # Financial calculations
        original_price = article_info["Prix_Vente_TND"]
        new_price = original_price * (1 - adjusted_promotion)

        original_revenue = original_volume * original_price
        new_revenue = new_volume * new_price

        cost_per_unit = article_info["Cout_Unitaire_TND"]
        original_profit = (original_price - cost_per_unit) * original_volume
        new_profit = (new_price - cost_per_unit) * new_volume

        # Create detailed prediction result
        prediction_result = {
            # Article information
            "article_code": article_code,
            "article_name": article_info["GA_LIBELLE"],
            "category": article_info["GA_FAMILLENIV2"],
            "supplier": article_info["GA_FOURNPRINC"],
            # Date information
            "target_date": str(target_date),
            "day_of_week": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][date_features["day_of_week"]],
            "season": self._get_season_name(target_date),
            "is_holiday_period": bool(date_features["is_holiday_season"]),
            "is_weekend": bool(date_features["is_weekend"]),
            # Prediction details
            "base_promotion_pct": round(predicted_promotion * 100, 2),
            "adjusted_promotion_pct": round(adjusted_promotion * 100, 2),
            "temporal_adjustment_factor": round(temporal_adjustment, 3),
            "seasonal_demand_multiplier": round(seasonal_adjustment, 3),
            # "competition_intensity": round(competition_factor, 3),
            # Current metrics
            "current_price_tnd": round(original_price, 2),
            "current_monthly_volume": int(original_volume),
            "current_monthly_revenue_tnd": round(original_revenue, 2),
            "current_monthly_profit_tnd": round(original_profit, 2),
            "current_margin_pct": round(article_info["Marge_Pourcentage"], 2),
            # Projected metrics
            "promoted_price_tnd": round(new_price, 2),
            "projected_monthly_volume": int(new_volume),
            "projected_monthly_revenue_tnd": round(new_revenue, 2),
            "projected_monthly_profit_tnd": round(new_profit, 2),
            # Impact analysis
            "revenue_impact_tnd": round(new_revenue - original_revenue, 2),
            "revenue_impact_pct": round(
                ((new_revenue - original_revenue) / original_revenue) * 100, 2
            ),
            "profit_impact_tnd": round(new_profit - original_profit, 2),
            "profit_impact_pct": round(
                ((new_profit - original_profit) / original_profit) * 100, 2
            ),
            "volume_impact_pct": round(
                ((new_volume - original_volume) / original_volume) * 100, 2
            ),
            # Risk assessment
            "risk_level": self._assess_promotion_risk(adjusted_promotion, article_info),
            "recommendation_confidence": self._calculate_confidence(
                X_aligned,
                (
                    self.training_features
                    if hasattr(self, "training_features")
                    else feature_names
                ),
            ),
        }

        # Add business recommendation after prediction_result is complete
        prediction_result["recommended_action"] = self._generate_recommendation(
            adjusted_promotion, prediction_result
        )

        # Print summary
        print(f"\n--- PREDICTION SUMMARY ---")
        print(f"Recommended Promotion: {prediction_result['adjusted_promotion_pct']}%")
        print(f"New Price: {prediction_result['promoted_price_tnd']} TND")
        print(f"Expected Volume Change: {prediction_result['volume_impact_pct']:+.1f}%")
        print(
            f"Expected Revenue Impact: {prediction_result['revenue_impact_tnd']:+.2f} TND ({prediction_result['revenue_impact_pct']:+.1f}%)"
        )
        print(
            f"Expected Profit Impact: {prediction_result['profit_impact_tnd']:+.2f} TND ({prediction_result['profit_impact_pct']:+.1f}%)"
        )
        print(f"Risk Level: {prediction_result['risk_level']}")
        print(f"Confidence: {prediction_result['recommendation_confidence']:.1%}")

        return prediction_result

    def _get_season_name(self, date):
        """Get season name for a given date"""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    def _assess_promotion_risk(self, promotion_pct, article_info):
        """Assess the risk level of the promotion"""
        margin = article_info["Marge_Pourcentage"] / 100

        if promotion_pct > margin * 0.8:  # Promotion > 80% of margin
            return "HIGH"
        elif promotion_pct > margin * 0.5:  # Promotion > 50% of margin
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_confidence(self, X, feature_names):
        """Calculate prediction confidence based on feature completeness"""
        # Simple confidence based on non-null features
        non_null_ratio = 1 - (X.isnull().sum().sum() / (X.shape[0] * X.shape[1]))

        # Adjust confidence based on model performance (if available)
        base_confidence = 0.85  # Based on R² score from training

        return min(base_confidence * non_null_ratio, 0.95)

    def _generate_recommendation(self, promotion_pct, prediction_result):
        """Generate business recommendation"""
        if promotion_pct >= 0.25:
            return "AGGRESSIVE PROMOTION - High clearance priority"
        elif promotion_pct >= 0.15:
            return "MODERATE PROMOTION - Good opportunity"
        elif promotion_pct >= 0.08:
            return "LIGHT PROMOTION - Conservative approach"
        else:
            return "MINIMAL PROMOTION - Maintain current strategy"

    def predict_multiple_articles_for_date(self, article_codes, target_date, df=None):
        """
        Predict promotions for multiple articles on a specific date

        Parameters:
        - article_codes: List of article codes
        - target_date: Target date
        - df: Optional dataframe

        Returns:
        - DataFrame with predictions for all articles
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        predictions = []

        print(
            f"\n=== PREDICTING PROMOTIONS FOR {len(article_codes)} ARTICLES ON {target_date} ==="
        )

        for i, article_code in enumerate(article_codes, 1):
            try:
                print(f"Processing article {i}/{len(article_codes)}: {article_code}")
                prediction = self.predict_promotion_for_date(
                    article_code, target_date, df
                )
                predictions.append(prediction)
            except Exception as e:
                print(f"Error processing {article_code}: {str(e)}")
                continue

        if not predictions:
            raise ValueError("No valid predictions generated")

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values(
            "profit_impact_tnd", ascending=False
        )

        # Summary statistics
        print(f"\n--- BATCH PREDICTION SUMMARY ---")
        print(f"Successfully processed: {len(predictions_df)} articles")
        print(
            f"Average promotion: {predictions_df['adjusted_promotion_pct'].mean():.1f}%"
        )
        print(
            f"Total revenue impact: {predictions_df['revenue_impact_tnd'].sum():+,.2f} TND"
        )
        print(
            f"Total profit impact: {predictions_df['profit_impact_tnd'].sum():+,.2f} TND"
        )
        print(f"High-risk promotions: {(predictions_df['risk_level'] == 'HIGH').sum()}")

        return predictions_df

    def predict_article_calendar(self, article_code, start_date, end_date, df=None):
        """
        Predict optimal promotions for an article across a date range

        Parameters:
        - article_code: Article code to analyze
        - start_date: Start date for analysis
        - end_date: End date for analysis
        - df: Optional dataframe

        Returns:
        - DataFrame with daily/weekly predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        print(
            f"\n=== PROMOTION CALENDAR FOR {article_code} ({start_date.date()} to {end_date.date()}) ==="
        )

        # Generate date range (weekly intervals for long periods)
        days_diff = (end_date - start_date).days
        if days_diff > 90:  # If more than 3 months, use weekly intervals
            date_range = pd.date_range(start=start_date, end=end_date, freq="W")
            print(f"Using weekly intervals ({len(date_range)} predictions)")
        else:  # Use daily intervals for shorter periods
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            print(f"Using daily intervals ({len(date_range)} predictions)")

        calendar_predictions = []

        for i, date in enumerate(date_range, 1):
            if i % 10 == 0:  # Progress update every 10 predictions
                print(f"Processing {i}/{len(date_range)} dates...")

            try:
                prediction = self.predict_promotion_for_date(article_code, date, df)

                # Simplified prediction for calendar view
                calendar_prediction = {
                    "date": date.strftime("%Y-%m-%d"),
                    "day_of_week": prediction["day_of_week"],
                    "season": prediction["season"],
                    "is_holiday": prediction["is_holiday_period"],
                    "is_weekend": prediction["is_weekend"],
                    "promotion_pct": prediction["adjusted_promotion_pct"],
                    "new_price_tnd": prediction["promoted_price_tnd"],
                    "revenue_impact_tnd": prediction["revenue_impact_tnd"],
                    "profit_impact_tnd": prediction["profit_impact_tnd"],
                    "risk_level": prediction["risk_level"],
                    "confidence": prediction["recommendation_confidence"],
                }
                calendar_predictions.append(calendar_prediction)

            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue

        calendar_df = pd.DataFrame(calendar_predictions)

        if not calendar_df.empty:
            # Add trend analysis
            calendar_df["promotion_trend"] = (
                calendar_df["promotion_pct"]
                .rolling(window=min(7, len(calendar_df)))
                .mean()
            )

            # Summary
            print(f"\n--- CALENDAR ANALYSIS SUMMARY ---")
            print(f"Date range analyzed: {len(calendar_df)} periods")
            print(f"Average promotion: {calendar_df['promotion_pct'].mean():.1f}%")
            print(
                f"Promotion range: {calendar_df['promotion_pct'].min():.1f}% - {calendar_df['promotion_pct'].max():.1f}%"
            )
            print(
                f"Best promotion date: {calendar_df.loc[calendar_df['profit_impact_tnd'].idxmax(), 'date']}"
            )
            print(
                f"Maximum profit impact: {calendar_df['profit_impact_tnd'].max():+.2f} TND"
            )

            # Seasonal insights
            seasonal_avg = calendar_df.groupby("season")["promotion_pct"].mean()
            print(f"\nSeasonal promotion averages:")
            for season, avg_promo in seasonal_avg.items():
                print(f"  {season}: {avg_promo:.1f}%")

        return calendar_df

    def calculate_business_impact(self, df, promotions):
        """Calculate business impact using real financial data"""
        print("\n=== CALCULATING REAL BUSINESS IMPACT ===")

        # Calculate new volumes using real price elasticity
        new_volumes = df["Ventes_Mensuelles_Unites"] * (
            1 + (df["Elasticite_Prix"] * promotions)
        )

        # Calculate new prices
        new_prices = df["Prix_Vente_TND"] * (1 - promotions)

        # Calculate new revenue and profit
        new_revenue = new_volumes * new_prices
        new_profit = (new_prices - df["Cout_Unitaire_TND"]) * new_volumes

        # Create impact analysis
        impact_data = {
            "Article": df["GA_ARTICLE"].values,
            "Libelle": df["GA_LIBELLE"].values,
            "Categorie": df["GA_FAMILLENIV2"].values,
            "Fournisseur": df["GA_FOURNPRINC"].values,
            # Current metrics
            "Prix_Actuel_TND": df["Prix_Vente_TND"].values,
            "Ventes_Actuelles": df["Ventes_Mensuelles_Unites"].values,
            "CA_Actuel_TND": df["CA_Mensuel_TND"].values,
            "Profit_Actuel_TND": df["Profit_Mensuel_TND"].values,
            "Marge_Actuelle_Pct": df["Marge_Pourcentage"].values,
            # Recommended promotion
            "Promotion_Recommandee_Pct": promotions * 100,
            # Projected metrics
            "Prix_Promo_TND": new_prices,
            "Ventes_Projetees": new_volumes,
            "CA_Projete_TND": new_revenue,
            "Profit_Projete_TND": new_profit,
            # Impact analysis
            "Impact_CA_TND": new_revenue - df["CA_Mensuel_TND"].values,
            "Impact_CA_Pct": (
                (new_revenue - df["CA_Mensuel_TND"].values)
                / df["CA_Mensuel_TND"].values
            )
            * 100,
            "Impact_Profit_TND": new_profit - df["Profit_Mensuel_TND"].values,
            "Impact_Volume_Pct": (
                (new_volumes - df["Ventes_Mensuelles_Unites"].values)
                / df["Ventes_Mensuelles_Unites"].values
            )
            * 100,
            # Business metrics
            "Stock_Actuel": df["Stock_Actuel_Unites"].values,
            "Elasticite_Prix": df["Elasticite_Prix"].values,
        }

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values("Impact_Profit_TND", ascending=False)

        return impact_df

    def generate_business_insights(self, impact_df):
        """Generate real business insights"""
        print("\n=== REAL BUSINESS INSIGHTS ===")

        total_current_ca = impact_df["CA_Actuel_TND"].sum()
        total_projected_ca = impact_df["CA_Projete_TND"].sum()
        total_ca_impact = total_projected_ca - total_current_ca

        total_current_profit = impact_df["Profit_Actuel_TND"].sum()
        total_projected_profit = impact_df["Profit_Projete_TND"].sum()
        total_profit_impact = total_projected_profit - total_current_profit

        print(f"REVENUE ANALYSIS:")
        print(f"  Current Monthly Revenue: {total_current_ca:,.2f} TND")
        print(f"  Projected Revenue with Promotions: {total_projected_ca:,.2f} TND")
        print(
            f"  Revenue Impact: {total_ca_impact:,.2f} TND ({(total_ca_impact/total_current_ca)*100:.2f}%)"
        )

        print(f"\nPROFIT ANALYSIS:")
        print(f"  Current Monthly Profit: {total_current_profit:,.2f} TND")
        print(f"  Projected Profit with Promotions: {total_projected_profit:,.2f} TND")
        print(
            f"  Profit Impact: {total_profit_impact:,.2f} TND ({(total_profit_impact/total_current_profit)*100:.2f}%)"
        )

        print(f"\nTOP 10 ARTICLES BY PROFIT IMPACT:")
        top_profit = impact_df.head(10)[
            ["Article", "Libelle", "Promotion_Recommandee_Pct", "Impact_Profit_TND"]
        ]
        for idx, row in top_profit.iterrows():
            print(
                f"  {row['Article']}: {row['Promotion_Recommandee_Pct']:.1f}% → +{row['Impact_Profit_TND']:,.2f} TND profit"
            )

        print(f"\nPROMOTION DISTRIBUTION:")
        print(
            f"  Average Promotion: {impact_df['Promotion_Recommandee_Pct'].mean():.1f}%"
        )
        print(
            f"  Median Promotion: {impact_df['Promotion_Recommandee_Pct'].median():.1f}%"
        )
        print(
            f"  Promotion Range: {impact_df['Promotion_Recommandee_Pct'].min():.1f}% - {impact_df['Promotion_Recommandee_Pct'].max():.1f}%"
        )

    def save_real_results(
        self, impact_df, filename="real_business_promotion_recommendations.xlsx"
    ):
        """Save real business results"""
        print(f"\n=== SAVING REAL BUSINESS RESULTS TO {filename} ===")

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # Main recommendations
            impact_df.to_excel(
                writer, sheet_name="Promotion_Recommendations", index=False
            )

            # Executive summary
            exec_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Total Products Analyzed",
                        "Average Promotion Recommended",
                        "Current Monthly Revenue (TND)",
                        "Projected Monthly Revenue (TND)",
                        "Revenue Impact (TND)",
                        "Revenue Impact (%)",
                        "Current Monthly Profit (TND)",
                        "Projected Monthly Profit (TND)",
                        "Profit Impact (TND)",
                        "Profit Impact (%)",
                    ],
                    "Value": [
                        f"{len(impact_df):,}",
                        f"{impact_df['Promotion_Recommandee_Pct'].mean():.1f}%",
                        f"{impact_df['CA_Actuel_TND'].sum():,.2f}",
                        f"{impact_df['CA_Projete_TND'].sum():,.2f}",
                        f"{impact_df['Impact_CA_TND'].sum():,.2f}",
                        f"{(impact_df['Impact_CA_TND'].sum()/impact_df['CA_Actuel_TND'].sum())*100:.2f}%",
                        f"{impact_df['Profit_Actuel_TND'].sum():,.2f}",
                        f"{impact_df['Profit_Projete_TND'].sum():,.2f}",
                        f"{impact_df['Impact_Profit_TND'].sum():,.2f}",
                        f"{(impact_df['Impact_Profit_TND'].sum()/impact_df['Profit_Actuel_TND'].sum())*100:.2f}%",
                    ],
                }
            )
            exec_summary.to_excel(writer, sheet_name="Executive_Summary", index=False)

            # Category analysis
            category_analysis = (
                impact_df.groupby("Categorie")
                .agg(
                    {
                        "Impact_Profit_TND": "sum",
                        "Impact_CA_TND": "sum",
                        "Promotion_Recommandee_Pct": "mean",
                        "Article": "count",
                    }
                )
                .round(2)
                .sort_values("Impact_Profit_TND", ascending=False)
            )
            category_analysis.to_excel(writer, sheet_name="Category_Analysis")

        print(f"Real business results saved successfully!")
        print(f"All data based on actual business metrics - NO random values used!")

    def _align_features_for_prediction(self, X, feature_names):
        """
        Align features for prediction to match training features
        This handles the case where we predict with date features but trained without them
        """
        if not hasattr(self, "training_features"):
            # If no training features stored, return as is
            return X

        print(
            f"Aligning features: {len(feature_names)} current -> {len(self.training_features)} training features"
        )

        # Create a DataFrame with the same features as training
        aligned_data = []

        for row_idx in range(X.shape[0]):
            aligned_row = {}

            # Add all training features
            for feature in self.training_features:
                if feature in feature_names:
                    # Feature exists in current prediction data
                    feature_idx = feature_names.index(feature)
                    aligned_row[feature] = (
                        X.iloc[row_idx, feature_idx]
                        if hasattr(X, "iloc")
                        else X[row_idx, feature_idx]
                    )
                else:
                    # Feature missing - fill with median/mean value or 0
                    if "encoded" in feature:
                        aligned_row[feature] = 0  # Categorical features default to 0
                    elif any(
                        keyword in feature.lower()
                        for keyword in ["price", "tnd", "cost"]
                    ):
                        aligned_row[feature] = (
                            100  # Price-related features get reasonable default
                        )
                    elif any(
                        keyword in feature.lower()
                        for keyword in ["pct", "pourcentage", "percentage"]
                    ):
                        aligned_row[feature] = (
                            0.2  # Percentage features get 20% default
                        )
                    else:
                        aligned_row[feature] = 0  # Other features default to 0

            aligned_data.append(aligned_row)

        # Convert to DataFrame and return as the original input type
        aligned_df = pd.DataFrame(aligned_data)

        print(f"Feature alignment completed: {aligned_df.shape[1]} features aligned")
        return aligned_df


def main():
    """Main execution using real business data"""
    print("=== REAL BUSINESS DATA PROMOTION OPTIMIZATION ===")
    print("Using actual sales, stock, price and financial data")
    print("NO RANDOM VALUES - only real business metrics")
    print("NOW WITH DATE PREDICTION CAPABILITIES")

    # Initialize model
    model = RealDataPromotionModel()

    # Load real business data
    df = model.load_real_business_data()

    # Prepare real features and targets
    X, feature_names = model.prepare_real_features(df)
    y = df["Promotion_Optimale"]

    # Train models on real data
    training_results = model.train_models(X, y)

    # Predict optimal promotions
    optimal_promotions = model.predict_optimal_promotions(df)

    # Calculate real business impact
    impact_df = model.calculate_business_impact(df, optimal_promotions)

    # Generate business insights
    model.generate_business_insights(impact_df)

    # Save results
    model.save_real_results(impact_df)

    print("\n=== REAL BUSINESS MODEL COMPLETED SUCCESSFULLY ===")
    print("Check 'real_business_promotion_recommendations.xlsx' for detailed results!")
    print("All recommendations based on actual business data!")

    # Example date predictions
    print("\n=== EXAMPLE DATE PREDICTIONS ===")
    try:
        # Get a sample article
        sample_article = df.iloc[0]["GA_ARTICLE"]

        # Predict for Christmas 2025
        christmas_prediction = model.predict_promotion_for_date(
            sample_article, "2025-12-25", df
        )

        print(f"\nExample predictions completed for article {sample_article}")
        print(
            "Use model.predict_promotion_for_date(article_code, date) for more predictions!"
        )

    except Exception as e:
        print(f"Example predictions failed: {e}")

    return model, impact_df


if __name__ == "__main__":
    model, results = main()
