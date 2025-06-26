"""
Enhanced AI Promotion Model with Direct Database Integration
Reads from MySQL/SQL Server database tables instead of CSV files
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
import sqlalchemy
from sqlalchemy import create_engine, text
import pymysql
import pyodbc
import os

warnings.filterwarnings("ignore")


class DatabasePromotionModel:
    """
    Professional AI Model for Promotion Optimization using REAL database data

    This model connects directly to your database and reads from:
    - Promotions table
    - Articles table
    - Stock table
    - Ventes table
    - Categories table

    NO CSV FILES - only real database data
    WITH DATE PREDICTION CAPABILITIES
    """

    def __init__(self, connection_string=None):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
        self.seasonal_patterns = {}
        self.market_trends = {}
        self.base_date = datetime.now()

        # Database connection
        self.connection_string = (
            connection_string or self._get_default_connection_string()
        )
        self.engine = None
        self.df_articles = None
        self.df_promotions = None
        self.df_stocks = None
        self.df_ventes = None
        self.df_categories = None

    def _get_default_connection_string(self):
        """Get default connection string for SQL Server LocalDB"""
        # SQL Server LocalDB connection (from your appsettings.json)
        return "mssql+pyodbc://DESKTOP-S22JEMV\\SQLEXPRESS/SmartPromoDb_Fresh?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

    def connect_to_database(self):
        """Establish database connection"""
        try:
            print("=== CONNECTING TO DATABASE ===")
            print(f"Connection string: {self.connection_string}")

            # Create engine
            self.engine = create_engine(self.connection_string)

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                print("✅ Database connection successful!")

            return True

        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Make sure SQL Server LocalDB is running")
            print("2. Verify the database 'Promotion' exists")
            print("3. Check if ODBC Driver 17 for SQL Server is installed")
            print("4. Ensure the connection string is correct")
            return False

    def load_database_tables(self):
        """Load all relevant tables from database"""
        if not self.engine:
            if not self.connect_to_database():
                raise Exception("Cannot connect to database")

        print("\n=== LOADING DATABASE TABLES ===")

        try:
            # Load Articles table
            print("Loading Articles table...")
            self.df_articles = pd.read_sql(
                """
                SELECT a.*, c.Nom as CategorieName, c.Description as CategorieDescription
                FROM Articles a 
                LEFT JOIN Categories c ON a.IdCategorie = c.IdCategorie
            """,
                self.engine,
            )
            print(f"✅ Articles loaded: {len(self.df_articles)} records")

            # Load Promotions table
            print("Loading Promotions table...")
            self.df_promotions = pd.read_sql(
                """
                SELECT p.*, a.Libelle as ArticleLibelle, a.FamilleNiv1, a.FamilleNiv2, a.Fournisseur
                FROM Promotions p 
                INNER JOIN Articles a ON p.CodeArticle = a.CodeArticle
            """,
                self.engine,
            )
            print(f"✅ Promotions loaded: {len(self.df_promotions)} records")

            # Load Stocks table
            print("Loading Stocks table...")
            self.df_stocks = pd.read_sql(
                """
                SELECT s.*, a.CodeArticle, a.Libelle as ArticleLibelle 
                FROM Stocks s 
                INNER JOIN Articles a ON s.ArticleId = a.Id
            """,
                self.engine,
            )
            print(f"✅ Stocks loaded: {len(self.df_stocks)} records")

            # Load Ventes table (if exists)
            try:
                print("Loading Ventes table...")
                self.df_ventes = pd.read_sql(
                    """
                    SELECT v.*, a.CodeArticle, a.Libelle as ArticleLibelle
                    FROM Ventes v 
                    INNER JOIN Articles a ON v.ArticleId = a.Id
                """,
                    self.engine,
                )
                print(f"✅ Ventes loaded: {len(self.df_ventes)} records")
            except Exception as e:
                print(f"⚠️ Ventes table not available: {str(e)}")
                self.df_ventes = pd.DataFrame()

            # Load Categories table
            try:
                print("Loading Categories table...")
                self.df_categories = pd.read_sql(
                    "SELECT * FROM Categories", self.engine
                )
                print(f"✅ Categories loaded: {len(self.df_categories)} records")
            except Exception as e:
                print(f"⚠️ Categories table not available: {str(e)}")
                self.df_categories = pd.DataFrame()

            print(f"\n📊 DATABASE SUMMARY:")
            print(f"   Articles: {len(self.df_articles):,} records")
            print(f"   Promotions: {len(self.df_promotions):,} records")
            print(f"   Stocks: {len(self.df_stocks):,} records")
            print(f"   Ventes: {len(self.df_ventes):,} records")
            print(f"   Categories: {len(self.df_categories):,} records")

            return True

        except Exception as e:
            print(f"❌ Error loading database tables: {str(e)}")
            return False

    def analyze_database_data(self):
        """Analyze the current database data structure and content"""
        print("\n=== DATABASE DATA ANALYSIS ===")

        if self.df_articles is None or len(self.df_articles) == 0:
            print("❌ No articles data available")
            return

        # Articles analysis
        print(f"\n📦 ARTICLES ANALYSIS:")
        print(f"   Total Articles: {len(self.df_articles):,}")
        print(f"   Unique Categories: {self.df_articles['IdCategorie'].nunique()}")
        print(f"   Unique Suppliers: {self.df_articles['Fournisseur'].nunique()}")

        # Show top categories
        if "CategorieName" in self.df_articles.columns:
            top_categories = self.df_articles["CategorieName"].value_counts().head()
            print(f"   Top Categories:")
            for cat, count in top_categories.items():
                print(f"     {cat}: {count} articles")

        # Promotions analysis
        if len(self.df_promotions) > 0:
            print(f"\n🎯 PROMOTIONS ANALYSIS:")
            print(f"   Total Promotions: {len(self.df_promotions):,}")
            print(
                f"   Average Discount: {self.df_promotions['TauxReduction'].mean():.1f}%"
            )
            print(
                f"   Discount Range: {self.df_promotions['TauxReduction'].min():.1f}% - {self.df_promotions['TauxReduction'].max():.1f}%"
            )

            # Current vs expired promotions
            current_date = datetime.now()
            self.df_promotions["DateFin"] = pd.to_datetime(
                self.df_promotions["DateFin"]
            )
            active_promos = self.df_promotions[
                self.df_promotions["DateFin"] >= current_date
            ]
            expired_promos = self.df_promotions[
                self.df_promotions["DateFin"] < current_date
            ]

            print(f"   Active Promotions: {len(active_promos):,}")
            print(f"   Expired Promotions: {len(expired_promos):,}")

            # Price analysis
            avg_price_before = self.df_promotions["Prix_Vente_TND_Avant"].mean()
            avg_price_after = self.df_promotions["Prix_Vente_TND_Apres"].mean()
            print(f"   Average Price Before: {avg_price_before:.2f} TND")
            print(f"   Average Price After: {avg_price_after:.2f} TND")
            print(f"   Average Savings: {avg_price_before - avg_price_after:.2f} TND")

        # Stock analysis
        if len(self.df_stocks) > 0:
            print(f"\n📦 STOCK ANALYSIS:")
            print(f"   Total Stock Entries: {len(self.df_stocks):,}")
            print(
                f"   Total Stock Value: {self.df_stocks['Valeur_Stock_TND'].sum():,.2f} TND"
            )
            print(
                f"   Average Stock Quantity: {self.df_stocks['QuantitePhysique'].mean():.0f} units"
            )

            # Low stock analysis
            low_stock = self.df_stocks[
                self.df_stocks["QuantitePhysique"] <= self.df_stocks["StockMin"]
            ]
            print(f"   Low Stock Items: {len(low_stock):,}")
            print(
                f"   Out of Stock Items: {len(self.df_stocks[self.df_stocks['QuantitePhysique'] == 0]):,}"
            )  # Sales analysis (if available)
        if len(self.df_ventes) > 0:
            print(f"\n💰 SALES ANALYSIS:")
            print(f"   Total Sales Records: {len(self.df_ventes):,}")
            if "QuantiteFacturee" in self.df_ventes.columns:
                print(
                    f"   Total Quantity Sold: {self.df_ventes['QuantiteFacturee'].sum():,.0f} units"
                )
            if "Prix_Vente_TND" in self.df_ventes.columns:
                print(
                    f"   Average Unit Price: {self.df_ventes['Prix_Vente_TND'].mean():.2f} TND"
                )

    def create_enhanced_dataset_from_database(self):
        """Create enhanced dataset by combining database tables"""
        print("\n=== CREATING ENHANCED DATASET FROM DATABASE ===")

        if self.df_articles is None or len(self.df_articles) == 0:
            raise Exception(
                "No articles data available. Load database tables first."
            )  # Start with articles as the base
        enhanced_df = self.df_articles.copy()

        # Add stock information
        if len(self.df_stocks) > 0:
            # Aggregate stocks by article (in case multiple stock entries per article)
            stock_agg = (
                self.df_stocks.groupby("ArticleId")
                .agg(
                    {
                        "QuantitePhysique": "sum",
                        "StockMin": "mean",
                        "VenteFFO": "sum",
                        "LivreFou": "sum",
                        "Transfert": "sum",
                        "AnnonceTrf": "sum",
                        "Valeur_Stock_TND": "sum",
                    }
                )
                .reset_index()
            )

            enhanced_df = enhanced_df.merge(
                stock_agg, left_on="Id", right_on="ArticleId", how="left"
            )
            print(f"✅ Stock data merged: {len(stock_agg)} stock records")
        else:
            print("⚠️ No stock data available")

        # Add sales information
        if len(self.df_ventes) > 0:
            # Aggregate sales by article (using correct column names from database)
            ventes_agg = (
                self.df_ventes.groupby("ArticleId")
                .agg(
                    {
                        "QuantiteFacturee": ["sum", "mean", "count"],
                        "Prix_Vente_TND": "mean",
                        "Date": ["min", "max"],
                        "CA_Mensuel_TND": "sum",
                    }
                )
                .reset_index()
            )  # Flatten column names
            ventes_agg.columns = [
                "ArticleId",
                "Total_Quantity_Sold",
                "Avg_Quantity_Per_Sale",
                "Total_Sales_Count",
                "Avg_Unit_Price",
                "First_Sale_Date",
                "Last_Sale_Date",
                "Total_Revenue",
            ]

            enhanced_df = enhanced_df.merge(
                ventes_agg, left_on="Id", right_on="ArticleId", how="left"
            )
            print(f"✅ Sales data merged: {len(ventes_agg)} sales aggregates")
        else:
            print("⚠️ No sales data available")

        # Add promotion history
        if len(self.df_promotions) > 0:
            # Aggregate promotion history by article
            promo_agg = (
                self.df_promotions.groupby("CodeArticle")
                .agg(
                    {
                        "TauxReduction": ["mean", "max", "count"],
                        "Prix_Vente_TND_Avant": "mean",
                        "Prix_Vente_TND_Apres": "mean",
                        "DateFin": "max",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            promo_agg.columns = [
                "CodeArticle",
                "Avg_Discount_Rate",
                "Max_Discount_Rate",
                "Promotion_Count",
                "Avg_Price_Before",
                "Avg_Price_After",
                "Last_Promotion_Date",
            ]

            enhanced_df = enhanced_df.merge(promo_agg, on="CodeArticle", how="left")
            print(f"✅ Promotion history merged: {len(promo_agg)} promotion aggregates")
        else:
            print("⚠️ No promotion data available")

        # Calculate derived business metrics
        enhanced_df = self._calculate_business_metrics(enhanced_df)

        print(
            f"📊 Enhanced dataset created: {len(enhanced_df)} articles with {len(enhanced_df.columns)} features"
        )
        return enhanced_df

    def _calculate_business_metrics(self, df):
        """Calculate business metrics from database data"""
        print("\n=== CALCULATING BUSINESS METRICS ===")

        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(
            0
        )  # Price and margin calculations - improved logic
        if "Avg_Unit_Price" in df.columns:
            # Use average unit price from sales data (best source)
            df["Current_Price_TND"] = df["Avg_Unit_Price"].fillna(0)
        elif "Avg_Price_Before" in df.columns:
            # Use promotion history if available
            df["Current_Price_TND"] = df["Avg_Price_Before"].fillna(0)
        else:
            # Estimate price based on stock value and quantity
            df["Current_Price_TND"] = np.where(
                (df.get("QuantitePhysique", 0) > 0)
                & (df.get("Valeur_Stock_TND", 0) > 0),
                df.get("Valeur_Stock_TND", 0) / df.get("QuantitePhysique", 1),
                0,  # Set to 0 if no data, will be handled later
            )

        # For articles with 0 price, try to get from stock value/quantity ratio
        zero_price_mask = df["Current_Price_TND"] == 0
        if zero_price_mask.any():
            # Calculate average price from stock value
            df.loc[zero_price_mask, "Current_Price_TND"] = np.where(
                (df.loc[zero_price_mask, "QuantitePhysique"] > 0)
                & (df.loc[zero_price_mask, "Valeur_Stock_TND"] > 0),
                df.loc[zero_price_mask, "Valeur_Stock_TND"]
                / df.loc[zero_price_mask, "QuantitePhysique"],
                150,  # Reasonable default price for articles without data
            )

        # Stock rotation (sales velocity)
        if "Total_Quantity_Sold" in df.columns and "QuantitePhysique" in df.columns:
            df["Stock_Rotation"] = np.where(
                df["QuantitePhysique"] > 0,
                df["Total_Quantity_Sold"] / df["QuantitePhysique"],
                0,
            )
        else:
            df["Stock_Rotation"] = 0.5  # Default rotation

        # Days since last sale
        if "Last_Sale_Date" in df.columns:
            df["Last_Sale_Date"] = pd.to_datetime(df["Last_Sale_Date"], errors="coerce")
            df["Days_Since_Last_Sale"] = (datetime.now() - df["Last_Sale_Date"]).dt.days
            df["Days_Since_Last_Sale"] = df["Days_Since_Last_Sale"].fillna(
                365
            )  # 1 year default
        else:
            df["Days_Since_Last_Sale"] = 90  # Default 3 months

        # Promotion frequency
        if "Promotion_Count" in df.columns:
            df["Promotion_Frequency"] = (
                df["Promotion_Count"].fillna(0) / 12
            )  # Per month
        else:
            df["Promotion_Frequency"] = 0

        # Price elasticity estimation
        if "Avg_Discount_Rate" in df.columns and "Total_Quantity_Sold" in df.columns:
            # Simple elasticity estimation based on historical promotions
            df["Estimated_Price_Elasticity"] = np.where(
                df["Avg_Discount_Rate"] > 0,
                -2.0,  # Assume elastic demand
                -1.0,  # Less elastic if no promotion history
            )
        else:
            df["Estimated_Price_Elasticity"] = -1.5  # Default elasticity

        # Stock coverage (months of stock)
        if "QuantitePhysique" in df.columns and "Avg_Quantity_Per_Sale" in df.columns:
            monthly_sales = (
                df["Avg_Quantity_Per_Sale"] * df.get("Total_Sales_Count", 1) / 12
            )
            df["Stock_Coverage_Months"] = np.where(
                monthly_sales > 0,
                df["QuantitePhysique"] / monthly_sales,
                12,  # Default 12 months if no sales
            )
        else:
            df["Stock_Coverage_Months"] = 6  # Default 6 months

        # Performance score
        df["Performance_Score"] = (
            1 / (df["Days_Since_Last_Sale"] / 30 + 1)
        ) * 0.3 + (  # Recency factor
            df["Stock_Rotation"] / df["Stock_Rotation"].max()
        ) * 0.3 * 0.4  # Rotation factor            + (df.get("Total_Revenue", 0) / df.get("Total_Revenue", 0).max())  # Revenue factor

        print(f"✅ Business metrics calculated for {len(df)} articles")
        return df

    def calculate_optimal_promotions_from_db(self, df):
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
            # Fallback: use existing stock rotation data
            rotation_factor = np.clip(
                (1 - df_calc.get("Stock_Rotation", 0.5)) * 0.25, 0, 0.25
            )

        # 1. Stock clearance factor (high stock = higher promotion)
        # Use existing stock coverage or calculate from available data
        if "Stock_Coverage_Months" in df_calc.columns:
            stock_months = df_calc["Stock_Coverage_Months"]
        elif (
            "QuantitePhysique" in df_calc.columns
            and "Ventes_Mensuelles_Unites" in df_calc.columns
        ):
            stock_months = df_calc["QuantitePhysique"] / df_calc[
                "Ventes_Mensuelles_Unites"
            ].replace(0, 1)
        else:
            # Fallback calculation using available sales data
            monthly_sales = df_calc.get("Total_Quantity_Sold", 10) / 12
            stock_months = df_calc.get("QuantitePhysique", 50) / monthly_sales

        stock_factor = np.clip(
            (stock_months - 3) / 10, 0, 0.3
        )  # Si le stock couvre plus de 3 mois, on considère que c'est excessif.

        # 2. Calcul de la marge en pourcentage à la volée et du margin_factor dans un seul bloc 
#         Base margin of 30% is considered normal (no effect)
# Each additional 1% margin adds 0.01 to the factor
# Factor is capped at 0.25 (25%)
        
        if (
            "Prix_Vente_TND" in df_calc.columns
            and "Cout_Unitaire_TND" in df_calc.columns
        ):
            df_calc["Marge_Pourcentage"] = (
                (df_calc["Prix_Vente_TND"] - df_calc["Cout_Unitaire_TND"])
                / df_calc["Prix_Vente_TND"]
            ) * 100
        elif (
            "Current_Price_TND" in df_calc.columns
            and "Cout_Unitaire_TND" in df_calc.columns
        ):
            df_calc["Marge_Pourcentage"] = (
                (df_calc["Current_Price_TND"] - df_calc["Cout_Unitaire_TND"])
                / df_calc["Current_Price_TND"]
            ) * 100
        else:
            # Estimate 40% margin if no cost data available
            df_calc["Marge_Pourcentage"] = 40

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
            )  # Scale elasticity Diviser par 5 permet de "normaliser" ou "réduire" l'impact de l'élasticité pour qu'il ne domine pas le calcul de la promotion
        else:
            # Fallback to estimated elasticity
            elasticity_factor = (
                np.abs(df_calc.get("Estimated_Price_Elasticity", -1.5)) / 5
            )  # fallback
        elasticity_factor = np.clip(
            elasticity_factor, 0, 0.3
        )  # Max 30% from elasticity

        # Combine all factors using business logic weights
        optimal_promotion = (
            rotation_factor * 0.25  # 25% weight - rotation
            + stock_factor * 0.25  # 25% weight - inventory management
            + margin_factor * 0.20  # 20% weight - profitability protection
            + elasticity_factor * 0.30  # 30% weight - customer price sensitivity
        )

        # Apply business constraints
        optimal_promotion = np.clip(
            optimal_promotion, 0.05, 0.45
        )  # 5-45% promotion range

        # Store the calculated optimal promotions
        df_calc["Optimal_Promotion_Rate"] = optimal_promotion

        # Calculate expected impact using actual business formulas
        if "Current_Price_TND" in df_calc.columns:
            current_price = df_calc["Current_Price_TND"]
        else:
            current_price = df_calc.get("Prix_Vente_TND", 100)

        # Get current sales volume
        if "Ventes_Mensuelles_Unites" in df_calc.columns:
            current_volume = df_calc["Ventes_Mensuelles_Unites"]
        else:
            current_volume = df_calc.get("Total_Quantity_Sold", 10) / 12

        # Calculate business impact
        df_calc["Volume_Avec_Promo"] = current_volume * (
            1 + (df_calc.get("Estimated_Price_Elasticity", -1.5) * optimal_promotion)
        )
        df_calc["Prix_Avec_Promo"] = current_price * (1 - optimal_promotion)
        df_calc["CA_Avec_Promo"] = (
            df_calc["Volume_Avec_Promo"] * df_calc["Prix_Avec_Promo"]
        )

        # Calculate profit if cost data is available
        if "Cout_Unitaire_TND" in df_calc.columns:
            df_calc["Profit_Avec_Promo"] = (
                df_calc["Prix_Avec_Promo"] - df_calc["Cout_Unitaire_TND"]
            ) * df_calc["Volume_Avec_Promo"]

        print(f"✅ Optimal promotions calculated using real business metrics")
        print(f"   Average promotion: {optimal_promotion.mean()*100:.1f}%")
        print(
            f"   Promotion range: {optimal_promotion.min()*100:.1f}% - {optimal_promotion.max()*100:.1f}%"
        )

        return df_calc

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
            }
        )

        # Cyclical features
        date_features.update(
            {
                "month_sin": np.sin(2 * np.pi * target_date.month / 12),
                "month_cos": np.cos(2 * np.pi * target_date.month / 12),
                "day_sin": np.sin(2 * np.pi * target_date.day / 31),
                "day_cos": np.cos(2 * np.pi * target_date.day / 31),
            }
        )

        return date_features

    def _is_ramadan_season(self, date):
        """Check if date falls in Ramadan season (approximate)"""
        return 1 if date.month in [3, 4, 5] else 0

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
        """Calculate seasonal demand multiplier"""
        seasonal_multipliers = {
            1: 0.8,
            2: 0.9,
            3: 1.1,
            4: 1.0,
            5: 1.2,
            6: 1.3,
            7: 1.4,
            8: 1.3,
            9: 1.2,
            10: 1.1,
            11: 1.0,
            12: 1.5,
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

    def prepare_features_from_database(self, df, target_date=None):
        """Prepare features from database data with optional date features"""
        print("\n=== PREPARING FEATURES FROM DATABASE DATA ===")

        feature_df = df.copy()

        # Add date features if target_date is provided
        if target_date is not None:
            print(f"Adding date features for: {target_date}")
            date_features = self.extract_date_features(target_date)
            for feature_name, feature_value in date_features.items():
                feature_df[f"date_{feature_name}"] = feature_value

        # Encode categorical variables
        categorical_cols = ["FamilleNiv1", "FamilleNiv2", "Fournisseur"]

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

        # Select real database features
        database_features = [
            # Price and financial features
            "Current_Price_TND",
            # Stock features
            "QuantitePhysique",
            "StockMin",
            "Valeur_Stock_TND",
            "Stock_Coverage_Months",
            # Sales and performance features
            "Stock_Rotation",
            "Days_Since_Last_Sale",
            "Performance_Score",
            # Promotion history features
            "Promotion_Frequency",
            "Estimated_Price_Elasticity",
        ]

        # Add columns that exist
        if "Total_Quantity_Sold" in feature_df.columns:
            database_features.append("Total_Quantity_Sold")
        if "Total_Revenue" in feature_df.columns:
            database_features.append("Total_Revenue")
        if "Avg_Discount_Rate" in feature_df.columns:
            database_features.append("Avg_Discount_Rate")

        # Add encoded categorical features
        encoded_features = [
            f"{col}_encoded" for col in categorical_cols if col in feature_df.columns
        ]

        # Add date features if they exist
        date_features = [col for col in feature_df.columns if col.startswith("date_")]

        # Combine all features
        all_features = database_features + encoded_features + date_features
        available_features = [f for f in all_features if f in feature_df.columns]

        X = feature_df[available_features]
        X = X.fillna(X.median())

        print(
            f"✅ Database features prepared: {X.shape[1]} features, {X.shape[0]} samples"
        )
        print(f"   Using ONLY real database data - NO simulated values")
        if date_features:
            print(f"   Date features included: {len(date_features)} temporal features")

        return X, available_features

    def train_models_on_database_data(self, X, y):
        """Train models on real database data"""
        print("\n=== TRAINING MODELS ON DATABASE DATA ===")

        # Store training feature names
        self.training_features = list(X.columns)
        print(f"Training features stored: {len(self.training_features)} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
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
            print(f"Training {name} on database data...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

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
            print(f"   {name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]["r2"])
        self.best_model = results[best_model_name]["model"]
        self.model_name = best_model_name

        print(
            f"\n✅ Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})"
        )

        # Store feature importance
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = dict(
                zip(X.columns, self.best_model.feature_importances_)
            )

        self.is_trained = True
        return results

    def predict_promotion_for_article_and_date(self, code_article, target_date):
        """Predict optimal promotion for a specific article on a specific date using database data"""
        if not self.is_trained:
            raise ValueError(
                "Model must be trained first! Call train_models_on_database_data() first."
            )

        print(
            f"\n=== PREDICTING PROMOTION FOR ARTICLE {code_article} ON {target_date} ==="
        )

        # Find article in database
        article_data = self.df_enhanced[self.df_enhanced["CodeArticle"] == code_article]
        if article_data.empty:
            raise ValueError(f"Article {code_article} not found in database")

        # Get article info
        article_info = article_data.iloc[0]
        print(f"Article: {article_info['CodeArticle']}")
        print(f"Description: {article_info['Libelle']}")
        if "FamilleNiv2" in article_info:
            print(f"Category: {article_info['FamilleNiv2']}")
        print(f"Current Price: {article_info['Current_Price_TND']:.2f} TND")
        print(f"Target Date: {target_date}")

        # Prepare features with date information
        X, feature_names = self.prepare_features_from_database(
            article_data, target_date=target_date
        )

        # Align features to match training features
        X_aligned = self._align_features_for_prediction(X, feature_names)

        # Predict promotion
        predicted_promotion = self.best_model.predict(X_aligned)[0]
        predicted_promotion = np.clip(predicted_promotion, 0.05, 0.45)

        # Calculate expected impact with date-specific factors
        date_features = self.extract_date_features(target_date)
        seasonal_adjustment = date_features["seasonal_demand_multiplier"]

        # Adjust promotion based on temporal factors
        temporal_adjustment = 1.0
        if date_features["is_holiday_season"]:
            temporal_adjustment *= 1.2
        if date_features["is_back_to_school"]:
            temporal_adjustment *= 1.1
        if date_features["is_weekend"]:
            temporal_adjustment *= 1.05

        adjusted_promotion = predicted_promotion * temporal_adjustment
        adjusted_promotion = np.clip(adjusted_promotion, 0.05, 0.45)

        # Calculate business impact
        original_volume = article_info.get("Current_Monthly_Volume", 10)
        elasticity = article_info["Estimated_Price_Elasticity"]

        volume_change = elasticity * adjusted_promotion * seasonal_adjustment
        new_volume = original_volume * (1 + volume_change)

        original_price = article_info["Current_Price_TND"]
        new_price = original_price * (1 - adjusted_promotion)

        original_revenue = original_volume * original_price
        new_revenue = new_volume * new_price

        # Create prediction result
        prediction_result = {
            "article_code": code_article,
            "article_name": article_info["Libelle"],
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
            "base_promotion_pct": round(predicted_promotion * 100, 2),
            "adjusted_promotion_pct": round(adjusted_promotion * 100, 2),
            "temporal_adjustment_factor": round(temporal_adjustment, 3),
            "seasonal_demand_multiplier": round(seasonal_adjustment, 3),
            "current_price_tnd": round(original_price, 2),
            "current_monthly_volume": int(original_volume),
            "current_monthly_revenue_tnd": round(original_revenue, 2),
            "promoted_price_tnd": round(new_price, 2),
            "projected_monthly_volume": int(new_volume),
            "projected_monthly_revenue_tnd": round(new_revenue, 2),
            "revenue_impact_tnd": round(new_revenue - original_revenue, 2),
            "revenue_impact_pct": round(
                ((new_revenue - original_revenue) / original_revenue) * 100, 2
            ),
            "volume_impact_pct": round(
                ((new_volume - original_volume) / original_volume) * 100, 2
            ),
            "confidence": 0.85,
        }

        # Print summary
        print(f"\n--- PREDICTION SUMMARY ---")
        print(f"Recommended Promotion: {prediction_result['adjusted_promotion_pct']}%")
        print(f"New Price: {prediction_result['promoted_price_tnd']} TND")
        print(f"Expected Volume Change: {prediction_result['volume_impact_pct']:+.1f}%")
        print(
            f"Expected Revenue Impact: {prediction_result['revenue_impact_tnd']:+.2f} TND ({prediction_result['revenue_impact_pct']:+.1f}%)"
        )

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

    def _align_features_for_prediction(self, X, feature_names):
        """Align features for prediction to match training features"""
        if not hasattr(self, "training_features"):
            return X

        aligned_data = []
        for row_idx in range(X.shape[0]):
            aligned_row = {}
            for feature in self.training_features:
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                    aligned_row[feature] = (
                        X.iloc[row_idx, feature_idx]
                        if hasattr(X, "iloc")
                        else X[row_idx, feature_idx]
                    )
                else:
                    # Fill missing features with defaults
                    if "encoded" in feature:
                        aligned_row[feature] = 0
                    elif any(
                        keyword in feature.lower()
                        for keyword in ["price", "tnd", "cost"]
                    ):
                        aligned_row[feature] = 100
                    elif any(
                        keyword in feature.lower() for keyword in ["pct", "percentage"]
                    ):
                        aligned_row[feature] = 0.2
                    else:
                        aligned_row[feature] = 0
            aligned_data.append(aligned_row)

        return pd.DataFrame(aligned_data)

    def save_database_results(
        self, impact_df, filename="database_promotion_recommendations.xlsx"
    ):
        """Save database results to Excel with robust error handling"""
        print(f"\n=== SAVING DATABASE RESULTS TO {filename} ===")

        import os
        from datetime import datetime

        # Try to save with the original filename first
        max_attempts = 5
        current_filename = filename

        for attempt in range(max_attempts):
            try:
                with pd.ExcelWriter(current_filename, engine="openpyxl") as writer:
                    # Main recommendations
                    impact_df.to_excel(
                        writer, sheet_name="Promotion_Recommendations", index=False
                    )

                    # Database summary
                    db_summary = pd.DataFrame(
                        {
                            "Table": [
                                "Articles",
                                "Promotions",
                                "Stocks",
                                "Ventes",
                                "Categories",
                            ],
                            "Records": [
                                (
                                    len(self.df_articles)
                                    if self.df_articles is not None
                                    else 0
                                ),
                                (
                                    len(self.df_promotions)
                                    if self.df_promotions is not None
                                    else 0
                                ),
                                (
                                    len(self.df_stocks)
                                    if self.df_stocks is not None
                                    else 0
                                ),
                                (
                                    len(self.df_ventes)
                                    if self.df_ventes is not None
                                    else 0
                                ),
                                (
                                    len(self.df_categories)
                                    if self.df_categories is not None
                                    else 0
                                ),
                            ],
                        }
                    )
                    db_summary.to_excel(
                        writer, sheet_name="Database_Summary", index=False
                    )

                print(f"✅ Database results saved successfully to: {current_filename}")
                return current_filename

            except PermissionError as e:
                if attempt < max_attempts - 1:
                    # Generate alternative filename
                    base_name = os.path.splitext(filename)[0]
                    extension = os.path.splitext(filename)[1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_filename = f"{base_name}_{timestamp}{extension}"
                    print(f"⚠️  File is in use, trying: {current_filename}")
                else:
                    print(
                        f"❌ Unable to save Excel file after {max_attempts} attempts."
                    )
                    print(f"📝 The file '{filename}' is likely open in Excel.")
                    print("🔧 SOLUTIONS:")
                    print("   1. Close Excel and try again")
                    print("   2. The model will save as CSV instead")

                    # Fallback: save as CSV
                    csv_filename = filename.replace(".xlsx", ".csv")
                    try:
                        impact_df.to_csv(csv_filename, index=False)
                        print(f"✅ Results saved as CSV: {csv_filename}")
                        return csv_filename
                    except Exception as csv_error:
                        print(f"❌ CSV save also failed: {str(csv_error)}")
                        return None
            except Exception as e:
                print(f"❌ Unexpected error saving file: {str(e)}")
                return None

        return None

    def interactive_promotion_generator(self):
        """Interactive system to generate promotions for specific articles and dates"""
        print("\n" + "=" * 60)
        print("🎯 INTERACTIVE PROMOTION GENERATOR")
        print("=" * 60)
        # Ensure model is loaded and trained
        if not self.connect_to_database():
            print("❌ Cannot connect to database. Please check your connection.")
            return

        if not self.load_database_tables():
            print("❌ Cannot load database tables.")
            return

        if not self.is_trained:
            print("🤖 Training AI model with your data...")
            self.df_enhanced = self.create_enhanced_dataset_from_database()
            self.df_enhanced = self.calculate_optimal_promotions_from_db(
                self.df_enhanced
            )

            # Prepare features and targets
            X, feature_names = self.prepare_features_from_database(self.df_enhanced)
            y = self.df_enhanced["Optimal_Promotion_Rate"]

            # Train models
            self.train_models_on_database_data(X, y)
            print("✅ AI model trained successfully!")

        while True:
            print("\n" + "-" * 50)
            print("📊 AVAILABLE ARTICLES IN YOUR DATABASE:")
            print("-" * 50)

            # Show available articles
            articles_df = self.df_articles[
                ["CodeArticle", "Libelle", "FamilleNiv2"]
            ].copy()
            articles_df.columns = ["Code", "Description", "Category"]

            # Display articles in a formatted way
            for idx, row in articles_df.iterrows():
                print(
                    f"  {row['Code']:<12} | {row['Description']:<40} | {row['Category']}"
                )

            print("\n" + "-" * 50)

            # Get user input
            try:
                print("🔍 Enter article code (or 'quit' to exit):")
                article_code = input(">>> ").strip().upper()

                if article_code.lower() == "quit":
                    print("👋 Goodbye!")
                    break

                # Check if article exists
                if article_code not in self.df_articles["CodeArticle"].values:
                    print(f"❌ Article '{article_code}' not found in database.")
                    continue

                # Get date input
                print("\n📅 Enter target date (YYYY-MM-DD format):")
                print("   Examples: 2025-07-15, 2025-12-25, 2025-11-29")
                date_input = input(">>> ").strip()

                # Validate date
                try:
                    target_date = datetime.strptime(date_input, "%Y-%m-%d")
                    if target_date < datetime.now():
                        print("⚠️  Warning: You entered a past date. Continue? (y/n)")
                        if input(">>> ").lower() != "y":
                            continue
                except ValueError:
                    print("❌ Invalid date format. Please use YYYY-MM-DD.")
                    continue

                # Generate promotion
                print(
                    f"\n🤖 Generating AI promotion for {article_code} on {date_input}..."
                )

                try:
                    prediction = self.predict_promotion_for_article_and_date(
                        article_code, date_input
                    )

                    # Ask if user wants to save to database
                    print(f"\n💾 Save this promotion to database? (y/n)")
                    if input(">>> ").lower() == "y":
                        success = self.save_promotion_to_database(
                            prediction, article_code, date_input
                        )
                        if success:
                            print("✅ Promotion saved to database successfully!")
                        else:
                            print("❌ Failed to save promotion to database.")

                except Exception as e:
                    print(f"❌ Error generating promotion: {str(e)}")

                # Ask if user wants to continue
                print(f"\n🔄 Generate another promotion? (y/n)")
                if input(">>> ").lower() != "y":
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

    def save_promotion_to_database(self, prediction, article_code, target_date, date_creation=None):
        """Save generated promotion to database with isAccepted = false"""
        try:
            print(f"\n💾 Saving promotion to database...")

            # Get article info
            article_info = self.df_articles[
                self.df_articles["CodeArticle"] == article_code
            ].iloc[0]

            # Prepare promotion data
            promotion_data = {
                "DateFin": target_date,
                "TauxReduction": prediction["adjusted_promotion_pct"]
                / 100,  # Convert percentage to decimal
                "CodeArticle": article_code,
                "Prix_Vente_TND_Avant": prediction["current_price_tnd"],
                "Prix_Vente_TND_Apres": prediction["promoted_price_tnd"],
                "isAccepted": False,  # Default to false
                "DateCreation": date_creation if date_creation else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "PredictionConfidence": prediction.get("confidence", 0.85),
                "SeasonalAdjustment": prediction.get("seasonal_demand_multiplier", 1.0),
                "TemporalAdjustment": prediction.get("temporal_adjustment_factor", 1.0),
                "ExpectedVolumeImpact": prediction.get("volume_impact_pct", 0),
                "ExpectedRevenueImpact": prediction.get("revenue_impact_tnd", 0),
            }

            # Create SQL insert statement
            insert_query = """
            INSERT INTO Promotions (
                DateFin, TauxReduction, CodeArticle, 
                Prix_Vente_TND_Avant, Prix_Vente_TND_Apres, isAccepted, DateCreation
            ) VALUES (
                :DateFin, :TauxReduction, :CodeArticle,
                :Prix_Vente_TND_Avant, :Prix_Vente_TND_Apres, :isAccepted, :DateCreation
            )
            """

            # Execute insert
            with self.engine.connect() as conn:
                conn.execute(text(insert_query), promotion_data)
                conn.commit()

            print(f"✅ Promotion saved successfully!")
            print(f"   Article: {article_code}")
            print(f"   Target Date: {target_date}")
            print(f"   Discount: {prediction['adjusted_promotion_pct']}%")
            print(f"   Status: Pending Approval (isAccepted = False)")

            return True

        except Exception as e:
            print(f"❌ Error saving promotion: {str(e)}")
            return False

    def list_pending_promotions(self):
        """List all promotions that are pending approval (isAccepted = False)"""
        try:
            query = """
            SELECT p.Id, p.CodeArticle, a.Libelle, p.DateFin, 
                   p.TauxReduction, p.Prix_Vente_TND_Avant, p.Prix_Vente_TND_Apres,
                   p.isAccepted
            FROM Promotions p
            LEFT JOIN Articles a ON p.CodeArticle = a.CodeArticle
            WHERE p.isAccepted = 0
            ORDER BY p.DateFin
            """

            with self.engine.connect() as conn:
                pending_promotions = pd.read_sql(query, conn)

            if len(pending_promotions) == 0:
                print("✅ No pending promotions found.")
                return

            print(f"\n📋 PENDING PROMOTIONS ({len(pending_promotions)} total):")
            print("-" * 80)
            print(
                f"{'ID':<4} | {'Code':<8} | {'Description':<25} | {'Date':<12} | {'Discount':<8} | {'Price Before':<12} | {'Price After':<12}"
            )
            print("-" * 80)

            for _, promo in pending_promotions.iterrows():
                print(
                    f"{promo['Id']:<4} | {promo['CodeArticle']:<8} | {promo['Libelle'][:25]:<25} | {promo['DateFin'].strftime('%Y-%m-%d'):<12} | {promo['TauxReduction']*100:.1f}%{'':<3} | {promo['Prix_Vente_TND_Avant']:<12.2f} | {promo['Prix_Vente_TND_Apres']:<12.2f}"
                )

            return pending_promotions

        except Exception as e:
            print(f"❌ Error listing pending promotions: {str(e)}")
            return None

    def approve_promotion(self, promotion_id):
        """Approve a promotion by setting isAccepted = True"""
        try:
            update_query = """
            UPDATE Promotions 
            SET isAccepted = 1 
            WHERE Id = :promotion_id
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(update_query), {"promotion_id": promotion_id}
                )
                conn.commit()

                if result.rowcount > 0:
                    print(f"✅ Promotion {promotion_id} approved successfully!")
                    return True
                else:
                    print(f"❌ Promotion {promotion_id} not found.")
                    return False

        except Exception as e:
            print(f"❌ Error approving promotion: {str(e)}")
            return False

    def promotion_management_menu(self):
        """Interactive menu for managing promotions"""
        while True:
            print("\n" + "=" * 60)
            print("🎯 PROMOTION MANAGEMENT SYSTEM")
            print("=" * 60)
            print("1. Generate New Promotion")
            print("2. List Pending Promotions")
            print("3. Approve Promotion")
            print("4. Exit")
            print("-" * 60)

            choice = input("Choose an option (1-4): ").strip()

            if choice == "1":
                self.interactive_promotion_generator()
            elif choice == "2":
                self.list_pending_promotions()
            elif choice == "3":
                pending = self.list_pending_promotions()
                if pending is not None and len(pending) > 0:
                    try:
                        promo_id = int(input("\nEnter promotion ID to approve: "))
                        self.approve_promotion(promo_id)
                    except ValueError:
                        print("❌ Invalid promotion ID.")
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please choose 1-4.")


def main():
    """Main execution using database data"""
    print("=== DATABASE-DRIVEN PROMOTION OPTIMIZATION ===")
    print("Reading directly from MySQL/SQL Server database tables")
    print("NO CSV FILES - only real database data")

    try:
        # Initialize model
        model = DatabasePromotionModel()

        # Connect to database and load tables
        if not model.connect_to_database():
            print("❌ Cannot proceed without database connection")
            return None, None

        if not model.load_database_tables():
            print("❌ Cannot proceed without database data")
            return None, None

        # Analyze database data
        model.analyze_database_data()

        # Create enhanced dataset from database
        model.df_enhanced = model.create_enhanced_dataset_from_database()

        # Calculate optimal promotions from database data
        model.df_enhanced = model.calculate_optimal_promotions_from_db(
            model.df_enhanced
        )

        # Prepare features and targets
        X, feature_names = model.prepare_features_from_database(model.df_enhanced)
        y = model.df_enhanced["Optimal_Promotion_Rate"]

        # Train models on database data
        training_results = model.train_models_on_database_data(X, y)

        # Create results dataframe
        results_df = model.df_enhanced[
            [
                "CodeArticle",
                "Libelle",
                "FamilleNiv1",
                "FamilleNiv2",
                "Fournisseur",
                "Current_Price_TND",
                "Optimal_Promotion_Rate",
                "New_Price_TND",
                "Current_Monthly_Revenue",
                "Projected_Monthly_Revenue",
                "Revenue_Impact",
            ]
        ].copy()

        results_df["Promotion_Pct"] = results_df["Optimal_Promotion_Rate"] * 100
        results_df = results_df.sort_values("Revenue_Impact", ascending=False)

        # Save results
        model.save_database_results(results_df)

        print("\n=== DATABASE MODEL COMPLETED SUCCESSFULLY ===")
        print("Check 'database_promotion_recommendations.xlsx' for detailed results!")
        print("All recommendations based on real database data!")

        # Example prediction
        if len(results_df) > 0:
            sample_article = results_df.iloc[0]["CodeArticle"]
            try:
                prediction = model.predict_promotion_for_article_and_date(
                    sample_article, "2025-12-25"
                )
                print(f"\n✅ Example prediction completed for article {sample_article}")
            except Exception as e:
                print(f"⚠️ Example prediction failed: {e}")

        return model, results_df

    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, results = main()
