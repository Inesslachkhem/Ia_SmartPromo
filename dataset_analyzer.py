"""
SmartPromo Dataset Analyzer
Analyzes the generated datasets and provides insights for the AI promotion model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")


class SmartPromoDatasetAnalyzer:
    """
    Analyzes SmartPromo datasets and generates insights for AI model training
    """

    def __init__(self, data_dir="datasets_csv"):
        self.data_dir = data_dir
        self.datasets = {}
        self.insights = {}

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def load_datasets(self):
        """Load all CSV datasets"""
        dataset_files = {
            "categories": "categories.csv",
            "articles": "articles.csv",
            "etablissements": "etablissements.csv",
            "users": "users.csv",
            "stocks": "stocks.csv",
            "promotions": "promotions.csv",
            "ventes": "ventes.csv",
            "stock_etablissements": "stocketablissements.csv",
        }

        print("📂 Loading datasets...")
        for name, filename in dataset_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                self.datasets[name] = pd.read_csv(filepath)
                print(f"✅ Loaded {name}: {len(self.datasets[name])} records")
            else:
                print(f"⚠️  File not found: {filepath}")

    def analyze_sales_patterns(self):
        """Analyze sales patterns and trends"""
        if "ventes" not in self.datasets:
            print("❌ Ventes dataset not found")
            return

        ventes = self.datasets["ventes"].copy()
        ventes["Date"] = pd.to_datetime(ventes["Date"])
        ventes["Month"] = ventes["Date"].dt.month
        ventes["DayOfWeek"] = ventes["Date"].dt.dayofweek
        ventes["IsWeekend"] = ventes["DayOfWeek"].isin([5, 6])

        insights = {}

        # Monthly sales analysis
        monthly_sales = (
            ventes.groupby("Month")
            .agg(
                {
                    "QuantiteFacturee": "sum",
                    "CA_Mensuel_TND": "mean",
                    "Profit_Mensuel_TND": "mean",
                }
            )
            .round(2)
        )

        insights["monthly_patterns"] = {
            "best_month": monthly_sales["QuantiteFacturee"].idxmax(),
            "worst_month": monthly_sales["QuantiteFacturee"].idxmin(),
            "avg_monthly_revenue": monthly_sales["CA_Mensuel_TND"].mean(),
            "avg_monthly_profit": monthly_sales["Profit_Mensuel_TND"].mean(),
        }

        # Day of week analysis
        dow_sales = ventes.groupby("DayOfWeek")["QuantiteFacturee"].sum()
        insights["day_patterns"] = {
            "best_day": dow_sales.idxmax(),
            "worst_day": dow_sales.idxmin(),
            "weekend_vs_weekday": ventes.groupby("IsWeekend")["QuantiteFacturee"]
            .mean()
            .to_dict(),
        }

        # Top performing articles
        article_performance = (
            ventes.groupby("ArticleId")
            .agg(
                {
                    "QuantiteFacturee": "sum",
                    "CA_Mensuel_TND": "mean",
                    "Profit_Mensuel_TND": "mean",
                }
            )
            .sort_values("QuantiteFacturee", ascending=False)
        )

        insights["top_articles"] = {
            "top_10_by_quantity": article_performance.head(10).index.tolist(),
            "top_10_by_revenue": article_performance.sort_values(
                "CA_Mensuel_TND", ascending=False
            )
            .head(10)
            .index.tolist(),
        }

        self.insights["sales_analysis"] = insights

        # Create visualizations
        self.plot_sales_patterns(ventes, monthly_sales, dow_sales)

        return insights

    def analyze_promotion_effectiveness(self):
        """Analyze promotion effectiveness"""
        if "promotions" not in self.datasets or "ventes" not in self.datasets:
            print("❌ Required datasets not found")
            return

        promotions = self.datasets["promotions"].copy()
        ventes = self.datasets["ventes"].copy()

        # Convert dates
        promotions["DateCreation"] = pd.to_datetime(promotions["DateCreation"])
        promotions["DateFin"] = pd.to_datetime(promotions["DateFin"])
        ventes["Date"] = pd.to_datetime(ventes["Date"])

        insights = {}

        # Promotion acceptance rate
        acceptance_rate = promotions["IsAccepted"].mean()
        insights["acceptance_rate"] = round(acceptance_rate * 100, 2)

        # Average discount rate
        avg_discount = promotions["TauxReduction"].mean()
        insights["avg_discount_rate"] = round(avg_discount * 100, 2)

        # Discount distribution
        discount_bins = pd.cut(
            promotions["TauxReduction"],
            bins=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )
        insights["discount_distribution"] = discount_bins.value_counts().to_dict()

        # Promotion duration analysis
        promotions["Duration"] = (
            promotions["DateFin"] - promotions["DateCreation"]
        ).dt.days
        insights["avg_promotion_duration"] = promotions["Duration"].mean()

        self.insights["promotion_analysis"] = insights

        # Create visualizations
        self.plot_promotion_analysis(promotions)

        return insights

    def analyze_stock_levels(self):
        """Analyze stock levels and identify potential issues"""
        if "stocks" not in self.datasets:
            print("❌ Stocks dataset not found")
            return

        stocks = self.datasets["stocks"].copy()

        insights = {}

        # Stock level analysis
        low_stock_threshold = stocks["QuantitePhysique"] <= stocks["StockMin"]
        insights["low_stock_items"] = low_stock_threshold.sum()
        insights["low_stock_percentage"] = round(
            (low_stock_threshold.sum() / len(stocks)) * 100, 2
        )

        # Stock value analysis
        total_stock_value = stocks["Valeur_Stock_TND"].sum()
        insights["total_stock_value"] = round(total_stock_value, 2)
        insights["avg_stock_value_per_item"] = round(
            stocks["Valeur_Stock_TND"].mean(), 2
        )

        # Stock distribution
        stock_ranges = pd.cut(
            stocks["QuantitePhysique"],
            bins=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )
        insights["stock_distribution"] = stock_ranges.value_counts().to_dict()

        self.insights["stock_analysis"] = insights

        # Create visualizations
        self.plot_stock_analysis(stocks)

        return insights

    def analyze_product_categories(self):
        """Analyze product category performance"""
        if "articles" not in self.datasets or "ventes" not in self.datasets:
            print("❌ Required datasets not found")
            return

        articles = self.datasets["articles"].copy()
        ventes = self.datasets["ventes"].copy()

        # Merge articles with sales data
        article_sales = ventes.merge(
            articles[["Id", "FamilleNiv1", "Prix_Vente_TND"]],
            left_on="ArticleId",
            right_on="Id",
        )

        insights = {}

        # Category performance
        category_performance = (
            article_sales.groupby("FamilleNiv1")
            .agg(
                {
                    "QuantiteFacturee": "sum",
                    "CA_Mensuel_TND": "mean",
                    "Profit_Mensuel_TND": "mean",
                }
            )
            .sort_values("QuantiteFacturee", ascending=False)
        )

        insights["top_categories"] = category_performance.head(10).index.tolist()
        insights["category_performance"] = category_performance.to_dict()

        # Price analysis by category
        price_by_category = (
            articles.groupby("FamilleNiv1")["Prix_Vente_TND"]
            .agg(["mean", "median", "std"])
            .round(2)
        )
        insights["price_analysis"] = price_by_category.to_dict()

        self.insights["category_analysis"] = insights

        # Create visualizations
        self.plot_category_analysis(category_performance, price_by_category)

        return insights

    def plot_sales_patterns(self, ventes, monthly_sales, dow_sales):
        """Create sales pattern visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Monthly sales pattern
        monthly_sales["QuantiteFacturee"].plot(
            kind="bar", ax=axes[0, 0], color="skyblue"
        )
        axes[0, 0].set_title("Monthly Sales Quantity")
        axes[0, 0].set_xlabel("Month")
        axes[0, 0].set_ylabel("Quantity Sold")

        # Day of week pattern
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_sales.plot(kind="bar", ax=axes[0, 1], color="lightgreen")
        axes[0, 1].set_title("Sales by Day of Week")
        axes[0, 1].set_xlabel("Day of Week")
        axes[0, 1].set_ylabel("Quantity Sold")
        axes[0, 1].set_xticklabels(dow_labels, rotation=45)

        # Revenue vs Profit correlation
        axes[1, 0].scatter(
            ventes["CA_Mensuel_TND"],
            ventes["Profit_Mensuel_TND"],
            alpha=0.6,
            color="coral",
        )
        axes[1, 0].set_title("Revenue vs Profit Correlation")
        axes[1, 0].set_xlabel("Monthly Revenue (TND)")
        axes[1, 0].set_ylabel("Monthly Profit (TND)")

        # Sales distribution
        ventes["QuantiteFacturee"].hist(bins=30, ax=axes[1, 1], color="gold", alpha=0.7)
        axes[1, 1].set_title("Sales Quantity Distribution")
        axes[1, 1].set_xlabel("Quantity per Sale")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig("sales_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_promotion_analysis(self, promotions):
        """Create promotion analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Discount rate distribution
        promotions["TauxReduction"].hist(
            bins=20, ax=axes[0, 0], color="lightblue", alpha=0.7
        )
        axes[0, 0].set_title("Discount Rate Distribution")
        axes[0, 0].set_xlabel("Discount Rate")
        axes[0, 0].set_ylabel("Frequency")

        # Promotion acceptance
        acceptance_counts = promotions["IsAccepted"].value_counts()
        axes[0, 1].pie(
            acceptance_counts.values,
            labels=["Rejected", "Accepted"],
            autopct="%1.1f%%",
            colors=["salmon", "lightgreen"],
        )
        axes[0, 1].set_title("Promotion Acceptance Rate")

        # Promotion duration
        promotions["Duration"] = (
            pd.to_datetime(promotions["DateFin"])
            - pd.to_datetime(promotions["DateCreation"])
        ).dt.days
        promotions["Duration"].hist(
            bins=20, ax=axes[1, 0], color="mediumpurple", alpha=0.7
        )
        axes[1, 0].set_title("Promotion Duration Distribution")
        axes[1, 0].set_xlabel("Duration (Days)")
        axes[1, 0].set_ylabel("Frequency")

        # Price before vs after promotion
        axes[1, 1].scatter(
            promotions["Prix_Vente_TND_Avant"],
            promotions["Prix_Vente_TND_Apres"],
            alpha=0.6,
            color="orange",
        )
        axes[1, 1].plot(
            [0, promotions["Prix_Vente_TND_Avant"].max()],
            [0, promotions["Prix_Vente_TND_Avant"].max()],
            "r--",
            alpha=0.5,
        )
        axes[1, 1].set_title("Price Before vs After Promotion")
        axes[1, 1].set_xlabel("Price Before (TND)")
        axes[1, 1].set_ylabel("Price After (TND)")

        plt.tight_layout()
        plt.savefig("promotion_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_stock_analysis(self, stocks):
        """Create stock analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Stock level distribution
        stocks["QuantitePhysique"].hist(
            bins=30, ax=axes[0, 0], color="steelblue", alpha=0.7
        )
        axes[0, 0].set_title("Stock Level Distribution")
        axes[0, 0].set_xlabel("Physical Quantity")
        axes[0, 0].set_ylabel("Frequency")

        # Stock value distribution
        stocks["Valeur_Stock_TND"].hist(
            bins=30, ax=axes[0, 1], color="darkgreen", alpha=0.7
        )
        axes[0, 1].set_title("Stock Value Distribution")
        axes[0, 1].set_xlabel("Stock Value (TND)")
        axes[0, 1].set_ylabel("Frequency")

        # Low stock identification
        low_stock = stocks["QuantitePhysique"] <= stocks["StockMin"]
        stock_status = ["Normal Stock", "Low Stock"]
        stock_counts = [len(stocks) - low_stock.sum(), low_stock.sum()]
        axes[1, 0].pie(
            stock_counts,
            labels=stock_status,
            autopct="%1.1f%%",
            colors=["lightgreen", "salmon"],
        )
        axes[1, 0].set_title("Stock Status Distribution")

        # Stock min vs physical quantity
        axes[1, 1].scatter(
            stocks["StockMin"], stocks["QuantitePhysique"], alpha=0.6, color="purple"
        )
        axes[1, 1].plot(
            [0, stocks["StockMin"].max()],
            [0, stocks["StockMin"].max()],
            "r--",
            alpha=0.5,
        )
        axes[1, 1].set_title("Minimum Stock vs Physical Stock")
        axes[1, 1].set_xlabel("Minimum Stock")
        axes[1, 1].set_ylabel("Physical Stock")

        plt.tight_layout()
        plt.savefig("stock_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_category_analysis(self, category_performance, price_by_category):
        """Create category analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Top categories by sales
        top_10_categories = category_performance.head(10)
        top_10_categories["QuantiteFacturee"].plot(
            kind="barh", ax=axes[0, 0], color="teal"
        )
        axes[0, 0].set_title("Top 10 Categories by Sales Volume")
        axes[0, 0].set_xlabel("Total Quantity Sold")

        # Category revenue
        top_10_categories["CA_Mensuel_TND"].plot(
            kind="barh", ax=axes[0, 1], color="orange"
        )
        axes[0, 1].set_title("Top 10 Categories by Revenue")
        axes[0, 1].set_xlabel("Average Monthly Revenue (TND)")

        # Price distribution by category (top 10)
        top_categories = price_by_category.head(10)
        top_categories["mean"].plot(kind="bar", ax=axes[1, 0], color="lightcoral")
        axes[1, 0].set_title("Average Price by Category (Top 10)")
        axes[1, 0].set_xlabel("Category")
        axes[1, 0].set_ylabel("Average Price (TND)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Profit analysis
        top_10_categories["Profit_Mensuel_TND"].plot(
            kind="barh", ax=axes[1, 1], color="gold"
        )
        axes[1, 1].set_title("Top 10 Categories by Profit")
        axes[1, 1].set_xlabel("Average Monthly Profit (TND)")

        plt.tight_layout()
        plt.savefig("category_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def generate_ai_training_features(self):
        """Generate features for AI model training"""
        if not {"articles", "ventes", "stocks", "promotions"}.issubset(
            self.datasets.keys()
        ):
            print("❌ Required datasets not found for AI feature generation")
            return None

        print("🤖 Generating AI training features...")

        # Base features from articles
        features_df = self.datasets["articles"][
            ["Id", "Prix_Vente_TND", "Prix_Achat_TND", "FamilleNiv1"]
        ].copy()
        features_df.rename(columns={"Id": "ArticleId"}, inplace=True)

        # Add sales features
        ventes = self.datasets["ventes"].copy()
        ventes["Date"] = pd.to_datetime(ventes["Date"])

        sales_features = (
            ventes.groupby("ArticleId")
            .agg(
                {
                    "QuantiteFacturee": ["sum", "mean", "std"],
                    "CA_Mensuel_TND": ["mean", "std"],
                    "Profit_Mensuel_TND": ["mean", "std"],
                    "Date": ["min", "max", "count"],
                }
            )
            .round(2)
        )

        # Flatten column names
        sales_features.columns = ["_".join(col) for col in sales_features.columns]
        sales_features.reset_index(inplace=True)

        # Add stock features
        stock_features = (
            self.datasets["stocks"]
            .groupby("ArticleId")
            .agg(
                {
                    "QuantitePhysique": "mean",
                    "StockMin": "mean",
                    "Valeur_Stock_TND": "sum",
                }
            )
            .round(2)
        )
        stock_features.reset_index(inplace=True)

        # Add promotion features
        promotions = self.datasets["promotions"].copy()
        promo_features = (
            promotions.merge(
                self.datasets["articles"][["CodeArticle", "Id"]], on="CodeArticle"
            )
            .groupby("Id")
            .agg({"TauxReduction": ["mean", "count"], "IsAccepted": "mean"})
            .round(2)
        )

        promo_features.columns = ["_".join(col) for col in promo_features.columns]
        promo_features.columns = ["AvgDiscountRate", "PromoCount", "AcceptanceRate"]
        promo_features.reset_index(inplace=True)
        promo_features.rename(columns={"Id": "ArticleId"}, inplace=True)

        # Merge all features
        ai_features = features_df.merge(sales_features, on="ArticleId", how="left")
        ai_features = ai_features.merge(stock_features, on="ArticleId", how="left")
        ai_features = ai_features.merge(promo_features, on="ArticleId", how="left")

        # Fill NaN values
        ai_features.fillna(0, inplace=True)

        # Calculate derived features
        ai_features["ProfitMargin"] = (
            ai_features["Prix_Vente_TND"] - ai_features["Prix_Achat_TND"]
        ) / ai_features["Prix_Vente_TND"]
        ai_features["StockTurnover"] = ai_features["QuantiteFacturee_sum"] / (
            ai_features["QuantitePhysique"] + 1
        )
        ai_features["RevenuePerUnit"] = ai_features["CA_Mensuel_TND_mean"] / (
            ai_features["QuantiteFacturee_mean"] + 1
        )

        # Save AI features
        ai_features.to_csv("ai_training_features.csv", index=False)
        print(
            f"✅ Generated AI training features: {len(ai_features)} articles, {len(ai_features.columns)} features"
        )

        return ai_features

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 60)
        print("🔍 SMARTPROMO DATASET ANALYSIS REPORT")
        print("=" * 60)

        # Perform all analyses
        sales_insights = self.analyze_sales_patterns()
        promotion_insights = self.analyze_promotion_effectiveness()
        stock_insights = self.analyze_stock_levels()
        category_insights = self.analyze_product_categories()

        # Generate AI features
        ai_features = self.generate_ai_training_features()

        # Print comprehensive insights
        print("\n📊 SALES INSIGHTS:")
        print(
            f"   • Best selling month: {sales_insights['monthly_patterns']['best_month']}"
        )
        print(
            f"   • Average monthly revenue: {sales_insights['monthly_patterns']['avg_monthly_revenue']:,.2f} TND"
        )
        print(
            f"   • Average monthly profit: {sales_insights['monthly_patterns']['avg_monthly_profit']:,.2f} TND"
        )

        print("\n🎯 PROMOTION INSIGHTS:")
        print(
            f"   • Promotion acceptance rate: {promotion_insights['acceptance_rate']}%"
        )
        print(f"   • Average discount rate: {promotion_insights['avg_discount_rate']}%")
        print(
            f"   • Average promotion duration: {promotion_insights['avg_promotion_duration']:.1f} days"
        )

        print("\n📦 STOCK INSIGHTS:")
        print(
            f"   • Low stock items: {stock_insights['low_stock_items']} ({stock_insights['low_stock_percentage']}%)"
        )
        print(f"   • Total stock value: {stock_insights['total_stock_value']:,.2f} TND")
        print(
            f"   • Average stock value per item: {stock_insights['avg_stock_value_per_item']:.2f} TND"
        )

        print("\n🏷️ CATEGORY INSIGHTS:")
        print(
            f"   • Top performing categories: {category_insights['top_categories'][:3]}"
        )

        print("\n🤖 AI MODEL READINESS:")
        if ai_features is not None:
            print(f"   • Training dataset: {len(ai_features)} articles")
            print(f"   • Feature count: {len(ai_features.columns)}")
            print("   • Ready for machine learning model training ✅")

        print("\n🎉 Analysis completed successfully!")
        print("📁 Generated files:")
        print("   - sales_analysis.png")
        print("   - promotion_analysis.png")
        print("   - stock_analysis.png")
        print("   - category_analysis.png")
        print("   - ai_training_features.csv")

        return {
            "sales": sales_insights,
            "promotions": promotion_insights,
            "stock": stock_insights,
            "categories": category_insights,
            "ai_features": ai_features,
        }


def main():
    """Main function to run the complete analysis"""
    print("🔬 SmartPromo Dataset Analyzer")
    print("===============================")

    # Initialize analyzer
    analyzer = SmartPromoDatasetAnalyzer()

    # Load datasets
    analyzer.load_datasets()

    # Generate comprehensive report
    results = analyzer.generate_comprehensive_report()

    return results


if __name__ == "__main__":
    results = main()
