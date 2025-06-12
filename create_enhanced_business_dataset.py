#!/usr/bin/env python3
"""
Enhanced Business Dataset Generator with Date-based Analysis

This script creates a comprehensive business dataset that includes:
- Purchase quantities with purchase dates
- Sales dates for elasticity analysis
- Historical sales data (before/after) for proper elasticity calculation
- Price history for elasticity analysis
- Temporal features for date-based promotion optimization

Columns to add:
- Date_Achat: Purchase date for each article
- Date_Derniere_Vente: Last sales date
- Ventes_Mensuelles_Unites_Avant: Sales before promotion/price change
- Ventes_Mensuelles_Unites_Apres: Sales after promotion/price change
- Prix_Vente_TND_Avant: Price before change
- Prix_Vente_TND_Apres: Price after change

Columns to remove:
- Rotation_Stock_Mois (will be calculated differently)
- Marge_Pourcentage (calculated on-the-fly in the model)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_enhanced_business_dataset(num_products=2500):
    """
    Generate enhanced business dataset with date-based features for elasticity analysis
    """
    print(f"Generating enhanced business dataset with {num_products} products...")

    # Define business categories and suppliers
    categories_l1 = [
        "BIJOUTERIE",
        "MONTRES",
        "MAROQUINERIE",
        "PARFUMERIE",
        "COSMETIQUE",
        "ACCESSOIRES",
        "ELECTROMENAGER",
        "TEXTILE",
        "OPTIQUE",
        "HORLOGERIE",
    ]

    categories_l2 = {
        "BIJOUTERIE": [
            "BAGUES",
            "COLLIERS",
            "BRACELETS",
            "BOUCLES_OREILLES",
            "PARURES",
        ],
        "MONTRES": ["MONTRES_HOMME", "MONTRES_FEMME", "MONTRES_SPORT", "MONTRES_LUXE"],
        "MAROQUINERIE": ["SACS_MAIN", "PORTEFEUILLES", "CEINTURES", "BAGAGES"],
        "PARFUMERIE": ["PARFUMS_HOMME", "PARFUMS_FEMME", "EAUX_TOILETTE", "COFFRETS"],
        "COSMETIQUE": ["MAQUILLAGE", "SOINS_VISAGE", "SOINS_CORPS", "VERNIS"],
        "ACCESSOIRES": ["FOULARDS", "LUNETTES", "CHAPEAUX", "GANTS"],
        "ELECTROMENAGER": ["PETIT_ELECTROMENAGER", "BEAUTE_ELECTRIQUE", "CUISINE"],
        "TEXTILE": ["VETEMENTS_FEMME", "VETEMENTS_HOMME", "SOUS_VETEMENTS"],
        "OPTIQUE": ["LUNETTES_VUE", "LUNETTES_SOLEIL", "LENTILLES"],
        "HORLOGERIE": ["HORLOGES", "REVEILS", "PENDULES", "CHRONOMETRES"],
    }

    suppliers = [
        "CHANEL",
        "DIOR",
        "LOUIS_VUITTON",
        "CARTIER",
        "ROLEX",
        "OMEGA",
        "LANCOME",
        "ESTEE_LAUDER",
        "YVES_SAINT_LAURENT",
        "GUCCI",
        "HERMÈS",
        "PRADA",
        "BULGARI",
        "TIFFANY",
        "PANDORA",
        "SWATCH",
        "CITIZEN",
        "CASIO",
        "TAG_HEUER",
        "BREITLING",
        "CLINIQUE",
        "SHISEIDO",
        "MAC",
        "NARS",
        "URBAN_DECAY",
        "RAY_BAN",
        "OAKLEY",
        "PERSOL",
        "VERSACE",
        "ARMANI",
    ]

    # Generate date ranges for realistic business simulation
    base_date = datetime(2023, 1, 1)
    current_date = datetime(2025, 6, 12)  # Current date

    # Generate base product data
    products = []

    for i in range(num_products):
        # Select category hierarchy
        cat_l1 = random.choice(categories_l1)
        cat_l2 = random.choice(categories_l2[cat_l1])
        supplier = random.choice(suppliers)

        # Generate article code
        article_code = f"{cat_l1[:3]}{i+1:05d}"

        # Generate product name
        product_names = {
            "BIJOUTERIE": ["Bague", "Collier", "Bracelet", "Boucles", "Parure"],
            "MONTRES": ["Montre", "Chrono", "Automatique", "Quartz"],
            "MAROQUINERIE": ["Sac", "Portefeuille", "Ceinture", "Valise"],
            "PARFUMERIE": ["Parfum", "Eau de Toilette", "Coffret"],
            "COSMETIQUE": ["Rouge", "Fond de teint", "Mascara", "Palette"],
            "ACCESSOIRES": ["Foulard", "Lunettes", "Chapeau"],
            "ELECTROMENAGER": ["Sèche-cheveux", "Lisseur", "Bouilloire"],
            "TEXTILE": ["Robe", "Chemise", "Pantalon", "Veste"],
            "OPTIQUE": ["Lunettes", "Monture", "Lentilles"],
            "HORLOGERIE": ["Horloge", "Réveil", "Pendule"],
        }

        base_name = random.choice(product_names[cat_l1])
        product_name = f"{base_name} {supplier} {random.choice(['Premium', 'Classic', 'Deluxe', 'Elite'])}"

        # Price logic based on category and supplier
        price_ranges = {
            "BIJOUTERIE": (50, 2000),
            "MONTRES": (100, 5000),
            "MAROQUINERIE": (80, 1500),
            "PARFUMERIE": (30, 300),
            "COSMETIQUE": (15, 150),
            "ACCESSOIRES": (25, 400),
            "ELECTROMENAGER": (40, 800),
            "TEXTILE": (30, 500),
            "OPTIQUE": (50, 600),
            "HORLOGERIE": (40, 300),
        }

        # Luxury brands get price multiplier
        luxury_brands = [
            "CHANEL",
            "DIOR",
            "LOUIS_VUITTON",
            "CARTIER",
            "ROLEX",
            "HERMÈS",
            "BULGARI",
            "TIFFANY",
        ]
        price_multiplier = 2.0 if supplier in luxury_brands else 1.0

        min_price, max_price = price_ranges[cat_l1]
        current_price = np.random.uniform(min_price, max_price) * price_multiplier

        # Generate historical pricing for elasticity analysis
        # Price change occurred 3-6 months ago
        price_change_date = current_date - timedelta(days=random.randint(90, 180))

        # Previous price (higher or lower by 5-25%)
        price_change_direction = random.choice(
            [-1, 1]
        )  # -1 for decrease, 1 for increase
        price_change_pct = np.random.uniform(0.05, 0.25)  # 5-25% change
        previous_price = current_price * (
            1 + (price_change_direction * price_change_pct)
        )

        # Cost calculation (realistic margin structure)
        if cat_l1 in ["BIJOUTERIE", "MONTRES"]:
            margin_pct = np.random.uniform(45, 65)  # High margin luxury items
        elif cat_l1 in ["PARFUMERIE", "COSMETIQUE"]:
            margin_pct = np.random.uniform(35, 55)  # Medium-high margin
        else:
            margin_pct = np.random.uniform(25, 45)  # Standard margin

        cost = current_price * (1 - margin_pct / 100)

        # Sales volume logic with elasticity consideration
        if current_price < 50:
            base_monthly_sales = np.random.poisson(150) + 50  # High volume, low price
        elif current_price < 200:
            base_monthly_sales = np.random.poisson(80) + 20  # Medium volume
        elif current_price < 500:
            base_monthly_sales = np.random.poisson(40) + 10  # Lower volume
        else:
            base_monthly_sales = np.random.poisson(15) + 5  # Low volume, high price

        current_monthly_sales = max(
            1, int(base_monthly_sales * np.random.uniform(0.7, 1.3))
        )

        # Calculate previous sales based on price elasticity
        # Price elasticity (luxury items less elastic)
        if cat_l1 in ["BIJOUTERIE", "MONTRES"] and supplier in luxury_brands:
            elasticity = np.random.uniform(-0.3, -0.8)  # Less elastic
        elif cat_l1 in ["COSMETIQUE", "PARFUMERIE"]:
            elasticity = np.random.uniform(-0.8, -1.5)  # Medium elastic
        else:
            elasticity = np.random.uniform(-1.2, -2.0)  # More elastic

        # Calculate historical sales based on price change and elasticity
        price_change_ratio = (current_price - previous_price) / previous_price
        volume_change_ratio = elasticity * price_change_ratio
        previous_monthly_sales = int(current_monthly_sales / (1 + volume_change_ratio))
        previous_monthly_sales = max(1, previous_monthly_sales)

        # Generate purchase date (2-8 months ago)
        purchase_date = current_date - timedelta(days=random.randint(60, 240))

        # Generate last sales date (within last 30 days)
        last_sale_date = current_date - timedelta(days=random.randint(1, 30))

        # Stock levels (business logic: 2-6 months of stock)
        stock_months = np.random.uniform(2, 6)
        current_stock = int(current_monthly_sales * stock_months)

        # Purchase quantity (3-8 months supply)
        purchase_quantity = int(current_monthly_sales * np.random.uniform(3, 8))

        # Historical promotion data
        promo_frequency = np.random.poisson(4) + 1  # 1-8 promotions per year
        avg_promo_pct = np.random.uniform(10, 35)  # 10-35% average discount
        last_promo_performance = np.random.uniform(
            0.8, 1.8
        )  # 80% to 180% of expected performance

        # Market share and brand strength
        if supplier in luxury_brands:
            market_share = np.random.uniform(3, 15)  # Higher market share for luxury
            brand_strength = np.random.uniform(0.7, 0.95)
        else:
            market_share = np.random.uniform(0.5, 8)
            brand_strength = np.random.uniform(0.4, 0.8)

        # Sales trend (3-month trend)
        trend_factor = np.random.uniform(-0.3, 0.4)  # -30% to +40% trend

        # Seasonal demand factor
        seasonal_categories = {
            "PARFUMERIE": 1.3,  # High seasonal demand
            "COSMETIQUE": 1.2,
            "BIJOUTERIE": 1.4,  # Very seasonal (holidays)
            "MONTRES": 1.3,
            "MAROQUINERIE": 1.1,
            "ACCESSOIRES": 1.2,
            "TEXTILE": 1.1,
            "OPTIQUE": 0.9,  # Less seasonal
            "ELECTROMENAGER": 0.95,
            "HORLOGERIE": 1.0,
        }
        seasonal_demand = seasonal_categories.get(cat_l1, 1.0)

        # Customer satisfaction and return rate
        satisfaction = np.random.uniform(3.5, 4.8)  # Out of 5
        return_rate = np.random.uniform(0.5, 8.0)  # 0.5% to 8% return rate

        # Calculate derived metrics
        monthly_revenue = current_monthly_sales * current_price
        monthly_profit = (current_price - cost) * current_monthly_sales
        stock_value = current_stock * cost

        # Create product record
        product = {
            # Basic product info
            "GA_ARTICLE": article_code,
            "GA_LIBELLE": product_name,
            "GA_FAMILLENIV1": cat_l1,
            "GA_FAMILLENIV2": cat_l2,
            "GA_FOURNPRINC": supplier,
            # Pricing data (current)
            "Prix_Vente_TND": round(current_price, 2),
            "Cout_Unitaire_TND": round(cost, 2),
            # Note: Marge_Pourcentage will be calculated on-the-fly in the model
            # Historical pricing for elasticity analysis
            "Prix_Vente_TND_Avant": round(previous_price, 2),
            "Prix_Vente_TND_Apres": round(current_price, 2),
            # Sales data (current)
            "Ventes_Mensuelles_Unites": current_monthly_sales,
            "CA_Mensuel_TND": round(monthly_revenue, 2),
            "Profit_Mensuel_TND": round(monthly_profit, 2),
            # Historical sales for elasticity analysis
            "Ventes_Mensuelles_Unites_Avant": previous_monthly_sales,
            "Ventes_Mensuelles_Unites_Apres": current_monthly_sales,
            # Stock data
            "Stock_Actuel_Unites": current_stock,
            "Valeur_Stock_TND": round(stock_value, 2),
            # Purchase data with dates
            "Quantite_Achat": purchase_quantity,
            "Date_Achat": purchase_date.strftime("%Y-%m-%d"),
            # Sales dates
            "Date_Derniere_Vente": last_sale_date.strftime("%Y-%m-%d"),
            # Historical promotion data
            "Historique_Promo_Freq_12M": promo_frequency,
            "Historique_Promo_Moyenne_Pct": round(avg_promo_pct, 1),
            "Derniere_Promo_Performance": round(last_promo_performance, 2),
            # Market data
            "Elasticite_Prix": round(elasticity, 2),
            "Demande_Saisonniere": round(seasonal_demand, 2),
            "Part_Marche_Pct": round(market_share, 2),
            "Force_Marque": round(brand_strength, 2),
            # Performance metrics
            "Tendance_Ventes_3M": round(trend_factor, 3),
            "Satisfaction_Client": round(satisfaction, 1),
            "Taux_Retour_Pct": round(return_rate, 1),
        }

        products.append(product)

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} products...")

    return pd.DataFrame(products)


def add_calculated_metrics(df):
    """Add calculated metrics that depend on the base data"""
    print("Adding calculated business metrics...")

    # Calculate stock rotation (using new purchase data)
    df["Rotation_Mensuelle"] = (
        df["Ventes_Mensuelles_Unites"] / df["Stock_Actuel_Unites"]
    )

    # Calculate months of stock coverage
    df["Mois_Stock_Couverture"] = (
        df["Stock_Actuel_Unites"] / df["Ventes_Mensuelles_Unites"]
    )

    # Calculate purchase-to-sales ratio
    df["Ratio_Achat_Ventes"] = df["Quantite_Achat"] / df["Ventes_Mensuelles_Unites"]

    # Calculate ROI based on stock value
    df["ROI_Mensuel_Pct"] = (df["Profit_Mensuel_TND"] / df["Valeur_Stock_TND"]) * 100

    # Calculate price change percentage
    df["Variation_Prix_Pct"] = (
        (df["Prix_Vente_TND_Apres"] - df["Prix_Vente_TND_Avant"])
        / df["Prix_Vente_TND_Avant"]
    ) * 100

    # Calculate volume change percentage
    df["Variation_Volume_Pct"] = (
        (df["Ventes_Mensuelles_Unites_Apres"] - df["Ventes_Mensuelles_Unites_Avant"])
        / df["Ventes_Mensuelles_Unites_Avant"]
    ) * 100

    # Calculate actual price elasticity from the data
    df["Elasticite_Calculee"] = df["Variation_Volume_Pct"] / df["Variation_Prix_Pct"]
    df["Elasticite_Calculee"] = df["Elasticite_Calculee"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Days since last purchase
    current_date = datetime.now()
    df["Jours_Depuis_Achat"] = (
        pd.to_datetime(current_date) - pd.to_datetime(df["Date_Achat"])
    ).dt.days

    # Days since last sale
    df["Jours_Depuis_Derniere_Vente"] = (
        pd.to_datetime(current_date) - pd.to_datetime(df["Date_Derniere_Vente"])
    ).dt.days

    # Handle infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def validate_enhanced_dataset(df):
    """Validate the enhanced dataset"""
    print("\n=== ENHANCED DATASET VALIDATION ===")

    # Check for required columns for elasticity analysis
    required_columns = [
        "GA_ARTICLE",
        "GA_LIBELLE",
        "GA_FAMILLENIV1",
        "GA_FAMILLENIV2",
        "GA_FOURNPRINC",
        "Prix_Vente_TND",
        "Cout_Unitaire_TND",
        "Ventes_Mensuelles_Unites",
        "CA_Mensuel_TND",
        "Stock_Actuel_Unites",
        "Quantite_Achat",
        "Date_Achat",
        "Date_Derniere_Vente",
        "Prix_Vente_TND_Avant",
        "Prix_Vente_TND_Apres",
        "Ventes_Mensuelles_Unites_Avant",
        "Ventes_Mensuelles_Unites_Apres",
        "Elasticite_Prix",
        "Historique_Promo_Freq_12M",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
    else:
        print("✅ All required columns present")

    # Validate data ranges and business logic
    validations = [
        (
            df["Prix_Vente_TND"] > df["Cout_Unitaire_TND"],
            "Current selling price > Cost",
        ),
        (df["Ventes_Mensuelles_Unites"] > 0, "Positive current sales"),
        (df["Ventes_Mensuelles_Unites_Avant"] > 0, "Positive historical sales"),
        (df["Stock_Actuel_Unites"] >= 0, "Non-negative stock"),
        (df["Elasticite_Prix"] < 0, "Negative price elasticity"),
        (df["Quantite_Achat"] > 0, "Positive purchase quantities"),
        (
            pd.to_datetime(df["Date_Achat"]) <= datetime.now(),
            "Purchase dates in the past",
        ),
        (
            pd.to_datetime(df["Date_Derniere_Vente"]) <= datetime.now(),
            "Sales dates in the past",
        ),
    ]

    for condition, description in validations:
        if condition.all():
            print(f"✅ {description}: Valid")
        else:
            invalid_count = (~condition).sum()
            print(f"⚠️  {description}: {invalid_count} invalid records")

    # Data summary
    print(f"\n=== ENHANCED DATASET SUMMARY ===")
    print(f"Total products: {len(df):,}")
    print(f"Total categories (L1): {df['GA_FAMILLENIV1'].nunique()}")
    print(f"Total subcategories (L2): {df['GA_FAMILLENIV2'].nunique()}")
    print(f"Total suppliers: {df['GA_FOURNPRINC'].nunique()}")
    print(
        f"Price range: {df['Prix_Vente_TND'].min():.2f} - {df['Prix_Vente_TND'].max():.2f} TND"
    )
    print(f"Total monthly revenue: {df['CA_Mensuel_TND'].sum():,.2f} TND")
    print(
        f"Date range for purchases: {df['Date_Achat'].min()} to {df['Date_Achat'].max()}"
    )
    print(
        f"Date range for sales: {df['Date_Derniere_Vente'].min()} to {df['Date_Derniere_Vente'].max()}"
    )

    # Elasticity analysis summary
    print(f"\n=== ELASTICITY ANALYSIS SUMMARY ===")
    print(f"Average price elasticity: {df['Elasticite_Prix'].mean():.2f}")
    print(
        f"Price change range: {df['Variation_Prix_Pct'].min():.1f}% to {df['Variation_Prix_Pct'].max():.1f}%"
    )
    print(
        f"Volume change range: {df['Variation_Volume_Pct'].min():.1f}% to {df['Variation_Volume_Pct'].max():.1f}%"
    )

    return df


def save_enhanced_dataset(df, filename="enhanced_business_dataset_with_dates.xlsx"):
    """Save the enhanced dataset to Excel with multiple sheets"""
    print(f"\nSaving enhanced dataset to {filename}...")

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Main business data
        df.to_excel(writer, sheet_name="Business_Data", index=False)

        # Summary by category
        category_summary = (
            df.groupby("GA_FAMILLENIV1")
            .agg(
                {
                    "CA_Mensuel_TND": "sum",
                    "Profit_Mensuel_TND": "sum",
                    "Ventes_Mensuelles_Unites": "sum",
                    "Prix_Vente_TND": "mean",
                    "Elasticite_Prix": "mean",
                    "GA_ARTICLE": "count",
                }
            )
            .round(2)
        )
        category_summary.columns = [
            "Total_Revenue",
            "Total_Profit",
            "Total_Sales",
            "Avg_Price",
            "Avg_Elasticity",
            "Product_Count",
        ]
        category_summary.to_excel(writer, sheet_name="Category_Summary")

        # Supplier analysis
        supplier_summary = (
            df.groupby("GA_FOURNPRINC")
            .agg(
                {
                    "CA_Mensuel_TND": "sum",
                    "Profit_Mensuel_TND": "sum",
                    "Prix_Vente_TND": "mean",
                    "Elasticite_Prix": "mean",
                    "GA_ARTICLE": "count",
                }
            )
            .round(2)
            .sort_values("CA_Mensuel_TND", ascending=False)
        )
        supplier_summary.columns = [
            "Total_Revenue",
            "Total_Profit",
            "Avg_Price",
            "Avg_Elasticity",
            "Product_Count",
        ]
        supplier_summary.to_excel(writer, sheet_name="Supplier_Analysis")

        # Elasticity analysis
        elasticity_analysis = df[
            [
                "GA_ARTICLE",
                "GA_LIBELLE",
                "GA_FAMILLENIV1",
                "Prix_Vente_TND_Avant",
                "Prix_Vente_TND_Apres",
                "Ventes_Mensuelles_Unites_Avant",
                "Ventes_Mensuelles_Unites_Apres",
                "Variation_Prix_Pct",
                "Variation_Volume_Pct",
                "Elasticite_Prix",
                "Elasticite_Calculee",
            ]
        ].copy()
        elasticity_analysis.to_excel(
            writer, sheet_name="Elasticity_Analysis", index=False
        )

        # Temporal analysis
        temporal_analysis = df[
            [
                "GA_ARTICLE",
                "GA_LIBELLE",
                "Date_Achat",
                "Date_Derniere_Vente",
                "Jours_Depuis_Achat",
                "Jours_Depuis_Derniere_Vente",
                "Quantite_Achat",
                "Ratio_Achat_Ventes",
            ]
        ].copy()
        temporal_analysis.to_excel(writer, sheet_name="Temporal_Analysis", index=False)

        # Data dictionary
        data_dict = pd.DataFrame(
            {
                "Column_Name": [
                    "GA_ARTICLE",
                    "GA_LIBELLE",
                    "GA_FAMILLENIV1",
                    "GA_FAMILLENIV2",
                    "GA_FOURNPRINC",
                    "Prix_Vente_TND",
                    "Cout_Unitaire_TND",
                    "Prix_Vente_TND_Avant",
                    "Prix_Vente_TND_Apres",
                    "Ventes_Mensuelles_Unites",
                    "CA_Mensuel_TND",
                    "Profit_Mensuel_TND",
                    "Ventes_Mensuelles_Unites_Avant",
                    "Ventes_Mensuelles_Unites_Apres",
                    "Stock_Actuel_Unites",
                    "Valeur_Stock_TND",
                    "Quantite_Achat",
                    "Date_Achat",
                    "Date_Derniere_Vente",
                    "Historique_Promo_Freq_12M",
                    "Historique_Promo_Moyenne_Pct",
                    "Derniere_Promo_Performance",
                    "Elasticite_Prix",
                    "Demande_Saisonniere",
                    "Part_Marche_Pct",
                    "Force_Marque",
                    "Tendance_Ventes_3M",
                    "Satisfaction_Client",
                    "Taux_Retour_Pct",
                ],
                "Description": [
                    "Article code (unique identifier)",
                    "Product description/name",
                    "Main category (Level 1)",
                    "Subcategory (Level 2)",
                    "Main supplier",
                    "Current selling price in TND",
                    "Unit cost in TND",
                    "Previous selling price (before change)",
                    "Current selling price (after change)",
                    "Current monthly sales in units",
                    "Current monthly revenue in TND",
                    "Current monthly profit in TND",
                    "Monthly sales before price change",
                    "Monthly sales after price change",
                    "Current stock level in units",
                    "Current stock value in TND",
                    "Last purchase quantity",
                    "Date of last purchase",
                    "Date of last sale",
                    "Historical promotion frequency (12 months)",
                    "Average historical promotion percentage",
                    "Last promotion performance factor",
                    "Price elasticity coefficient",
                    "Seasonal demand factor",
                    "Market share percentage",
                    "Brand strength factor",
                    "Sales trend over 3 months",
                    "Customer satisfaction (1-5)",
                    "Return rate percentage",
                ],
            }
        )
        data_dict.to_excel(writer, sheet_name="Data_Dictionary", index=False)

    print(f"✅ Enhanced dataset saved successfully to {filename}")
    print(
        f"📊 Sheets created: Business_Data, Category_Summary, Supplier_Analysis, Elasticity_Analysis, Temporal_Analysis, Data_Dictionary"
    )


def main():
    """Main function to generate the enhanced business dataset"""
    print("=" * 70)
    print("ENHANCED BUSINESS DATASET GENERATOR WITH DATE-BASED ANALYSIS")
    print("=" * 70)

    # Generate the dataset
    num_products = 2500  # You can adjust this number
    df = generate_enhanced_business_dataset(num_products)

    # Add calculated metrics
    df = add_calculated_metrics(df)

    # Validate the dataset
    df = validate_enhanced_dataset(df)

    # Save to Excel
    save_enhanced_dataset(df)

    print("\n" + "=" * 70)
    print("ENHANCED DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNew features added:")
    print("✅ Purchase dates (Date_Achat) for each article")
    print("✅ Sales dates (Date_Derniere_Vente) for temporal analysis")
    print("✅ Historical sales data (before/after) for elasticity calculation")
    print("✅ Historical pricing data (before/after) for elasticity analysis")
    print("✅ Calculated elasticity from actual data changes")
    print("✅ Temporal metrics (days since purchase/sale)")
    print("✅ Enhanced business logic with realistic date-based patterns")
    print("\nRemoved columns:")
    print("❌ Rotation_Stock_Mois (replaced with better rotation metrics)")
    print("❌ Marge_Pourcentage (calculated on-the-fly in the model)")
    print("\nYour AI promotion model can now:")
    print("🎯 Analyze true price elasticity from historical data")
    print("🎯 Make date-based promotion predictions")
    print("🎯 Use temporal patterns for better recommendations")
    print("🎯 Track purchase and sales cycles")

    return df


if __name__ == "__main__":
    dataset = main()
