"""
SmartPromo Dataset Generator
Generates realistic test data for all entities in the SmartPromo system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from faker import Faker
import json

# Initialize Faker for generating realistic data
fake = Faker(["fr_FR", "en_US"])  # French and English locales


class SmartPromoDatasetGenerator:
    """
    Generates comprehensive datasets for all SmartPromo entities
    """

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

        # Configuration for data generation
        self.num_categories = 50
        self.num_articles = 1000
        self.num_etablissements = 25
        self.num_users = 50
        self.num_stocks = 1200
        self.num_promotions = 300
        self.num_ventes = 5000

        # Date ranges
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2024, 12, 31)

        # Price ranges (in TND - Tunisian Dinar)
        self.min_price = 5.0
        self.max_price = 2000.0

        # Initialize data containers
        self.datasets = {}

        # Product categories in French (Tunisia context)
        self.product_categories = [
            "Alimentation",
            "Boissons",
            "Hygiène",
            "Cosmétiques",
            "Vêtements",
            "Électronique",
            "Électroménager",
            "Maison & Jardin",
            "Sport & Loisirs",
            "Automobiles",
            "Librairie",
            "Santé",
            "Bébé & Enfant",
            "Informatique",
            "Téléphonie",
            "Jouets",
            "Bijouterie",
            "Chaussures",
            "Maroquinerie",
            "Parfumerie",
        ]

        # Tunisian cities
        self.tunisian_cities = [
            "Tunis",
            "Sfax",
            "Sousse",
            "Kairouan",
            "Bizerte",
            "Gabès",
            "Ariana",
            "Gafsa",
            "Monastir",
            "Ben Arous",
            "Kasserine",
            "Médenine",
            "Nabeul",
            "Tataouine",
            "Béja",
            "Jendouba",
            "Mahdia",
            "Siliana",
            "Manouba",
            "Zaghouan",
            "Kébili",
            "Tozeur",
            "Sidi Bouzid",
            "Le Kef",
        ]

        # Supplier names (French/Tunisian context)
        self.suppliers = [
            "Groupe Poulina",
            "Délice Danone",
            "SOTUMAG",
            "Yazaki Tunisia",
            "Leoni AG",
            "One Tech Group",
            "Tunisie Telecom",
            "Orange Tunisie",
            "Monoprix",
            "Carrefour Tunisie",
            "Géant",
            "Aziza",
            "Senia",
            "UTIC",
            "Modern Leasing",
            "Attijari Bank",
            "BIAT",
            "STB",
            "Amen Bank",
            "Arab Tunisian Bank",
        ]

    def generate_categories(self):
        """Generate Categories dataset"""
        categories = []

        for i in range(self.num_categories):
            category = {
                "IdCategorie": f"CAT{i+1:03d}",
                "Nom": random.choice(self.product_categories)
                + f" {random.randint(1, 10)}",
                "Description": fake.text(max_nb_chars=200),
            }
            categories.append(category)

        df = pd.DataFrame(categories)
        self.datasets["Categories"] = df
        return df

    def generate_articles(self):
        """Generate Articles dataset"""
        if "Categories" not in self.datasets:
            self.generate_categories()

        articles = []
        categories = self.datasets["Categories"]["IdCategorie"].tolist()

        for i in range(self.num_articles):
            # Generate realistic product names
            product_type = random.choice(
                [
                    "Smartphone",
                    "Ordinateur",
                    "Télévision",
                    "Réfrigérateur",
                    "Machine à laver",
                    "Chaise",
                    "Table",
                    "Lampe",
                    "Livre",
                    "Stylo",
                    "Cahier",
                    "Sac",
                    "Chaussures",
                    "T-shirt",
                    "Pantalon",
                    "Parfum",
                    "Shampoing",
                    "Yaourt",
                    "Pain",
                    "Fromage",
                    "Eau",
                    "Jus",
                    "Café",
                    "Thé",
                ]
            )

            brand = random.choice(
                [
                    "Samsung",
                    "Apple",
                    "LG",
                    "Sony",
                    "Philips",
                    "Bosch",
                    "IKEA",
                    "Nike",
                    "Adidas",
                    "L'Oréal",
                    "Nivea",
                    "Danone",
                    "Nestlé",
                    "Coca-Cola",
                ]
            )

            article = {
                "Id": i + 1,
                "CodeArticle": f"ART{i+1:06d}",
                "CodeBarre": fake.ean13(),
                "Libelle": f"{brand} {product_type} {fake.word()}",
                "CodeDim1": f"DIM1-{random.randint(1, 100)}",
                "LibelleDim1": fake.word(),
                "CodeDim2": f"DIM2-{random.randint(1, 100)}",
                "LibelleDim2": fake.word(),
                "Fournisseur": random.choice(self.suppliers),
                "FamilleNiv1": random.choice(self.product_categories),
                "FamilleNiv2": fake.word(),
                "FamilleNiv3": fake.word(),
                "FamilleNiv4": fake.word(),
                "FamilleNiv5": fake.word(),
                "FamilleNiv6": fake.word(),
                "FamilleNiv7": fake.word(),
                "FamilleNiv8": fake.word(),
                "Quantite_Achat": str(random.randint(1, 100)),
                "DateLibre": fake.date_between(
                    start_date=self.start_date, end_date=self.end_date
                ),
                "Prix_Vente_TND": round(
                    random.uniform(self.min_price, self.max_price), 2
                ),
                "Prix_Achat_TND": 0,  # Will be calculated as 60-80% of selling price
                "IdCategorie": random.choice(categories),
            }

            # Calculate purchase price (60-80% of selling price)
            article["Prix_Achat_TND"] = round(
                article["Prix_Vente_TND"] * random.uniform(0.6, 0.8), 2
            )

            articles.append(article)

        df = pd.DataFrame(articles)
        self.datasets["Articles"] = df
        return df

    def generate_etablissements(self):
        """Generate Etablissements (Establishments) dataset"""
        etablissements = []

        establishment_types = [
            "Magasin",
            "Supermarché",
            "Hypermarché",
            "Boutique",
            "Entrepôt",
        ]
        sectors = [
            "Centre Ville",
            "Banlieue",
            "Zone Industrielle",
            "Centre Commercial",
            "Quartier Résidentiel",
        ]

        for i in range(self.num_etablissements):
            etablissement = {
                "Id": i + 1,
                "Code": f"ETB{i+1:03d}",
                "Libelle": f"{random.choice(establishment_types)} {fake.company()}",
                "Adresse": fake.address(),
                "Ville": random.choice(self.tunisian_cities),
                "Type": random.choice(establishment_types),
                "Secteur": random.choice(sectors),
            }
            etablissements.append(etablissement)

        df = pd.DataFrame(etablissements)
        self.datasets["Etablissements"] = df
        return df

    def generate_users(self):
        """Generate Users dataset"""
        users = []

        for i in range(self.num_users):
            user_type = random.choice([0, 1, 2])  # 0=Admin, 1=Manager, 2=Employee

            user = {
                "Id": i + 1,
                "Nom": fake.last_name(),
                "Prenom": fake.first_name(),
                "Email": fake.email(),
                "Type": user_type,
                "PasswordHash": fake.sha256(),
                "CreatedAt": fake.date_time_between(
                    start_date=self.start_date, end_date=self.end_date
                ),
                "UpdatedAt": fake.date_time_between(
                    start_date=self.start_date, end_date=self.end_date
                ),
                "LastLogin": fake.date_time_between(
                    start_date=self.start_date, end_date=self.end_date
                ),
                "IsActive": random.choice([True, False]),
                "RefreshToken": fake.uuid4() if random.random() > 0.3 else None,
                "RefreshTokenExpiryTime": (
                    fake.future_datetime() if random.random() > 0.3 else None
                ),
            }
            users.append(user)

        df = pd.DataFrame(users)
        self.datasets["Users"] = df
        return df

    def generate_stocks(self):
        """Generate Stock dataset"""
        if "Articles" not in self.datasets:
            self.generate_articles()

        stocks = []
        articles = self.datasets["Articles"]["Id"].tolist()

        for i in range(self.num_stocks):
            article_id = random.choice(articles)
            article = self.datasets["Articles"][
                self.datasets["Articles"]["Id"] == article_id
            ].iloc[0]

            quantite_physique = random.randint(0, 1000)
            stock_min = random.randint(10, 100)

            stock = {
                "Id": i + 1,
                "QuantitePhysique": quantite_physique,
                "StockMin": stock_min,
                "VenteFFO": random.randint(0, 50),
                "LivreFou": random.randint(0, 100),
                "Transfert": random.randint(0, 20),
                "AnnonceTrf": random.randint(0, 30),
                "Valeur_Stock_TND": round(
                    quantite_physique * article["Prix_Achat_TND"], 2
                ),
                "ArticleId": article_id,
            }
            stocks.append(stock)

        df = pd.DataFrame(stocks)
        self.datasets["Stocks"] = df
        return df

    def generate_promotions(self):
        """Generate Promotions dataset"""
        if "Articles" not in self.datasets:
            self.generate_articles()

        promotions = []
        articles = self.datasets["Articles"]

        for i in range(self.num_promotions):
            article = articles.sample(n=1).iloc[0]

            # Generate promotion dates
            date_creation = fake.date_time_between(
                start_date=self.start_date, end_date=self.end_date
            )
            date_fin = date_creation + timedelta(days=random.randint(7, 90))

            # Generate discount rate (5% to 50%)
            taux_reduction = round(random.uniform(0.05, 0.50), 2)

            # Calculate prices
            prix_avant = article["Prix_Vente_TND"]
            prix_apres = round(prix_avant * (1 - taux_reduction), 2)

            promotion = {
                "Id": i + 1,
                "DateFin": date_fin,
                "TauxReduction": taux_reduction,
                "CodeArticle": article["CodeArticle"],
                "Prix_Vente_TND_Avant": prix_avant,
                "Prix_Vente_TND_Apres": prix_apres,
                "IsAccepted": random.choice([True, False]),
                "DateCreation": date_creation,
                "DateApproval": (
                    date_creation + timedelta(days=random.randint(1, 7))
                    if random.random() > 0.3
                    else None
                ),
                "ApprovedBy": (
                    f"User{random.randint(1, 10)}" if random.random() > 0.3 else None
                ),
            }
            promotions.append(promotion)

        df = pd.DataFrame(promotions)
        self.datasets["Promotions"] = df
        return df

    def generate_ventes(self):
        """Generate Ventes (Sales) dataset"""
        if "Articles" not in self.datasets:
            self.generate_articles()

        ventes = []
        articles = self.datasets["Articles"]

        for i in range(self.num_ventes):
            article = articles.sample(n=1).iloc[0]

            # Generate sale data
            date_vente = fake.date_time_between(
                start_date=self.start_date, end_date=self.end_date
            )
            quantite_facturee = random.randint(1, 50)
            prix_vente = article["Prix_Vente_TND"]

            # Add some price variation (±10%)
            prix_vente = round(prix_vente * random.uniform(0.9, 1.1), 2)

            # Calculate monthly figures
            ventes_mensuelles = random.randint(50, 500)
            ca_mensuel = round(ventes_mensuelles * prix_vente, 2)
            profit_mensuel = round(ca_mensuel * random.uniform(0.1, 0.3), 2)

            vente = {
                "Id": i + 1,
                "Date": date_vente,
                "QuantiteFacturee": quantite_facturee,
                "NumeroFacture": f"FAC{random.randint(100000, 999999)}",
                "NumLigne": f"{random.randint(1, 99):02d}",
                "Prix_Vente_TND": prix_vente,
                "Cout_Unitaire_TND": article["Prix_Achat_TND"],
                "Ventes_Mensuelles_Unites": ventes_mensuelles,
                "CA_Mensuel_TND": ca_mensuel,
                "Profit_Mensuel_TND": profit_mensuel,
                "Date_Derniere_Vente": date_vente,
                "ArticleId": article["Id"],
            }
            ventes.append(vente)

        df = pd.DataFrame(ventes)
        self.datasets["Ventes"] = df
        return df

    def generate_stock_etablissements(self):
        """Generate StockEtablissement (Many-to-Many relationship) dataset"""
        if "Stocks" not in self.datasets:
            self.generate_stocks()
        if "Etablissements" not in self.datasets:
            self.generate_etablissements()

        stock_etablissements = []
        stocks = self.datasets["Stocks"]["Id"].tolist()
        etablissements = self.datasets["Etablissements"]["Id"].tolist()

        # Generate relationships (each stock can be in multiple establishments)
        for stock_id in stocks:
            # Each stock appears in 1-5 establishments
            num_etablissements = random.randint(1, 5)
            selected_etablissements = random.sample(etablissements, num_etablissements)

            for etab_id in selected_etablissements:
                stock_etab = {
                    "StockId": stock_id,
                    "EtablissementId": etab_id,
                    "QuantiteAllouee": random.randint(10, 200),
                    "DateAffectation": fake.date_between(
                        start_date=self.start_date, end_date=self.end_date
                    ),
                }
                stock_etablissements.append(stock_etab)

        df = pd.DataFrame(stock_etablissements)
        self.datasets["StockEtablissements"] = df
        return df

    def generate_all_datasets(self):
        """Generate all datasets"""
        print("🚀 Generating SmartPromo datasets...")

        print("📦 Generating Categories...")
        self.generate_categories()

        print("🛍️ Generating Articles...")
        self.generate_articles()

        print("🏢 Generating Etablissements...")
        self.generate_etablissements()

        print("👥 Generating Users...")
        self.generate_users()

        print("📊 Generating Stocks...")
        self.generate_stocks()

        print("🎯 Generating Promotions...")
        self.generate_promotions()

        print("💰 Generating Ventes...")
        self.generate_ventes()

        print("🔗 Generating Stock-Etablissement relationships...")
        self.generate_stock_etablissements()

        print("✅ All datasets generated successfully!")
        return self.datasets

    def save_datasets(self, output_format="csv", output_dir="datasets"):
        """Save all datasets to files"""
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for entity_name, df in self.datasets.items():
            if output_format == "csv":
                filename = f"{output_dir}/{entity_name.lower()}.csv"
                df.to_csv(filename, index=False, encoding="utf-8")
                print(
                    f"💾 Saved {entity_name} dataset to {filename} ({len(df)} records)"
                )

            elif output_format == "json":
                filename = f"{output_dir}/{entity_name.lower()}.json"
                df.to_json(filename, orient="records", date_format="iso", indent=2)
                print(
                    f"💾 Saved {entity_name} dataset to {filename} ({len(df)} records)"
                )

            elif output_format == "excel":
                filename = f"{output_dir}/{entity_name.lower()}.xlsx"
                df.to_excel(filename, index=False)
                print(
                    f"💾 Saved {entity_name} dataset to {filename} ({len(df)} records)"
                )

    def get_dataset_summary(self):
        """Get summary statistics for all datasets"""
        summary = {}

        for entity_name, df in self.datasets.items():
            summary[entity_name] = {
                "Records": len(df),
                "Columns": len(df.columns),
                "Memory_Usage_MB": round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "Column_Names": list(df.columns),
            }

        return summary

    def print_dataset_info(self):
        """Print information about all generated datasets"""
        print("\n" + "=" * 60)
        print("📊 SMARTPROMO DATASET SUMMARY")
        print("=" * 60)

        total_records = 0
        for entity_name, df in self.datasets.items():
            total_records += len(df)
            print(
                f"🗂️  {entity_name:20} | {len(df):6,} records | {len(df.columns):2} columns"
            )

        print("-" * 60)
        print(f"📈 TOTAL RECORDS: {total_records:,}")
        print(f"🏷️  TOTAL ENTITIES: {len(self.datasets)}")
        print("=" * 60)


def main():
    """Main function to generate and save datasets"""
    print("🎯 SmartPromo Dataset Generator")
    print("================================")

    # Initialize generator
    generator = SmartPromoDatasetGenerator(seed=42)

    # Generate all datasets
    datasets = generator.generate_all_datasets()

    # Print summary
    generator.print_dataset_info()

    # Save datasets in multiple formats
    print("\n💾 Saving datasets...")
    generator.save_datasets(output_format="csv", output_dir="datasets_csv")
    generator.save_datasets(output_format="json", output_dir="datasets_json")

    # Show detailed summary
    print("\n📋 Detailed Dataset Summary:")
    summary = generator.get_dataset_summary()
    for entity, info in summary.items():
        print(f"\n🗂️  {entity}:")
        print(f"   Records: {info['Records']:,}")
        print(f"   Columns: {info['Columns']}")
        print(f"   Memory: {info['Memory_Usage_MB']} MB")

    print("\n🎉 Dataset generation completed successfully!")
    print("📁 Files saved in:")
    print("   - datasets_csv/ (CSV format)")
    print("   - datasets_json/ (JSON format)")

    return datasets


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import faker
    except ImportError:
        print("Installing required packages...")
        import subprocess

        subprocess.check_call(["pip", "install", "faker"])
        import faker

    datasets = main()
