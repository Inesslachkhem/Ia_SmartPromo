"""
Create import-ready CSV files for all SmartPromo entities
"""

import pandas as pd
import csv
from datetime import datetime, timedelta
import random


def create_articles_csv():
    """Create articles CSV"""
    articles = []
    categories = [f"CAT{i+1:03d}" for i in range(20)]

    suppliers = [
        "Groupe Poulina",
        "Délice Danone",
        "SOTUMAG",
        "Yazaki Tunisia",
        "Leoni AG",
        "One Tech Group",
        "Tunisie Telecom",
        "Orange Tunisie",
    ]

    families = [
        "Alimentation",
        "Boissons",
        "Hygiène",
        "Cosmétiques",
        "Vêtements",
        "Électronique",
        "Électroménager",
        "Maison & Jardin",
        "Sport & Loisirs",
    ]

    for i in range(100):  # Create 100 articles
        article = {
            "Id": i + 1,
            "CodeArticle": f"ART{i+1:06d}",
            "CodeBarre": f"123456{i+1:06d}",
            "Libelle": f"Article {i+1}",
            "CodeDim1": f"DIM1-{i+1}",
            "LibelleDim1": f"Dimension 1-{i+1}",
            "CodeDim2": f"DIM2-{i+1}",
            "LibelleDim2": f"Dimension 2-{i+1}",
            "Fournisseur": random.choice(suppliers),
            "FamilleNiv1": random.choice(families),
            "FamilleNiv2": f"Sous-famille {random.randint(1, 5)}",
            "FamilleNiv3": f"Catégorie {random.randint(1, 3)}",
            "FamilleNiv4": f"Type {random.randint(1, 2)}",
            "FamilleNiv5": "",
            "FamilleNiv6": "",
            "FamilleNiv7": "",
            "FamilleNiv8": "",
            "Quantite_Achat": str(random.randint(1, 100)),
            "DateLibre": datetime.now().strftime("%Y-%m-%d"),
            "Prix_Vente_TND": round(random.uniform(5.0, 500.0), 2),
            "Prix_Achat_TND": 0,  # Will calculate
            "IdCategorie": random.choice(categories),
        }

        # Calculate purchase price (70% of selling price)
        article["Prix_Achat_TND"] = round(article["Prix_Vente_TND"] * 0.7, 2)
        articles.append(article)

    df = pd.DataFrame(articles)
    df.to_csv("articles_import.csv", index=False, encoding="utf-8")
    print(f"✅ Created articles_import.csv with {len(articles)} articles")


def create_etablissements_csv():
    """Create etablissements CSV"""
    cities = [
        "Tunis",
        "Sfax",
        "Sousse",
        "Kairouan",
        "Bizerte",
        "Gabès",
        "Ariana",
        "Monastir",
        "Ben Arous",
        "Nabeul",
    ]

    types = ["Magasin", "Supermarché", "Hypermarché", "Boutique", "Entrepôt"]
    sectors = ["Centre Ville", "Banlieue", "Zone Industrielle", "Centre Commercial"]

    etablissements = []
    for i in range(10):
        etablissement = {
            "Id": i + 1,
            "Code": f"ETB{i+1:03d}",
            "Libelle": f"Établissement {i+1}",
            "Adresse": f"Adresse {i+1}",
            "Ville": random.choice(cities),
            "Type": random.choice(types),
            "Secteur": random.choice(sectors),
        }
        etablissements.append(etablissement)

    df = pd.DataFrame(etablissements)
    df.to_csv("etablissements_import.csv", index=False, encoding="utf-8")
    print(
        f"✅ Created etablissements_import.csv with {len(etablissements)} établissements"
    )


def create_users_csv():
    """Create users CSV"""
    users = []
    first_names = [
        "Ahmed",
        "Fatma",
        "Mohamed",
        "Amal",
        "Karim",
        "Salma",
        "Omar",
        "Nadia",
        "Sami",
        "Leila",
    ]
    last_names = [
        "Ben Ali",
        "Trabelsi",
        "Mahmoudi",
        "Bouazizi",
        "Gharbi",
        "Mansouri",
        "Khelifi",
        "Sassi",
        "Jaziri",
        "Ferchichi",
    ]

    for i in range(20):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        user = {
            "Id": i + 1,
            "Nom": last_name,
            "Prenom": first_name,
            "Email": f"{first_name.lower()}.{last_name.lower()}@smartpromo.tn",
            "Type": random.choice([0, 1, 2]),  # 0=Admin, 1=Manager, 2=Employee
            "PasswordHash": "hashed_password_placeholder",
            "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "UpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "LastLogin": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IsActive": True,
            "RefreshToken": None,
            "RefreshTokenExpiryTime": None,
        }
        users.append(user)

    df = pd.DataFrame(users)
    df.to_csv("users_import.csv", index=False, encoding="utf-8")
    print(f"✅ Created users_import.csv with {len(users)} users")


def create_stocks_csv():
    """Create stocks CSV"""
    stocks = []
    for i in range(50):  # 50 stock entries
        article_id = random.randint(1, 100)
        quantite = random.randint(0, 1000)

        stock = {
            "Id": i + 1,
            "QuantitePhysique": quantite,
            "StockMin": random.randint(10, 100),
            "VenteFFO": random.randint(0, 50),
            "LivreFou": random.randint(0, 100),
            "Transfert": random.randint(0, 20),
            "AnnonceTrf": random.randint(0, 30),
            "Valeur_Stock_TND": round(quantite * random.uniform(5.0, 50.0), 2),
            "ArticleId": article_id,
        }
        stocks.append(stock)

    df = pd.DataFrame(stocks)
    df.to_csv("stocks_import.csv", index=False, encoding="utf-8")
    print(f"✅ Created stocks_import.csv with {len(stocks)} stocks")


def create_promotions_csv():
    """Create promotions CSV"""
    promotions = []
    for i in range(30):  # 30 promotions
        date_creation = datetime.now() - timedelta(days=random.randint(1, 90))
        date_fin = date_creation + timedelta(days=random.randint(7, 30))

        prix_avant = round(random.uniform(10.0, 200.0), 2)
        taux_reduction = round(random.uniform(0.05, 0.50), 2)
        prix_apres = round(prix_avant * (1 - taux_reduction), 2)

        promotion = {
            "Id": i + 1,
            "DateFin": date_fin.strftime("%Y-%m-%d"),
            "TauxReduction": taux_reduction,
            "CodeArticle": f"ART{random.randint(1, 100):06d}",
            "Prix_Vente_TND_Avant": prix_avant,
            "Prix_Vente_TND_Apres": prix_apres,
            "IsAccepted": random.choice([True, False]),
            "DateCreation": date_creation.strftime("%Y-%m-%d"),
            "DateApproval": (
                date_creation.strftime("%Y-%m-%d") if random.random() > 0.3 else None
            ),
            "ApprovedBy": (
                f"User{random.randint(1, 10)}" if random.random() > 0.3 else None
            ),
        }
        promotions.append(promotion)

    df = pd.DataFrame(promotions)
    df.to_csv("promotions_import.csv", index=False, encoding="utf-8")
    print(f"✅ Created promotions_import.csv with {len(promotions)} promotions")


def create_ventes_csv():
    """Create ventes CSV"""
    ventes = []
    for i in range(200):  # 200 sales records
        date_vente = datetime.now() - timedelta(days=random.randint(1, 365))
        quantite = random.randint(1, 20)
        prix_vente = round(random.uniform(5.0, 200.0), 2)

        vente = {
            "Id": i + 1,
            "Date": date_vente.strftime("%Y-%m-%d"),
            "QuantiteFacturee": quantite,
            "NumeroFacture": f"FAC{random.randint(100000, 999999)}",
            "NumLigne": f"{random.randint(1, 99):02d}",
            "Prix_Vente_TND": prix_vente,
            "Cout_Unitaire_TND": round(prix_vente * 0.7, 2),
            "Ventes_Mensuelles_Unites": random.randint(50, 500),
            "CA_Mensuel_TND": round(random.uniform(1000.0, 10000.0), 2),
            "Profit_Mensuel_TND": round(random.uniform(100.0, 2000.0), 2),
            "Date_Derniere_Vente": date_vente.strftime("%Y-%m-%d"),
            "ArticleId": random.randint(1, 100),
        }
        ventes.append(vente)

    df = pd.DataFrame(ventes)
    df.to_csv("ventes_import.csv", index=False, encoding="utf-8")
    print(f"✅ Created ventes_import.csv with {len(ventes)} ventes")


def main():
    """Create all import CSV files"""
    print("📊 Creating import-ready CSV files for SmartPromo entities...")
    print("=" * 60)

    create_articles_csv()
    create_etablissements_csv()
    create_users_csv()
    create_stocks_csv()
    create_promotions_csv()
    create_ventes_csv()

    print("=" * 60)
    print("🎉 All import CSV files created successfully!")
    print("\n📁 Files created:")
    print("   - categories_import.csv (20 categories)")
    print("   - articles_import.csv (100 articles)")
    print("   - etablissements_import.csv (10 établissements)")
    print("   - users_import.csv (20 users)")
    print("   - stocks_import.csv (50 stocks)")
    print("   - promotions_import.csv (30 promotions)")
    print("   - ventes_import.csv (200 ventes)")

    print("\n🚀 Ready to import via API endpoints!")


if __name__ == "__main__":
    main()
