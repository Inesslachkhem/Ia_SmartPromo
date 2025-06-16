"""
Fixed Dataset Generator for SmartPromo
Creates properly formatted CSV files for Articles and Categories
"""

import csv
import random
from datetime import datetime, timedelta

class FixedDatasetGenerator:
    def __init__(self):
        # Product categories
        self.categories = [
            {"IdCategorie": "CAT001", "Nom": "Électronique", "Description": "Produits électroniques et électroménagers"},
            {"IdCategorie": "CAT002", "Nom": "Vêtements", "Description": "Vêtements et accessoires de mode"},
            {"IdCategorie": "CAT003", "Nom": "Alimentation", "Description": "Produits alimentaires et boissons"},
            {"IdCategorie": "CAT004", "Nom": "Hygiène", "Description": "Produits d'hygiène et de beauté"},
            {"IdCategorie": "CAT005", "Nom": "Maison", "Description": "Articles pour la maison et décoration"},
            {"IdCategorie": "CAT006", "Nom": "Sport", "Description": "Articles de sport et loisirs"},
            {"IdCategorie": "CAT007", "Nom": "Automobile", "Description": "Pièces et accessoires automobiles"},
            {"IdCategorie": "CAT008", "Nom": "Informatique", "Description": "Matériel informatique et accessoires"},
            {"IdCategorie": "CAT009", "Nom": "Téléphonie", "Description": "Téléphones et accessoires"},
            {"IdCategorie": "CAT010", "Nom": "Santé", "Description": "Produits de santé et bien-être"},
        ]
        
        # Suppliers
        self.suppliers = [
            "Samsung Tunisia", "LG Electronics", "Sony Middle East", "Apple MENA",
            "Nestlé Tunisia", "Danone Tunisia", "L'Oréal Tunisie", "Unilever North Africa",
            "Adidas MENA", "Nike Middle East", "Puma Tunisia", "Decathlon Tunisia",
            "Carrefour Tunisie", "Monoprix", "Géant Tunisia"
        ]
        
        # Product templates by category
        self.product_templates = {
            "CAT001": [  # Électronique
                {"name": "Télévision LED", "price_range": (800, 3000), "brands": ["Samsung", "LG", "Sony"]},
                {"name": "Réfrigérateur", "price_range": (1200, 2500), "brands": ["Samsung", "LG", "Beko"]},
                {"name": "Machine à laver", "price_range": (900, 2000), "brands": ["LG", "Samsung", "Bosch"]},
            ],
            "CAT002": [  # Vêtements
                {"name": "T-shirt", "price_range": (25, 80), "brands": ["Zara", "H&M", "Bershka"]},
                {"name": "Pantalon", "price_range": (50, 150), "brands": ["Zara", "Mango", "Pull&Bear"]},
                {"name": "Robe", "price_range": (60, 200), "brands": ["H&M", "Zara", "Mango"]},
            ],
            "CAT003": [  # Alimentation
                {"name": "Huile d'olive", "price_range": (15, 35), "brands": ["Zituna", "Safi", "Olio"]},
                {"name": "Pâtes", "price_range": (3, 8), "brands": ["Barilla", "Panzani", "Mamma"]},
                {"name": "Yaourt", "price_range": (2, 6), "brands": ["Délice", "Danone", "Vitalait"]},
            ],
            "CAT004": [  # Hygiène
                {"name": "Shampoing", "price_range": (12, 40), "brands": ["L'Oréal", "Head&Shoulders", "Pantene"]},
                {"name": "Dentifrice", "price_range": (8, 25), "brands": ["Colgate", "Signal", "Sensodyne"]},
                {"name": "Savon", "price_range": (5, 15), "brands": ["Dove", "Nivea", "Palmolive"]},
            ],
            "CAT005": [  # Maison
                {"name": "Oreiller", "price_range": (30, 80), "brands": ["IKEA", "Casa", "Mobilia"]},
                {"name": "Lampe", "price_range": (40, 120), "brands": ["IKEA", "Philips", "Casa"]},
                {"name": "Tapis", "price_range": (60, 200), "brands": ["Casa", "Mobilia", "IKEA"]},
            ],
            "CAT006": [  # Sport
                {"name": "Chaussures de sport", "price_range": (80, 300), "brands": ["Nike", "Adidas", "Puma"]},
                {"name": "Ballon de football", "price_range": (25, 60), "brands": ["Nike", "Adidas", "Puma"]},
                {"name": "Raquette de tennis", "price_range": (150, 400), "brands": ["Wilson", "Head", "Babolat"]},
            ],
            "CAT007": [  # Automobile
                {"name": "Pneu", "price_range": (200, 500), "brands": ["Michelin", "Bridgestone", "Continental"]},
                {"name": "Batterie", "price_range": (150, 300), "brands": ["Varta", "Bosch", "Exide"]},
                {"name": "Huile moteur", "price_range": (40, 80), "brands": ["Mobil", "Castrol", "Shell"]},
            ],
            "CAT008": [  # Informatique
                {"name": "Ordinateur portable", "price_range": (1500, 4000), "brands": ["HP", "Dell", "Lenovo"]},
                {"name": "Souris", "price_range": (25, 80), "brands": ["Logitech", "Microsoft", "HP"]},
                {"name": "Clavier", "price_range": (30, 120), "brands": ["Logitech", "Microsoft", "Corsair"]},
            ],
            "CAT009": [  # Téléphonie
                {"name": "Smartphone", "price_range": (400, 2500), "brands": ["Samsung", "Apple", "Huawei"]},
                {"name": "Écouteurs", "price_range": (30, 200), "brands": ["Samsung", "Apple", "Sony"]},
                {"name": "Chargeur", "price_range": (15, 50), "brands": ["Samsung", "Apple", "Anker"]},
            ],
            "CAT010": [  # Santé
                {"name": "Vitamines", "price_range": (20, 60), "brands": ["Centrum", "Supradyn", "Pharmaton"]},
                {"name": "Thermomètre", "price_range": (25, 80), "brands": ["Braun", "Omron", "Beurer"]},
                {"name": "Masque facial", "price_range": (5, 20), "brands": ["3M", "Medicom", "Kimberly"]},
            ],
        }

    def generate_categories_csv(self, filename="categories_import.csv"):
        """Generate categories CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["IdCategorie", "Nom", "Description"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for category in self.categories:
                writer.writerow(category)
        print(f"Created categories CSV: {filename}")

    def generate_articles_csv(self, filename="articles_import.csv", num_articles=200):
        """Generate articles CSV file"""
        articles = []
        
        for i in range(1, num_articles + 1):
            # Select random category
            category = random.choice(self.categories)
            category_id = category["IdCategorie"]
            
            # Get product template for this category
            templates = self.product_templates.get(category_id, [])
            if templates:
                template = random.choice(templates)
                product_name = template["name"]
                price_range = template["price_range"]
                brands = template["brands"]
                brand = random.choice(brands)
                product_full_name = f"{brand} {product_name}"
            else:
                product_full_name = f"Produit {i}"
                price_range = (10, 100)
            
            # Generate prices
            prix_vente = round(random.uniform(*price_range), 2)
            prix_achat = round(prix_vente * random.uniform(0.6, 0.8), 2)  # 20-40% margin
            
            # Generate barcode (13 digits)
            barcode = f"123456{i:07d}"  # Ensure it's exactly 13 digits
            
            # Generate dates
            base_date = datetime(2025, 6, 16)
            date_libre = (base_date + timedelta(days=random.randint(-365, 365))).strftime('%Y-%m-%d')
            
            article = {
                "Id": i,
                "CodeArticle": f"ART{i:06d}",
                "CodeBarre": barcode,
                "Libelle": product_full_name,
                "CodeDim1": f"DIM1-{i}",
                "LibelleDim1": f"Dimension 1 Article {i}",
                "CodeDim2": f"DIM2-{i}",
                "LibelleDim2": f"Dimension 2 Article {i}",
                "Fournisseur": random.choice(self.suppliers),
                "FamilleNiv1": f"Famille1-{i}",
                "FamilleNiv2": f"Famille2-{i}",
                "FamilleNiv3": f"Famille3-{i}",
                "FamilleNiv4": f"Famille4-{i}",
                "FamilleNiv5": f"Famille5-{i}",
                "FamilleNiv6": f"Famille6-{i}",
                "FamilleNiv7": f"Famille7-{i}",
                "FamilleNiv8": f"Famille8-{i}",
                "Quantite_Achat": str(random.randint(1, 100)),
                "DateLibre": date_libre,
                "Prix_Vente_TND": prix_vente,
                "Prix_Achat_TND": prix_achat,
                "IdCategorie": category_id
            }
            articles.append(article)
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "Id", "CodeArticle", "CodeBarre", "Libelle", 
                "CodeDim1", "LibelleDim1", "CodeDim2", "LibelleDim2", 
                "Fournisseur", "FamilleNiv1", "FamilleNiv2", "FamilleNiv3", 
                "FamilleNiv4", "FamilleNiv5", "FamilleNiv6", "FamilleNiv7", 
                "FamilleNiv8", "Quantite_Achat", "DateLibre", 
                "Prix_Vente_TND", "Prix_Achat_TND", "IdCategorie"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in articles:
                writer.writerow(article)
        
        print(f"Created articles CSV: {filename} with {num_articles} articles")

    def generate_all(self):
        """Generate both categories and articles CSV files"""
        print("Generating SmartPromo datasets...")
        self.generate_categories_csv()
        self.generate_articles_csv()
        print("Dataset generation complete!")

# Run if executed directly
if __name__ == "__main__":
    generator = FixedDatasetGenerator()
    generator.generate_all()
