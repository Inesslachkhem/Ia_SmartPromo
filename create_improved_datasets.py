"""
SmartPromo Improved Dataset Generator
Generates realistic test data for Articles and Categories with the latest model structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from faker import Faker
import json
import csv

# Initialize Faker for generating realistic data
fake = Faker(['fr_FR', 'en_US'])  # French and English locales

class ImprovedDatasetGenerator:
    """
    Generates improved datasets for Articles and Categories with the latest model structure
    """

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

        # Configuration for data generation
        self.num_categories = 100  # More categories for better diversity
        self.num_articles = 2000   # More articles for comprehensive testing

        # Date ranges (current date in 2025)
        self.current_date = datetime(2025, 6, 16)
        self.start_date = self.current_date - timedelta(days=365*2)  # 2 years ago
        self.end_date = self.current_date + timedelta(days=365)      # 1 year ahead

        # Price ranges (in TND - Tunisian Dinar)
        self.min_price = 5.0
        self.max_price = 2000.0

        # Product categories organized by department (more structured)
        self.departments = {
            "Alimentation": ["Conserves", "Produits laitiers", "Boulangerie", "Épicerie", "Viandes", "Poissons", "Fruits et légumes", "Surgelés"],
            "Boissons": ["Eau", "Jus", "Sodas", "Thé", "Café", "Boissons alcoolisées"],
            "Hygiène": ["Soins corporels", "Soins capillaires", "Soins dentaires", "Soins intimes"],
            "Cosmétiques": ["Maquillage", "Parfums", "Soins visage", "Soins anti-âge"],
            "Vêtements": ["Hommes", "Femmes", "Enfants", "Bébés", "Sport", "Accessoires"],
            "Électronique": ["TV", "Audio", "Photo", "Accessoires", "Gaming"],
            "Informatique": ["Ordinateurs", "Périphériques", "Composants", "Réseau", "Stockage"],
            "Téléphonie": ["Smartphones", "Accessoires", "Tablettes", "Objets connectés"],
            "Électroménager": ["Gros électroménager", "Petit électroménager", "Cuisine", "Entretien"],
            "Maison & Jardin": ["Décoration", "Linge de maison", "Mobilier", "Bricolage", "Jardinage"],
            "Santé": ["Médicaments", "Compléments alimentaires", "Premiers secours", "Matériel médical"],
            "Bébé & Enfant": ["Puériculture", "Jouets", "Vêtements", "Alimentation", "Hygiène"]
        }

        # Supplier names (expanded list with international and local suppliers)
        self.suppliers = [
            "Groupe Poulina", "Délice Danone", "SOTUMAG", "Yazaki Tunisia", "Leoni AG", 
            "One Tech Group", "Tunisie Telecom", "Orange Tunisie", "Monoprix", "Carrefour Tunisie",
            "Géant", "Aziza", "Senia", "UTIC", "Samsung Electronics", "LG Tunisia", 
            "Huawei Technologies Tunisia", "Microsoft Tunisia", "Apple MENA", "Sony Middle East",
            "Philips North Africa", "Nestlé Tunisia", "Coca-Cola Tunisia", "PepsiCo Maghreb",
            "L'Oréal Tunisie", "Unilever North Africa", "P&G Maghreb", "Adidas MENA",
            "Nike Middle East", "Decathlon Tunisia", "IKEA Tunisia", "Toyota Tunisia"
        ]

        # Initialize data containers
        self.datasets = {}

    def generate_categories(self):
        """Generate improved Categories dataset with more structure"""
        categories = []
        cat_id = 1
        
        # Generate structured categories based on departments and subcategories
        for department, subcategories in self.departments.items():
            # Create main department category
            dept_category = {
                "IdCategorie": f"CAT{cat_id:03d}",
                "Nom": department,
                "Description": f"Département {department} - Catégorie principale regroupant tous les produits {department.lower()}"
            }
            categories.append(dept_category)
            cat_id += 1
            
            # Create subcategories for each department
            for subcategory in subcategories:
                sub_category = {
                    "IdCategorie": f"CAT{cat_id:03d}",
                    "Nom": f"{department} - {subcategory}",
                    "Description": f"Produits {subcategory.lower()} du département {department.lower()}"
                }
                categories.append(sub_category)
                cat_id += 1
        
        # Add some additional random categories to reach the target number
        while len(categories) < self.num_categories:
            random_dept = random.choice(list(self.departments.keys()))
            category = {
                "IdCategorie": f"CAT{cat_id:03d}",
                "Nom": f"{random_dept} - {fake.word().capitalize()}",
                "Description": fake.text(max_nb_chars=150)
            }
            categories.append(category)
            cat_id += 1
            
        df = pd.DataFrame(categories)
        self.datasets["Categories"] = df
        return df

    def generate_articles(self):
        """Generate improved Articles dataset with more realistic data"""
        if "Categories" not in self.datasets:
            self.generate_categories()
            
        articles = []
        categories = self.datasets["Categories"]["IdCategorie"].tolist()
        
        # Product types with brands for more realistic product names
        product_catalog = {
            "Smartphone": ["Samsung", "Apple", "Huawei", "Xiaomi", "Oppo", "Vivo", "Realme", "OnePlus"],
            "Ordinateur": ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Toshiba"],
            "Télévision": ["Samsung", "LG", "Sony", "Philips", "Hisense", "Panasonic", "TCL"],
            "Réfrigérateur": ["Samsung", "LG", "Bosch", "Whirlpool", "Electrolux", "Beko", "Haier"],
            "Machine à laver": ["LG", "Samsung", "Bosch", "Whirlpool", "Electrolux", "Beko", "Siemens"],
            "Vêtement": ["Zara", "H&M", "Bershka", "Pull & Bear", "Mango", "C&A", "LC Waikiki"],
            "Chaussures": ["Nike", "Adidas", "Puma", "New Balance", "Asics", "Reebok", "Converse"],
            "Parfum": ["Dior", "Chanel", "Guerlain", "Yves Saint Laurent", "Hugo Boss", "Armani", "Lancôme"],
            "Produit laitier": ["Délice", "Danone", "Vitalait", "Président", "Yoplait", "Candia", "Activia"],
            "Boisson gazeuse": ["Coca-Cola", "Pepsi", "Fanta", "Sprite", "7Up", "Boga", "Hamoud Boualem"],
            "Huile": ["Zituna", "Safi", "Olio", "Borges", "Lesieur", "Cristal", "Oleoliva"],
            "Céréales": ["Kellogg's", "Nestlé", "Quaker", "Trésor", "Chocapic", "Fitness", "Corn Flakes"]
        }
        
        for i in range(self.num_articles):
            # Select random product type and brand
            product_type = random.choice(list(product_catalog.keys()))
            brand = random.choice(product_catalog[product_type])
            
            # Determine category based on product type (more logical assignment)
            if product_type in ["Smartphone", "Ordinateur"]:
                category_prefix = "CAT" + str(random.randint(65, 75)).zfill(3)  # Electronique/Informatique categories
            elif product_type in ["Télévision", "Réfrigérateur", "Machine à laver"]:
                category_prefix = "CAT" + str(random.randint(76, 85)).zfill(3)  # Electroménager categories
            elif product_type in ["Vêtement", "Chaussures"]:
                category_prefix = "CAT" + str(random.randint(30, 40)).zfill(3)  # Vêtements categories
            elif product_type in ["Parfum"]:
                category_prefix = "CAT" + str(random.randint(20, 30)).zfill(3)  # Cosmétiques categories
            elif product_type in ["Produit laitier", "Boisson gazeuse", "Huile", "Céréales"]:
                category_prefix = "CAT" + str(random.randint(1, 20)).zfill(3)  # Alimentation categories
            else:
                category_prefix = random.choice(categories)
                
            # Match with an existing category or fallback to random
            matching_categories = [cat for cat in categories if cat.startswith(category_prefix[:5])]
            selected_category = random.choice(matching_categories) if matching_categories else random.choice(categories)
            
            # Generate model/variant for the product
            models = ["Pro", "Max", "Plus", "Ultra", "Lite", "Standard", "Premium", "Basic", "Elite", "Advanced"]
            sizes = ["S", "M", "L", "XL", "XXL", "Mini", "Maxi"] if product_type in ["Vêtement"] else []
            sizes.extend(["32\"", "43\"", "55\"", "65\"", "75\""] if product_type == "Télévision" else [])
            sizes.extend(["13\"", "14\"", "15\"", "17\""] if product_type == "Ordinateur" else [])
            
            model = ""
            if random.random() > 0.3:  # 70% chance to have a model
                model = random.choice(models) if models else ""
            
            size = ""
            if sizes and random.random() > 0.5:  # 50% chance to have a size if applicable
                size = random.choice(sizes)
            
            # Generate realistic libellé (product name)
            if size and model:
                libelle = f"{brand} {product_type} {model} {size} {fake.word().capitalize()}"
            elif size:
                libelle = f"{brand} {product_type} {size} {fake.word().capitalize()}"
            elif model:
                libelle = f"{brand} {product_type} {model} {fake.word().capitalize()}"
            else:
                libelle = f"{brand} {product_type} {fake.word().capitalize()}"
            
            # Generate realistic dimensions
            dim_types = {
                "Smartphone": ["Couleur", "Stockage"],
                "Ordinateur": ["CPU", "RAM"],
                "Télévision": ["Résolution", "Smart TV"],
                "Vêtement": ["Couleur", "Matière"],
                "Chaussures": ["Couleur", "Pointure"],
            }
            
            dim1_type = dim_types.get(product_type, ["Type", "Variant"])[0]
            dim2_type = dim_types.get(product_type, ["Type", "Variant"])[1]
            
            dim1_values = {
                "Couleur": ["Noir", "Blanc", "Bleu", "Rouge", "Vert", "Gris", "Or", "Argent"],
                "Stockage": ["64GB", "128GB", "256GB", "512GB", "1TB"],
                "CPU": ["Intel i3", "Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"],
                "RAM": ["4GB", "8GB", "16GB", "32GB"],
                "Résolution": ["HD", "Full HD", "4K", "8K"],
                "Smart TV": ["Oui", "Non"],
                "Matière": ["Coton", "Polyester", "Laine", "Lin", "Soie"],
                "Pointure": [str(p) for p in range(36, 47)],
                "Type": ["Standard", "Deluxe", "Basic"],
                "Variant": ["V1", "V2", "V3", "2023", "2024", "2025"]
            }
            
            dim1_value = random.choice(dim1_values.get(dim1_type, ["Standard"]))
            dim2_value = random.choice(dim1_values.get(dim2_type, ["Standard"]))
            
            # Calculate realistic prices based on product type
            base_prices = {
                "Smartphone": (300, 3000),
                "Ordinateur": (800, 5000),
                "Télévision": (500, 8000),
                "Réfrigérateur": (700, 3500),
                "Machine à laver": (800, 2500),
                "Vêtement": (20, 200),
                "Chaussures": (50, 300),
                "Parfum": (80, 500),
                "Produit laitier": (1, 15),
                "Boisson gazeuse": (1, 8),
                "Huile": (10, 30),
                "Céréales": (5, 20)
            }
            
            min_price, max_price = base_prices.get(product_type, (self.min_price, self.max_price))
            prix_vente = round(random.uniform(min_price, max_price), 2)
            
            # More realistic purchase price based on product type margins
            margin_ranges = {
                "Smartphone": (0.65, 0.8),  # Higher margins
                "Ordinateur": (0.7, 0.85),
                "Télévision": (0.7, 0.85),
                "Réfrigérateur": (0.7, 0.8),
                "Machine à laver": (0.7, 0.8),
                "Vêtement": (0.3, 0.5),    # Lower margins
                "Chaussures": (0.4, 0.6),
                "Parfum": (0.2, 0.4),      # Very high margins (low purchase cost)
                "Produit laitier": (0.7, 0.85),
                "Boisson gazeuse": (0.6, 0.75),
                "Huile": (0.8, 0.9),
                "Céréales": (0.7, 0.85)
            }
            
            margin_min, margin_max = margin_ranges.get(product_type, (0.6, 0.8))
            prix_achat = round(prix_vente * random.uniform(margin_min, margin_max), 2)
            
            # Generate article with all properties
            article = {
                "Id": i + 1,
                "CodeArticle": f"ART{i+1:06d}",
                "CodeBarre": fake.ean13(),
                "Libelle": libelle[:200],  # Ensure it fits within 200 chars
                "CodeDim1": f"DIM1-{dim1_type}",
                "LibelleDim1": dim1_value,
                "CodeDim2": f"DIM2-{dim2_type}",
                "LibelleDim2": dim2_value,
                "Fournisseur": random.choice(self.suppliers),
                "FamilleNiv1": product_type,
                "FamilleNiv2": brand,
                "FamilleNiv3": model if model else "",
                "FamilleNiv4": size if size else "",
                "FamilleNiv5": dim1_value,
                "FamilleNiv6": "",
                "FamilleNiv7": "",
                "FamilleNiv8": "",
                "Quantite_Achat": str(random.randint(1, 100)),
                "DateLibre": fake.date_between(start_date=self.start_date, end_date=self.end_date).strftime('%Y-%m-%d'),
                "Prix_Vente_TND": prix_vente,
                "Prix_Achat_TND": prix_achat,
                "IdCategorie": selected_category
            }
            
            articles.append(article)
            
        df = pd.DataFrame(articles)
        self.datasets["Articles"] = df
        return df
        
    def save_datasets(self, output_dir="datasets_csv", format="csv"):
        """Save datasets to CSV/JSON files"""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate datasets if not already generated
        if "Categories" not in self.datasets:
            self.generate_categories()
        if "Articles" not in self.datasets:
            self.generate_articles()
            
        # Save each dataset
        for name, df in self.datasets.items():
            if format.lower() == "csv":
                filepath = os.path.join(output_dir, f"{name.lower()}_import.csv")
                df.to_csv(filepath, index=False)
                print(f"Saved {name} dataset to {filepath}")
            elif format.lower() == "json":
                filepath = os.path.join(output_dir, f"{name.lower()}_import.json")
                df.to_json(filepath, orient="records", indent=2)
                print(f"Saved {name} dataset to {filepath}")
            else:
                print(f"Unsupported format: {format}")    def create_csv_for_import(self, output_dir="datasets_csv"):
        """Create CSV files formatted specifically for import in SmartPromo"""
        # Create output directories if they don't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate datasets if not already generated
        if "Categories" not in self.datasets:
            self.generate_categories()
        if "Articles" not in self.datasets:
            self.generate_articles()
            
        # Save articles with the expected format for import
        articles_df = self.datasets["Articles"]
        categories_df = self.datasets["Categories"]
        
        # Format articles CSV as expected by the import function
        articles_csv_path = os.path.join(output_dir, "articles_import.csv")
        
        # Create CSV with headers matching the expected format
        with open(articles_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
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
            
            for _, row in articles_df.iterrows():
                writer.writerow(row.to_dict())
                
        print(f"Saved articles import CSV to {articles_csv_path}")
          # Format categories CSV as expected by the import function
        categories_csv_path = os.path.join(output_dir, "categories_import.csv")
        
        # Create CSV with headers matching the expected format  
        with open(categories_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["IdCategorie", "Nom", "Description"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for _, row in categories_df.iterrows():
                writer.writerow(row.to_dict())
                
        print(f"Saved categories import CSV to {categories_csv_path}")

        # Also create a unique categories file with just the IDs for testing
        categories_unique_path = os.path.join(output_dir, "categories_unique_import.csv")
        with open(categories_unique_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["IdCategorie"])
            
            for cat_id in categories_df["IdCategorie"]:
                writer.writerow([cat_id])
                
        print(f"Saved unique categories CSV to {categories_unique_path}")
        
# Run the generator if executed directly
if __name__ == "__main__":
    print("Generating improved datasets for Articles and Categories...")
    generator = ImprovedDatasetGenerator()
    generator.create_csv_for_import()
    generator.save_datasets(format="json", output_dir="datasets_json")
    print("Dataset generation complete!")
