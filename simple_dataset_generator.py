"""
Simple data generator for articles and categories
"""
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os
import csv

def create_categories_csv(num_categories=20):
    """Generate category data and save to CSV"""
    categories = []
    for i in range(1, num_categories+1):
        category = {
            "IdCategorie": f"CAT{i:03d}",
            "Nom": f"Catégorie {i}",
            "Description": f"Description de la catégorie {i}"
        }
        categories.append(category)
    
    # Create output directory if it doesn't exist
    os.makedirs("datasets_csv", exist_ok=True)
    
    # Save to CSV
    categories_csv_path = os.path.join("datasets_csv", "categories_import.csv")
    with open(categories_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["IdCategorie", "Nom", "Description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for category in categories:
            writer.writerow(category)
    
    print(f"Created {num_categories} categories in {categories_csv_path}")
    return categories

def create_articles_csv(categories, num_articles=100):
    """Generate article data and save to CSV"""
    articles = []
    category_ids = [c["IdCategorie"] for c in categories]
    
    for i in range(1, num_articles+1):
        # Basic product info
        product_name = f"Produit {i}"
        price = round(random.uniform(10, 1000), 2)
        purchase_price = round(price * 0.7, 2)  # 30% margin
        
        article = {
            "Id": i,
            "CodeArticle": f"ART{i:06d}",
            "CodeBarre": f"123456{i:09d}",
            "Libelle": product_name,
            "CodeDim1": f"DIM1-{i}",
            "LibelleDim1": "Dimension 1",
            "CodeDim2": f"DIM2-{i}",
            "LibelleDim2": "Dimension 2", 
            "Fournisseur": "Supplier",
            "FamilleNiv1": "Famille 1",
            "FamilleNiv2": "Famille 2",
            "FamilleNiv3": "Famille 3",
            "FamilleNiv4": "Famille 4",
            "FamilleNiv5": "Famille 5",
            "FamilleNiv6": "Famille 6",
            "FamilleNiv7": "Famille 7",
            "FamilleNiv8": "Famille 8",
            "Quantite_Achat": str(random.randint(1, 100)),
            "DateLibre": datetime(2025, 6, 16).strftime('%Y-%m-%d'),
            "Prix_Vente_TND": price,
            "Prix_Achat_TND": purchase_price,
            "IdCategorie": random.choice(category_ids)
        }
        articles.append(article)
    
    # Create output directory if it doesn't exist
    os.makedirs("datasets_csv", exist_ok=True)
    
    # Save to CSV
    articles_csv_path = os.path.join("datasets_csv", "articles_import.csv") 
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
        for article in articles:
            writer.writerow(article)
    
    print(f"Created {num_articles} articles in {articles_csv_path}")
    return articles

def main():
    print("Generating simple dataset for testing...")
    categories = create_categories_csv(num_categories=20)
    articles = create_articles_csv(categories, num_articles=100)
    print("Done!")

if __name__ == "__main__":
    main()
