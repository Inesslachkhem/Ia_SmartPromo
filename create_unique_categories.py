"""
Create unique categories CSV that won't conflict with existing data
"""

import pandas as pd
import csv
import random


def create_unique_categories_csv():
    """Create a categories CSV with unique IDs"""

    categories = [
        {
            "IdCategorie": "IMPORT001",
            "Nom": "Alimentation Bio",
            "Description": "Produits alimentaires biologiques et naturels",
        },
        {
            "IdCategorie": "IMPORT002",
            "Nom": "Hygiène Corporelle",
            "Description": "Produits d'hygiène pour le corps",
        },
        {
            "IdCategorie": "IMPORT003",
            "Nom": "Cosmétiques Premium",
            "Description": "Produits de beauté haut de gamme",
        },
        {
            "IdCategorie": "IMPORT004",
            "Nom": "Mode Femme",
            "Description": "Vêtements et accessoires pour femmes",
        },
        {
            "IdCategorie": "IMPORT005",
            "Nom": "Mode Homme",
            "Description": "Vêtements et accessoires pour hommes",
        },
        {
            "IdCategorie": "IMPORT006",
            "Nom": "Électronique Grand Public",
            "Description": "Appareils électroniques pour particuliers",
        },
        {
            "IdCategorie": "IMPORT007",
            "Nom": "Électroménager Cuisine",
            "Description": "Appareils électroménagers pour la cuisine",
        },
        {
            "IdCategorie": "IMPORT008",
            "Nom": "Décoration Intérieure",
            "Description": "Articles de décoration pour la maison",
        },
        {
            "IdCategorie": "IMPORT009",
            "Nom": "Jardinage",
            "Description": "Outils et produits pour le jardin",
        },
        {
            "IdCategorie": "IMPORT010",
            "Nom": "Sports Collectifs",
            "Description": "Équipements pour sports d'équipe",
        },
        {
            "IdCategorie": "IMPORT011",
            "Nom": "Fitness",
            "Description": "Équipements de fitness et musculation",
        },
        {
            "IdCategorie": "IMPORT012",
            "Nom": "Automobile",
            "Description": "Pièces et accessoires automobiles",
        },
        {
            "IdCategorie": "IMPORT013",
            "Nom": "Moto",
            "Description": "Pièces et accessoires pour motos",
        },
        {
            "IdCategorie": "IMPORT014",
            "Nom": "Littérature",
            "Description": "Livres et romans",
        },
        {
            "IdCategorie": "IMPORT015",
            "Nom": "Fournitures Bureau",
            "Description": "Matériel de bureau et papeterie",
        },
    ]

    # Create DataFrame
    df = pd.DataFrame(categories)

    # Save to CSV with proper formatting
    df.to_csv(
        "categories_unique_import.csv",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    print("✅ Created categories_unique_import.csv with 15 unique categories")
    print("📁 File ready for safe import via API")

    # Print sample content
    print("\n📋 Sample content:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    create_unique_categories_csv()
