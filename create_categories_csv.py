"""
Create a properly formatted categories CSV file for import
"""

import pandas as pd
import csv


def create_categories_csv():
    """Create a clean categories CSV file suitable for import"""

    categories = [
        {
            "IdCategorie": "CAT001",
            "Nom": "Alimentation",
            "Description": "Produits alimentaires et boissons",
        },
        {
            "IdCategorie": "CAT002",
            "Nom": "Hygiène",
            "Description": "Produits d'hygiène personnelle",
        },
        {
            "IdCategorie": "CAT003",
            "Nom": "Cosmétiques",
            "Description": "Produits de beauté et cosmétiques",
        },
        {
            "IdCategorie": "CAT004",
            "Nom": "Vêtements",
            "Description": "Vêtements et accessoires de mode",
        },
        {
            "IdCategorie": "CAT005",
            "Nom": "Électronique",
            "Description": "Appareils électroniques et gadgets",
        },
        {
            "IdCategorie": "CAT006",
            "Nom": "Électroménager",
            "Description": "Appareils électroménagers pour la maison",
        },
        {
            "IdCategorie": "CAT007",
            "Nom": "Maison & Jardin",
            "Description": "Articles pour la maison et le jardin",
        },
        {
            "IdCategorie": "CAT008",
            "Nom": "Sport & Loisirs",
            "Description": "Équipements sportifs et loisirs",
        },
        {
            "IdCategorie": "CAT009",
            "Nom": "Automobiles",
            "Description": "Accessoires et pièces automobiles",
        },
        {
            "IdCategorie": "CAT010",
            "Nom": "Librairie",
            "Description": "Livres et fournitures de bureau",
        },
        {
            "IdCategorie": "CAT011",
            "Nom": "Santé",
            "Description": "Produits de santé et bien-être",
        },
        {
            "IdCategorie": "CAT012",
            "Nom": "Bébé & Enfant",
            "Description": "Produits pour bébés et enfants",
        },
        {
            "IdCategorie": "CAT013",
            "Nom": "Informatique",
            "Description": "Matériel informatique et logiciels",
        },
        {
            "IdCategorie": "CAT014",
            "Nom": "Téléphonie",
            "Description": "Téléphones et accessoires",
        },
        {
            "IdCategorie": "CAT015",
            "Nom": "Jouets",
            "Description": "Jouets et jeux pour enfants",
        },
        {
            "IdCategorie": "CAT016",
            "Nom": "Bijouterie",
            "Description": "Bijoux et montres",
        },
        {
            "IdCategorie": "CAT017",
            "Nom": "Chaussures",
            "Description": "Chaussures pour tous âges",
        },
        {
            "IdCategorie": "CAT018",
            "Nom": "Maroquinerie",
            "Description": "Sacs et accessoires en cuir",
        },
        {
            "IdCategorie": "CAT019",
            "Nom": "Parfumerie",
            "Description": "Parfums et fragrances",
        },
        {
            "IdCategorie": "CAT020",
            "Nom": "Boissons",
            "Description": "Boissons alcoolisées et non-alcoolisées",
        },
    ]

    # Create DataFrame
    df = pd.DataFrame(categories)

    # Save to CSV with proper formatting
    df.to_csv(
        "categories_import.csv",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    print("✅ Created categories_import.csv with 20 categories")
    print("📁 File ready for import via API")

    # Print sample content
    print("\n📋 Sample content:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    create_categories_csv()
