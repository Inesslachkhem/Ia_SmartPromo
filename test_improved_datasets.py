"""
Simple script to demonstrate the improved dataset generator
"""

from create_improved_datasets import ImprovedDatasetGenerator
import pandas as pd
import os

def main():
    print("===== SMART PROMO IMPROVED DATASET GENERATOR =====")
    print("Generating new datasets for Articles and Categories...")
    
    # Create the generator
    generator = ImprovedDatasetGenerator()
    
    # Generate and save all datasets
    generator.create_csv_for_import()
    generator.save_datasets(format="json", output_dir="datasets_json")
    generator.save_datasets(format="csv", output_dir="datasets_csv")
    
    # Display some sample data
    print("\n===== SAMPLE DATA =====")
    
    # Show sample categories
    categories_df = generator.datasets["Categories"]
    print(f"\nCategories sample (total: {len(categories_df)}):")
    print(categories_df.head(5))
    
    # Show sample articles
    articles_df = generator.datasets["Articles"]
    print(f"\nArticles sample (total: {len(articles_df)}):")
    print(articles_df[["Id", "CodeArticle", "Libelle", "Prix_Vente_TND", "Prix_Achat_TND", "IdCategorie"]].head(5))

    # Verify files created
    print("\n===== FILES CREATED =====")
    print("1. For direct import to SmartPromo:")
    print(f"   - articles_import.csv ({os.path.getsize('articles_import.csv')/1024:.2f} KB)")
    print(f"   - categories_import.csv ({os.path.getsize('categories_import.csv')/1024:.2f} KB)")
    print(f"   - categories_unique_import.csv ({os.path.getsize('categories_unique_import.csv')/1024:.2f} KB)")
    
    print("\n2. CSV format datasets:")
    for file in os.listdir("datasets_csv"):
        print(f"   - datasets_csv/{file} ({os.path.getsize(os.path.join('datasets_csv', file))/1024:.2f} KB)")
        
    print("\n3. JSON format datasets:")
    for file in os.listdir("datasets_json"):
        print(f"   - datasets_json/{file} ({os.path.getsize(os.path.join('datasets_json', file))/1024:.2f} KB)")
    
    print("\n===== IMPORT INSTRUCTIONS =====")
    print("1. First, import categories using 'categories_import.csv'")
    print("2. Then, import articles using 'articles_import.csv'")
    print("3. If you only need to test category IDs, use 'categories_unique_import.csv'")
    
    print("\nGenerator completed successfully!")
    
if __name__ == "__main__":
    main()
