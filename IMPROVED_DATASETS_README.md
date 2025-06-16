# SmartPromo Improved Dataset Generator

This tool generates high-quality test datasets for the SmartPromo application, specifically focused on Articles and Categories with the latest model structure.

## Features

- Generates realistic article and category data matching the latest model
- Creates CSV files ready for direct import into SmartPromo
- Creates both CSV and JSON formats for different use cases
- Supports automatic category creation in the application
- Ensures price fields and required fields meet application requirements

## Datasets Generated

1. **Categories** - Contains structured category data with:
   - Proper IdCategorie format (CAT001, CAT002, etc.)
   - Realistic names and descriptions
   - Organized by departments and subcategories

2. **Articles** - Contains comprehensive article data with:
   - All required fields from the Article model
   - Realistic product names, codes, and descriptions
   - Properly formatted price fields (Prix_Vente_TND, Prix_Achat_TND)
   - Valid category references
   - Realistic supplier information
   - Proper date formatting

## Files Generated

Running the generator creates the following files:

### For Direct Import

- `articles_import.csv` - Article data ready for import into SmartPromo
- `categories_import.csv` - Category data ready for import into SmartPromo
- `categories_unique_import.csv` - Just category IDs for testing purposes

### Additional Dataset Formats

- `datasets_csv/articles_import.csv` - Article data in CSV format
- `datasets_csv/categories_import.csv` - Category data in CSV format
- `datasets_json/articles_import.json` - Article data in JSON format
- `datasets_json/categories_import.json` - Category data in JSON format

## Usage

### Running the Generator

```bash
python create_improved_datasets.py
```

Or run the test script for a demonstration:

```bash
python test_improved_datasets.py
```

### Importing Data into SmartPromo

1. First, import categories:
   - Use `categories_import.csv` with the Categories import function
   
2. Then, import articles:
   - Use `articles_import.csv` with the Articles import function
   
3. For testing category auto-creation:
   - Import articles directly without importing categories first
   - The application should now automatically create missing categories

## Dataset Structure

### Categories

- **IdCategorie**: Unique identifier (e.g., "CAT001")
- **Nom**: Category name
- **Description**: Detailed description of the category

### Articles

- **Id**: Numeric identifier
- **CodeArticle**: Article code (e.g., "ART000001")
- **CodeBarre**: EAN-13 barcode
- **Libelle**: Product name/description
- **CodeDim1, LibelleDim1**: First dimension code and label
- **CodeDim2, LibelleDim2**: Second dimension code and label
- **Fournisseur**: Supplier name
- **FamilleNiv1-8**: Hierarchical family classification
- **Quantite_Achat**: Purchase quantity
- **DateLibre**: Free date field
- **Prix_Vente_TND**: Selling price in TND
- **Prix_Achat_TND**: Purchase price in TND
- **IdCategorie**: Foreign key to Categories

## Customization

You can modify the generator by adjusting the following parameters in `create_improved_datasets.py`:

- `num_categories`: Number of categories to generate
- `num_articles`: Number of articles to generate
- `min_price`, `max_price`: Price ranges
- `departments`: Product category structure
- `suppliers`: List of supplier names

## Requirements

- Python 3.6+
- pandas
- numpy
- faker

Install requirements with:

```bash
pip install pandas numpy faker
```
