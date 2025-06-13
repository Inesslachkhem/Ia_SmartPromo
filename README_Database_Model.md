# Database-Driven AI Promotion Model

This is an enhanced AI promotion optimization model that reads directly from your SQL Server database instead of CSV files. It analyzes real data from your database tables to generate intelligent promotion recommendations.

## 🎯 Features

- **Direct Database Integration**: Reads from SQL Server LocalDB/Express
- **Real-time Data**: Uses current data from your database tables
- **Multi-table Analysis**: Combines Articles, Promotions, Stocks, Ventes, and Categories
- **Date-based Predictions**: Generates promotions for specific dates with seasonal adjustments
- **Business Intelligence**: Calculates stock rotation, performance scores, and price elasticity
- **Machine Learning**: Uses Random Forest, XGBoost, and LightGBM for predictions

## 📊 Database Tables Used

| Table          | Purpose               | Key Columns                                                                     |
| -------------- | --------------------- | ------------------------------------------------------------------------------- |
| **Articles**   | Product catalog       | CodeArticle, Libelle, FamilleNiv1, FamilleNiv2, Fournisseur                     |
| **Promotions** | Historical promotions | CodeArticle, TauxReduction, Prix_Vente_TND_Avant, Prix_Vente_TND_Apres, DateFin |
| **Stocks**     | Inventory data        | ArticleId, QuantitePhysique, StockMin, Valeur_Stock_TND                         |
| **Ventes**     | Sales history         | ArticleId, Quantite, PrixUnitaire, DateVente, Total                             |
| **Categories** | Product categories    | IdCategorie, Nom, Description                                                   |

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd "c:\Users\belha\Desktop\AI integration v2\Ia_SmartPromo"
python setup_database_model.py
```

### Step 2: Test Database Connection

```bash
python test_database_model.py
```

### Step 3: Run Full Model

```bash
python database_promotion_model.py
```

## 💻 Usage Examples

### Basic Model Execution

```python
from database_promotion_model import DatabasePromotionModel

# Initialize model
model = DatabasePromotionModel()

# Connect and load data
model.connect_to_database()
model.load_database_tables()

# Create enhanced dataset
model.df_enhanced = model.create_enhanced_dataset_from_database()
model.df_enhanced = model.calculate_optimal_promotions_from_db(model.df_enhanced)

# Train the model
X, features = model.prepare_features_from_database(model.df_enhanced)
y = model.df_enhanced["Optimal_Promotion_Rate"]
model.train_models_on_database_data(X, y)
```

### Predict Promotion for Specific Article and Date

```python
# Predict promotion for a specific article on Christmas
prediction = model.predict_promotion_for_article_and_date(
    code_article="ART001",
    target_date="2025-12-25"
)

print(f"Recommended promotion: {prediction['adjusted_promotion_pct']}%")
print(f"New price: {prediction['promoted_price_tnd']} TND")
print(f"Expected revenue impact: {prediction['revenue_impact_tnd']} TND")
```

### Batch Predictions for Multiple Articles

```python
# Get all article codes
article_codes = model.df_enhanced['CodeArticle'].tolist()

# Predict for summer season
for article in article_codes[:10]:  # First 10 articles
    try:
        prediction = model.predict_promotion_for_article_and_date(
            article, "2025-07-15"
        )
        print(f"{article}: {prediction['adjusted_promotion_pct']}% promotion")
    except Exception as e:
        print(f"Error for {article}: {e}")
```

## 📈 Business Intelligence Features

### Stock Analysis

- **Stock Rotation**: Measures how fast products sell relative to inventory
- **Stock Coverage**: Calculates months of inventory remaining
- **Low Stock Detection**: Identifies items below minimum stock levels

### Performance Metrics

- **Performance Score**: Combines recency, rotation, and revenue factors
- **Price Elasticity**: Estimates customer price sensitivity
- **Promotion Frequency**: Tracks historical promotion patterns

### Seasonal Intelligence

- **Holiday Seasons**: Adjusts for Christmas, summer, Ramadan periods
- **Weekend Effects**: Considers day-of-week impact
- **Competitive Periods**: Accounts for high-competition times

## 🎛️ Configuration

### Database Connection

The model uses your existing SQL Server LocalDB connection:

```
Server=(localdb)\MSSQLLocalDB;Database=Promotion;Trusted_Connection=True;
```

To use a different database, modify the connection string:

```python
model = DatabasePromotionModel(
    connection_string="your_custom_connection_string"
)
```

### Model Parameters

You can adjust the promotion calculation weights:

```python
# In calculate_optimal_promotions_from_db method
optimal_promotion = (
    stock_factor * 0.25 +        # Inventory management weight
    rotation_factor * 0.20 +     # Sales velocity weight
    time_factor * 0.15 +         # Freshness weight
    hist_factor * 0.15 +         # Historical success weight
    elasticity_factor * 0.15 +   # Price sensitivity weight
    performance_factor * 0.10    # Overall performance weight
)
```

## 📊 Output Files

The model generates several output files:

### 1. database_promotion_recommendations.xlsx

- **Promotion_Recommendations**: Main results with optimal promotions
- **Database_Summary**: Overview of database tables and record counts

### 2. Console Output

- Database connection status
- Data analysis summary
- Model training results
- Feature importance rankings
- Business insights

## 🛠️ Troubleshooting

### Database Connection Issues

1. **SQL Server not running**: Start SQL Server LocalDB service
2. **Database doesn't exist**: Run Entity Framework migrations first
3. **Permission issues**: Ensure Windows authentication works
4. **Driver missing**: Install ODBC Driver 17 for SQL Server

### Data Issues

1. **No data**: Import CSV data into database first using controllers
2. **Missing tables**: Ensure all Entity Framework migrations are applied
3. **Empty results**: Check that articles exist in Articles table

### Python Issues

1. **Package errors**: Run `python setup_database_model.py` again
2. **Import errors**: Ensure all packages in requirements_database.txt are installed
3. **Memory issues**: Reduce dataset size or increase system memory

## 🔧 Advanced Usage

### Custom Date Features

Add your own seasonal patterns:

```python
def _is_custom_season(self, date):
    # Add your custom business seasons
    return 1 if date.month in [your_months] else 0
```

### Custom Business Logic

Modify promotion calculations:

```python
def calculate_custom_promotions(self, df):
    # Your custom promotion logic
    custom_factor = your_calculation(df)
    return custom_factor
```

### Integration with Web API

Use the model in your ASP.NET Core application:

```python
# Create a Python service endpoint
# Call from C# using Process.Start or HTTP API
```

## 📞 Support

For issues and questions:

1. Check the console output for detailed error messages
2. Run the test script: `python test_database_model.py`
3. Verify database connectivity and data availability
4. Check that all required Python packages are installed

## 🚀 Next Steps

1. **Import your data**: Use the ASP.NET Core controllers to import CSV data
2. **Test the model**: Run basic predictions to verify functionality
3. **Customize**: Adjust business logic weights and seasonal factors
4. **Integrate**: Connect the model to your web application
5. **Monitor**: Track promotion performance and adjust model parameters

The model is designed to learn from your actual business data and improve recommendations over time!
