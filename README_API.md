# AI Promotion Generator API

🤖 **REST API for AI-powered promotion generation and optimization**

This Flask-based API provides endpoints to access the AI promotion generator functionality, allowing you to predict optimal promotions for articles and dates through HTTP requests.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MySQL/SQL Server database with promotion data
- Required Python packages (see requirements_database.txt)

### Installation

1. **Install dependencies:**

```bash
pip install -r requirements_database.txt
```

2. **Configure database connection** in `database_promotion_model.py`

3. **Start the API server:**

```bash
python api_server.py
```

4. **Server will start on:** `http://localhost:5000`

## 📡 API Endpoints

### Health Check

```http
GET /
```

Returns server status and model information.

### Get Articles

```http
GET /api/articles
```

Returns list of available articles with codes, names, and categories.

**Response:**

```json
{
  "status": "success",
  "data": [
    {
      "CodeArticle": "ART001",
      "Libelle": "Product Name",
      "FamilleNiv2": "Category",
      "Prix_Vente_TND": 25.5
    }
  ],
  "count": 150
}
```

### Get Categories

```http
GET /api/categories
```

Returns list of article categories.

### Predict Promotion

```http
POST /api/predict
Content-Type: application/json

{
  "article_code": "ART001",
  "target_date": "2025-07-15"
}
```

Generates AI prediction for optimal promotion on specific article and date.

**Response:**

```json
{
  "status": "success",
  "data": {
    "article_code": "ART001",
    "article_info": {
      "libelle": "Product Name",
      "famille": "Category",
      "prix_original": 25.5
    },
    "target_date": "2025-07-15",
    "prediction": {
      "promotion_rate": 0.15,
      "new_price": 21.68,
      "expected_sales_boost": 1.25,
      "confidence_score": 0.87
    }
  }
}
```

### Generate and Save Promotion

```http
POST /api/generate-and-save
Content-Type: application/json

{
  "article_code": "ART001",
  "target_date": "2025-07-15",
  "auto_save": true
}
```

Generates AI promotion and optionally saves it to the database.

### Batch Prediction

```http
POST /api/batch-predict
Content-Type: application/json

{
  "requests": [
    {
      "article_code": "ART001",
      "target_date": "2025-07-15"
    },
    {
      "article_code": "ART002",
      "target_date": "2025-07-20"
    }
  ]
}
```

Generates predictions for multiple articles and dates in a single request.

### Model Information

```http
GET /api/model-info
```

Returns information about the AI model status and capabilities.

## 🧪 Testing

**Run the test client:**

```bash
python test_api_client.py
```

This will test all API endpoints and show you how to use them.

## 💡 Usage Examples

### Python Client Example

```python
import requests
import json

# Predict promotion
response = requests.post('http://localhost:5000/api/predict',
    json={
        'article_code': 'ART001',
        'target_date': '2025-07-15'
    })

data = response.json()
if data['status'] == 'success':
    prediction = data['data']['prediction']
    print(f"Recommended discount: {prediction['promotion_rate']:.1%}")
    print(f"New price: {prediction['new_price']:.2f} TND")
```

### cURL Example

```bash
# Get all articles
curl -X GET "http://localhost:5000/api/articles"

# Predict promotion
curl -X POST "http://localhost:5000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"article_code": "ART001", "target_date": "2025-07-15"}'
```

### JavaScript/Fetch Example

```javascript
// Predict promotion
fetch("http://localhost:5000/api/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    article_code: "ART001",
    target_date: "2025-07-15",
  }),
})
  .then((response) => response.json())
  .then((data) => {
    if (data.status === "success") {
      console.log("Prediction:", data.data.prediction);
    }
  });
```

## 🔧 Configuration

### CORS Settings

The API includes CORS support for cross-origin requests. You can modify CORS settings in `api_server.py`:

```python
CORS(app, origins=['http://localhost:4200'])  # Restrict to specific origins
```

### Port Configuration

Change the port in `api_server.py`:

```python
app.run(host='0.0.0.0', port=8000, debug=False)  # Production settings
```

## 🛡️ Error Handling

All endpoints return consistent error responses:

```json
{
  "status": "error",
  "message": "Description of the error",
  "traceback": "Detailed error trace (in debug mode)"
}
```

Common HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (article not found)
- `500` - Internal Server Error

## 🔄 Integration with Angular Frontend

This API is designed to work with your Angular SmartPromo frontend. You can call these endpoints from your Angular services:

```typescript
// In your Angular service
predictPromotion(articleCode: string, targetDate: string) {
  return this.http.post('http://localhost:5000/api/predict', {
    article_code: articleCode,
    target_date: targetDate
  });
}
```

## 📊 Performance

- **Prediction speed:** ~100-500ms per request
- **Batch processing:** Up to 100 requests per batch
- **Concurrent requests:** Supports multiple simultaneous predictions
- **Memory usage:** ~200-500MB (depends on dataset size)

## 🚨 Important Notes

1. **Database Connection:** Ensure your database is accessible and contains the required tables
2. **Model Training:** The AI model trains automatically on first startup (may take 1-5 minutes)
3. **Data Format:** Dates must be in YYYY-MM-DD format
4. **Article Codes:** Must match exactly with database entries (case-insensitive)

## 🆘 Troubleshooting

### Common Issues:

**Model not initialized:**

- Check database connection settings
- Ensure required database tables exist
- Check Python dependencies

**Prediction errors:**

- Verify article code exists in database
- Check date format (YYYY-MM-DD)
- Ensure sufficient historical data for training

**Port conflicts:**

- Change port in `api_server.py`
- Kill existing processes on port 5000

---

🎯 **Ready to use!** Start the server and test with the provided client.
