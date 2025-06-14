"""
Test Client for AI Promotion Generator API
Demonstrates how to use the API endpoints
"""

import requests
import json
from datetime import datetime, timedelta

# API Base URL
API_BASE_URL = "http://localhost:5000"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_articles():
    """Test getting available articles"""
    print("\n📦 Testing get articles...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/articles")
        data = response.json()
        print(f"Status: {response.status_code}")
        if data["status"] == "success":
            print(f"Found {data['count']} articles")
            # Show first 5 articles
            for article in data["data"][:5]:
                print(
                    f"  - {article['CodeArticle']}: {article['Libelle']} ({article['FamilleNiv2']})"
                )
        else:
            print(f"Error: {data['message']}")
        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_categories():
    """Test getting article categories"""
    print("\n🏷️ Testing get categories...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/categories")
        data = response.json()
        print(f"Status: {response.status_code}")
        if data["status"] == "success":
            print(f"Found {data['count']} categories:")
            for category in data["data"][:10]:  # Show first 10
                print(f"  - {category}")
        else:
            print(f"Error: {data['message']}")
        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict_promotion():
    """Test predicting a promotion"""
    print("\n🤖 Testing promotion prediction...")
    try:
        # First get an article to test with
        articles_response = requests.get(f"{API_BASE_URL}/api/articles")
        articles_data = articles_response.json()

        if articles_data["status"] != "success" or not articles_data["data"]:
            print("❌ No articles available for testing")
            return False

        # Use the first article
        test_article = articles_data["data"][0]["CodeArticle"]
        test_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"Testing with article: {test_article}, date: {test_date}")

        payload = {"article_code": test_article, "target_date": test_date}

        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        print(f"Status: {response.status_code}")

        if data["status"] == "success":
            prediction_data = data["data"]
            print(f"Article: {prediction_data['article_info']['libelle']}")
            print(
                f"Original Price: {prediction_data['article_info']['prix_original']} TND"
            )
            print(f"Target Date: {prediction_data['target_date']}")
            print(
                f"AI Prediction: {json.dumps(prediction_data['prediction'], indent=2)}"
            )
        else:
            print(f"Error: {data['message']}")

        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_generate_and_save():
    """Test generating and saving a promotion"""
    print("\n💾 Testing generate and save promotion...")
    try:
        # First get an article to test with
        articles_response = requests.get(f"{API_BASE_URL}/api/articles")
        articles_data = articles_response.json()

        if articles_data["status"] != "success" or not articles_data["data"]:
            print("❌ No articles available for testing")
            return False

        # Use the second article if available, or first
        test_article = (
            articles_data["data"][1]["CodeArticle"]
            if len(articles_data["data"]) > 1
            else articles_data["data"][0]["CodeArticle"]
        )
        test_date = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        print(f"Testing with article: {test_article}, date: {test_date}")

        payload = {
            "article_code": test_article,
            "target_date": test_date,
            "auto_save": True,
        }

        response = requests.post(
            f"{API_BASE_URL}/api/generate-and-save",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        print(f"Status: {response.status_code}")

        if data["status"] == "success":
            result = data["data"]
            print(f"Article: {result['article_info']['libelle']}")
            print(f"Original Price: {result['article_info']['prix_original']} TND")
            print(f"Target Date: {result['target_date']}")
            print(f"Saved to Database: {result['saved_to_database']}")
            print(f"Message: {result['message']}")
            print(f"AI Prediction: {json.dumps(result['prediction'], indent=2)}")
        else:
            print(f"Error: {data['message']}")

        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch_predict():
    """Test batch prediction"""
    print("\n📊 Testing batch prediction...")
    try:
        # First get articles to test with
        articles_response = requests.get(f"{API_BASE_URL}/api/articles")
        articles_data = articles_response.json()

        if articles_data["status"] != "success" or len(articles_data["data"]) < 2:
            print("❌ Not enough articles available for batch testing")
            return False

        # Create batch request with first 3 articles
        batch_requests = []
        for i in range(min(3, len(articles_data["data"]))):
            article = articles_data["data"][i]
            test_date = (datetime.now() + timedelta(days=60 + i * 7)).strftime(
                "%Y-%m-%d"
            )
            batch_requests.append(
                {"article_code": article["CodeArticle"], "target_date": test_date}
            )

        payload = {"requests": batch_requests}

        response = requests.post(
            f"{API_BASE_URL}/api/batch-predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        print(f"Status: {response.status_code}")

        if data["status"] == "success":
            print(f"Total Requests: {data['total_requests']}")
            print(f"Successful Predictions: {data['successful_predictions']}")

            for result in data["data"]:
                if result["status"] == "success":
                    pred_data = result["data"]
                    print(
                        f"\n  ✅ {pred_data['article_code']}: {pred_data['article_info']['libelle']}"
                    )
                    print(f"     Target Date: {pred_data['target_date']}")
                    if "promotion_rate" in pred_data["prediction"]:
                        print(
                            f"     Recommended Discount: {pred_data['prediction']['promotion_rate']:.1%}"
                        )
                else:
                    print(f"\n  ❌ Request {result['index']}: {result['message']}")
        else:
            print(f"Error: {data['message']}")

        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_info():
    """Test getting model information"""
    print("\n🔧 Testing model info...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/model-info")
        data = response.json()
        print(f"Status: {response.status_code}")

        if data["status"] == "success":
            info = data["data"]
            print(f"Model Trained: {info['is_trained']}")
            print(f"Database Connected: {info['database_connected']}")
            print(f"Available Models: {info['models_available']}")
            if "articles_count" in info:
                print(f"Articles Count: {info['articles_count']}")
            if "promotions_count" in info:
                print(f"Promotions Count: {info['promotions_count']}")
        else:
            print(f"Error: {data['message']}")

        return response.status_code == 200 and data["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 AI PROMOTION API TESTING")
    print("=" * 50)

    tests = [
        ("Health Check", test_health_check),
        ("Get Articles", test_get_articles),
        ("Get Categories", test_get_categories),
        ("Model Info", test_model_info),
        ("Predict Promotion", test_predict_promotion),
        ("Generate and Save", test_generate_and_save),
        ("Batch Predict", test_batch_predict),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20}")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your AI Promotion API is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the API server and database connection.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted!")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
