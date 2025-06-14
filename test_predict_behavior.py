"""
Quick test to verify API behavior - check if /api/predict saves to database
"""

import requests
import json


def test_predict_vs_database():
    """Test if /api/predict actually saves to database"""

    print("🔍 Testing if /api/predict saves to database...")

    # First, get current promotions count
    try:
        print("1. Getting current promotion count...")
        model_info_response = requests.get("http://localhost:5000/api/model-info")
        if model_info_response.status_code == 200:
            current_count = (
                model_info_response.json().get("data", {}).get("promotions_count", 0)
            )
            print(f"   Current promotions in DB: {current_count}")
        else:
            print("   Could not get current count")
            current_count = None
    except Exception as e:
        print(f"   Error getting count: {e}")
        current_count = None

    # Call /api/predict
    print("2. Calling /api/predict...")
    try:
        predict_payload = {"article_code": "ART001", "target_date": "2025-06-20"}

        predict_response = requests.post(
            "http://localhost:5000/api/predict",
            json=predict_payload,
            headers={"Content-Type": "application/json"},
        )

        print(f"   Status: {predict_response.status_code}")
        if predict_response.status_code == 200:
            data = predict_response.json()
            if data.get("status") == "success":
                print("   ✅ Prediction successful")
                prediction = data["data"]["prediction"]
                print(
                    f"   Recommended discount: {prediction.get('adjusted_promotion_pct', 0):.1f}%"
                )
            else:
                print(f"   ❌ Prediction failed: {data.get('message')}")
        else:
            print(f"   ❌ HTTP Error: {predict_response.text}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Check promotions count again
    print("3. Checking promotion count after prediction...")
    try:
        model_info_response2 = requests.get("http://localhost:5000/api/model-info")
        if model_info_response2.status_code == 200:
            new_count = (
                model_info_response2.json().get("data", {}).get("promotions_count", 0)
            )
            print(f"   New promotions in DB: {new_count}")

            if current_count is not None:
                if new_count > current_count:
                    print(
                        f"   🚨 WARNING: Promotion count increased by {new_count - current_count}!"
                    )
                    print("   🚨 /api/predict IS saving to database!")
                    return True
                else:
                    print("   ✅ Promotion count unchanged")
                    print("   ✅ /api/predict is NOT saving to database")
                    return False
            else:
                print("   ⚠️  Could not compare counts")
                return None
        else:
            print("   Could not get new count")
            return None

    except Exception as e:
        print(f"   Error getting new count: {e}")
        return None


if __name__ == "__main__":
    result = test_predict_vs_database()

    if result is True:
        print("\n🚨 CONFIRMED: /api/predict is incorrectly saving to database!")
        print("   This should be fixed - predict should only predict, not save.")
    elif result is False:
        print(
            "\n✅ CONFIRMED: /api/predict is working correctly - not saving to database."
        )
    else:
        print("\n⚠️  Could not determine behavior definitively.")
