"""
Flask API Server for AI Promotion Generator
Provides REST endpoints to access AI promotion functionality
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from database_promotion_model import DatabasePromotionModel
from datetime import datetime
import json
import traceback
import pandas as pd

app = Flask(__name__)

# Configure CORS to allow requests from Angular dev server
CORS(
    app,
    origins=[
        "http://localhost:4200",  # Angular dev server default port
        "http://localhost:4201",  # Alternative Angular port
        "http://127.0.0.1:4200",  # Alternative localhost format
        "http://127.0.0.1:4201",  # Alternative localhost format
    ],
    supports_credentials=True,
)

# Global AI model instance
ai_model = None


def initialize_ai_model():
    """Initialize and train the AI model"""
    global ai_model

    try:
        print("🤖 Initializing AI Promotion Model...")
        ai_model = DatabasePromotionModel()

        # Connect to database
        if not ai_model.connect_to_database():
            raise Exception("Cannot connect to database")

        # Load data
        if not ai_model.load_database_tables():
            raise Exception("Cannot load database tables")

        # Train model if not already trained
        if not ai_model.is_trained:
            print("🎯 Training AI model...")
            ai_model.df_enhanced = ai_model.create_enhanced_dataset_from_database()
            ai_model.df_enhanced = ai_model.calculate_optimal_promotions_from_db(
                ai_model.df_enhanced
            )

            # Prepare features and targets
            X, feature_names = ai_model.prepare_features_from_database(
                ai_model.df_enhanced
            )
            y = ai_model.df_enhanced["Optimal_Promotion_Rate"]

            # Train models
            ai_model.train_models_on_database_data(X, y)
            print("✅ AI model trained successfully!")

        return True
    except Exception as e:
        print(f"❌ Error initializing AI model: {str(e)}")
        return False


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "success",
            "message": "AI Promotion Generator API is running",
            "model_trained": ai_model.is_trained if ai_model else False,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/articles", methods=["GET"])
def get_articles():
    """Get list of available articles"""
    try:
        if not ai_model or not ai_model.is_trained:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AI model not initialized or trained",
                    }
                ),
                500,
            )

        articles = ai_model.df_articles[
            ["CodeArticle", "Libelle", "FamilleNiv2", "Prix_Vente_TND"]
        ].to_dict("records")

        return jsonify({"status": "success", "data": articles, "count": len(articles)})

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/predict", methods=["POST"])
def predict_promotion():
    """Predict promotion for specific article and date"""
    try:
        if not ai_model or not ai_model.is_trained:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AI model not initialized or trained",
                    }
                ),
                500,
            )

        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        article_code = data.get("article_code", "").strip().upper()
        target_date = data.get("target_date", "").strip()

        if not article_code or not target_date:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Both article_code and target_date are required",
                    }
                ),
                400,
            )

        # Validate article exists
        if article_code not in ai_model.df_articles["CodeArticle"].values:
            return (
                jsonify(
                    {"status": "error", "message": f"Article {article_code} not found"}
                ),
                404,
            )

        # Validate date format
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Invalid date format. Use YYYY-MM-DD",
                    }
                ),
                400,
            )

        # Generate prediction
        prediction = ai_model.predict_promotion_for_article_and_date(
            article_code, target_date
        )

        # Get article details
        article_info = (
            ai_model.df_articles[ai_model.df_articles["CodeArticle"] == article_code]
            .iloc[0]
            .to_dict()
        )

        return jsonify(
            {
                "status": "success",
                "data": {
                    "article_code": article_code,
                    "article_info": {
                        "libelle": article_info.get("Libelle", ""),
                        "famille": article_info.get("FamilleNiv2", ""),
                        "prix_original": article_info.get("Prix_Vente_TND", 0),
                    },
                    "target_date": target_date,
                    "prediction": prediction,
                },
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/generate-and-save", methods=["POST"])
def generate_and_save_promotion():
    """Generate AI promotion and save it to database"""
    try:
        if not ai_model or not ai_model.is_trained:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AI model not initialized or trained",
                    }
                ),
                500,
            )

        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        article_code = data.get("article_code", "").strip().upper()
        target_date = data.get("target_date", "").strip()
        auto_save = data.get("auto_save", True)

        if not article_code or not target_date:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Both article_code and target_date are required",
                    }
                ),
                400,
            )

        # Validate article exists
        if article_code not in ai_model.df_articles["CodeArticle"].values:
            return (
                jsonify(
                    {"status": "error", "message": f"Article {article_code} not found"}
                ),
                404,
            )

        # Validate date format
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Invalid date format. Use YYYY-MM-DD",
                    }
                ),
                400,
            )

        # Generate prediction
        prediction = ai_model.predict_promotion_for_article_and_date(
            article_code, target_date
        )

        # Get article details
        article_info = (
            ai_model.df_articles[ai_model.df_articles["CodeArticle"] == article_code]
            .iloc[0]
            .to_dict()
        )

        result = {
            "article_code": article_code,
            "article_info": {
                "libelle": article_info.get("Libelle", ""),
                "famille": article_info.get("FamilleNiv2", ""),
                "prix_original": article_info.get("Prix_Vente_TND", 0),
            },
            "target_date": target_date,
            "prediction": prediction,
            "saved_to_database": False,
        }

        # Save to database if requested
        if auto_save:
            success = ai_model.save_promotion_to_database(
                prediction, article_code, target_date
            )
            result["saved_to_database"] = success
            if success:
                result["message"] = (
                    "Promotion generated and saved successfully (Pending approval)"
                )
            else:
                result["message"] = "Promotion generated but failed to save to database"
        else:
            result["message"] = "Promotion generated successfully (Not saved)"

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Get information about the AI model"""
    try:
        if not ai_model:
            return (
                jsonify({"status": "error", "message": "AI model not initialized"}),
                500,
            )

        info = {
            "is_trained": ai_model.is_trained,
            "models_available": list(ai_model.models.keys()) if ai_model.models else [],
            "feature_importance": (
                ai_model.feature_importance
                if hasattr(ai_model, "feature_importance")
                else {}
            ),
            "database_connected": hasattr(ai_model, "df_articles")
            and ai_model.df_articles is not None,
        }

        if hasattr(ai_model, "df_articles") and ai_model.df_articles is not None:
            info["articles_count"] = len(ai_model.df_articles)

        if hasattr(ai_model, "df_promotions") and ai_model.df_promotions is not None:
            info["promotions_count"] = len(ai_model.df_promotions)

        return jsonify({"status": "success", "data": info})

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Get list of article categories"""
    try:
        if not ai_model or not ai_model.is_trained:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AI model not initialized or trained",
                    }
                ),
                500,
            )

        categories = ai_model.df_articles["FamilleNiv2"].unique().tolist()
        categories = [cat for cat in categories if pd.notna(cat)]  # Remove NaN values

        return jsonify(
            {"status": "success", "data": sorted(categories), "count": len(categories)}
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Predict promotions for multiple articles and dates"""
    try:
        if not ai_model or not ai_model.is_trained:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AI model not initialized or trained",
                    }
                ),
                500,
            )

        data = request.get_json()
        if not data or "requests" not in data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": 'No requests provided. Expected format: {"requests": [{"article_code": "...", "target_date": "..."}]}',
                    }
                ),
                400,
            )

        requests_list = data["requests"]
        if not isinstance(requests_list, list):
            return (
                jsonify({"status": "error", "message": "requests must be a list"}),
                400,
            )

        results = []

        for i, req in enumerate(requests_list):
            try:
                article_code = req.get("article_code", "").strip().upper()
                target_date = req.get("target_date", "").strip()

                if not article_code or not target_date:
                    results.append(
                        {
                            "index": i,
                            "status": "error",
                            "message": "Both article_code and target_date are required",
                        }
                    )
                    continue

                # Validate article exists
                if article_code not in ai_model.df_articles["CodeArticle"].values:
                    results.append(
                        {
                            "index": i,
                            "status": "error",
                            "message": f"Article {article_code} not found",
                        }
                    )
                    continue

                # Validate date format
                try:
                    datetime.strptime(target_date, "%Y-%m-%d")
                except ValueError:
                    results.append(
                        {
                            "index": i,
                            "status": "error",
                            "message": "Invalid date format. Use YYYY-MM-DD",
                        }
                    )
                    continue

                # Generate prediction
                prediction = ai_model.predict_promotion_for_article_and_date(
                    article_code, target_date
                )

                # Get article details
                article_info = (
                    ai_model.df_articles[
                        ai_model.df_articles["CodeArticle"] == article_code
                    ]
                    .iloc[0]
                    .to_dict()
                )

                results.append(
                    {
                        "index": i,
                        "status": "success",
                        "data": {
                            "article_code": article_code,
                            "article_info": {
                                "libelle": article_info.get("Libelle", ""),
                                "famille": article_info.get("FamilleNiv2", ""),
                                "prix_original": article_info.get("Prix_Vente_TND", 0),
                            },
                            "target_date": target_date,
                            "prediction": prediction,
                        },
                    }
                )

            except Exception as e:
                results.append({"index": i, "status": "error", "message": str(e)})

        return jsonify(
            {
                "status": "success",
                "data": results,
                "total_requests": len(requests_list),
                "successful_predictions": len(
                    [r for r in results if r["status"] == "success"]
                ),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    print("🚀 Starting AI Promotion Generator API Server...")
    print("=" * 60)

    # Initialize AI model
    if initialize_ai_model():
        print("✅ AI Model initialized successfully!")
        print("🌐 Starting Flask server...")
        print("📡 API Endpoints:")
        print("   GET  /                     - Health check")
        print("   GET  /api/articles         - Get available articles")
        print("   GET  /api/categories       - Get article categories")
        print("   GET  /api/model-info       - Get AI model information")
        print("   POST /api/predict          - Predict promotion for article")
        print("   POST /api/generate-and-save - Generate and save promotion")
        print("   POST /api/batch-predict    - Batch predict multiple promotions")
        print("=" * 60)
        print("🎯 Server running on: http://localhost:5000")
        print("💡 Use Ctrl+C to stop the server")

        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("❌ Failed to initialize AI model. Server not started.")
