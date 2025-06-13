"""
Simple Promotion Generator
Choose article and date, generate AI promotion, save to database
"""

from database_promotion_model import DatabasePromotionModel
from datetime import datetime


def generate_and_save_promotion():
    """Simple function to generate and save a promotion"""
    print("🎯 AI PROMOTION GENERATOR")
    print("=" * 50)

    # Initialize the AI model
    model = DatabasePromotionModel()

    # Connect to database
    print("🔗 Connecting to database...")
    if not model.connect_to_database():
        print("❌ Cannot connect to database!")
        return

    # Load data
    print("📊 Loading business data...")
    if not model.load_database_tables():
        print("❌ Cannot load database tables!")
        return

    # Train the AI model if needed
    if not model.is_trained:
        print("🤖 Training AI model with your business data...")
        try:
            model.df_enhanced = model.create_enhanced_dataset_from_database()
            model.df_enhanced = model.calculate_optimal_promotions_from_db(
                model.df_enhanced
            )

            # Prepare features and targets
            X, feature_names = model.prepare_features_from_database(model.df_enhanced)
            y = model.df_enhanced["Optimal_Promotion_Rate"]

            # Train models
            model.train_models_on_database_data(X, y)
            print("✅ AI model trained successfully!")
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return

    # Show available articles
    print("\n📦 AVAILABLE ARTICLES:")
    print("-" * 60)
    articles_df = model.df_articles[["CodeArticle", "Libelle", "FamilleNiv2"]].copy()

    for idx, row in articles_df.iterrows():
        print(
            f"  {row['CodeArticle']:<10} | {row['Libelle']:<35} | {row['FamilleNiv2']}"
        )

    print("-" * 60)

    # Get user input for article
    while True:
        article_code = input("\n🎯 Enter article code: ").strip().upper()

        if article_code in model.df_articles["CodeArticle"].values:
            break
        else:
            print(f"❌ Article '{article_code}' not found. Please try again.")

    # Get user input for date
    while True:
        date_input = input("📅 Enter target date (YYYY-MM-DD): ").strip()

        try:
            target_date = datetime.strptime(date_input, "%Y-%m-%d")
            break
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD (e.g., 2025-07-15)")

    # Generate AI promotion
    print(f"\n🤖 Generating AI promotion for {article_code} on {date_input}...")

    try:
        prediction = model.predict_promotion_for_article_and_date(
            article_code, date_input
        )

        # Ask if user wants to save
        print(f"\n💾 Save this promotion to database? (y/n): ", end="")
        save_choice = input().strip().lower()

        if save_choice == "y":
            success = model.save_promotion_to_database(
                prediction, article_code, date_input
            )
            if success:
                print("✅ Promotion saved to database successfully!")
                print("📋 Status: Pending approval (isAccepted = False)")
            else:
                print("❌ Failed to save promotion.")
        else:
            print("ℹ️  Promotion not saved.")

    except Exception as e:
        print(f"❌ Error generating promotion: {str(e)}")


def main():
    """Main function"""
    try:
        generate_and_save_promotion()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
