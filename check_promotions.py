"""
Check saved promotions in database
"""

from database_promotion_model import DatabasePromotionModel


def check_promotions():
    """Check what promotions are in the database"""
    model = DatabasePromotionModel()

    if model.connect_to_database():
        print("📋 CHECKING SAVED PROMOTIONS:")
        print("=" * 60)

        # Check all promotions
        model.list_pending_promotions()

        # Also check all promotions (including approved ones)
        try:
            query = """
            SELECT p.Id, p.CodeArticle, a.Libelle, p.DateFin, 
                   p.TauxReduction, p.Prix_Vente_TND_Avant, p.Prix_Vente_TND_Apres,
                   p.isAccepted, p.DateCreation
            FROM Promotions p
            LEFT JOIN Articles a ON p.CodeArticle = a.CodeArticle
            ORDER BY p.DateCreation DESC
            """

            import pandas as pd
            from sqlalchemy import text

            with model.engine.connect() as conn:
                all_promotions = pd.read_sql(query, conn)

            if len(all_promotions) > 0:
                print(f"\n📊 ALL PROMOTIONS IN DATABASE ({len(all_promotions)} total):")
                print("-" * 80)
                for _, promo in all_promotions.iterrows():
                    status = "✅ Approved" if promo["isAccepted"] else "⏳ Pending"
                    print(
                        f"ID: {promo['Id']} | {promo['CodeArticle']} | {promo['Libelle'][:30]} | {promo['DateFin'].strftime('%Y-%m-%d')} | {promo['TauxReduction']*100:.1f}% | {status}"
                    )
            else:
                print("❌ No promotions found in database.")

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    check_promotions()
