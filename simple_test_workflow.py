"""
Simple Test for Promotion Approval Workflow
"""

import pyodbc
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta


def test_promotion_approval():
    """Test the promotion approval workflow"""
    connection_string = "mssql+pyodbc://(localdb)\\MSSQLLocalDB/Promotion?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    engine = create_engine(connection_string)

    print("🚀 TESTING PROMOTION APPROVAL WORKFLOW")
    print("=" * 50)

    try:
        with engine.connect() as conn:
            # Step 1: Create test article if it doesn't exist
            print("1. Creating test article...")
            result = conn.execute(
                text(
                    "SELECT COUNT(*) as count FROM Articles WHERE CodeArticle = 'TEST001'"
                )
            )
            if result.fetchone().count == 0:
                conn.execute(
                    text(
                        """
                    INSERT INTO Articles (
                        CodeArticle, Libelle, Fournisseur, IdCategorie,
                        Prix_Vente_TND, Prix_Achat_TND,
                        CodeBarre, CodeDim1, LibelleDim1, CodeDim2, LibelleDim2,
                        FamilleNiv1, FamilleNiv2, FamilleNiv3, FamilleNiv4, FamilleNiv5, FamilleNiv6, FamilleNiv7, FamilleNiv8,
                        Quantite_Achat, DateLibre
                    ) VALUES (
                        'TEST001', 'Test Article for Promotion', 'Test Supplier', 'CAT001',
                        100.0, 70.0,
                        '1234567890', '', '', '', '',
                        'Electronics', 'Smartphones', '', '', '', '', '', '',
                        '1', GETDATE()
                    )
                """
                    )
                )
                print("   ✅ Test article created")
            else:
                print("   ✅ Test article already exists")

            # Step 2: Create test promotion
            print("2. Creating test promotion...")
            conn.execute(
                text(
                    """
                INSERT INTO Promotions (
                    DateFin, TauxReduction, CodeArticle, 
                    Prix_Vente_TND_Avant, Prix_Vente_TND_Apres, 
                    IsAccepted
                ) VALUES (
                    :DateFin, :TauxReduction, :CodeArticle,
                    :Prix_Vente_TND_Avant, :Prix_Vente_TND_Apres, 
                    :IsAccepted
                )
            """
                ),
                {
                    "DateFin": datetime.now() + timedelta(days=30),
                    "TauxReduction": 15.0,
                    "CodeArticle": "TEST001",
                    "Prix_Vente_TND_Avant": 100.0,
                    "Prix_Vente_TND_Apres": 85.0,
                    "IsAccepted": False,
                },
            )

            # Get the promotion ID
            result = conn.execute(
                text(
                    "SELECT TOP 1 Id FROM Promotions WHERE CodeArticle = 'TEST001' ORDER BY Id DESC"
                )
            )
            promotion_id = result.fetchone().Id
            print(f"   ✅ Test promotion created with ID: {promotion_id}")

            # Step 3: Check current state
            print("3. Checking current state...")
            result = conn.execute(
                text(
                    "SELECT Prix_Vente_TND FROM Articles WHERE CodeArticle = 'TEST001'"
                )
            )
            original_price = result.fetchone().Prix_Vente_TND
            print(f"   Original article price: {original_price} TND")

            result = conn.execute(
                text("SELECT IsAccepted FROM Promotions WHERE Id = :id"),
                {"id": promotion_id},
            )
            is_accepted = result.fetchone().IsAccepted
            print(f"   Promotion status: {'Approved' if is_accepted else 'Pending'}")

            # Step 4: Approve the promotion
            print("4. Approving promotion...")
            conn.execute(
                text(
                    """
                UPDATE Promotions 
                SET IsAccepted = 1, DateApproval = GETDATE(), ApprovedBy = 'Test Script'
                WHERE Id = :promotion_id
            """
                ),
                {"promotion_id": promotion_id},
            )

            # Get the new price from promotion
            result = conn.execute(
                text("SELECT Prix_Vente_TND_Apres FROM Promotions WHERE Id = :id"),
                {"id": promotion_id},
            )
            new_price = result.fetchone().Prix_Vente_TND_Apres

            # Update article price
            conn.execute(
                text(
                    """
                UPDATE Articles 
                SET Prix_Vente_TND = :new_price
                WHERE CodeArticle = 'TEST001'
            """
                ),
                {"new_price": new_price},
            )

            conn.commit()
            print(
                f"   ✅ Promotion approved and article price updated to {new_price} TND"
            )

            # Step 5: Verify the changes
            print("5. Verifying changes...")
            result = conn.execute(
                text(
                    """
                SELECT p.IsAccepted, p.DateApproval, p.ApprovedBy, a.Prix_Vente_TND
                FROM Promotions p
                INNER JOIN Articles a ON p.CodeArticle = a.CodeArticle
                WHERE p.Id = :id
            """
                ),
                {"id": promotion_id},
            )

            row = result.fetchone()
            print(f"   Promotion approved: {row.IsAccepted}")
            print(f"   Approval date: {row.DateApproval}")
            print(f"   Approved by: {row.ApprovedBy}")
            print(f"   Current article price: {row.Prix_Vente_TND} TND")

            # Step 6: Show statistics
            print("6. Current promotion statistics...")
            result = conn.execute(
                text(
                    """
                SELECT 
                    COUNT(*) as Total,
                    SUM(CASE WHEN IsAccepted = 1 THEN 1 ELSE 0 END) as Approved,
                    SUM(CASE WHEN IsAccepted = 0 THEN 1 ELSE 0 END) as Pending
                FROM Promotions
            """
                )
            )

            stats = result.fetchone()
            print(f"   Total promotions: {stats.Total}")
            print(f"   Approved: {stats.Approved}")
            print(f"   Pending: {stats.Pending}")

            # Ask if user wants to clean up
            cleanup = (
                input("\\nDo you want to clean up test data? (y/N): ").lower().strip()
            )
            if cleanup == "y":
                conn.execute(
                    text("DELETE FROM Promotions WHERE CodeArticle = 'TEST001'")
                )
                conn.execute(text("DELETE FROM Articles WHERE CodeArticle = 'TEST001'"))
                conn.commit()
                print("✅ Test data cleaned up")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

    print("\\n🎉 PROMOTION APPROVAL WORKFLOW TEST COMPLETED SUCCESSFULLY!")
    return True


if __name__ == "__main__":
    test_promotion_approval()
