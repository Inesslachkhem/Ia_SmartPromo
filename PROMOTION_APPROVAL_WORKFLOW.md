# Promotion Approval Workflow Implementation

## Overview

This document outlines the complete implementation of the AI-powered promotion recommendation system with an approval workflow that automatically updates article prices when promotions are approved.

## System Components

### 1. Database Schema Updates

#### Promotion Model (C#)

- Added `IsAccepted` field (bool) - tracks approval status
- Added `DateCreation` field (DateTime) - tracks when promotion was created
- Added `DateApproval` field (DateTime?) - tracks when promotion was approved
- Added `ApprovedBy` field (string?) - tracks who approved the promotion

#### Article Model (C#)

- Added `Prix_Vente_TND` field (float) - current selling price
- Added `Prix_Achat_TND` field (float) - purchase price

### 2. Business Logic Implementation

#### PromotionService (C#)

Located: `SmartPromo_Back/Services/PromotionService.cs`

**New Methods Added:**

- `GetPendingPromotionsAsync()` - Gets all promotions awaiting approval
- `GetApprovedPromotionsAsync()` - Gets all approved promotions
- `ApprovePromotionAsync(int promotionId, string approvedBy)` - Approves a promotion and updates article price
- `RejectPromotionAsync(int promotionId, string rejectedBy)` - Rejects and removes a promotion
- `BulkApprovePromotionsAsync(List<int> promotionIds, string approvedBy)` - Bulk approve multiple promotions
- `GetApprovalStatisticsAsync()` - Gets approval statistics

**Core Approval Logic:**

```csharp
public async Task<bool> ApprovePromotionAsync(int promotionId, string approvedBy = "System")
{
    using var transaction = await _context.Database.BeginTransactionAsync();

    try
    {
        // Get the promotion with its associated article
        var promotion = await _context.Promotions
            .Include(p => p.Article)
            .FirstOrDefaultAsync(p => p.Id == promotionId);

        if (promotion == null || promotion.IsAccepted)
            return false;

        // Update promotion status
        promotion.IsAccepted = true;
        promotion.DateApproval = DateTime.Now;
        promotion.ApprovedBy = approvedBy;

        // Update the article's current price to the promotion price
        if (promotion.Article != null)
        {
            promotion.Article.Prix_Vente_TND = promotion.Prix_Vente_TND_Apres;
            _context.Articles.Update(promotion.Article);
        }

        _context.Promotions.Update(promotion);
        await _context.SaveChangesAsync();

        await transaction.CommitAsync();
        return true;
    }
    catch (Exception ex)
    {
        await transaction.RollbackAsync();
        throw new InvalidOperationException($"Failed to approve promotion: {ex.Message}", ex);
    }
}
```

### 3. API Endpoints

#### PromotionController (C#)

Located: `SmartPromo_Back/Controllers/PromotionController.cs`

**New Endpoints Added:**

1. **GET /api/promotion/pending**

   - Returns all pending promotions
   - Response: List of promotions with `IsAccepted = false`

2. **GET /api/promotion/approved**

   - Returns all approved promotions
   - Response: List of promotions with `IsAccepted = true`

3. **POST /api/promotion/{id}/approve**

   - Approves a specific promotion
   - Body: `{ "approvedBy": "username" }` (optional)
   - Action: Sets `IsAccepted = true`, updates article price

4. **POST /api/promotion/{id}/reject**

   - Rejects and removes a promotion
   - Body: `{ "approvedBy": "username" }` (optional)

5. **POST /api/promotion/bulk-approve**

   - Bulk approves multiple promotions
   - Body: `{ "promotionIds": [1,2,3], "approvedBy": "username" }`

6. **GET /api/promotion/approval-stats**
   - Returns approval statistics
   - Response: `{ "totalPromotions": 10, "approvedPromotions": 5, "pendingPromotions": 5, "approvalRate": 50.0 }`

### 4. Python AI Integration

#### Database Promotion Model

Located: `Ia_SmartPromo/database_promotion_model.py`

The AI model automatically creates promotions with `isAccepted = False` (pending status). These promotions must be approved through the C# API or directly in the database before the article prices are updated.

**Key Features:**

- Generates intelligent promotion recommendations
- Saves promotions with pending status
- Calculates optimal pricing based on sales data, stock levels, and market trends
- Provides confidence scores and impact predictions

### 5. Database Schema

#### Promotions Table Structure

```sql
CREATE TABLE Promotions (
    Id int IDENTITY(1,1) PRIMARY KEY,
    DateFin datetime2 NOT NULL,
    TauxReduction float NOT NULL,
    CodeArticle nvarchar(50) NOT NULL,
    Prix_Vente_TND_Avant real NOT NULL,
    Prix_Vente_TND_Apres real NOT NULL,
    isAccepted bit NOT NULL DEFAULT 0,           -- Used by Python scripts
    IsAccepted bit NOT NULL DEFAULT 0,           -- Used by C# application
    DateCreation datetime2 DEFAULT GETDATE(),
    DateApproval datetime2 NULL,
    ApprovedBy nvarchar(255) NULL,
    -- Additional AI metadata fields
    PredictionConfidence float DEFAULT 0.85,
    SeasonalAdjustment float DEFAULT 1.0,
    TemporalAdjustment float DEFAULT 1.0,
    ExpectedVolumeImpact float DEFAULT 0,
    ExpectedRevenueImpact float DEFAULT 0
);
```

#### Articles Table Structure

```sql
ALTER TABLE Articles ADD Prix_Vente_TND real NOT NULL DEFAULT 0;
ALTER TABLE Articles ADD Prix_Achat_TND real NOT NULL DEFAULT 0;
```

## Workflow Process

### 1. AI Recommendation Generation

1. Python AI model analyzes sales data, stock levels, and market trends
2. Generates optimal promotion recommendations
3. Saves promotions to database with `IsAccepted = false`

### 2. Approval Process

1. Business users review pending promotions via API endpoints
2. Promotions can be approved or rejected
3. Upon approval:
   - `IsAccepted` is set to `true`
   - `DateApproval` is set to current timestamp
   - `ApprovedBy` is set to the approver's identifier
   - Article's `Prix_Vente_TND` is updated to the promotion price

### 3. Price Update Mechanism

When a promotion is approved, the system automatically:

1. Updates the article's current selling price (`Prix_Vente_TND`) to the promotion price (`Prix_Vente_TND_Apres`)
2. Maintains transaction integrity using database transactions
3. Logs the approval details for audit purposes

## Usage Examples

### Approve a Promotion (C# API)

```csharp
// Via service
var promotionService = new PromotionService(context);
bool success = await promotionService.ApprovePromotionAsync(promotionId, "manager123");

// Via API endpoint
POST /api/promotion/5/approve
{
    "approvedBy": "manager123"
}
```

### Get Pending Promotions

```csharp
// Via service
var pendingPromotions = await promotionService.GetPendingPromotionsAsync();

// Via API endpoint
GET /api/promotion/pending
```

### Generate AI Promotions (Python)

```python
from database_promotion_model import DatabasePromotionModel

model = DatabasePromotionModel()
model.connect_to_database()
model.load_database_tables()

# Generate promotion for specific article
recommendation = model.generate_promotion_for_article("ART001", target_date="2025-07-01")
```

## Testing

A comprehensive test suite is available in `Ia_SmartPromo/simple_test_workflow.py` that:

1. Creates test articles and promotions
2. Tests the approval workflow
3. Verifies price updates
4. Provides cleanup functionality

## Security Considerations

1. **Transaction Safety**: All approval operations use database transactions
2. **Audit Trail**: Complete audit trail with timestamps and approver information
3. **Validation**: Input validation for all API endpoints
4. **Error Handling**: Comprehensive error handling with rollback capabilities

## Performance Features

1. **Bulk Operations**: Support for bulk approval of multiple promotions
2. **Efficient Queries**: Optimized database queries with proper indexing
3. **Transaction Management**: Proper transaction scoping for data consistency

## Monitoring and Analytics

1. **Approval Statistics**: Real-time approval rate tracking
2. **Performance Metrics**: AI recommendation accuracy monitoring
3. **Business Intelligence**: Integration-ready for BI dashboards

This implementation provides a complete, production-ready promotion approval workflow with automated price updates, ensuring business control over AI-generated recommendations while maintaining operational efficiency.
