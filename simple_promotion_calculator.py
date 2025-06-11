#!/usr/bin/env python3
"""
Simple Promotion Calculator - Easy Interface for Single Article Promotion Predictions

This script provides a streamlined interface to calculate promotion recommendations
for a single article on a specific date using the enhanced AI model.

Usage Examples:
    python simple_promotion_calculator.py ARTICLE123 2025-12-25
    python simple_promotion_calculator.py MONTRES001 2026-07-15
"""

import sys
import os
from datetime import datetime
import argparse
from real_business_promotion_model_with_dates import RealDataPromotionModel


def main():
    """Main function for simple promotion calculation"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate promotion recommendation for a single article on a specific date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ARTICLE123 2025-12-25
  %(prog)s MONTRES001 2026-07-15
  %(prog)s BIJOUX999 2025-09-15

Date formats supported:
  - YYYY-MM-DD (e.g., 2025-12-25)
  - DD/MM/YYYY (e.g., 25/12/2025)
  - DD-MM-YYYY (e.g., 25-12-2025)
        """
    )
    
    parser.add_argument('article_code', 
                       help='The article code to calculate promotion for')
    parser.add_argument('date', 
                       help='Target date for promotion calculation (YYYY-MM-DD format)')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Show detailed analysis and model training info')
    parser.add_argument('--save', '-s',
                       action='store_true', 
                       help='Save detailed results to Excel file')
    
    # If no arguments provided, show help and interactive mode
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        return interactive_mode()
    
    args = parser.parse_args()
    
    # Validate and parse date
    try:
        target_date = parse_date(args.date)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Calculate promotion
    return calculate_promotion(args.article_code, target_date, args.verbose, args.save)


def interactive_mode():
    """Interactive mode for easy input"""
    print("Enter promotion calculation details:")
    
    # Get article code
    while True:
        article_code = input("\nArticle Code: ").strip()
        if article_code:
            break
        print("❌ Please enter a valid article code")
    
    # Get date
    while True:
        date_input = input("Target Date (YYYY-MM-DD): ").strip()
        try:
            target_date = parse_date(date_input)
            break
        except ValueError as e:
            print(f"❌ {e}")
            print("Please use format: YYYY-MM-DD (e.g., 2025-12-25)")
    
    # Get options
    verbose = input("\nShow detailed analysis? (y/N): ").strip().lower() in ['y', 'yes']
    save_results = input("Save results to Excel? (y/N): ").strip().lower() in ['y', 'yes']
    
    print("\n" + "="*60)
    print("CALCULATING PROMOTION...")
    print("="*60)
    
    return calculate_promotion(article_code, target_date, verbose, save_results)


def parse_date(date_string):
    """Parse date string in various formats"""
    date_string = date_string.strip()
    
    # Try different date formats
    formats = [
        '%Y-%m-%d',     # 2025-12-25
        '%d/%m/%Y',     # 25/12/2025
        '%d-%m-%Y',     # 25-12-2025
        '%Y/%m/%d',     # 2025/12/25
        '%m/%d/%Y',     # 12/25/2025
    ]
    
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_string, fmt)
            return parsed_date
        except ValueError:
            continue
    
    raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD format.")


def calculate_promotion(article_code, target_date, verbose=False, save_results=False):
    """Calculate promotion for the given article and date"""
    
    try:
        print(f"🔍 Initializing AI Promotion Model...")
        
        # Initialize model
        model = RealDataPromotionModel()
        
        # Load and prepare data
        if not verbose:
            # Suppress detailed training output for clean interface
            import warnings
            warnings.filterwarnings("ignore")
        
        print(f"📊 Loading business data...")
        df = model.load_real_business_data()
        
        # Check if article exists
        article_data = df[df['GA_ARTICLE'] == article_code]
        if article_data.empty:
            print(f"❌ Article '{article_code}' not found in database")
            
            # Show similar articles if any
            similar_articles = df[df['GA_ARTICLE'].str.contains(article_code[:5], case=False, na=False)]['GA_ARTICLE'].head(5)
            if not similar_articles.empty:
                print(f"\n💡 Similar articles found:")
                for article in similar_articles:
                    print(f"   - {article}")
            return 1
        
        print(f"🤖 Training AI model...")
        
        # Prepare and train model
        X, feature_names = model.prepare_real_features(df)
        y = df['Promotion_Optimale'] if 'Promotion_Optimale' in df.columns else model.calculate_real_optimal_promotions(df)['Promotion_Optimale']
        model.train_models(X, y)
        
        print(f"🎯 Calculating promotion for {article_code} on {target_date.strftime('%Y-%m-%d')}...")
        
        # Get prediction
        result = model.predict_promotion_for_date(article_code, target_date)
        
        # Display results
        display_results(result, verbose)
        
        # Save results if requested
        if save_results:
            save_detailed_results(result, article_code, target_date)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def display_results(result, verbose=False):
    """Display promotion calculation results in a clean format"""
    
    print("\n" + "="*70)
    print("🎯 PROMOTION RECOMMENDATION")
    print("="*70)
    
    # Basic article info
    print(f"📦 Article: {result['article_code']}")
    print(f"📝 Description: {result['article_name']}")
    print(f"🏷️  Category: {result['category']}")
    print(f"🏢 Supplier: {result['supplier']}")
    
    # Date context
    print(f"\n📅 Target Date: {result['target_date']} ({result['day_of_week']})")
    print(f"🌍 Season: {result['season']}")
    if result['is_holiday_period']:
        print(f"🎉 Holiday Period: Yes")
    if result['is_weekend']:
        print(f"🎯 Weekend: Yes")
    
    # Main recommendation
    print(f"\n" + "="*70)
    print(f"💰 RECOMMENDED PROMOTION: {result['adjusted_promotion_pct']}%")
    print(f"💵 NEW PRICE: {result['promoted_price_tnd']} TND")
    print(f"⚠️  RISK LEVEL: {result['risk_level']}")
    print(f"🎯 CONFIDENCE: {result['recommendation_confidence']:.1%}")
    print("="*70)
    
    # Financial impact
    print(f"\n📊 EXPECTED IMPACT:")
    print(f"   📈 Volume Change: {result['volume_impact_pct']:+.1f}%")
    print(f"   💰 Revenue Impact: {result['revenue_impact_tnd']:+,.2f} TND ({result['revenue_impact_pct']:+.1f}%)")
    print(f"   💎 Profit Impact: {result['profit_impact_tnd']:+,.2f} TND ({result['profit_impact_pct']:+.1f}%)")
    
    # Current vs projected metrics
    print(f"\n📋 CURRENT vs PROJECTED:")
    print(f"   Price:    {result['current_price_tnd']:.2f} TND → {result['promoted_price_tnd']:.2f} TND")
    print(f"   Volume:   {result['current_monthly_volume']:,} → {result['projected_monthly_volume']:,} units/month")
    print(f"   Revenue:  {result['current_monthly_revenue_tnd']:,.2f} → {result['projected_monthly_revenue_tnd']:,.2f} TND/month")
    print(f"   Profit:   {result['current_monthly_profit_tnd']:,.2f} → {result['projected_monthly_profit_tnd']:,.2f} TND/month")
    
    # Business recommendation
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {result.get('recommended_action', 'No specific recommendation available')}")
    
    # Detailed analysis (if verbose)
    if verbose:
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"   Base Promotion: {result['base_promotion_pct']}%")
        print(f"   Temporal Adjustment: {result['temporal_adjustment_factor']:.3f}")
        print(f"   Seasonal Multiplier: {result['seasonal_demand_multiplier']:.3f}")
        print(f"   Competition Intensity: {result['competition_intensity']:.3f}")
        print(f"   Current Margin: {result['current_margin_pct']:.1f}%")
    
    # Risk warning
    if result['risk_level'] == 'HIGH':
        print(f"\n⚠️  HIGH RISK WARNING:")
        print(f"   This promotion is aggressive and may significantly impact margins.")
        print(f"   Consider reviewing business constraints before implementation.")
    elif result['risk_level'] == 'MEDIUM':
        print(f"\n⚡ MEDIUM RISK NOTICE:")
        print(f"   This promotion carries moderate risk. Monitor performance closely.")
    
    print(f"\n" + "="*70)


def save_detailed_results(result, article_code, target_date):
    """Save detailed results to Excel file"""
    
    try:
        import pandas as pd
        
        # Prepare data for Excel
        excel_data = {
            'Metric': [],
            'Value': [],
            'Unit': []
        }
        
        # Add all result data
        for key, value in result.items():
            if isinstance(value, (int, float)):
                excel_data['Metric'].append(key.replace('_', ' ').title())
                excel_data['Value'].append(value)
                excel_data['Unit'].append('')
            elif isinstance(value, str):
                excel_data['Metric'].append(key.replace('_', ' ').title())
                excel_data['Value'].append(value)
                excel_data['Unit'].append('')
        
        df_results = pd.DataFrame(excel_data)
        
        # Create filename
        date_str = target_date.strftime('%Y%m%d')
        filename = f"promotion_calc_{article_code}_{date_str}.xlsx"
        
        # Save to Excel
        df_results.to_excel(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Could not save to Excel: {str(e)}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
