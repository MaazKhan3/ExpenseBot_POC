"""
Intelligent Tools for Expense Processing and SQL Generation
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from datetime import datetime, timedelta

from app import crud, models
from app.services import llm_service

logger = logging.getLogger("expensebot.intelligent_agent.tools")

class ExpenseTools:
    """Enhanced tools for expense processing and analysis"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_advanced_sql(self, query_type: str, user_id: int, **kwargs) -> str:
        """Generate advanced SQL queries for complex analysis"""
        
        if query_type == "max_expense":
            time_filter = self._get_time_filter(kwargs.get("time_period", "all"))
            return f"""
            SELECT 
                e.amount,
                c.name as category,
                e.timestamp,
                e.note
            FROM expenses e
            JOIN categories c ON e.category_id = c.id
            WHERE e.user_id = {user_id} {time_filter}
            ORDER BY e.amount DESC
            LIMIT 1
            """
        
        elif query_type == "top_expenses":
            limit = kwargs.get("limit", 5)
            time_filter = self._get_time_filter(kwargs.get("time_period", "all"))
            return f"""
            SELECT 
                e.amount,
                c.name as category,
                e.timestamp,
                e.note
            FROM expenses e
            JOIN categories c ON e.category_id = c.id
            WHERE e.user_id = {user_id} {time_filter}
            ORDER BY e.amount DESC
            LIMIT {limit}
            """
        
        elif query_type == "category_breakdown":
            time_filter = self._get_time_filter(kwargs.get("time_period", "all"))
            return f"""
            SELECT 
                c.name as category,
                SUM(e.amount) as total_amount,
                COUNT(*) as transaction_count,
                AVG(e.amount) as avg_amount
            FROM expenses e
            JOIN categories c ON e.category_id = c.id
            WHERE e.user_id = {user_id} {time_filter}
            GROUP BY c.name
            ORDER BY total_amount DESC
            """
        
        elif query_type == "daily_average":
            time_filter = self._get_time_filter(kwargs.get("time_period", "all"))
            return f"""
            SELECT 
                DATE(e.timestamp) as date,
                SUM(e.amount) as daily_total,
                COUNT(*) as transactions
            FROM expenses e
            WHERE e.user_id = {user_id} {time_filter}
            GROUP BY DATE(e.timestamp)
            ORDER BY date DESC
            """
        
        elif query_type == "spending_trend":
            days = kwargs.get("days", 7)
            return f"""
            SELECT 
                DATE(e.timestamp) as date,
                SUM(e.amount) as daily_total
            FROM expenses e
            WHERE e.user_id = {user_id}
            AND e.timestamp >= NOW() - INTERVAL '{days} days'
            GROUP BY DATE(e.timestamp)
            ORDER BY date
            """
        
        else:
            # Fallback to basic query
            return f"SELECT SUM(amount) FROM expenses WHERE user_id = {user_id}"
    
    def _get_time_filter(self, time_period: str) -> str:
        """Generate time filter for SQL queries"""
        if time_period == "today":
            return "AND DATE(timestamp) = CURRENT_DATE"
        elif time_period == "week":
            return "AND timestamp >= NOW() - INTERVAL '7 days'"
        elif time_period == "month":
            return "AND timestamp >= NOW() - INTERVAL '1 month'"
        elif time_period == "year":
            return "AND timestamp >= NOW() - INTERVAL '1 year'"
        else:
            return ""
    
    def execute_sql_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            result = self.db.execute(text(sql))
            rows = result.fetchall()
            
            # Get column names from result
            columns = result.keys()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {}
                for i, column in enumerate(columns):
                    row_dict[str(column)] = row[i]
                results.append(row_dict)
            
            return results
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return []
    
    def get_max_expense(self, user_id: int, time_period: str = "all") -> Optional[Dict[str, Any]]:
        """Get the maximum expense for a user"""
        sql = self.generate_advanced_sql("max_expense", user_id, time_period=time_period)
        results = self.execute_sql_query(sql)
        return results[0] if results else None
    
    def get_top_expenses(self, user_id: int, limit: int = 5, time_period: str = "all") -> List[Dict[str, Any]]:
        """Get top expenses for a user"""
        sql = self.generate_advanced_sql("top_expenses", user_id, limit=limit, time_period=time_period)
        return self.execute_sql_query(sql)
    
    def get_category_breakdown(self, user_id: int, time_period: str = "all") -> List[Dict[str, Any]]:
        """Get category breakdown for a user"""
        sql = self.generate_advanced_sql("category_breakdown", user_id, time_period=time_period)
        return self.execute_sql_query(sql)
    
    def get_daily_average(self, user_id: int, time_period: str = "all") -> float:
        """Get daily average spending"""
        sql = self.generate_advanced_sql("daily_average", user_id, time_period=time_period)
        results = self.execute_sql_query(sql)
        
        if not results:
            return 0.0
        
        total_amount = sum(row["daily_total"] for row in results)
        days = len(results)
        return total_amount / days if days > 0 else 0.0
    
    def format_expense_response(self, data: List[Dict[str, Any]], query_type: str) -> str:
        """Format expense data into natural language response"""
        
        if not data:
            return "I couldn't find any expenses matching your query."
        
        if query_type == "max_expense":
            expense = data[0]
            return f"Your most expensive purchase was {expense['amount']:,.0f} PKR on {expense['category']}."
        
        elif query_type == "top_expenses":
            response = "Here are your top expenses:\n"
            for i, expense in enumerate(data, 1):
                response += f"{i}. {expense['amount']:,.0f} PKR - {expense['category']}\n"
            return response
        
        elif query_type == "category_breakdown":
            response = "📊 Your spending breakdown:\n\n"
            total = sum(row["total_amount"] for row in data)
            response += f"Total: {total:,.0f} PKR\n\n"
            
            for row in data:
                emoji = self._get_category_emoji(row["category"])
                response += f"{emoji} {row['category'].title()}: {row['total_amount']:,.0f} PKR ({row['transaction_count']} transactions)\n"
            return response
        
        else:
            return f"Found {len(data)} results for your query."
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emoji_map = {
            "transport": "🚗", "electronics": "💻", "lunch": "🍔", 
            "purchases": "🛒", "groceries": "🛍️", "entertainment": "🎬", 
            "health": "💊", "food": "🍕", "coffee": "☕", "shopping": "🛍️",
            "clothing": "👕", "utilities": "⚡", "rent": "🏠"
        }
        return emoji_map.get(category.lower(), "💰")
    
    def process_expense_logging(self, user_id: int, expenses_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Process expense logging with enhanced validation"""
        confirmations = []
        errors = []
        
        for expense in expenses_data:
            try:
                amount = expense.get("amount")
                category = expense.get("category")
                note = expense.get("note", "")
                
                if not amount or not category:
                    errors.append(f"Missing amount or category for expense")
                    continue
                
                # Create or get category
                db_category = crud.get_or_create_category(self.db, user_id, category)
                
                # Create expense
                db_expense = crud.create_expense(
                    self.db,
                    user_id=user_id,
                    category_id=getattr(db_category, "id"),
                    amount=amount,
                    note=note
                )
                
                confirmations.append(f"{db_expense.amount:,.0f} PKR for {db_category.name}")
                
            except Exception as e:
                logger.error(f"Error processing expense: {e}")
                errors.append(f"Failed to process expense: {str(e)}")
        
        return confirmations, errors 