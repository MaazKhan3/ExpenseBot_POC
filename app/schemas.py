# schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# Expense Schemas
class ExpenseBase(BaseModel):
    amount: float
    category_id: int
    note: Optional[str] = None

class ExpenseCreate(ExpenseBase):
    pass

class Expense(ExpenseBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Category Schemas
class CategoryBase(BaseModel):
    name: str
    is_custom: bool = False

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    id: int
    user_id: Optional[int] = None
    expenses: List[Expense] = []

    class Config:
        orm_mode = True

# User Schemas
class UserBase(BaseModel):
    phone_number: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    expenses: List[Expense] = []
    categories: List[Category] = []

    class Config:
        orm_mode = True