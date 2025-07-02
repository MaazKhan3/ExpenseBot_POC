# crud.py

from sqlalchemy.orm import Session
from . import models, schemas

def get_user_by_phone_number(db: Session, phone_number: str):
    return db.query(models.User).filter(models.User.phone_number == phone_number).first()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(phone_number=user.phone_number)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_or_create_category(db: Session, user_id: int, category_name: str):
    category = db.query(models.Category).filter(
        models.Category.user_id == user_id,
        models.Category.name.ilike(category_name)
    ).first()
    if not category:
        category = models.Category(name=category_name, user_id=user_id, is_custom=True)
        db.add(category)
        db.commit()
        db.refresh(category)
    return category

def create_expense(db: Session, user_id: int, category_id: int, amount: float, note: str = ""):
    expense = models.Expense(user_id=user_id, category_id=category_id, amount=amount, note=note)
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return expense
