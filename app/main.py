# main.py

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from . import crud, models, schemas
from .database import SessionLocal, engine, get_db
from .services import llm_service, whatsapp_service
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
from app.intelligent_agent_v3.agent_v3 import process_message_with_agent_v3
import os
from dotenv import load_dotenv

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

class WebhookPayload(BaseModel):
    phone_number: str
    message_body: str
    timestamp: str # Or use datetime

@app.get("/")
def read_root():
    return {"message": "Welcome to the Expense Tracker API"}

@app.post("/webhook")
def handle_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    print(f"🔍 DEBUG: Using intelligent agent V3 for message: {payload.message_body}")
    print(db)
    agent_response = process_message_with_agent_v3(
        phone_number=payload.phone_number,
        message=payload.message_body,
        db=db
    )
    print(f"🔍 DEBUG: Agent V3 response: {agent_response}")
    if agent_response and agent_response.get("message"):
        print(f"🔍 DEBUG: Using intelligent agent V3 response: {agent_response['message']}")
        whatsapp_service.send_whatsapp_message(to=str(payload.phone_number), message=agent_response["message"])
        return {"status": "ok", "message": agent_response["message"], "intent": agent_response.get("intent")}
    else:
        msg = "Sorry, I couldn't process your request. Please try again."
        whatsapp_service.send_whatsapp_message(to=str(payload.phone_number), message=msg)
        return {"status": "error", "message": msg}

@app.get("/expenses")
def list_expenses(phone_number: str, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_phone_number(db, phone_number=phone_number)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    expenses = db.query(models.Expense).filter(models.Expense.user_id == db_user.id).all()
    result = []
    for exp in expenses:
        category = db.query(models.Category).filter(models.Category.id == exp.category_id).first()
        result.append({
            "amount": exp.amount,
            "category": category.name if category else None,
            "note": exp.note,
            "timestamp": exp.timestamp
        })
    return {"phone_number": phone_number, "expenses": result}

@app.get("/trigger_summary")
def trigger_summary(phone_number: str, summary_type: str = "weekly", db: Session = Depends(get_db)):
    db_user = crud.get_user_by_phone_number(db, phone_number=phone_number)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = getattr(db_user, "id")
    now = datetime.utcnow()
    if summary_type == "monthly":
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = now
        period_label = "Monthly"
    else:  # weekly by default
        end_date = now
        start_date = end_date - timedelta(days=7)
        period_label = "Weekly"
    expenses: list[models.Expense] = db.query(models.Expense).filter(
        models.Expense.user_id == user_id,
        models.Expense.timestamp >= start_date,
        models.Expense.timestamp < end_date
    ).all()
    if not expenses:
        msg = f"No expenses found for the {summary_type} period."
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=msg)
        return {"status": "ok", "summary": msg}
    # Aggregate by category
    category_totals = {}
    for exp in expenses:
        category = db.query(models.Category).filter(models.Category.id == exp.category_id).first()
        cat_name = category.name if category else "unknown"
        category_totals[cat_name] = category_totals.get(cat_name, 0) + exp.amount
    total = sum(category_totals.values())
    # Top categories
    top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    # Biggest single expense
    amounts = [float(getattr(e, "amount", 0.0)) for e in expenses]
    max_index = amounts.index(max(amounts))
    biggest_expense = expenses[max_index]
    biggest_expense_cat = db.query(models.Category).filter(models.Category.id == biggest_expense.category_id).first()
    # Average per day
    days = max((end_date - start_date).days, 1)
    avg_per_day = total / days
    # Prepare bullet points
    bullet_points = []
    emoji_map = {"transport": "🚗", "electronics": "💻", "lunch": "🍔", "purchases": "🛒", "groceries": "🛍️", "entertainment": "🎬", "health": "💊"}
    for cat, amt in top_categories:
        emoji = emoji_map.get(cat.lower(), "•")
        bullet_points.append(f"{emoji} {cat.title()}: PKR {amt:,.0f}")
    # LLM prompt for concise, bullet-pointed summary
    summary_prompt = (
        f"{period_label} Expense Summary:\n"
        f"Total: PKR {total:,.0f}\n"
        f"Top Categories:\n" + "\n".join(bullet_points) + "\n"
        f"Biggest single expense: PKR {biggest_expense.amount:,.0f} ({biggest_expense_cat.name.title() if biggest_expense_cat else 'unknown'})\n"
        f"Average per day: PKR {avg_per_day:,.0f}\n"
        f"Respond ONLY with a concise, bullet-pointed WhatsApp message using emojis and line breaks."
    )
    summary = llm_service.format_summary_with_llm(summary_prompt)
    whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=summary)
    return {"status": "ok", "summary": summary, "raw": summary_prompt}

@app.get("/chat")
def chat_page():
    return FileResponse("static/index.html")
