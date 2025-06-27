from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from . import crud, models, schemas
from .database import SessionLocal, engine, get_db
from .services import llm_service, whatsapp_service
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
from app.intelligent_agent import process_message_safely
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
    # Debug: Check if intelligent agent is enabled
    load_dotenv()
    agent_enabled = os.getenv("USE_INTELLIGENT_AGENT", "false").lower() == "true"
    print(f"🔍 DEBUG: USE_INTELLIGENT_AGENT = {os.getenv('USE_INTELLIGENT_AGENT')}")
    print(f"🔍 DEBUG: Agent enabled = {agent_enabled}")
    
    # Try the intelligent agent first
    agent_response = None
    if agent_enabled:
        try:
            print(f"🔍 DEBUG: Trying intelligent agent for message: {payload.message_body}")
            agent_response = process_message_safely(
                phone_number=payload.phone_number,
                message=payload.message_body,
                db=db
            )
            print(f"🔍 DEBUG: Agent response: {agent_response}")
            
            # Check if agent response is valid
            if agent_response and agent_response.get("message"):
                print(f"🔍 DEBUG: Using intelligent agent response: {agent_response['message']}")
                whatsapp_service.send_whatsapp_message(to=str(payload.phone_number), message=agent_response["message"])
                return {"status": "ok", "message": agent_response["message"], "intent": agent_response.get("intent")}
            else:
                print(f"🔍 DEBUG: Agent returned no valid response, falling back")
        except Exception as e:
            print(f"🔍 DEBUG: Agent error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"🔍 DEBUG: Falling back to legacy logic")
    # Fallback to original logic
    db_user = crud.get_user_by_phone_number(db, phone_number=payload.phone_number)
    if not db_user:
        db_user = crud.create_user(db, user=schemas.UserCreate(phone_number=payload.phone_number))

    llm_result = llm_service.process_user_message(payload.message_body)
    user_id = getattr(db_user, "id")

    # Clarification loop for expense logging
    if llm_result.get("intent") == "expense_logging" and llm_result.get("expenses"):
        missing_info = []
        for idx, exp in enumerate(llm_result["expenses"], 1):
            if exp.get("amount") is None or exp.get("category") in (None, "", "unclear", "unknown"):
                missing = []
                if exp.get("amount") is None:
                    missing.append("amount")
                if exp.get("category") in (None, "", "unclear", "unknown"):
                    missing.append("category")
                missing_info.append(f"Expense {idx}: missing {', '.join(missing)}")
        if missing_info:
            clarification = "I need a bit more info to log your expense. " + "; ".join(missing_info) + ". Could you please clarify?"
            whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=clarification)
            return {"status": "clarification_needed", "clarification": clarification, "llm_result": llm_result}

        # All info present, log expenses
        confirmations = []
        try:
            for exp in llm_result["expenses"]:
                category = crud.get_or_create_category(db, user_id, exp["category"])
                category_id = getattr(category, "id")
                expense = crud.create_expense(
                    db,
                    user_id=user_id,
                    category_id=category_id,
                    amount=exp["amount"],
                    note=exp.get("note") or ""
                )
                confirmations.append(f"{expense.amount} PKR for {category.name}")
            confirmation_msg = "Logged: " + ", ".join(confirmations) + " ✅"
            whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=confirmation_msg)
            return {"status": "ok", "confirmation": confirmation_msg, "llm_result": llm_result}
        except Exception as e:
            error_msg = "I couldn't save your expense. Please try again with a different format."
            whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=error_msg)
            return {"status": "error", "error": error_msg, "llm_result": llm_result}

    elif llm_result.get("intent") == "query":
        sql = llm_service.generate_sql_from_query(payload.message_body, user_id)
        result = None
        try:
            result = db.execute(text(sql)).fetchall()
            if result:
                # Try to format as a conversational summary
                if len(result[0]) == 1:
                    value = list(result[0])[0]
                    formatted = f"You spent {value} PKR in the requested period." if value is not None else "No expenses found for your query."
                else:
                    rows = [dict(row) for row in result]
                    formatted = f"Query result: {rows}"
            else:
                formatted = "No data found for your query."
        except Exception as e:
            formatted = "I couldn't process your query. Could you please rephrase it or try asking something else?"
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=formatted)
        return {"status": "ok", "query_result": formatted, "llm_result": llm_result, "sql": sql}

    elif llm_result.get("intent") == "breakdown":
        # Handle breakdown requests
        try:
            # Determine time period from message
            message_lower = payload.message_body.lower()
            if "week" in message_lower or "this week" in message_lower:
                time_period = "week"
            elif "month" in message_lower or "this month" in message_lower:
                time_period = "month"
            else:
                time_period = "all"
            
            sql = llm_service.generate_breakdown_sql(user_id, time_period)
            result = db.execute(text(sql)).fetchall()
            formatted = llm_service.format_breakdown_result(result, time_period)
        except Exception as e:
            formatted = "I couldn't generate your spending breakdown right now. Please try again in a moment."
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=formatted)
        return {"status": "ok", "breakdown_result": formatted, "llm_result": llm_result}

    else:
        msg = "I'm not sure what you meant. You can:\n• Log expenses: '500 for groceries'\n• Ask queries: 'How much did I spend this week?'\n• Get breakdowns: 'Show me my spending breakdown'\n• Get summaries: Use the summary trigger"
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=msg)
        return {"status": "unclear", "message": msg, "llm_result": llm_result}

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
