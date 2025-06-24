from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from . import crud, models, schemas
from .database import SessionLocal, engine, get_db
from .services import llm_service, whatsapp_service
from pydantic import BaseModel

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
            formatted = f"Sorry, I couldn't process your query: {e}"
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=formatted)
        return {"status": "ok", "query_result": formatted, "llm_result": llm_result, "sql": sql}

    else:
        msg = "I'm not sure what you meant. Please try rephrasing your message."
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
