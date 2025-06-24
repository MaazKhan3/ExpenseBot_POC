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
    confirmation = None
    logged_expenses = []
    user_id = getattr(db_user, "id")

    if llm_result.get("intent") == "expense_logging" and llm_result.get("expenses"):
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
            logged_expenses.append(f"{expense.amount} for {category.name}")
        confirmation = f"Logged: {', '.join(logged_expenses)}"
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=confirmation)
        return {"status": "ok", "confirmation": confirmation, "llm_result": llm_result}

    elif llm_result.get("intent") == "query":
        # Generate SQL using LLM
        sql = llm_service.generate_sql_from_query(payload.message_body, user_id)
        result = None
        try:
            result = db.execute(text(sql)).fetchall()
            # Format result as a string for WhatsApp and API
            if result:
                columns = result[0].keys() if hasattr(result[0], 'keys') else []
                rows = [dict(row) for row in result]
                formatted = f"Query result: {rows}"
            else:
                formatted = "No data found for your query."
        except Exception as e:
            formatted = f"Error executing query: {e}"
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=formatted)
        return {"status": "ok", "query_result": formatted, "llm_result": llm_result, "sql": sql}

    else:
        confirmation = "No expenses logged."
        whatsapp_service.send_whatsapp_message(to=str(db_user.phone_number), message=confirmation)
        return {"status": "ok", "confirmation": confirmation, "llm_result": llm_result}

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
