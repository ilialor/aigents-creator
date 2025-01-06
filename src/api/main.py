from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..llm.orchestrator import LLMOrchestrator
from ..search.semantic_search import SemanticSearch

app = FastAPI(
    title="Aigents Creator Service",
    description="Service for creating new practices using LLM",
    version="1.0.0"
)

orchestrator = LLMOrchestrator()
search = SemanticSearch()

class PracticeIdea(BaseModel):
    title: str
    description: str
    domain: Optional[str] = None
    tags: Optional[List[str]] = []
    additional_details: Optional[Dict[str, Any]] = {}

@app.get("/")
async def root():
    """
    Корневой эндпоинт с информацией о сервисе
    """
    return {
        "service": "Aigents Creator",
        "version": "1.0.0",
        "endpoints": {
            "POST /practice/create": "Create new practice from idea",
            "GET /practice/status/{practice_id}": "Get practice creation status"
        }
    }

@app.post("/practice/create", response_model=Dict[str, Any])
async def create_practice(idea: PracticeIdea):
    """
    Создает новую практику на основе идеи и описания
    """
    try:
        # Преобразуем Pydantic модель в словарь
        idea_dict = idea.model_dump()
        
        # Проверяем похожие практики
        similar = await search.find_similar(f"{idea_dict['title']} {idea_dict['description']}")
        
        # Генерируем практику через LLM
        practice = await orchestrator.generate_practice(
            idea=idea_dict,
            similar_practices=similar
        )
        
        return practice
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/practice/status/{practice_id}")
async def get_practice_status(practice_id: str):
    """
    Получает статус генерации практики
    """
    status = await orchestrator.get_status(practice_id)
    if not status:
        raise HTTPException(status_code=404, detail="Practice not found")
    return status

# Добавляем обработчик ошибок
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": str(exc),
        "type": type(exc).__name__
    } 