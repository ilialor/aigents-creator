import aiohttp
import json
from typing import Dict, List, Any
import redis
from uuid import uuid4
from jsonschema import validate
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class LLMOrchestrator:
    def __init__(self):
        self.dify_url = os.getenv('DIFY_API_URL')
        self.dify_token = os.getenv('DIFY_API_KEY')
        
        # Получаем базовый URL и путь отдельно
        self.storage_base_url = os.getenv('STORAGE_API_URL', 'http://aigents-storage-api-1:8000')
        self.storage_practices_path = os.getenv('STORAGE_PRACTICES_PATH', '/api/v1/practices')
        
        if not all([self.dify_url, self.dify_token, self.storage_base_url]):
            raise Exception("Missing required environment variables")
            
        # Загружаем схему для валидации
        with open('src/schemas/practice_schema.json') as f:
            self.schema = json.load(f)

    def _normalize_response(self, response: Dict) -> Dict:
        """
        Нормализует ответ от LLM в соответствии со схемой API
        """
        # Получаем domain как строку
        domain = response.get("domain", "general")
        if isinstance(domain, list):
            domain = domain[0] if domain else "general"
            # Добавляем остальные домены в sub_domains если они есть
            sub_domains = response.get("sub_domains", [])
            if len(domain) > 1:
                sub_domains.extend(domain[1:])
        
        # Значения по умолчанию для implementation_steps
        default_steps = [
            {
                "order": 1,
                "title": "Plan and Prepare",
                "description": "Define objectives and gather necessary resources for implementation. Create detailed timeline and assign responsibilities."
            },
            {
                "order": 2,
                "title": "Execute and Monitor",
                "description": "Implement the practice while monitoring progress and results. Make adjustments based on feedback and performance metrics."
            }
        ]

        normalized = {
            "title": response.get("title", "Systematic Process Improvement Framework"),
            "summary": response.get("summary", "A comprehensive methodology for analyzing and improving organizational processes through systematic assessment, implementation, and continuous monitoring of improvements. This approach ensures sustainable positive changes."),
            "problem": response.get("problem", "Organizations often struggle with inefficient processes and lack of systematic approaches to improvement, leading to reduced productivity and missed opportunities for optimization."),
            "solution": response.get("solution", "Implementation of a structured framework that guides organizations through process assessment, improvement planning, and systematic implementation of changes, supported by clear metrics and continuous monitoring."),
            "visual_type": response.get("visual_type", ""),
            "visual_url": response.get("visual_url", ""),
            "visual_alt_text": response.get("visual_alt_text", ""),
            "benefits": response.get("benefits", [
                "Significant improvement in process efficiency and effectiveness through systematic implementation",
                "Enhanced quality and consistency of outputs through standardized approaches"
            ]),
            "limitations": response.get("limitations", [
                "Requires significant time and resource investment for proper implementation and training",
                "May face resistance to change and require extensive change management efforts"
            ]),
            "implementation_steps": response.get("implementation_steps", default_steps),
            "implementation_requirements": response.get("implementation_requirements", [
                "Experienced process improvement specialists",
                "Appropriate analysis and monitoring tools"
            ]),
            "estimated_resources": response.get("estimated_resources", [
                "Process documentation and analysis tools",
                "Training materials and facilitation resources"
            ]),
            "domain": domain,
            "sub_domains": response.get("sub_domains", ["process_improvement"]),
            "tags": response.get("tags", ["best_practice", "process_improvement"]),
            "creator": response.get("creator", "system"),
            "estimated_financial_cost_value": 0,
            "estimated_financial_cost_currency": "USD",
            "estimated_time_cost_minutes": 60,
            "language": "en"
        }
        return normalized

    async def generate_practice(self, idea: Dict, similar_practices: List[Dict]) -> Dict:
        practice_id = str(uuid4())
        creator_id = idea.get("creator", "system")
        
        try:
            # Отправляем title + description
            prompt = f"{idea['title']}\n\n{idea['description']}"
            
            # Получаем JSON практики
            practice_json_str = await self._call_llm(prompt, creator_id)
            
            # Парсим JSON
            try:
                practice_json = json.loads(practice_json_str)
                practice_json["creator"] = creator_id
                
                # Нормализуем ответ перед валидацией
                practice_json = self._normalize_response(practice_json)
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Raw JSON string: {practice_json_str}")
                raise Exception("Invalid JSON generated by LLM")
            
            # Валидируем JSON
            try:
                validate(instance=practice_json, schema=self.schema)
            except Exception as e:
                print(f"JSON validation failed: {e}")
                raise Exception("Generated practice does not match schema")
            
            # Сохраняем практику
            practice_url = f"{self.storage_base_url}{self.storage_practices_path}"
            print(f"Saving practice to {practice_url}")
            print(f"Practice JSON: {json.dumps(practice_json, indent=2)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    practice_url,
                    json=practice_json,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response_text = await response.text()
                    
                    # Проверяем успешные статусы
                    if response.status in [200, 201]:
                        print(f"Successfully saved practice with status {response.status}")
                        result = json.loads(response_text)
                        print(f"Save response: {result}")
                        return result  # Возвращаем ответ от API вместо practice_json
                    else:
                        print(f"Failed to save practice: Status {response.status}")
                        print(f"Response: {response_text}")
                        print(f"Headers: {response.headers}")
                        raise Exception(f"Failed to save practice: {response_text}")
            
        except Exception as e:
            print(f"Error generating practice {practice_id}: {str(e)}")
            raise
    
    async def _call_llm(self, prompt: str, creator_id: str = "system") -> Any:
        """
        Вызывает LLM через Dify API
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.dify_token}",
                "Content-Type": "application/json",
                "Origin": "http://localhost:8080",
                "Access-Control-Allow-Origin": "*"
            }
            
            data = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "conversation_id": "",
                "user": creator_id
            }
            
            try:
                async with session.post(
                    self.dify_url,
                    headers=headers,
                    json=data,
                    ssl=False
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Dify API error: {error_text}")
                        raise Exception(f"Dify API error: {response.status}")
                    
                    result = await response.json()
                    return result.get("answer", "")
            except aiohttp.ClientError as e:
                print(f"Network error calling Dify API: {e}")
                raise Exception(f"Network error: {e}") 