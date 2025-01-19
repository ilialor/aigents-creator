from typing import Dict, Any
import aiohttp
import os
from dotenv import load_dotenv
from .retriever import HybridRetriever

load_dotenv()

class ContextAugmenter:
    def __init__(self):
        self.dify_url = os.getenv('DIFY_API_URL')
        self.dify_token = os.getenv('DIFY_API_KEY')
        self.retriever = HybridRetriever()
        
    async def augment_with_context(self, idea: Dict[str, Any]) -> str:
        """
        Обогащает промпт контекстом из похожих практик
        """
        # Получаем контекст из похожих практик
        query = f"{idea['title']} {idea['description']}"
        similar_context = await self.retriever.get_context(query)
        
        # Формируем промпт с контекстом
        prompt = f"""You are an AI assistant specialized in creating high-quality practices.
Your task is to create a new practice based on the idea provided, while considering similar existing practices for context and ensuring uniqueness.

IDEA:
Title: {idea['title']}
Description: {idea['description']}

SIMILAR PRACTICES FOR CONTEXT:
{similar_context}

Based on the idea and similar practices, create a new unique practice that:
1. Addresses the core problem in a novel way
2. Doesn't duplicate existing solutions
3. Provides clear implementation steps
4. Considers limitations and benefits
5. Is practical and actionable

Please provide the practice as a raw JSON object (no code block markers, no ```json, just the JSON object) with the following structure:
{{
    "title": "Practice title",
    "summary": "Brief overview",
    "problem": "Problem description",
    "solution": "Detailed solution",
    "benefits": ["benefit1", "benefit2", ...],
    "limitations": ["limitation1", "limitation2", ...],
    "implementation_steps": [
        {{"order": 1, "title": "Step 1", "description": "Step 1 details"}},
        ...
    ],
    "implementation_requirements": ["requirement1", "requirement2", ...],
    "estimated_resources": ["resource1", "resource2", ...],
    "domain": "main_domain",
    "sub_domains": ["subdomain1", "subdomain2"],
    "tags": ["tag1", "tag2"],
    "category": "concept" or "implementation" // Must be one of these two English values
}}

IMPORTANT:
1. The response must be in Russian language, except for:
   - domain names
   - tags
   - category (must be "concept" or "implementation" in English)
2. Make sure all arrays have at least one item
3. All text fields must be meaningful and detailed
4. Return ONLY the JSON object, without any markdown code block markers or other text"""

        return prompt

    async def generate_with_rag(self, prompt: str) -> str:
        """
        Генерирует контент используя Dify API с RAG
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.dify_token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "conversation_id": "",
                "user": "system"
            }
            
            async with session.post(
                self.dify_url,
                headers=headers,
                json=data,
                ssl=False
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Dify API error: {error_text}")
                
                result = await response.json()
                return result.get("answer", "") 