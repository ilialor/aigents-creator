from typing import List, Dict, Any
import aiohttp
import asyncpg
import numpy as np
import uuid
from decimal import Decimal

class SemanticSearch:
    def __init__(self):
        # Используем креды из docker-compose aigents-storage
        self.db_url = "postgresql://aigents:aigents_secret@aigents-storage-db-1:5432/aigents_practices"
        self.ollama_url = "http://aigents-storage-ollama-1:11434"
        
    def _serialize_value(self, value: Any) -> Any:
        """
        Сериализует различные типы данных в JSON-совместимый формат
        """
        if isinstance(value, uuid.UUID):
            return str(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, 'isoformat'):  # для datetime, date, time
            return value.isoformat()
        return value
        
    async def find_similar(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        Находит похожие практики используя векторный поиск в PostgreSQL
        """
        embedding = await self._get_embedding(text)
        
        conn = await asyncpg.connect(self.db_url)
        
        try:
            # Проверяем существование таблицы
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'ollama'
            """)
            print("Available tables:", [t['table_name'] for t in tables])
            
            # Проверяем структуру таблицы
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'practice_embeddings'
                AND table_schema = 'ollama'
            """)
            print("Practice embeddings columns:", [(c['column_name'], c['data_type']) for c in columns])
            
            # Ищем похожие практики используя скалярное произведение через unnest
            similar = await conn.fetch("""
                WITH similarity_calc AS (
                    SELECT p.*, pe.embedding,
                           1 - SUM(v1.value::float8 * v2.value::float8) as similarity
                    FROM practices p
                    JOIN ollama.practice_embeddings pe ON pe.id = p.id
                    CROSS JOIN LATERAL unnest(pe.embedding) WITH ORDINALITY AS v1(value, idx)
                    CROSS JOIN LATERAL unnest($1::float8[]) WITH ORDINALITY AS v2(value, idx)
                    WHERE pe.embedding IS NOT NULL
                    AND v1.idx = v2.idx
                    GROUP BY p.id, pe.embedding
                )
                SELECT *
                FROM similarity_calc
                ORDER BY similarity
                LIMIT $2
            """, embedding, top_k)
            
            # Сериализуем все значения
            return [
                {k: self._serialize_value(v) for k, v in practice.items()}
                for practice in similar
            ]
            
        finally:
            await conn.close()
        
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг текста через Storage API
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "llama2",
                    "prompt": text
                }
            ) as response:
                result = await response.json()
                return result["embedding"] 