from typing import List, Dict, Any
import asyncpg
import numpy as np
from rank_bm25 import BM25Okapi
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

class HybridRetriever:
    def __init__(self):
        self.db_url = os.getenv('POSTGRES_URL', "postgresql://aigents:aigents_secret@aigents-storage-db-1:5432/aigents_practices")
        self.ollama_url = os.getenv('OLLAMA_URL', "http://aigents-storage-ollama-1:11434")
        self.bm25_weight = 0.3
        self.vector_weight = 0.7
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг текста через Ollama API"""
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

    def _calculate_bm25_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """Вычисляет BM25 scores для документов"""
        # Подготавливаем корпус для BM25
        corpus = [
            f"{doc['title']} {doc['summary']} {doc['problem']} {doc['solution']}"
            for doc in documents
        ]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Инициализируем BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Получаем scores
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        return scores.tolist()

    async def _calculate_vector_scores(self, query_embedding: List[float], documents: List[Dict]) -> List[float]:
        """Вычисляет векторные scores через косинусное сходство"""
        scores = []
        
        for doc in documents:
            # Получаем эмбеддинг документа из БД
            doc_embedding = doc.get('embedding')
            if not doc_embedding:
                scores.append(0.0)
                continue
                
            # Вычисляем косинусное сходство
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append(float(similarity))
            
        return scores

    def _combine_scores(self, bm25_scores: List[float], vector_scores: List[float]) -> List[float]:
        """Комбинирует BM25 и векторные scores с весами"""
        # Нормализуем scores
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        max_vector = max(vector_scores) if vector_scores else 1.0
        
        # Защита от деления на ноль
        if max_bm25 == 0:
            max_bm25 = 1.0
        if max_vector == 0:
            max_vector = 1.0
        
        normalized_bm25 = [score/max_bm25 for score in bm25_scores]
        normalized_vector = [score/max_vector for score in vector_scores]
        
        # Комбинируем с весами
        combined_scores = [
            self.bm25_weight * bm25 + self.vector_weight * vector
            for bm25, vector in zip(normalized_bm25, normalized_vector)
        ]
        
        return combined_scores

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Выполняет гибридный поиск похожих практик
        """
        # Получаем все практики из БД
        conn = await asyncpg.connect(self.db_url)
        try:
            practices = await conn.fetch("""
                SELECT p.*, pe.embedding
                FROM practices p
                LEFT JOIN ollama.practice_embeddings pe ON pe.id = p.id
            """)
            
            # Конвертируем в список словарей
            documents = [dict(practice) for practice in practices]
            
            if not documents:
                return []
                
            # Получаем эмбеддинг запроса
            query_embedding = await self._get_embedding(query)
            
            # Вычисляем BM25 scores
            bm25_scores = self._calculate_bm25_scores(query, documents)
            
            # Вычисляем векторные scores
            vector_scores = await self._calculate_vector_scores(query_embedding, documents)
            
            # Комбинируем scores
            combined_scores = self._combine_scores(bm25_scores, vector_scores)
            
            # Сортируем документы по combined score
            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Возвращаем top_k документов
            return [doc for doc, _ in scored_docs[:top_k]]
            
        finally:
            await conn.close()

    async def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Получает контекст из похожих практик для RAG
        """
        similar_practices = await self.retrieve(query, top_k)
        
        if not similar_practices:
            return ""
            
        context = []
        for practice in similar_practices:
            context.append(
                f"Title: {practice['title']}\n"
                f"Summary: {practice['summary']}\n"
                f"Problem: {practice['problem']}\n"
                f"Solution: {practice['solution']}\n"
                "---"
            )
            
        return "\n".join(context) 