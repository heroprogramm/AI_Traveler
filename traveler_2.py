from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import logging
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import defaultdict
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import requests
import traceback

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "mistralai/mixtral-8x7b-instruct"

app = FastAPI(
    title="Intelligent AI Traveling Agent",
    description="Self-learning travel assistant with dynamic knowledge expansion",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embeddings model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant setup
try:
    # ✅ Correct Qdrant Cloud URL (no :6333, default is 443 with HTTPS)
    qdrant_client = QdrantClient(
        url="https://34ea31d2-09b4-46ed-8b64-c8d302a524e1.us-east-1-1.aws.cloud.qdrant.io",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Vlc3MiOiJtIn0.qW5tN8juEp_DbMKHAddAdBln2brLqK-0V0siJ7uiQ98"
    )
    logger.info("✅ Qdrant client connected")
except Exception as e:
    logger.error(f"❌ Failed to connect Qdrant: {e}")
    qdrant_client = None

COLLECTION_NAME = "travel_knowledge"

# ----------------------------
# Data ingestion
def ingest_travel_data(travel_docs: List[str]) -> bool:
    if not embedder or not qdrant_client:
        logger.error("Required components not initialized")
        return False
    try:
        vectors = embedder.encode(travel_docs).tolist()
        
        # ✅ Use get_collection instead of collection_exists (for Qdrant Cloud)
        try:
            qdrant_client.get_collection(COLLECTION_NAME)
            qdrant_client.delete_collection(COLLECTION_NAME)
            logger.info("♻️ Old collection deleted")
        except Exception:
            logger.info("ℹ️ No existing collection found, creating a new one")

        # ✅ Recreate collection with correct vector size
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        
        # ✅ Insert documents as points
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=vectors[i], 
                payload={"doc": travel_docs[i], "doc_id": i}
            ) 
            for i in range(len(vectors))
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        # ✅ Build TF-IDF for keyword fallback
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = tfidf.fit_transform(travel_docs)
        app.state.vector_store = {"tfidf": tfidf, "tfidf_matrix": tfidf_matrix, "docs": travel_docs}
        
        logger.info("✅ Travel data ingestion completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error ingesting travel data: {e}")
        logger.error(traceback.format_exc())
        return False

COLLECTION_NAME = "travel_knowledge"

# Intelligence System State
class IntelligenceSystem:
    def __init__(self):
        self.unknown_places = defaultdict(int)  # Track frequency of unknown places
        self.learning_queue = set()             # Places to research
        self.recently_learned = set()           # Cache of recent additions
        self.user_contributions = []            # User-provided info
        self.last_cleanup = datetime.now()
        
    def track_unknown_place(self, place: str):
        """Track requests for unknown places"""
        self.unknown_places[place] += 1
        logger.info(f"Unknown place '{place}' requested {self.unknown_places[place]} times")
        
        # Auto-queue popular unknown places for research
        if self.unknown_places[place] >= 2:  # Threshold for research
            self.learning_queue.add(place)
            logger.info(f"Added '{place}' to learning queue")
    
    def mark_as_learned(self, place: str):
        """Mark place as learned"""
        if place in self.learning_queue:
            self.learning_queue.remove(place)
        self.recently_learned.add(place)
        if place in self.unknown_places:
            del self.unknown_places[place]
    
    def cleanup_old_data(self):
        """Clean up old tracking data"""
        if datetime.now() - self.last_cleanup > timedelta(hours=24):
            # Reset counters daily
            self.unknown_places.clear()
            self.recently_learned.clear()
            self.last_cleanup = datetime.now()

# Global intelligence system
intel_system = IntelligenceSystem()

# Pydantic Models
class QA(BaseModel):
    question: str
    answer: str

class QuestionInput(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence_level: str
    data_sources: List[str]
    learned_new_info: bool
    history: List[QA]

class ContributeInfo(BaseModel):
    place: str
    information: str
    user_id: Optional[str] = "anonymous"

# ----------------------------
# Enhanced Knowledge Management
# ----------------------------
def extract_place_names(text: str) -> List[str]:
    """Extract potential place names from text (simple implementation)"""
    # This is a basic implementation - you could use NLP libraries like spaCy for better extraction
    import re
    
    # Common place indicators
    place_indicators = ['in ', 'to ', 'from ', 'visit ', 'about ', 'around ']
    places = []
    
    for indicator in place_indicators:
        pattern = fr'{indicator}([A-Z][a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        places.extend([match.strip() for match in matches])
    
    # Also check for capitalized words (potential place names)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
    places.extend(capitalized)
    
    return list(set(places))  # Remove duplicates

async def search_web_for_place(place: str) -> List[str]:
    """Search web for travel information about a place"""
    try:
        # Simulate web search - replace with actual web search API
        search_queries = [
            f"{place} travel guide attractions",
            f"{place} tourism what to see",
            f"visiting {place} recommendations",
            f"{place} travel tips best places"
        ]
        
        # Mock web search results - replace with real implementation
        mock_results = [
            f"{place} is a popular destination known for its unique attractions and cultural heritage.",
            f"Travelers to {place} often recommend visiting during the best season for optimal weather.",
            f"The local cuisine in {place} features traditional dishes that reflect the regional culture.",
            f"Popular activities in {place} include sightseeing, cultural experiences, and outdoor adventures.",
            f"Transportation in {place} is accessible through various local and international connections."
        ]
        
        logger.info(f"Found web information for {place}")
        return mock_results
        
    except Exception as e:
        logger.error(f"Web search failed for {place}: {e}")
        return []

def ingest_travel_data(travel_docs: List[str], place: str = None):
    """Enhanced ingestion with place tracking"""
    try:
        vectors = embedder.encode(travel_docs).tolist()

        # Create collection if it doesn't exist
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
            )

        points = []
        for i, vector in enumerate(vectors):
            payload = {
                "doc": travel_docs[i],
                "timestamp": datetime.now().isoformat(),
                "place": place or "general",
                "source": "dynamic_learning" if place else "initial_data"
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        # Update TF-IDF store
        if hasattr(app.state, 'vector_store'):
            existing_docs = app.state.vector_store["docs"]
            all_docs = existing_docs + travel_docs
        else:
            all_docs = travel_docs
            
        tfidf = TfidfVectorizer().fit(all_docs)
        tfidf_matrix = tfidf.transform(all_docs)
        
        app.state.vector_store = {
            "tfidf": tfidf, 
            "tfidf_matrix": tfidf_matrix, 
            "docs": all_docs
        }
        
        if place:
            intel_system.mark_as_learned(place)
        
        logger.info(f"Successfully ingested {len(travel_docs)} documents" + 
                   (f" for {place}" if place else ""))
        return True
        
    except Exception as e:
        logger.error(f"Error in enhanced ingest: {e}")
        return False

# ----------------------------
# Intelligent Retrieval
# ----------------------------
def retrieve_with_intelligence(query: str, store: dict, docs: List[str], top_k=5) -> tuple[List[str], str, List[str]]:
    """Enhanced retrieval with confidence assessment"""
    try:
        # Semantic search via Qdrant
        query_vec = embedder.encode([query])[0]
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k * 2,  # Get more to filter
            with_payload=True,
            score_threshold=0.3  # Minimum relevance threshold
        )
        
        sem_docs = []
        sources = []
        scores = []
        
        for r in results:
            sem_docs.append(r.payload["doc"])
            sources.append(r.payload.get("source", "unknown"))
            scores.append(r.score)
        
        # Keyword search
        if "tfidf" in store:
            query_tfidf = store["tfidf"].transform([query])
            sim_scores = cosine_similarity(query_tfidf, store["tfidf_matrix"]).flatten()
            top_idx = sim_scores.argsort()[::-1][:top_k]
            keyword_docs = [docs[i] for i in top_idx if i < len(docs) and sim_scores[i] > 0.1]
        else:
            keyword_docs = []
        
        # Combine and deduplicate
        all_docs = sem_docs + keyword_docs
        unique_docs = list(dict.fromkeys(all_docs))[:top_k]
        
        # Assess confidence
        if len(scores) > 0:
            avg_score = sum(scores[:3]) / min(3, len(scores))  # Average of top 3
            if avg_score > 0.7:
                confidence = "high"
            elif avg_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "very_low"
        
        return unique_docs, confidence, list(set(sources))
        
    except Exception as e:
        logger.error(f"Intelligent retrieval failed: {e}")
        return [], "error", ["fallback"]

# ----------------------------
# Smart Prompt Engineering
# ----------------------------
def create_intelligent_prompt(question: str, context_docs: List[str], confidence: str, 
                            places: List[str]) -> str:
    """Create context-aware prompts based on confidence level"""
    
    context = "\n---\n".join(context_docs) if context_docs else "No specific information available."
    
    if confidence == "high":
        prompt = f"""You are an expert AI Travel Agent with comprehensive knowledge.

Context (High Confidence):
{context}

Question: {question}

Provide detailed, specific travel advice based on the reliable information above. 
Include practical tips, recommendations, and insider knowledge."""

    elif confidence == "medium":
        prompt = f"""You are a knowledgeable AI Travel Agent.

Context (Moderate Confidence):
{context}

Question: {question}

Based on available information, provide helpful travel advice. If certain details seem 
uncertain, mention that travelers should verify current information from official sources."""

    elif confidence == "low" or confidence == "very_low":
        unknown_places_text = ", ".join(places) if places else "the requested location"
        prompt = f"""You are a helpful AI Travel Agent. I have limited specific information about {unknown_places_text}.

Available Context:
{context}

Question: {question}

Provide general travel guidance and practical advice. Since I don't have comprehensive 
current information about this destination, please:
1. Give general travel tips applicable to most destinations
2. Suggest reliable sources for current, detailed information
3. Recommend standard travel preparations
4. Be honest about the limitations of available information

Focus on being helpful while being transparent about information gaps."""

    else:  # error case
        prompt = f"""You are a helpful AI Travel Agent experiencing technical difficulties.

Question: {question}

Provide general travel advice and suggest the user try again later or consult 
official tourism websites for the most current information."""

    return prompt

# ----------------------------
# Enhanced LLM Integration
# ----------------------------
def generate_intelligent_answer(question: str, context_docs: List[str], confidence: str, 
                               places: List[str]) -> str:
    """Generate answers with intelligence-aware prompting"""
    
    prompt = create_intelligent_prompt(question, context_docs, confidence, places)
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an intelligent, adaptive AI travel assistant that provides helpful advice while being honest about information limitations."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"OpenRouter error: {response.text}")
            return f"I'm experiencing technical difficulties. For travel information about your destination, I recommend checking official tourism websites or travel guides."
            
    except Exception as e:
        logger.error(f"OpenRouter request failed: {e}")
        return "I'm currently unable to process your request. Please try again later or consult reliable travel resources."

# ----------------------------
# Background Learning Tasks
# ----------------------------
async def background_learner():
    """Background task to learn about popular unknown places"""
    while True:
        try:
            # Clean up old data
            intel_system.cleanup_old_data()
            
            # Process learning queue
            if intel_system.learning_queue:
                place = intel_system.learning_queue.pop()
                logger.info(f"Learning about {place}...")
                
                # Fetch information
                new_info = await search_web_for_place(place)
                
                if new_info:
                    # Ingest new knowledge
                    ingest_travel_data(new_info, place)
                    logger.info(f"Successfully learned about {place}")
                else:
                    logger.warning(f"Could not find information about {place}")
            
            # Wait before next learning cycle
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Background learning error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

# ----------------------------
# API Endpoints
# ----------------------------
@app.on_event("startup")
async def startup_tasks():
    """Enhanced startup with initial knowledge and background tasks"""
    
    # Load initial travel knowledge
    travel_knowledge = [
        "Paris, France is famous for the Eiffel Tower, Louvre Museum, Seine River cruises, and charming café culture.",
        "Tokyo, Japan offers diverse attractions including Shibuya Crossing, Tokyo Tower, ancient temples, and modern technology districts.",
        "New York City features iconic landmarks like Times Square, Central Park, Statue of Liberty, and world-class Broadway shows.",
        "The Maldives is renowned for luxury overwater resorts, crystal-clear waters perfect for snorkeling and diving, and pristine white sandy beaches.",
        "Dubai, UAE showcases the Burj Khalifa, thrilling desert safaris, luxury shopping malls, and innovative architecture.",
        "Rome, Italy captivates visitors with the Colosseum, Vatican City, ancient Roman Forum, and authentic Italian cuisine.",
        "Bali, Indonesia is known for beautiful temples, terraced rice fields, volcanic landscapes, and wellness retreats.",
        "London, England offers Big Ben, British Museum, Tower Bridge, and rich royal history with modern cultural scenes.",
        "Thailand combines bustling Bangkok markets, serene temples, tropical beaches in Phuket, and delicious street food.",
        "Iceland provides stunning natural wonders including Northern Lights, geysers, waterfalls, and unique volcanic landscapes."
    ]
    
    success = ingest_travel_data(travel_knowledge)
    if success:
        logger.info("Successfully loaded initial travel knowledge base")
    else:
        logger.error("Failed to load initial knowledge base")
    
    # Start background learning task
    asyncio.create_task(background_learner())
    logger.info("Started background learning system")

@app.post("/ask", response_model=AnswerResponse)
async def intelligent_ask(request: Request, input: QuestionInput):
    """Enhanced Q&A endpoint with intelligence features"""
    try:
        # Extract potential place names from question
        places = extract_place_names(input.question)
        
        # Get vector store
        store = getattr(app.state, 'vector_store', {"docs": []})
        docs = store.get("docs", [])
        
        # Intelligent retrieval
        relevant_docs, confidence, sources = retrieve_with_intelligence(input.question, store, docs)
        
        # Track unknown places for learning
        learned_new_info = False
        if confidence in ["low", "very_low"] and places:
            for place in places:
                intel_system.track_unknown_place(place)
                
                # Try immediate learning for high-priority places
                if intel_system.unknown_places[place] >= 3:
                    new_info = await search_web_for_place(place)
                    if new_info:
                        ingest_travel_data(new_info, place)
                        # Re-retrieve with new information
                        relevant_docs, confidence, sources = retrieve_with_intelligence(input.question, app.state.vector_store, app.state.vector_store["docs"])
                        learned_new_info = True
                        logger.info(f"Immediately learned about {place}")
        
        # Generate intelligent answer
        answer = generate_intelligent_answer(input.question, relevant_docs, confidence, places)
        
        # Map confidence to user-friendly terms
        confidence_map = {
            "high": "High - Based on comprehensive information",
            "medium": "Medium - Based on available information", 
            "low": "Low - Limited specific information available",
            "very_low": "Very Low - General guidance provided",
            "error": "Error - Technical difficulties encountered"
        }
        
        return AnswerResponse(
            question=input.question,
            answer=answer,
            confidence_level=confidence_map.get(confidence, "Unknown"),
            data_sources=sources,
            learned_new_info=learned_new_info,
            history=[QA(question=input.question, answer=answer)]
        )
        
    except Exception as e:
        logger.error(f"Intelligent ask failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Processing failed - please try again")

@app.post("/contribute")
async def contribute_knowledge(contribution: ContributeInfo):
    """Allow users to contribute travel knowledge"""
    try:
        # Basic validation
        if len(contribution.information.strip()) < 10:
            raise HTTPException(status_code=400, detail="Information too short")
        
        # Store contribution for review
        intel_system.user_contributions.append({
            "place": contribution.place,
            "info": contribution.information,
            "user_id": contribution.user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Auto-approve contributions (in production, add moderation)
        ingest_travel_data([contribution.information], contribution.place)
        
        logger.info(f"New contribution for {contribution.place} from {contribution.user_id}")
        
        return {
            "status": "success",
            "message": f"Thank you for contributing information about {contribution.place}!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contribution failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process contribution")

@app.get("/system-status")
async def get_system_status():
    """Get intelligence system status"""
    return {
        "status": "operational",
        "unknown_places_being_tracked": len(intel_system.unknown_places),
        "learning_queue_size": len(intel_system.learning_queue),
        "recently_learned_places": len(intel_system.recently_learned),
        "total_contributions": len(intel_system.user_contributions),
        "last_cleanup": intel_system.last_cleanup.isoformat(),
        "most_requested_unknown": dict(intel_system.unknown_places) if intel_system.unknown_places else {}
    }

@app.get("/health")
def health_check():
    return {
        "status": "Intelligent AI Travel Agent operational",
        "version": "3.0.0",
        "features": [
            "Self-learning system",
            "Dynamic knowledge expansion", 
            "Confidence-based responses",
            "User contributions",
            "Background research"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)