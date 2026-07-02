import logging
from sentence_transformers import SentenceTransformer
from src.offline.text_cleaner import clean_text

logger = logging.getLogger(__name__)

class JDMemoryService:
    def __init__(self):
        logger.info("Initializing Memory Service: Loading embedding model...")
        # Load the AI brain into RAM once
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # This variable acts as your persistent in-memory storage for the JD
        self.stored_jd_embedding = None 
        logger.info("Service Ready. Ready to embed and store JD.")

    def update_jd_memory(self, jd_text: str):
        """Reads, embeds, and stores the JD vector in RAM."""
        logger.info("Cleaning and embedding JD text...")
        cleaned_jd = clean_text(jd_text)
        
        # Calculate and store the embedding directly in RAM
        self.stored_jd_embedding = self.model.encode(
            [cleaned_jd], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        logger.info("New JD embedding stored in RAM memory.")
memory_service = JDMemoryService()

memory_service.update_jd_memory("Senior ML Engineer, Python, PyTorch")
