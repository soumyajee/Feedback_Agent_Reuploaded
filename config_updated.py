
from pydantic import BaseModel
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class AgentConfig(BaseModel):
    # Groq API settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    LLM_MODEL: str = os.getenv("GROQ_MODEL", "gemma2-9b-it")
    LLM_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 7000))  # Increased for descriptive answers

    # Agent configuration
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", 0.5))
    CONTENT_CHUNK_SIZE: int = 1000
    CONTENT_CHUNK_OVERLAP: int = 200
    MAX_SOURCES: int = 6
    QUESTIONS_PER_BLOOM_LEVEL: int = 2
    MAX_QUESTIONS_PER_ASSESSMENT: int = 8
    SHORT_ANSWER_MAX_WORDS: int = 50
    DESCRIPTIVE_MIN_WORDS: int = 100
    DESCRIPTIVE_MAX_WORDS: int = 300
    SKILL_GAP_THRESHOLD: float = 0.7

    # Static sources
    STATIC_SOURCES: List[str] = [
        "https://en.wikipedia.org/wiki/AUTOSAR",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://arxiv.org/abs/2303.08774",
        "https://en.wikipedia.org/wiki/Electronic_control_unit",
        "https://www.sae.org/standards/content/j1939_201808/",
        "https://arxiv.org/abs/2006.06068"
    ]

    # Content categories and keywords
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "automotive": ["autosar", "ecu", "diagnostics", "fault detection", "prediction", "iso 26262", "can bus"],
        "artificial_intelligence": ["artificial intelligence", "ai", "machine learning", "neural network"]
    }

    # Quality scoring weights
    QUALITY_WEIGHTS: Dict[str, float] = {
        "content_length": 0.3,
        "title_quality": 0.2,
        "tags_relevance": 0.2,
        "category_assignment": 0.25,
        "source_reliability": 0.15
    }

    # Content processing settings
    MIN_CONTENT_LENGTH: int = 50
    MAX_CONTENT_LENGTH: int = 5000
    MIN_TITLE_LENGTH: int = 10
    MAX_TITLE_LENGTH: int = 100
    MAX_TAGS_PER_ITEM: int = 5

    # Reliable domains
    RELIABLE_DOMAINS: List[str] = [
        "wikipedia.org", "edu", "gov", "nature.com", "arxiv.org", "sae.org"
    ]

    # Assessment configuration
    BLOOM_TAXONOMY_LEVELS: List[str] = ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
    ASSESSMENT_QUESTION_TYPES: List[str] = ["mcq", "short_answer", "open_ended", "descriptive"]
    CURRICULUM_STANDARDS: Dict[str, List[str]] = {
        "automotive": ["Implement AUTOSAR standards", "Develop embedded systems", "Understand AI concepts"]
    }

    # Profile management
    PROFILE_DB_PATH: str = "profiles.db"
    MAX_PROFILE_ASSESSMENTS: int = 100

    def validate_config(self) -> List[str]:
        issues = []
        if not self.GROQ_API_KEY:
            issues.append("GROQ_API_KEY is not set")
        if not self.STATIC_SOURCES:
            issues.append("No static sources configured")
        if len(self.STATIC_SOURCES) > self.MAX_SOURCES:
            issues.append(f"Number of static sources ({len(self.STATIC_SOURCES)}) exceeds MAX_SOURCES ({self.MAX_SOURCES})")
        if self.QUALITY_THRESHOLD < 0 or self.QUALITY_THRESHOLD > 1:
            issues.append("Quality threshold must be between 0 and 1")
        if self.MAX_SOURCES <= 0:
            issues.append("Max sources must be positive")
        if self.MIN_CONTENT_LENGTH >= self.MAX_CONTENT_LENGTH:
            issues.append("Min content length must be less than max content length")
        if self.DESCRIPTIVE_MIN_WORDS >= self.DESCRIPTIVE_MAX_WORDS:
            issues.append("Descriptive min words must be less than max words")
        if sum(self.QUALITY_WEIGHTS.values()) != 1.0:
            issues.append(f"Quality weights sum to {sum(self.QUALITY_WEIGHTS.values())}, should be 1.0")
        return issues

def get_config() -> AgentConfig:
    return AgentConfig()

if __name__ == "__main__":
    config = get_config()
    print("Configuration Summary")
    print("="*50)
    print(f"LLM Model: {config.LLM_MODEL}")
    print(f"Max Tokens: {config.MAX_TOKENS}")
    print(f"Skill Gap Threshold: {config.SKILL_GAP_THRESHOLD}")
    print(f"Static Sources: {len(config.STATIC_SOURCES)}")
    print(f"Assessment Types: {', '.join(config.ASSESSMENT_QUESTION_TYPES)}")
    issues = config.validate_config()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid")

