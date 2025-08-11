#!/usr/bin/env python3
"""
Content Sourcing Agent using Groq API with Bloom's Taxonomy assessment and Assessment Agent
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import logging
import validators
import requests
import re
import sqlite3
import json
import argparse
from bs4 import BeautifulSoup
from groq import Groq, NotFoundError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from config_updated import get_config, AgentConfig

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ContentItem(BaseModel):
    id: str
    title: str
    content: str
    source_url: str
    category: str
    tags: List[str]
    timestamp: str
    quality_score: float
    metadata: Dict[str, Any]
    bloom_level: str = "Unknown"

class AssessmentItem(BaseModel):
    question_type: str
    question_text: str
    correct_answer: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    bloom_level: str = "unknown"
    objective: str = "general"
    curriculum_standard: str = "general"
    student_answer: Optional[str] = None
    score: Optional[float] = None

class StudentProfile(BaseModel):
    student_id: str
    name: str
    assessments_taken: List[AssessmentItem] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    average_score: float = 0.0
    total_assessments: int = 0

class TeacherProfile(BaseModel):
    teacher_id: str
    name: str
    students: List[str] = Field(default_factory=list)
    classwide_gaps: List[str] = Field(default_factory=list)
    assessment_summary: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    query: str = ""
    sources: List[str] = Field(default_factory=list)
    raw_content: List[Dict] = Field(default_factory=list)
    processed_content: List[ContentItem] = Field(default_factory=list)
    stored_content: List[str] = Field(default_factory=list)
    assessments: List[AssessmentItem] = Field(default_factory=list)
    current_step: str = "start"
    errors: List[str] = Field(default_factory=list)
    trigger: str = "manual"

class ProfileStorage:
    def __init__(self, db_path: str = "profiles.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT,
                    assessments_taken TEXT,
                    skill_gaps TEXT,
                    average_score REAL,
                    total_assessments INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teachers (
                    teacher_id TEXT PRIMARY KEY,
                    name TEXT,
                    students TEXT,
                    classwide_gaps TEXT,
                    assessment_summary TEXT
                )
            """)
            conn.commit()

    def save_student_profile(self, profile: StudentProfile):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO students (student_id, name, assessments_taken, skill_gaps, average_score, total_assessments)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile.student_id,
                profile.name,
                json.dumps([assessment.model_dump() for assessment in profile.assessments_taken]),
                json.dumps(profile.skill_gaps),
                profile.average_score,
                profile.total_assessments
            ))
            conn.commit()

    def save_teacher_profile(self, profile: TeacherProfile):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO teachers (teacher_id, name, students, classwide_gaps, assessment_summary)
                VALUES (?, ?, ?, ?, ?)
            """, (
                profile.teacher_id,
                profile.name,
                json.dumps(profile.students),
                json.dumps(profile.classwide_gaps),
                json.dumps(profile.assessment_summary)
            ))
            conn.commit()

    def get_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
            result = cursor.fetchone()
            if result:
                return StudentProfile(
                    student_id=result[0],
                    name=result[1],
                    assessments_taken=[AssessmentItem(**item) for item in json.loads(result[2])],
                    skill_gaps=json.loads(result[3]),
                    average_score=result[4],
                    total_assessments=result[5]
                )
        return None

    def get_teacher_profile(self, teacher_id: str) -> Optional[TeacherProfile]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teachers WHERE teacher_id = ?", (teacher_id,))
            result = cursor.fetchone()
            if result:
                return TeacherProfile(
                    teacher_id=result[0],
                    name=result[1],
                    students=json.loads(result[2]),
                    classwide_gaps=json.loads(result[3]),
                    assessment_summary=json.loads(result[4])
                )
        return None

class ContentAPI:
    def __init__(self):
        self.storage = {}
        self.counter = 0

    def store_content(self, content_item: ContentItem) -> str:
        self.counter += 1
        item_id = f"content_{self.counter}"
        self.storage[item_id] = content_item.model_dump()
        return item_id

    def get_content(self, content_id: str) -> Optional[Dict]:
        return self.storage.get(content_id)

    def list_all_content(self, query: Optional[str] = None) -> List[Dict]:
        if query:
            return self.search_content(query)
        return list(self.storage.values())

    def search_content(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for content in self.storage.values():
            if (query_lower in content['title'].lower() or
                query_lower in content['content'].lower() or
                any(query_lower in tag.lower() for tag in content['tags'])):
                results.append(content)
        return results

class ConfigurableLLM:
    def __init__(self, config: AgentConfig, api_key: str = "", model: str = "", base_url: str = "", max_tokens: int = 500):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        if not model:
            raise ValueError("GROQ_MODEL is required")
        if not base_url:
            raise ValueError("GROQ_BASE_URL is required")
        
        self.config = config
        self.client = Groq(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Initialized Groq LLM: {base_url} with model: {model}")

    def invoke(self, prompt: str, max_tokens: int = None, temperature: float = 0.7) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty or None")
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature
                )
                content = response.choices[0].message.content.strip()
                # Ensure response is valid JSON for grading prompts
                if "{" in content and "}" in content:
                    try:
                        json.loads(content)
                        return content
                    except json.JSONDecodeError:
                        # Attempt to fix incomplete JSON
                        content = content[:content.rfind("}") + 1]
                        try:
                            json.loads(content)
                            return content
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON, returning raw content: {content[:100]}...")
                            return content
                return content
            except NotFoundError as e:
                logger.error(f"Failed to invoke Groq API: {e}")
                raise
            except Exception as e:
                if '429' in str(e) and attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Error calling Groq LLM at {self.client.base_url}: {e}")
                return ""
        return ""

    def assess_bloom_taxonomy(self, content: str) -> str:
        if not content:
            return "Unknown"
        prompt = f"""
        Analyze the following content and determine the highest applicable Bloom's Taxonomy level:
        - Remembering
        - Understanding
        - Applying
        - Analyzing
        - Evaluating
        - Creating
        Return only the level name (e.g., 'remembering').
        Content: {content[:500]}...
        """
        try:
            response = self.invoke(prompt, max_tokens=50, temperature=0.3)
            response = response.strip().lower()
            bloom_levels = ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
            for level in bloom_levels:
                if level in response:
                    return level
            return "Unknown"
        except Exception as e:
            logger.error(f"Error assessing Bloom's Taxonomy: {e}")
            return "Unknown"

    def _check_answer_relevance(self, answer: str, question: str) -> float:
        automotive_keywords = [
            'ecu', 'autosar', 'diagnostics', 'fault detection', 'prediction', 
            'iso 26262', 'can bus', 'standardized interfaces', 'communication protocols', 
            'data exchange', 'fault prediction', 'reliability'
        ]
        question_lower = question.lower()
        answer_lower = answer.lower()
        matched_keywords = [kw for kw in automotive_keywords if kw in answer_lower]
        if len(matched_keywords) >= 3:
            return 0.9  # Very high relevance
        elif len(matched_keywords) >= 2:
            return 0.8  # High relevance
        elif len(matched_keywords) == 1:
            return 0.6  # Moderate relevance
        return 0.3  # Low relevance

    def generate_assessment(self, content: str, question_types: List[str], max_questions: int) -> List[AssessmentItem]:
        assessments = []
        bloom_levels = self.config.BLOOM_TAXONOMY_LEVELS
        questions_per_level = self.config.QUESTIONS_PER_BLOOM_LEVEL
        type_to_bloom = {
            'mcq': ['remembering', 'understanding'],
            'short_answer': ['applying'],
            'open_ended': ['analyzing', 'evaluating'],
            'descriptive': ['creating']
        }
        generated_types = {'mcq': 0, 'short_answer': 0, 'open_ended': 0, 'descriptive': 0}

        for bloom_level in bloom_levels:
            selected_types = [q_type for q_type, blooms in type_to_bloom.items() if bloom_level in blooms and q_type in question_types]
            if not selected_types:
                selected_types = question_types

            for q_type in selected_types:
                curriculum_standards_list = sum(
                    [[std] if isinstance(std, str) else std for std in self.config.CURRICULUM_STANDARDS.values()], []
                )
                prompt = f"""
                Generate {1 if q_type == 'descriptive' else questions_per_level} assessment question(s) of type '{q_type}' for Bloom's Taxonomy level: {bloom_level}.
                The question must focus on artificial intelligence in automotive systems, specifically AUTOSAR, ECU diagnostics, or fault prediction.
                Align with curriculum standard: {curriculum_standards_list[0]}.
                For MCQs, provide 4 options with one correct answer.
                For short_answer, provide a sample answer (1-2 sentences, max {self.config.SHORT_ANSWER_MAX_WORDS} words).
                For open_ended, provide a sample answer (max 200 words).
                For descriptive, provide a sample answer ({self.config.DESCRIPTIVE_MIN_WORDS}-{self.config.DESCRIPTIVE_MAX_WORDS} words).
                Return a JSON list of dictionaries with: "type", "question", "correct_answer", "options" (for mcq), "bloom_level", "objective", and "curriculum_standard" (string).
                Ensure valid JSON with no additional text or markdown.
                Content: {content[:500]}...
                """
                max_tokens = 6000 if q_type == 'descriptive' else (3000 if q_type == 'open_ended' else self.config.MAX_TOKENS)
                temperature = 0.2 if q_type == 'descriptive' else 0.5

                logger.debug(f"Assessment prompt for {bloom_level}, type {q_type}:\n{prompt}")
                for attempt in range(3):
                    try:
                        response = self.invoke(prompt, max_tokens=max_tokens, temperature=temperature)
                        logger.debug(f"Raw API response for {bloom_level}, type {q_type}: {response[:100]}...")
                        response = re.sub(r'```json\n|```', '', response).strip()
                        try:
                            questions = json.loads(response)
                            if not isinstance(questions, list):
                                questions = [questions]
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON response: {e}. Response: {response[:100]}...")
                            if attempt == 2:
                                assessments.append(self._create_default_question(bloom_level, q_type))
                                generated_types[q_type] += 1
                            continue
                        for q in questions[:questions_per_level]:
                            if q.get('bloom_level') != bloom_level:
                                q['bloom_level'] = bloom_level
                            if q.get('type') == 'mcq' and (not q.get('correct_answer') or not q.get('options') or len(q.get('options')) != 4):
                                logger.warning(f"Skipping invalid MCQ: {q.get('question')}")
                                continue
                            if q.get('type') in ['short_answer', 'open_ended', 'descriptive'] and not q.get('correct_answer'):
                                q['correct_answer'] = self._create_default_question(bloom_level, q_type).correct_answer
                            if q.get('type') == 'descriptive':
                                word_count = len(q.get('correct_answer', '').split())
                                if word_count < self.config.DESCRIPTIVE_MIN_WORDS or word_count > self.config.DESCRIPTIVE_MAX_WORDS:
                                    q['correct_answer'] = self._create_default_question(bloom_level, q_type).correct_answer
                            if q.get('type') == 'open_ended':
                                word_count = len(q.get('correct_answer', '').split())
                                if word_count > 200 or not self._check_answer_relevance(q.get('correct_answer', ''), q.get('question', '')):
                                    q['correct_answer'] = self._create_default_question(bloom_level, q_type).correct_answer
                            if q.get('type') == 'short_answer':
                                word_count = len(q.get('correct_answer', '').split())
                                if word_count > self.config.SHORT_ANSWER_MAX_WORDS:
                                    q['correct_answer'] = self._create_default_question(bloom_level, q_type).correct_answer
                            if isinstance(q.get('curriculum_standard'), list):
                                q['curriculum_standard'] = q['curriculum_standard'][0] if q['curriculum_standard'] else "Understand AI concepts"
                            assessments.append(AssessmentItem(
                                question_type=q.get('type', q_type),
                                question_text=q.get('question', 'Sample question'),
                                correct_answer=q.get('correct_answer'),
                                options=q.get('options', [] if q.get('type') != 'mcq' else ['a) Yes', 'b) No', 'c) Maybe', 'd) Unknown']),
                                bloom_level=q.get('bloom_level', bloom_level),
                                objective=q.get('objective', 'Understand AI concepts'),
                                curriculum_standard=q.get('curriculum_standard', 'Understand AI concepts')
                            ))
                            generated_types[q_type] += 1
                        break
                    except Exception as e:
                        logger.error(f"Error generating assessment: {e}")
                        if attempt == 2:
                            assessments.append(self._create_default_question(bloom_level, q_type))
                            generated_types[q_type] += 1

        required_types = {'mcq': 4, 'short_answer': 1, 'open_ended': 2, 'descriptive': 1}
        for q_type, min_count in required_types.items():
            if generated_types[q_type] < min_count:
                bloom_level = type_to_bloom[q_type][0]
                for _ in range(min_count - generated_types[q_type]):
                    assessments.append(self._create_default_question(bloom_level, q_type))
                    generated_types[q_type] += 1

        logger.info(f"Generated {len(assessments)} assessments: {generated_types}")
        return assessments[:max_questions]

    def _create_default_question(self, bloom_level: str, question_type: str) -> AssessmentItem:
        default_questions = {
            'mcq': {
                'question': f"What is a key application of AI in automotive systems at the {bloom_level} level?",
                'options': ['Data visualization', 'Complex decision-making', 'Basic pattern recognition', 'None'],
                'correct_answer': 'Complex decision-making',
                'objective': 'Understand AI applications',
                'curriculum_standard': 'Develop embedded systems'
            },
            'short_answer': {
                'question': f"Explain a use of AI in ECU diagnostics relevant to {bloom_level}.",
                'correct_answer': 'AUTOSAR’s standardized interfaces enable AI-driven ECU diagnostics by ensuring consistent data exchange across components.',
                'objective': 'Apply AI concepts',
                'curriculum_standard': 'Develop embedded systems'
            },
            'open_ended': {
                'question': f"Explain how ECU diagnostics can leverage AI techniques to improve fault detection at {bloom_level}.",
                'correct_answer': 'AUTOSAR’s communication protocols, like CAN, ensure reliable data exchange, enabling AI models to analyze real-time ECU data for accurate fault prediction.',
                'objective': 'Analyze AI applications',
                'curriculum_standard': 'Develop embedded systems'
            },
            'descriptive': {
                'question': f"Describe how AI can enhance AUTOSAR-based ECU development at the {bloom_level} level.",
                'correct_answer': 'AI enhances AUTOSAR-based ECU development by automating code generation for software components, ensuring compliance with ISO 26262 safety standards. Machine learning models analyze CAN bus data to predict failures, improving reliability. AUTOSAR’s standardized interfaces ensure seamless integration of AI models, enabling consistent diagnostics across ECUs. This streamlines development and enhances system robustness, reducing maintenance costs.',
                'objective': 'Develop advanced AI solutions',
                'curriculum_standard': 'Implement AUTOSAR standards'
            }
        }
        q = default_questions.get(question_type, default_questions['mcq'])
        return AssessmentItem(
            question_type=question_type,
            question_text=q['question'],
            correct_answer=q.get('correct_answer'),
            options=q.get('options', []),
            bloom_level=bloom_level,
            objective=q['objective'],
            curriculum_standard=q['curriculum_standard']
        )

class AssessmentAgent:
    def __init__(self, llm: ConfigurableLLM, config: AgentConfig, profile_storage: ProfileStorage):
        self.llm = llm
        self.config = config
        self.profile_storage = profile_storage

    def generate_assessment(self, content: List[ContentItem], trigger: str, teacher_id: str = None) -> List[AssessmentItem]:
        if not content:
            content = [ContentItem(
                id="default_1",
                title="Default Automotive AI Content",
                content="Artificial intelligence in automotive systems includes applications like AUTOSAR-based ECU development, predictive diagnostics, and autonomous driving.",
                source_url="http://default.example.com",
                category="automotive",
                tags=["AI", "AUTOSAR", "ECU"],
                timestamp=datetime.now().isoformat(),
                quality_score=1.0,
                metadata={'word_count': 50}
            )]
        combined_content = "\n".join(item.content for item in content)
        assessments = self.llm.generate_assessment(combined_content, self.config.ASSESSMENT_QUESTION_TYPES, self.config.MAX_QUESTIONS_PER_ASSESSMENT)
        if teacher_id:
            teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
            if teacher_profile:
                teacher_profile.assessment_summary['latest_assessments'] = [a.model_dump() for a in assessments]
                self.profile_storage.save_teacher_profile(teacher_profile)
        return assessments

    def grade_assessment(self, assessment: AssessmentItem, student_answer: str, student_id: str, teacher_id: str = None) -> float:
        logger.debug(f"Grading assessment: {assessment.question_text}, Type: {assessment.question_type}")
        result = {"score": 0.5, "feedback": "Default grading feedback"}
        if assessment.question_type == 'mcq':
            if not assessment.correct_answer or not assessment.options:
                logger.warning(f"Invalid MCQ: {assessment.question_text}")
                result = {"score": 0.5, "feedback": "Invalid MCQ configuration"}
                score = 0.5
            else:
                student_answer = student_answer.strip().lower()
                correct_answer = assessment.correct_answer.strip().lower()
                score = 1.0 if student_answer == correct_answer else 0.0
                result = {"score": score, "feedback": "MCQ graded based on exact match"}
        elif assessment.question_type in ['short_answer', 'open_ended', 'descriptive']:
            student_word_count = len(student_answer.split())
            if assessment.question_type == 'short_answer' and student_word_count > self.config.SHORT_ANSWER_MAX_WORDS:
                logger.warning(f"Short answer exceeds word limit: {student_word_count} words")
                result = {"score": 0.6, "feedback": "Answer exceeds word limit"}
                score = 0.6
            elif assessment.question_type == 'descriptive' and (student_word_count < self.config.DESCRIPTIVE_MIN_WORDS or student_word_count > self.config.DESCRIPTIVE_MAX_WORDS):
                logger.warning(f"Descriptive answer outside word limit: {student_word_count} words")
                result = {"score": 0.6, "feedback": "Answer outside word limit"}
                score = 0.6
            elif assessment.question_type == 'open_ended' and student_word_count > 200:
                logger.warning(f"Open-ended answer exceeds word limit: {student_word_count} words")
                result = {"score": 0.6, "feedback": "Answer exceeds word limit"}
                score = 0.6
            else:
                if assessment.correct_answer:
                    student_answer_clean = student_answer.strip().lower()
                    correct_answer_clean = assessment.correct_answer.strip().lower()
                    student_words = set(student_answer_clean.split())
                    correct_words = set(correct_answer_clean.split())
                    common_words = len(student_words & correct_words)
                    total_words = len(student_words | correct_words)
                    similarity = common_words / total_words if total_words > 0 else 0.0
                    if similarity >= 0.9 or student_answer_clean == correct_answer_clean:
                        score = 1.0
                        result = {"score": score, "feedback": "Exact or near-exact match with correct answer"}
                    elif similarity >= 0.7:
                        score = 0.8
                        result = {"score": score, "feedback": "Highly relevant answer with strong similarity to correct answer"}
                    else:
                        prompt = f"""
                        Assess the student answer for relevance and correctness based on the question and content domain (AI in automotive systems, focusing on AUTOSAR, ECU diagnostics, or fault prediction).
                        Question: {assessment.question_text}
                        Student Answer: {student_answer}
                        Correct Answer: {assessment.correct_answer or 'No specific correct answer; evaluate for relevance and depth.'}
                        Return a JSON object with: {{ "score": float, "feedback": str }}
                        Score from 0.0 to 1.0 based on accuracy, relevance, and adherence to word limits.
                        Award 1.0 for exact or near-exact matches, 0.7-0.9 for partially correct or relevant answers, 0.3-0.5 for irrelevant or incorrect answers.
                        For short_answer, expect 1-2 sentences (max {self.config.SHORT_ANSWER_MAX_WORDS} words).
                        For descriptive, expect {self.config.DESCRIPTIVE_MIN_WORDS}-{self.config.DESCRIPTIVE_MAX_WORDS} words.
                        For open_ended, expect max 200 words.
                        Ensure the response is valid JSON with no additional text or markdown.
                        """
                        try:
                            response = self.llm.invoke(prompt, max_tokens=300, temperature=0.3)
                            try:
                                result = json.loads(response)
                                score = float(result['score'])
                                score = max(0.0, min(1.0, score))
                                result['feedback'] = result.get('feedback', 'Graded based on LLM evaluation')
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON response: {response[:100]}...")
                                score = self.llm._check_answer_relevance(student_answer, assessment.question_text)
                                result = {"score": score, "feedback": f"Grading fallback: {'Very highly relevant' if score >= 0.9 else 'Highly relevant' if score >= 0.8 else 'Moderately relevant' if score >= 0.6 else 'Irrelevant'} answer"}
                        except Exception as e:
                            logger.error(f"Error grading {assessment.question_type}: {e}")
                            score = self.llm._check_answer_relevance(student_answer, assessment.question_text)
                            result = {"score": score, "feedback": f"Grading fallback: {'Very highly relevant' if score >= 0.9 else 'Highly relevant' if score >= 0.8 else 'Moderately relevant' if score >= 0.6 else 'Irrelevant'} answer"}
                else:
                    score = self.llm._check_answer_relevance(student_answer, assessment.question_text)
                    result = {"score": score, "feedback": f"Grading fallback: {'Very highly relevant' if score >= 0.9 else 'Highly relevant' if score >= 0.8 else 'Moderately relevant' if score >= 0.6 else 'Irrelevant'} answer"}
        else:
            logger.warning(f"Unknown question type: {assessment.question_type}")
            score = 0.5
            result = {"score": score, "feedback": "Unknown question type"}

        assessment.student_answer = student_answer
        assessment.score = score

        student_profile = self.profile_storage.get_student_profile(student_id)
        if not student_profile:
            student_profile = StudentProfile(student_id=student_id, name=f"Student_{student_id}")
        student_profile.assessments_taken.append(assessment)
        student_profile.total_assessments += 1
        total_score = sum(a.score for a in student_profile.assessments_taken if a.score is not None)
        student_profile.average_score = total_score / student_profile.total_assessments if student_profile.total_assessments else 0.0
        if score < 0.9:  # Modified: Capture skill gaps for scores below 0.9
            student_profile.skill_gaps.append(
                f"Score {score:.2f} on {assessment.question_text} (Bloom: {assessment.bloom_level}, Type: {assessment.question_type})"
            )
        logger.debug(f"Student {student_id} score: {score:.2f}, Skill gaps: {student_profile.skill_gaps}")
        self.profile_storage.save_student_profile(student_profile)

        if teacher_id:
            teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
            if not teacher_profile:
                teacher_profile = TeacherProfile(teacher_id=teacher_id, name=f"Teacher_{teacher_id}")
            if student_id not in teacher_profile.students:
                teacher_profile.students.append(student_id)
            all_assessments = []
            for sid in teacher_profile.students:
                s_profile = self.profile_storage.get_student_profile(sid)
                if s_profile:
                    all_assessments.extend(s_profile.assessments_taken)
            teacher_profile.classwide_gaps = self.flag_skill_gaps(all_assessments)['classwide']
            teacher_profile.assessment_summary['total_submissions'] = (
                teacher_profile.assessment_summary.get('total_submissions', 0) + 1
            )
            teacher_profile.assessment_summary['average_score'] = (
                sum(self.profile_storage.get_student_profile(sid).average_score for sid in teacher_profile.students) /
                len(teacher_profile.students)
            ) if teacher_profile.students else 0.0
            self.profile_storage.save_teacher_profile(teacher_profile)

        logger.info(f"Logged assessment for student {student_id}, score: {score:.2f}, feedback: {result['feedback']}")
        return score

    def flag_skill_gaps(self, assessments: List[AssessmentItem]) -> Dict[str, List[str]]:
        gaps = {'individual': [], 'classwide': []}
        bloom_counts = {level: {'count': 0, 'low_scores': 0} for level in self.config.BLOOM_TAXONOMY_LEVELS}
        type_counts = {q_type: {'count': 0, 'low_scores': 0} for q_type in self.config.ASSESSMENT_QUESTION_TYPES}

        for assessment in assessments:
            if assessment.score is not None:
                if assessment.score < 0.9:  # Modified: Capture gaps for scores below 0.9
                    gaps['individual'].append(
                        f"Score {assessment.score:.2f} on {assessment.question_text[:50]}... "
                        f"(Bloom: {assessment.bloom_level}, Type: {assessment.question_type})"
                    )
                bloom_counts[assessment.bloom_level]['count'] += 1
                type_counts[assessment.question_type]['count'] += 1
                if assessment.score < 0.9:
                    bloom_counts[assessment.bloom_level]['low_scores'] += 1
                    type_counts[assessment.question_type]['low_scores'] += 1

        for bloom_level, stats in bloom_counts.items():
            if stats['count'] > 0 and stats['low_scores'] / stats['count'] > 0.2:
                gaps['classwide'].append(f"Consistent low performance in {bloom_level} (Bloom level, {stats['low_scores']}/{stats['count']} low scores)")
        for q_type, stats in type_counts.items():
            if stats['count'] > 0 and stats['low_scores'] / stats['count'] > 0.2:
                gaps['classwide'].append(f"Consistent low performance in {q_type} questions ({stats['low_scores']}/{stats['count']} low scores)")

        if not gaps['individual']:
            gaps['individual'].append("No individual skill gaps identified (all scores >= 0.9)")
        if not gaps['classwide']:
            gaps['classwide'].append("No significant class-wide gaps identified")
        
        logger.debug(f"Skill gaps identified: Individual: {gaps['individual']}, Classwide: {gaps['classwide']}")
        return gaps

class ContentSourcingAgent:
    def __init__(self, config: AgentConfig, api_key: Optional[str] = None, model: str = "", base_url: str = "", max_tokens: int = None):
        self.content_api = ContentAPI()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CONTENT_CHUNK_SIZE,
            chunk_overlap=config.CONTENT_CHUNK_OVERLAP
        )
        self.config = config
        self.profile_storage = ProfileStorage(db_path=config.PROFILE_DB_PATH)
        api_key = api_key or self.config.GROQ_API_KEY or os.getenv('GROQ_API_KEY', '')
        
        self.assessments: List[AssessmentItem] = []
        
        try:
            self.llm = ConfigurableLLM(
                config=config,
                api_key=api_key,
                model=model or self.config.LLM_MODEL,
                base_url=base_url or self.config.LLM_BASE_URL,
                max_tokens=max_tokens or self.config.MAX_TOKENS
            )
            self.assessment_agent = AssessmentAgent(self.llm, self.config, self.profile_storage)
        except ValueError as e:
            logger.error(f"LLM initialization failed: {e}")
            self.llm = None
            self.assessment_agent = None
            logger.warning("Using rule-based processing")
        
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("initialize", self._initialize_sources)
        workflow.add_node("fetch_content", self._fetch_content)
        workflow.add_node("process_content", self._process_content)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("store_content", self._store_content)
        workflow.add_node("generate_assessment", self._generate_assessment)
        workflow.add_node("finalize", self._finalize)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "fetch_content")
        workflow.add_edge("fetch_content", "process_content")
        workflow.add_edge("process_content", "quality_check")
        workflow.add_edge("quality_check", "store_content")
        workflow.add_edge("store_content", "generate_assessment")
        workflow.add_edge("generate_assessment", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _initialize_sources(self, state: AgentState) -> AgentState:
        logger.info(f"Initializing sources for query: {state.query}")
        if not state.sources:
            raise ValueError("No sources provided")
        state.current_step = "initialize"
        state.sources = [url for url in state.sources if validators.url(url)]
        if not state.sources:
            raise ValueError("No valid URLs provided")
        return state

    def _fetch_content_from_url(self, url: str) -> Dict[str, str]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title').text.strip() if soup.find('title') else "Untitled"
            content = soup.find('body').get_text() if soup.find('body') else ""
            content = ' '.join(content.split())[:8000] + "..." if len(content) > 8000 else content
            if not content or len(content.strip()) < 50:
                raise Exception("Insufficient content")
            return {'title': title, 'content': content.strip()}
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {str(e)}")
            return {'title': 'Default Content', 'content': 'Artificial intelligence in automotive systems includes applications like AUTOSAR-based ECU development.'}

    def _fetch_content(self, state: AgentState) -> AgentState:
        state.current_step = "fetch_content"
        for source_url in state.sources:
            try:
                real_content = self._fetch_content_from_url(source_url)
                state.raw_content.append({
                    'url': source_url,
                    'content': real_content['content'],
                    'title': real_content['title']
                })
            except Exception as e:
                state.errors.append(f"Failed to fetch from {source_url}: {str(e)}")
        if not state.raw_content:
            state.raw_content.append({
                'url': 'http://default.example.com',
                'content': 'Artificial intelligence in automotive systems includes applications like AUTOSAR-based ECU development.',
                'title': 'Default Automotive AI Content'
            })
        return state

    def _process_content(self, state: AgentState) -> AgentState:
        state.current_step = "process_content"
        for item in state.raw_content:
            try:
                processed_item = ContentItem(
                    id=f"item_{len(state.processed_content) + 1}",
                    title=item['title'],
                    content=item['content'],
                    source_url=item['url'],
                    category=self._determine_category(item['content']),
                    tags=self._extract_tags(item['content']),
                    timestamp=datetime.now().isoformat(),
                    bloom_level=self._determine_bloom_level(item['content']),
                    quality_score=0.0,
                    metadata={'word_count': len(item['content'].split())}
                )
                state.processed_content.append(processed_item)
            except Exception as e:
                state.errors.append(f"Failed to process content: {str(e)}")
        if not state.processed_content:
            state.processed_content.append(ContentItem(
                id="default_1",
                title="Default Automotive AI Content",
                content="Artificial intelligence in automotive systems includes applications like AUTOSAR-based ECU development.",
                source_url="http://default.example.com",
                category="automotive",
                tags=["AI", "AUTOSAR", "ECU"],
                timestamp=datetime.now().isoformat(),
                quality_score=1.0,
                metadata={'word_count': 50}
            ))
        return state

    def _determine_bloom_level(self, content: str) -> str:
        if self.llm:
            return self.llm.assess_bloom_taxonomy(content)
        return "Unknown"

    def _determine_category(self, content: str) -> str:
        content_lower = content.lower()
        automotive_keywords = ['autosar', 'ecu', 'diagnostics', 'fault detection', 'prediction', 'iso 26262', 'can bus']
        if any(keyword in content_lower for keyword in automotive_keywords):
            return 'automotive'
        return 'artificial_intelligence'

    def _extract_tags(self, content: str) -> List[str]:
        content_lower = content.lower()
        automotive_tags = ['autosar', 'ecu', 'diagnostics', 'fault detection', 'prediction', 'iso 26262', 'can bus']
        tags = [tag for tag in automotive_tags if tag in content_lower]
        if not tags:
            tags = ['ai', 'machine learning']
        return tags[:5]

    def _quality_check(self, state: AgentState) -> AgentState:
        state.current_step = "quality_check"
        for item in state.processed_content:
            item.quality_score = self._calculate_quality_score(item)
        state.processed_content = [item for item in state.processed_content if item.quality_score >= self.config.QUALITY_THRESHOLD]
        return state

    def _calculate_quality_score(self, item: ContentItem) -> float:
        score = 0.0
        word_count = len(item.content.split())
        if 50 <= word_count <= 1000:
            score += 0.3
        if 10 < len(item.title) < 100:
            score += 0.2
        if len(item.tags) >= 2:
            score += 0.2
        if item.category == 'automotive':
            score += 0.25
        if any(domain in item.source_url for domain in self.config.RELIABLE_DOMAINS):
            score += 0.15
        return min(score, 1.0)

    def _store_content(self, state: AgentState) -> AgentState:
        state.current_step = "store_content"
        for item in state.processed_content:
            try:
                stored_id = self.content_api.store_content(item)
                state.stored_content.append(stored_id)
            except Exception as e:
                state.errors.append(f"Failed to store {item.title}: {str(e)}")
        return state

    def _generate_assessment(self, state: AgentState) -> AgentState:
        state.current_step = "generate_assessment"
        if self.assessment_agent and state.processed_content:
            state.assessments = self.assessment_agent.generate_assessment(state.processed_content, state.trigger)
            skill_gaps = self.assessment_agent.flag_skill_gaps(state.assessments)
            if skill_gaps['individual']:
                logger.warning("Individual skill gaps: " + ", ".join(skill_gaps['individual']))
            if skill_gaps['classwide']:
                logger.warning("Class-wide skill gaps: " + ", ".join(skill_gaps['classwide']))
        return state

    def _finalize(self, state: AgentState) -> AgentState:
        state.current_step = "finalize"
        print("\n" + "="*50)
        print("EXECUTION SUMMARY")
        print("="*50)
        print(f"Query: {state.query}")
        print(f"Sources processed: {len(state.sources)}")
        print(f"Content items fetched: {len(state.raw_content)}")
        print(f"Content items processed: {len(state.processed_content)}")
        print(f"Content items stored: {len(state.stored_content)}")
        print(f"Assessments generated: {len(state.assessments)}")
        print(f"Errors encountered: {len(state.errors)}")
        if state.errors:
            print("\nErrors:")
            for error in state.errors:
                print(f"  - {error}")
        if state.assessments:
            print("\nGenerated Assessments:")
            for i, assessment in enumerate(state.assessments, 1):
                print(f"\n{i}. Type: {assessment.question_type}")
                print(f"   Question: {assessment.question_text}")
                if assessment.options:
                    print(f"   Options: {', '.join(f'{opt}' for opt in assessment.options)}")
                print(f"   Correct Answer: {assessment.correct_answer or 'None'}")
                print(f"   Bloom Level: {assessment.bloom_level}")
                print(f"   Objective: {assessment.objective}")
                print(f"   Curriculum Standard: {assessment.curriculum_standard}")
        print("\nAgent execution completed successfully!")
        return state

    def run(self, query: str, sources: List[str], trigger: str = "manual") -> List[AssessmentItem]:
        logger.info(f"Starting Content Sourcing Agent with query: {query}, trigger: {trigger}")
        if not sources:
            raise ValueError("No sources provided")
        initial_state = AgentState(query=query, sources=sources, trigger=trigger)
        final_state_dict = self.workflow.invoke(initial_state)
        self.assessments = final_state_dict.get('assessments', [])
        return self.assessments

    def search_stored_content(self, query: str) -> List[Dict]:
        return self.content_api.search_content(query)

    def get_all_stored_content(self) -> List[Dict]:
        return self.content_api.list_all_content()

    def submit_assessment(self, assessment_id: int, student_answer: str, student_id: str, teacher_id: str = None) -> Dict:
        if not self.assessments or assessment_id >= len(self.assessments):
            return {"error": "Invalid assessment ID"}
        assessment = self.assessments[assessment_id]
        score = self.assessment_agent.grade_assessment(assessment, student_answer, student_id, teacher_id)
        return {
            "question": assessment.question_text,
            "score": score,
            "feedback": f"Score: {score:.2f}",
            "student_id": student_id
        }

    def get_student_report(self, student_id: str) -> Dict:
        profile = self.profile_storage.get_student_profile(student_id)
        if not profile:
            return {"error": "Student profile not found"}
        return {
            "student_id": profile.student_id,
            "name": profile.name,
            "total_assessments": profile.total_assessments,
            "average_score": profile.average_score,
            "skill_gaps": profile.skill_gaps,
            "assessments": [a.model_dump() for a in profile.assessments_taken]
        }

    def get_teacher_report(self, teacher_id: str) -> Dict:
        profile = self.profile_storage.get_teacher_profile(teacher_id)
        if not profile:
            return {"error": "Teacher profile not found"}
        return {
            "teacher_id": profile.teacher_id,
            "name": profile.name,
            "students": profile.students,
            "classwide_gaps": profile.classwide_gaps,
            "assessment_summary": profile.assessment_summary
        }

def main():
    print("Universal Content Sourcing Agent Demo")
    print("="*45)
    
    parser = argparse.ArgumentParser(description="Content Sourcing Agent Demo")
    parser.add_argument("--student-id", type=str, help="Student ID for assessment submission")
    parser.add_argument("--teacher-id", type=str, help="Teacher ID for assessment tracking")
    args = parser.parse_args()
    
    student_id = args.student_id or os.getenv('STUDENT_ID', 'student_001')
    teacher_id = args.teacher_id or os.getenv('TEACHER_ID', 'teacher_001')
    
    config = get_config()
    try:
        agent = ContentSourcingAgent(
            config=config,
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.MAX_TOKENS
        )
        query = os.getenv('TEST_QUERY', 'artificial intelligence in automotive systems')
        sources = config.STATIC_SOURCES
        assessments = agent.run(query, sources, trigger="manual")
        
        custom_answers = {
            0: "AUTOSAR is a software framework for automotive systems.",  # Partially correct, likely ~0.7-0.8
            1: "ECU diagnostics use AI to visualize data.",  # Incorrect, likely ~0.3-0.5
            2: "CAN bus is a network protocol.",  # Partially correct, likely ~0.6
            3: "ISO 26262 ensures safety in automotive systems.",  # Correct but brief, likely ~0.8
            4: "AUTOSAR's standardized interfaces simplify ECU diagnostics by providing consistent data formats across components.",
            5: "AUTOSAR improves fault prediction by standardizing communication protocols, enabling AI to analyze ECU data effectively.",
            6: "AUTOSAR’s protocols, like CAN, ensure reliable data exchange, enabling AI to predict faults with high accuracy.",
            7: "AUTOSAR-based diagnostics offer standardized interfaces, improving interoperability over manufacturer-specific methods."
        }
        
        if assessments:
            for i, assessment in enumerate(assessments, 1):
                print(f"\n{i}. Assessment Question: {assessment.question_text}")
                print(f"Question Type: {assessment.question_type}")
                if assessment.options:
                    print(f"Options: {', '.join(assessment.options)}")
                print(f"Correct Answer: {assessment.correct_answer or 'None'}")
                
                sample_answer = custom_answers.get(i-1, "AI improves ECU diagnostics by analyzing CAN bus data.")
                print(f"Submitting answer for student {student_id}: {sample_answer}")
                response = agent.submit_assessment(i-1, sample_answer, student_id, teacher_id)
                if "error" not in response:
                    print(f"Question: {response['question']}")
                    print(f"Score: {response['score']:.2f}")
                    print(f"Feedback: {response['feedback']}")
        
        print("\n" + "="*60)
        print("STUDENT REPORT")
        print("="*60)
        student_report = agent.get_student_report(student_id)
        if "error" not in student_report:
            print(f"Student ID: {student_report['student_id']}")
            print(f"Name: {student_report['name']}")
            print(f"Total Assessments: {student_report['total_assessments']}")
            print(f"Average Score: {student_report['average_score']:.2f}")
            print(f"Skill Gaps: {', '.join(student_report['skill_gaps']) if student_report['skill_gaps'] else 'None'}")
        
        print("\n" + "="*60)
        print("TEACHER REPORT")
        print("="*60)
        teacher_report = agent.get_teacher_report(teacher_id)
        if "error" not in teacher_report:
            print(f"Teacher ID: {teacher_report['teacher_id']}")
            print(f"Name: {teacher_report['name']}")
            print(f"Students: {', '.join(teacher_report['students']) if teacher_report['students'] else 'None'}")
            print(f"Class-wide Gaps: {', '.join(teacher_report['classwide_gaps']) if teacher_report['classwide_gaps'] else 'None'}")
            print(f"Assessment Summary: {teacher_report['assessment_summary']}")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    main()