#!/usr/bin/env python3
"""
Feedback Agent for summarizing assessment results with learning gap detection
"""

import csv
import os
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from pydantic import BaseModel
from agant_updated_assement import ProfileStorage, AssessmentItem, StudentProfile, TeacherProfile, ConfigurableLLM, get_config, AgentConfig
import json
import argparse
from statistics import mean, median

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackSummary(BaseModel):
    student_summaries: Dict[str, Dict] = {}
    teacher_summaries: Dict[str, Dict] = {}
    objective_summaries: Dict[str, Dict] = {}
    learning_gaps: Dict[str, Dict] = {}  # New field for learning gaps by objective
    low_performing_students: List[Dict] = []  # New field for low-performing students
    misunderstood_concepts: List[Dict] = []  # New field for misunderstood concepts

class FeedbackAgent:
    def __init__(self, config: AgentConfig, profile_storage: ProfileStorage, llm: ConfigurableLLM, csv_path: str = "assessment_data.csv"):
        self.config = config
        self.profile_storage = profile_storage
        self.llm = llm
        self.csv_path = csv_path
        self.assessments: List[Dict] = []
        # Create CSV file with embedded data
        self._create_csv_file()
        logger.info(f"Initialized FeedbackAgent with CSV path: {csv_path}")

    def _create_csv_file(self):
        """Create a CSV file with embedded data."""
        csv_data = """student_id,teacher_id,objective,question,answer
student_001,teacher_001,Understand AI concepts,What is the primary role of ECU diagnostics in vehicles?,ECU diagnostics monitor vehicle systems to detect faults and ensure performance.
student_002,teacher_002,Apply AI concepts,How do AUTOSAR protocols enhance ECU data processing?,AUTOSAR's protocols enable consistent data exchange, improving ECU data processing efficiency.
student_003,teacher_003,Analyze AI applications,What advantages does AI bring to AUTOSAR-based fault detection?,AI analyzes AUTOSAR data to predict faults with higher accuracy using real-time ECU inputs.
student_004,teacher_004,Understand AI concepts,Why are ECU diagnostics critical for automotive safety?,ECU diagnostics ensure vehicle reliability by identifying system faults early.
student_005,teacher_005,Apply AI concepts,Describe a way AUTOSAR improves diagnostic data consistency.,AUTOSAR provides uniform data formats, enhancing diagnostic data consistency across ECUs.
student_006,teacher_006,Analyze AI applications,How can AI fault prediction benefit from AUTOSAR interfaces?,AI fault prediction benefits from AUTOSAR interfaces by accessing standardized ECU data for analysis.
student_007,teacher_001,Understand AI concepts,What is the main purpose of ECU diagnostics in automotive systems?,ECU diagnostics identify faults to maintain vehicle performance and safety.
student_008,teacher_002,Apply AI concepts,Explain how AUTOSAR supports efficient ECU troubleshooting.,AUTOSAR ensures reliable data exchange, supporting efficient ECU troubleshooting processes.
student_009,teacher_003,Analyze AI applications,What role does AI play in enhancing AUTOSAR diagnostics?,AI enhances AUTOSAR diagnostics by processing ECU data for predictive maintenance.
student_010,teacher_004,Understand AI concepts,How does ECU diagnostics contribute to vehicle reliability?,ECU diagnostics detect issues to ensure consistent vehicle reliability.
student_011,teacher_005,Apply AI concepts,What is a key benefit of using AUTOSAR for ECU diagnostics?,AUTOSAR's standardized protocols streamline ECU diagnostic data handling.
student_012,teacher_006,Analyze AI applications,How can AI integrate with AUTOSAR to predict ECU failures?,AI integrates with AUTOSAR by analyzing CAN bus data to predict ECU failures accurately."""
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            csvfile.write(csv_data)

    def infer_bloom_level(self, question: str, answer: str) -> str:
        """Infer Bloom's Taxonomy level for a question and answer, aligned with objective."""
        prompt = f"""
        Infer the Bloom's Taxonomy level for the following question and answer, considering the question's objective and complexity.
        Question: {question}
        Answer: {answer}
        Guidelines:
        - 'Understand' for questions asking for definitions, purposes, or basic explanations (e.g., 'What is...', 'Why is...').
        - 'Apply' for questions asking for practical applications or methods (e.g., 'How does...', 'Describe a way...').
        - 'Analyze' for questions requiring analysis, comparison, or evaluation (e.g., 'What advantages...', 'What role...').
        - Consider the question's phrasing and objective explicitly.
        - Return only the Bloom's level as a string (e.g., 'understanding', 'applying', 'analyzing').
        """
        try:
            bloom_level = self.llm.invoke(prompt, max_tokens=50, temperature=0.1).strip().lower()
            logger.debug(f"Inferred Bloom level: {bloom_level} for question: {question[:50]}...")
            return bloom_level
        except Exception as e:
            logger.warning(f"Error inferring Bloom level: {e}. Using fallback.")
            # Fallback based on objective
            objective = question.split(',')[2] if ',' in question else ''
            if 'Understand' in objective:
                return 'understanding'
            elif 'Apply' in objective:
                return 'applying'
            elif 'Analyze' in objective:
                return 'analyzing'
            return 'understanding'  # Default

    def infer_question_type(self, question: str, answer: str) -> str:
        """Infer question type based on question and answer characteristics."""
        word_count = len(answer.split())
        if word_count <= self.config.SHORT_ANSWER_MAX_WORDS:
            return "short_answer"
        elif word_count <= 200:
            return "open_ended"
        elif self.config.DESCRIPTIVE_MIN_WORDS <= word_count <= self.config.DESCRIPTIVE_MAX_WORDS:
            return "descriptive"
        return "open_ended"  # Default fallback

    def compute_score(self, question: str, answer: str, objective: str) -> float:
        """Compute a score for the answer using LLM-based relevance checking with improved fallback."""
        prompt = f"""
        Assess the student answer for relevance and correctness based on the question and content domain (AI in automotive systems, focusing on AUTOSAR, ECU diagnostics, or fault prediction).
        Question: {question}
        Student Answer: {answer}
        Objective: {objective}
        Return a JSON object with: {{ "score": float, "feedback": str }}
        Score from 0.0 to 1.0 based on accuracy and relevance.
        Guidelines:
        - Award 0.9-1.0 for highly relevant, fully correct answers that address the core concept.
        - Award 0.7-0.89 for partially correct answers with minor omissions or slight rephrasing.
        - Award 0.5-0.69 for answers that are minimally relevant but related to the topic.
        - Award 0.0-0.49 for incorrect or irrelevant answers.
        - Be lenient with answers that use synonyms (e.g., 'reliability' for 'performance') or rephrase concepts accurately.
        - Prioritize alignment with the objective and conceptual accuracy over exact wording.
        Ensure the response is valid JSON with no additional text or markdown.
        """
        try:
            response = self.llm.invoke(prompt, max_tokens=500, temperature=0.2)
            result = json.loads(response)
            score = float(result['score'])
            logger.debug(f"LLM score: {score}, Feedback: {result['feedback']} for question: {question[:50]}... and answer: {answer[:50]}...")
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error computing score with LLM: {e}. Falling back to improved scoring.")
            # Improved fallback: Check for key concepts and answer length
            automotive_keywords = {
                'Understand AI concepts': ['ecu', 'diagnostics', 'fault', 'monitor', 'safety', 'reliability', 'performance'],
                'Apply AI concepts': ['autosar', 'data', 'exchange', 'protocols', 'consistency', 'troubleshooting', 'efficiency'],
                'Analyze AI applications': ['ai', 'autosar', 'fault', 'prediction', 'diagnostics', 'data', 'analysis', 'predictive']
            }
            answer_words = set(answer.lower().split())
            objective_keywords = set(automotive_keywords.get(objective, []))
            question_words = set(question.lower().split())
            
            # Base score: Proportion of relevant keywords
            common_keywords = len(answer_words & objective_keywords) / len(objective_keywords) if objective_keywords else 0.0
            base_score = common_keywords * 0.6  # Lower weight to avoid over-penalization
            
            # Bonus for answer completeness
            word_count = len(answer.split())
            completeness_bonus = min(word_count / 20, 0.3)  # Up to 0.3 bonus for longer, detailed answers
            
            # Bonus for question relevance
            question_overlap = len(answer_words & question_words) / len(question_words) if question_words else 0.0
            relevance_bonus = question_overlap * 0.2  # Up to 0.2 bonus for question alignment
            
            score = base_score + completeness_bonus + relevance_bonus
            logger.debug(f"Fallback score: {score:.2f} (base: {base_score:.2f}, completeness: {completeness_bonus:.2f}, relevance: {relevance_bonus:.2f}) for question: {question[:50]}... and answer: {answer[:50]}...")
            return max(0.0, min(1.0, score))

    def read_csv(self):
        """Read assessment data from CSV and update profiles with inferred attributes."""
        self.assessments = []
        # Clear existing profiles to ensure no stale data is used
        try:
            self.profile_storage.clear_profiles()  # Assumes ProfileStorage has a clear_profiles method
            logger.info("Cleared existing profiles to prevent stale data.")
        except AttributeError:
            logger.warning("ProfileStorage does not have clear_profiles method. Ensure profiles are managed correctly.")

        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                expected_fields = {'student_id', 'teacher_id', 'objective', 'question', 'answer'}
                if not all(field in reader.fieldnames for field in expected_fields):
                    raise ValueError(f"CSV must contain fields: {', '.join(expected_fields)}")
                logger.info(f"Reading CSV from: {os.path.abspath(self.csv_path)}")

                for row in reader:
                    student_id = row['student_id'].strip()
                    teacher_id = row['teacher_id'].strip()
                    objective = row['objective'].strip()
                    question = row['question'].strip()
                    answer = row['answer'].strip()

                    # Validate input data
                    if not all([student_id, teacher_id, objective, question, answer]):
                        logger.warning(f"Skipping invalid row with missing data: {row}")
                        continue

                    # Infer attributes
                    bloom_level = self.infer_bloom_level(question, answer)
                    question_type = self.infer_question_type(question, answer)
                    score = self.compute_score(question, answer, objective)

                    # Create assessment item
                    assessment = AssessmentItem(
                        question_type=question_type,
                        question_text=question,
                        student_answer=answer,
                        score=score,
                        bloom_level=bloom_level,
                        objective=objective,
                        curriculum_standard=objective
                    )
                    self.assessments.append({
                        'student_id': student_id,
                        'teacher_id': teacher_id,
                        'assessment': assessment
                    })

                    # Update or create student profile
                    student_profile = self.profile_storage.get_student_profile(student_id)
                    if not student_profile:
                        student_profile = StudentProfile(student_id=student_id, name=f"Student_{student_id}")
                        student_profile.assessments_taken = []
                    student_profile.assessments_taken.append(assessment)
                    student_profile.total_assessments = len(student_profile.assessments_taken)
                    student_profile.average_score = mean(a.score for a in student_profile.assessments_taken if a.score is not None)
                    self.profile_storage.save_student_profile(student_profile)
                    logger.debug(f"Updated student profile for {student_id} with score {score:.2f}")

                    # Update or create teacher profile
                    teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
                    if not teacher_profile:
                        teacher_profile = TeacherProfile(teacher_id=teacher_id, name=f"Teacher_{teacher_id}")
                        teacher_profile.students = []
                    if student_id not in teacher_profile.students:
                        teacher_profile.students.append(student_id)
                    self.profile_storage.save_teacher_profile(teacher_profile)
                    logger.debug(f"Updated teacher profile for {teacher_id} with student {student_id}")

                logger.info(f"Successfully read {len(self.assessments)} assessments from {self.csv_path}")
                if not self.assessments:
                    logger.warning("No assessments were loaded from the CSV file.")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise

    def generate_feedback(self) -> FeedbackSummary:
        """Generate feedback summarizing results, including learning gaps, low-performing students, and misunderstood concepts."""
        summary = FeedbackSummary()
        student_scores = defaultdict(list)
        teacher_scores = defaultdict(list)
        objective_scores = defaultdict(list)
        student_objective_scores = defaultdict(lambda: defaultdict(list))
        student_bloom_scores = defaultdict(lambda: defaultdict(list))
        question_scores = defaultdict(list)  # Track scores by question for concept analysis

        # Aggregate data from current assessments
        for entry in self.assessments:
            student_id = entry['student_id']
            teacher_id = entry['teacher_id']
            assessment = entry['assessment']
            score = assessment.score
            objective = assessment.objective
            bloom_level = assessment.bloom_level
            question = assessment.question_text

            student_scores[student_id].append(score)
            teacher_scores[teacher_id].append(score)
            objective_scores[objective].append(score)
            student_objective_scores[student_id][objective].append(score)
            student_bloom_scores[student_id][bloom_level].append(score)
            question_scores[question].append(score)

        # Summarize per student
        for student_id, scores in student_scores.items():
            profile = self.profile_storage.get_student_profile(student_id)
            if profile:
                obj_breakdown = {obj: mean(scores) if scores else 0.0 for obj, scores in student_objective_scores[student_id].items()}
                bloom_breakdown = {bloom: mean(scores) if scores else 0.0 for bloom, scores in student_bloom_scores[student_id].items()}
                trend = "Stable"
                if len(scores) > 1 and scores[-1] > scores[0]:
                    trend = "Improving"
                elif len(scores) > 1 and scores[-1] < scores[0]:
                    trend = "Declining"
                summary.student_summaries[student_id] = {
                    'name': profile.name,
                    'average_score': mean(scores) if scores else 0.0,
                    'total_assessments': len(scores),
                    'objective_breakdown': obj_breakdown,
                    'bloom_breakdown': bloom_breakdown,
                    'performance_trend': trend,
                    'assessments': [{
                        'question': a.question_text,
                        'answer': a.student_answer,
                        'score': a.score,
                        'bloom_level': a.bloom_level,
                        'question_type': a.question_type,
                        'objective': a.objective
                    } for a in profile.assessments_taken]
                }

        # Identify low-performing students (average score < 0.5)
        low_performing_threshold = 0.5
        for student_id, scores in student_scores.items():
            avg_score = mean(scores) if scores else 0.0
            if avg_score < low_performing_threshold:
                profile = self.profile_storage.get_student_profile(student_id)
                weak_objectives = {obj: mean(scores) for obj, scores in student_objective_scores[student_id].items() if scores and mean(scores) < low_performing_threshold}
                weak_blooms = {bloom: mean(scores) for bloom, scores in student_bloom_scores[student_id].items() if scores and mean(scores) < low_performing_threshold}
                summary.low_performing_students.append({
                    'student_id': student_id,
                    'name': profile.name if profile else f"Student_{student_id}",
                    'average_score': avg_score,
                    'weak_objectives': weak_objectives,
                    'weak_bloom_levels': weak_blooms,
                    'recommendation': f"Focus on {', '.join(weak_objectives.keys())} and {', '.join(weak_blooms.keys())} through targeted exercises."
                })

        # Summarize per teacher
        for teacher_id, scores in teacher_scores.items():
            teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
            if teacher_profile:
                all_assessments = []
                student_scores_list = []
                for sid in teacher_profile.students:
                    s_profile = self.profile_storage.get_student_profile(sid)
                    if s_profile:
                        all_assessments.extend(s_profile.assessments_taken)
                        student_scores_list.append(mean(a.score for a in s_profile.assessments_taken if a.score is not None) if s_profile.assessments_taken else 0.0)
                summary.teacher_summaries[teacher_id] = {
                    'teacher_name': teacher_profile.name,
                    'average_score': mean(scores) if scores else 0.0,
                    'total_students': len(teacher_profile.students),
                    'score_distribution': {
                        'min': min(student_scores_list) if student_scores_list else 0.0,
                        'max': max(student_scores_list) if student_scores_list else 0.0,
                        'median': median(student_scores_list) if student_scores_list else 0.0,
                        'average': mean(student_scores_list) if student_scores_list else 0.0
                    },
                    'top_students': sorted(teacher_profile.students, key=lambda sid: mean(a.score for a in self.profile_storage.get_student_profile(sid).assessments_taken if a.score is not None) if self.profile_storage.get_student_profile(sid).assessments_taken else 0.0, reverse=True)[:2],
                    'low_students': sorted(teacher_profile.students, key=lambda sid: mean(a.score for a in self.profile_storage.get_student_profile(sid).assessments_taken if a.score is not None) if self.profile_storage.get_student_profile(sid).assessments_taken else 0.0)[:2],
                    'summary': f"Teacher performance summary for {len(teacher_profile.students)} students with an average score of {mean(scores):.2f}."
                }

        # Summarize per objective and detect learning gaps
        total_students = len(set(entry['student_id'] for entry in self.assessments))
        for objective, scores in objective_scores.items():
            avg_score = mean(scores) if scores else 0.0
            passing_students = sum(1 for sid in student_scores.keys() if student_objective_scores[sid][objective] and mean(student_objective_scores[sid][objective]) >= 0.5)
            mastery_percentage = (passing_students / total_students * 100) if total_students else 0.0
            summary.objective_summaries[objective] = {
                'average_score': avg_score,
                'total_questions': len(scores),
                'mastery_percentage': mastery_percentage,
                'summary': f"Performance on {objective} with {len(scores)} questions, {mastery_percentage:.1f}% mastery."
            }
            # Detect learning gaps (objective with average score < 0.5 and low mastery)
            if avg_score < low_performing_threshold and mastery_percentage < 50.0:
                summary.learning_gaps[objective] = {
                    'average_score': avg_score,
                    'mastery_percentage': mastery_percentage,
                    'recommendation': f"Review {objective} with additional resources or practical exercises."
                }

        # Detect misunderstood concepts by analyzing low-scoring questions
        concept_keywords = {
            'ECU diagnostics': ['ecu', 'diagnostics', 'fault', 'monitor', 'reliability', 'safety'],
            'AUTOSAR protocols': ['autosar', 'protocols', 'data', 'exchange', 'consistency'],
            'AI fault prediction': ['ai', 'fault', 'prediction', 'analysis', 'predictive']
        }
        for question, scores in question_scores.items():
            if mean(scores) < low_performing_threshold:
                # Identify the concept based on keywords in the question
                question_words = set(question.lower().split())
                for concept, keywords in concept_keywords.items():
                    if any(keyword in question_words for keyword in keywords):
                        summary.misunderstood_concepts.append({
                            'concept': concept,
                            'question': question,
                            'average_score': mean(scores),
                            'recommendation': f"Clarify {concept} through targeted lessons or examples."
                        })
                        break

        logger.info("Generated feedback summary with learning gaps and misunderstood concepts")
        return summary

    def print_feedback(self, feedback: FeedbackSummary):
        """Print the feedback summary in a formatted way."""
        print("\n" + "="*60)
        print("FEEDBACK REPORT")
        print("="*60)

        print("\nStudent Summaries:")
        print("-"*40)
        for student_id, summary_data in feedback.student_summaries.items():
            print(f"Student ID: {student_id}")
            print(f"Name: {summary_data['name']}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Assessments: {summary_data['total_assessments']}")
            print(f"Performance Trend: {summary_data['performance_trend']}")
            print(f"Objective Breakdown: {', '.join(f'{obj}: {score:.2f}' for obj, score in summary_data['objective_breakdown'].items())}")
            print(f"Bloom Breakdown: {', '.join(f'{bloom}: {score:.2f}' for bloom, score in summary_data['bloom_breakdown'].items())}")
            print("Assessments:")
            for assessment in summary_data['assessments']:
                print(f"  Question: {assessment['question']}")
                print(f"  Answer: {assessment['answer']}")
                print(f"  Score: {assessment['score']:.2f}")
                print(f"  Bloom Level: {assessment['bloom_level']}")
                print(f"  Question Type: {assessment['question_type']}")
                print(f"  Objective: {assessment['objective']}")
            print("-"*40)

        print("\nTeacher Summaries (Teacher Performance):")
        print("-"*40)
        for teacher_id, summary_data in feedback.teacher_summaries.items():
            print(f"Teacher ID: {teacher_id}")
            print(f"Teacher Name: {summary_data['teacher_name']}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Students: {summary_data['total_students']}")
            print(f"Score Distribution: Min={summary_data['score_distribution']['min']:.2f}, Max={summary_data['score_distribution']['max']:.2f}, Median={summary_data['score_distribution']['median']:.2f}, Avg={summary_data['score_distribution']['average']:.2f}")
            print(f"Top Students: {', '.join(summary_data['top_students']) or 'None'}")
            print(f"Low Students: {', '.join(summary_data['low_students']) or 'None'}")
            print(f"Summary: {summary_data['summary']}")
            print("-"*40)

        print("\nObjective Summaries:")
        print("-"*40)
        for objective, summary_data in feedback.objective_summaries.items():
            print(f"Objective: {objective}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Questions: {summary_data['total_questions']}")
            print(f"Mastery Percentage: {summary_data['mastery_percentage']:.1f}%")
            print(f"Summary: {summary_data['summary']}")
            print("-"*40)

        print("\nLearning Gaps:")
        print("-"*40)
        if feedback.learning_gaps:
            for objective, gap_data in feedback.learning_gaps.items():
                print(f"Objective: {objective}")
                print(f"Average Score: {gap_data['average_score']:.2f}")
                print(f"Mastery Percentage: {gap_data['mastery_percentage']:.1f}%")
                print(f"Recommendation: {gap_data['recommendation']}")
                print("-"*40)
        else:
            print("No significant learning gaps detected.")
            print("-"*40)

        print("\nLow-Performing Students:")
        print("-"*40)
        if feedback.low_performing_students:
            for student in feedback.low_performing_students:
                print(f"Student ID: {student['student_id']}")
                print(f"Name: {student['name']}")
                print(f"Average Score: {student['average_score']:.2f}")
                print(f"Weak Objectives: {', '.join(f'{obj}: {score:.2f}' for obj, score in student['weak_objectives'].items()) or 'None'}")
                print(f"Weak Bloom Levels: {', '.join(f'{bloom}: {score:.2f}' for bloom, score in student['weak_bloom_levels'].items()) or 'None'}")
                print(f"Recommendation: {student['recommendation']}")
                print("-"*40)
        else:
            print("No low-performing students detected.")
            print("-"*40)

        print("\nMisunderstood Concepts:")
        print("-"*40)
        if feedback.misunderstood_concepts:
            for concept_data in feedback.misunderstood_concepts:
                print(f"Concept: {concept_data['concept']}")
                print(f"Question: {concept_data['question']}")
                print(f"Average Score: {concept_data['average_score']:.2f}")
                print(f"Recommendation: {concept_data['recommendation']}")
                print("-"*40)
        else:
            print("No misunderstood concepts detected.")
            print("-"*40)

def main():
    print("Feedback Agent Demo")
    print("="*45)

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Feedback Agent Demo")
    parser.add_argument("--csv-path", type=str, default='assessment_data.csv', help="Path to CSV file")
    args = parser.parse_args()

    config = get_config()
    profile_storage = ProfileStorage(db_path=config.PROFILE_DB_PATH)
    llm = ConfigurableLLM(
        config=config,
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        max_tokens=config.MAX_TOKENS
    )
    feedback_agent = FeedbackAgent(config=config, profile_storage=profile_storage, llm=llm, csv_path=args.csv_path)

    try:
        # Read CSV and update profiles
        feedback_agent.read_csv()

        # Generate and print feedback
        feedback = feedback_agent.generate_feedback()
        feedback_agent.print_feedback(feedback)

    except Exception as e:
        logger.error(f"Error running FeedbackAgent: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()