"""
Interactive Training and Assessment System
Provides video tutorials, interactive learning materials, and competency tracking
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

class TrainingLevel(Enum):
    MAINTENANCE_TECHNICIAN = "maintenance_technician"
    SYSTEM_ADMINISTRATOR = "system_administrator"
    ADVANCED_USER = "advanced_user"

class AssessmentType(Enum):
    KNOWLEDGE = "knowledge"
    PRACTICAL = "practical"
    SCENARIO = "scenario"

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SCENARIO_BASED = "scenario_based"
    PRACTICAL_TASK = "practical_task"

@dataclass
class TrainingModule:
    id: str
    title: str
    level: TrainingLevel
    duration_minutes: int
    prerequisites: List[str]
    learning_objectives: List[str]
    content_sections: List[str]
    video_url: Optional[str] = None
    interactive_elements: List[str] = field(default_factory=list)
    assessment_id: Optional[str] = None

@dataclass
class AssessmentQuestion:
    id: str
    question_type: QuestionType
    question_text: str
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    topic: str = ""
    points: int = 1

@dataclass
class Assessment:
    id: str
    title: str
    assessment_type: AssessmentType
    level: TrainingLevel
    questions: List[AssessmentQuestion]
    passing_score: float = 0.8
    time_limit_minutes: int = 60
    max_attempts: int = 3

@dataclass
class UserProgress:
    user_id: str
    module_id: str
    completion_percentage: float
    last_accessed: datetime
    completed: bool = False
    assessment_scores: List[float] = field(default_factory=list)
    certification_earned: Optional[str] = None

class TrainingSystem:
    """Interactive training and assessment system"""
    
    def __init__(self, db_path: str = "training.db"):
        self.db_path = db_path
        self.modules: Dict[str, TrainingModule] = {}
        self.assessments: Dict[str, Assessment] = {}
        self._init_database()
        self._load_training_content()
    
    def _init_database(self):
        """Initialize training database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                completion_percentage REAL DEFAULT 0,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed BOOLEAN DEFAULT FALSE,
                UNIQUE(user_id, module_id)
            )
        ''')
        
        # Assessment results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                assessment_id TEXT NOT NULL,
                score REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                attempt_number INTEGER NOT NULL,
                completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                time_taken_minutes INTEGER,
                answers TEXT  -- JSON string of user answers
            )
        ''')
        
        # Certifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS certifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                certification_name TEXT NOT NULL,
                level TEXT NOT NULL,
                earned_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                expiry_date DATETIME,
                certificate_id TEXT UNIQUE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_training_content(self):
        """Load training modules and assessments"""
        self._create_maintenance_technician_modules()
        self._create_system_administrator_modules()
        self._create_assessments()
    
    def _create_maintenance_technician_modules(self):
        """Create maintenance technician training modules"""
        
        # Module 1: Introduction to RUL Prediction
        self.modules["mt_intro"] = TrainingModule(
            id="mt_intro",
            title="Introduction to RUL Prediction",
            level=TrainingLevel.MAINTENANCE_TECHNICIAN,
            duration_minutes=30,
            prerequisites=[],
            learning_objectives=[
                "Understand RUL prediction concepts and benefits",
                "Identify equipment suitable for RUL monitoring",
                "Explain the difference between traditional and predictive maintenance"
            ],
            content_sections=[
                "What is RUL Prediction?",
                "Benefits for Maintenance Teams",
                "Traditional vs. Predictive Maintenance",
                "Equipment Types and Applications"
            ],
            video_url="/training/videos/mt_intro.mp4",
            interactive_elements=["equipment_identification_quiz", "benefit_calculator"],
            assessment_id="mt_intro_assessment"
        )
        
        # Module 2: Interpreting Predictions
        self.modules["mt_interpretation"] = TrainingModule(
            id="mt_interpretation",
            title="Understanding Prediction Results",
            level=TrainingLevel.MAINTENANCE_TECHNICIAN,
            duration_minutes=45,
            prerequisites=["mt_intro"],
            learning_objectives=[
                "Interpret RUL prediction results accurately",
                "Understand confidence intervals and degradation stages",
                "Make appropriate maintenance decisions based on predictions"
            ],
            content_sections=[
                "Reading Prediction Reports",
                "Degradation Stages Explained",
                "Confidence Intervals and Uncertainty",
                "Decision Making Framework"
            ],
            video_url="/training/videos/mt_interpretation.mp4",
            interactive_elements=["prediction_simulator", "decision_tree"],
            assessment_id="mt_interpretation_assessment"
        )
    
    def _create_system_administrator_modules(self):
        """Create system administrator training modules"""
        
        # Module 1: System Architecture
        self.modules["sa_architecture"] = TrainingModule(
            id="sa_architecture",
            title="RUL System Architecture",
            level=TrainingLevel.SYSTEM_ADMINISTRATOR,
            duration_minutes=60,
            prerequisites=[],
            learning_objectives=[
                "Understand system components and data flow",
                "Identify system requirements and dependencies",
                "Explain security and performance considerations"
            ],
            content_sections=[
                "System Components Overview",
                "Data Flow and Processing Pipeline",
                "Hardware and Software Requirements",
                "Security Architecture"
            ],
            video_url="/training/videos/sa_architecture.mp4",
            interactive_elements=["architecture_diagram", "requirements_calculator"],
            assessment_id="sa_architecture_assessment"
        )
    
    def _create_assessments(self):
        """Create training assessments"""
        
        # Maintenance Technician Introduction Assessment
        mt_intro_questions = [
            AssessmentQuestion(
                id="mt_intro_q1",
                question_type=QuestionType.MULTIPLE_CHOICE,
                question_text="What does RUL stand for in predictive maintenance?",
                options=[
                    "Remaining Useful Life",
                    "Rapid Update Logic",
                    "Real-time Usage Logging",
                    "Routine Utility Lifecycle"
                ],
                correct_answer="Remaining Useful Life",
                explanation="RUL stands for Remaining Useful Life, which predicts how many operational cycles equipment has before failure.",
                topic="Basic Concepts",
                points=1
            ),
            AssessmentQuestion(
                id="mt_intro_q2",
                question_type=QuestionType.TRUE_FALSE,
                question_text="RUL prediction systems have higher false positive rates than traditional monitoring.",
                options=["True", "False"],
                correct_answer="False",
                explanation="RUL prediction systems achieve <5% false positive rates, much lower than traditional systems (10-15%).",
                topic="System Performance",
                points=1
            )
        ]
        
        self.assessments["mt_intro_assessment"] = Assessment(
            id="mt_intro_assessment",
            title="Introduction to RUL Prediction - Assessment",
            assessment_type=AssessmentType.KNOWLEDGE,
            level=TrainingLevel.MAINTENANCE_TECHNICIAN,
            questions=mt_intro_questions,
            passing_score=0.8,
            time_limit_minutes=15,
            max_attempts=3
        )
    
    def get_user_progress(self, user_id: str) -> List[UserProgress]:
        """Get training progress for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT module_id, completion_percentage, last_accessed, completed
            FROM user_progress WHERE user_id = ?
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        progress_list = []
        for row in results:
            progress_list.append(UserProgress(
                user_id=user_id,
                module_id=row[0],
                completion_percentage=row[1],
                last_accessed=datetime.fromisoformat(row[2]),
                completed=bool(row[3])
            ))
        
        return progress_list
    
    def update_progress(self, user_id: str, module_id: str, completion_percentage: float):
        """Update user progress for a module"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_progress 
            (user_id, module_id, completion_percentage, last_accessed, completed)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, module_id, completion_percentage, datetime.now(), completion_percentage >= 100))
        
        conn.commit()
        conn.close()
    
    def get_assessment(self, assessment_id: str) -> Optional[Assessment]:
        """Get assessment by ID"""
        return self.assessments.get(assessment_id)
    
    def submit_assessment(self, user_id: str, assessment_id: str, answers: Dict[str, str], 
                         time_taken_minutes: int) -> Tuple[float, bool]:
        """Submit assessment and calculate score"""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        # Calculate score
        total_points = sum(q.points for q in assessment.questions)
        earned_points = 0
        
        for question in assessment.questions:
            user_answer = answers.get(question.id, "")
            if user_answer == question.correct_answer:
                earned_points += question.points
        
        score = earned_points / total_points if total_points > 0 else 0
        passed = score >= assessment.passing_score
        
        # Get attempt number
        attempt_number = self._get_next_attempt_number(user_id, assessment_id)
        
        # Store result
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assessment_results 
            (user_id, assessment_id, score, passed, attempt_number, time_taken_minutes, answers)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, assessment_id, score, passed, attempt_number, time_taken_minutes, json.dumps(answers)))
        
        conn.commit()
        conn.close()
        
        return score, passed
    
    def _get_next_attempt_number(self, user_id: str, assessment_id: str) -> int:
        """Get the next attempt number for an assessment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(attempt_number) FROM assessment_results 
            WHERE user_id = ? AND assessment_id = ?
        ''', (user_id, assessment_id))
        
        result = cursor.fetchone()
        conn.close()
        
        return (result[0] or 0) + 1
    
    def award_certification(self, user_id: str, certification_name: str, level: str, 
                           validity_months: int = 24) -> str:
        """Award certification to user"""
        certificate_id = self._generate_certificate_id(user_id, certification_name)
        expiry_date = datetime.now() + timedelta(days=validity_months * 30)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO certifications 
            (user_id, certification_name, level, expiry_date, certificate_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, certification_name, level, expiry_date, certificate_id))
        
        conn.commit()
        conn.close()
        
        return certificate_id
    
    def _generate_certificate_id(self, user_id: str, certification_name: str) -> str:
        """Generate unique certificate ID"""
        data = f"{user_id}_{certification_name}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12].upper()
    
    def get_user_certifications(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's certifications"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT certification_name, level, earned_date, expiry_date, certificate_id
            FROM certifications WHERE user_id = ?
            ORDER BY earned_date DESC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        certifications = []
        for row in results:
            certifications.append({
                'certification_name': row[0],
                'level': row[1],
                'earned_date': row[2],
                'expiry_date': row[3],
                'certificate_id': row[4],
                'is_expired': datetime.fromisoformat(row[3]) < datetime.now()
            })
        
        return certifications
    
    def get_learning_path(self, user_id: str, target_level: TrainingLevel) -> List[str]:
        """Get recommended learning path for user"""
        user_progress = self.get_user_progress(user_id)
        completed_modules = {p.module_id for p in user_progress if p.completed}
        
        # Get modules for target level
        target_modules = [m for m in self.modules.values() if m.level == target_level]
        
        # Sort by prerequisites and dependencies
        learning_path = []
        remaining_modules = {m.id: m for m in target_modules}
        
        while remaining_modules:
            # Find modules with satisfied prerequisites
            ready_modules = []
            for module_id, module in remaining_modules.items():
                if all(prereq in completed_modules or prereq in learning_path 
                      for prereq in module.prerequisites):
                    ready_modules.append(module_id)
            
            if not ready_modules:
                # Add remaining modules (circular dependencies or missing prereqs)
                ready_modules = list(remaining_modules.keys())
            
            # Add first ready module to path
            next_module = ready_modules[0]
            learning_path.append(next_module)
            del remaining_modules[next_module]
        
        return learning_path
    
    def generate_progress_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive progress report for user"""
        progress = self.get_user_progress(user_id)
        certifications = self.get_user_certifications(user_id)
        
        # Calculate overall statistics
        total_modules = len(self.modules)
        completed_modules = sum(1 for p in progress if p.completed)
        avg_completion = sum(p.completion_percentage for p in progress) / len(progress) if progress else 0
        
        # Get assessment results
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT assessment_id, score, passed, completed_at
            FROM assessment_results WHERE user_id = ?
            ORDER BY completed_at DESC
        ''', (user_id,))
        
        assessment_results = cursor.fetchall()
        conn.close()
        
        return {
            'user_id': user_id,
            'overall_progress': {
                'total_modules': total_modules,
                'completed_modules': completed_modules,
                'completion_percentage': avg_completion,
                'modules_in_progress': len([p for p in progress if 0 < p.completion_percentage < 100])
            },
            'module_progress': [
                {
                    'module_id': p.module_id,
                    'module_title': self.modules[p.module_id].title if p.module_id in self.modules else 'Unknown',
                    'completion_percentage': p.completion_percentage,
                    'completed': p.completed,
                    'last_accessed': p.last_accessed.isoformat()
                }
                for p in progress
            ],
            'assessment_results': [
                {
                    'assessment_id': result[0],
                    'score': result[1],
                    'passed': bool(result[2]),
                    'completed_at': result[3]
                }
                for result in assessment_results
            ],
            'certifications': certifications,
            'recommendations': self._get_recommendations(user_id, progress, certifications)
        }
    
    def _get_recommendations(self, user_id: str, progress: List[UserProgress], 
                           certifications: List[Dict[str, Any]]) -> List[str]:
        """Get personalized recommendations for user"""
        recommendations = []
        
        # Check for incomplete modules
        incomplete_modules = [p for p in progress if not p.completed]
        if incomplete_modules:
            recommendations.append(f"Complete {len(incomplete_modules)} remaining modules")
        
        # Check for expired certifications
        expired_certs = [c for c in certifications if c['is_expired']]
        if expired_certs:
            recommendations.append(f"Renew {len(expired_certs)} expired certifications")
        
        # Suggest next level if current level is complete
        mt_modules = [m for m in self.modules.values() if m.level == TrainingLevel.MAINTENANCE_TECHNICIAN]
        mt_completed = sum(1 for p in progress if p.completed and p.module_id in [m.id for m in mt_modules])
        
        if mt_completed == len(mt_modules):
            recommendations.append("Consider advancing to System Administrator certification")
        
        return recommendations

# Global training system instance
training_system = TrainingSystem()

def get_training_system() -> TrainingSystem:
    """Get the global training system instance"""
    return training_system