"""
Knowledge Management System
Provides searchable knowledge base, case studies, collaborative problem-solving,
and best practices repository for the RUL Prediction System
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from datetime import datetime
import hashlib
import re
from pathlib import Path

class ContentType(Enum):
    KNOWLEDGE_ARTICLE = "knowledge_article"
    CASE_STUDY = "case_study"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    LESSON_LEARNED = "lesson_learned"
    FAQ_ITEM = "faq_item"

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Status(Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    PUBLISHED = "published"
    ARCHIVED = "archived"

@dataclass
class KnowledgeItem:
    id: str
    title: str
    content_type: ContentType
    content: str
    tags: List[str]
    author: str
    created_date: datetime
    last_modified: datetime
    status: Status
    priority: Priority = Priority.MEDIUM
    views: int = 0
    rating: float = 0.0
    rating_count: int = 0
    related_items: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

@dataclass
class CaseStudy:
    id: str
    title: str
    equipment_type: str
    problem_description: str
    symptoms: List[str]
    root_cause: str
    solution: str
    outcome: str
    lessons_learned: List[str]
    author: str
    date_created: datetime
    anonymized: bool = True
    tags: List[str] = field(default_factory=list)

@dataclass
class BestPractice:
    id: str
    title: str
    category: str
    description: str
    implementation_steps: List[str]
    benefits: List[str]
    prerequisites: List[str]
    success_metrics: List[str]
    author: str
    date_created: datetime
    validated: bool = False
    validation_date: Optional[datetime] = None

@dataclass
class ProblemTicket:
    id: str
    title: str
    description: str
    category: str
    priority: Priority
    status: str
    reporter: str
    assignee: Optional[str]
    created_date: datetime
    last_updated: datetime
    resolution: Optional[str] = None
    resolution_date: Optional[datetime] = None
    related_knowledge: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)

class KnowledgeManagementSystem:
    """Comprehensive knowledge management system"""
    
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self._init_database()
        self._load_initial_content()
    
    def _init_database(self):
        """Initialize knowledge management database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Knowledge items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,  -- JSON array
                author TEXT NOT NULL,
                created_date DATETIME NOT NULL,
                last_modified DATETIME NOT NULL,
                status TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                views INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                related_items TEXT,  -- JSON array
                attachments TEXT     -- JSON array
            )
        ''')
        
        # Case studies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS case_studies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                equipment_type TEXT NOT NULL,
                problem_description TEXT NOT NULL,
                symptoms TEXT,  -- JSON array
                root_cause TEXT NOT NULL,
                solution TEXT NOT NULL,
                outcome TEXT NOT NULL,
                lessons_learned TEXT,  -- JSON array
                author TEXT NOT NULL,
                date_created DATETIME NOT NULL,
                anonymized BOOLEAN DEFAULT TRUE,
                tags TEXT  -- JSON array
            )
        ''')
        
        # Best practices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_practices (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                implementation_steps TEXT,  -- JSON array
                benefits TEXT,  -- JSON array
                prerequisites TEXT,  -- JSON array
                success_metrics TEXT,  -- JSON array
                author TEXT NOT NULL,
                date_created DATETIME NOT NULL,
                validated BOOLEAN DEFAULT FALSE,
                validation_date DATETIME
            )
        ''')
        
        # Problem tickets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problem_tickets (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                reporter TEXT NOT NULL,
                assignee TEXT,
                created_date DATETIME NOT NULL,
                last_updated DATETIME NOT NULL,
                resolution TEXT,
                resolution_date DATETIME,
                related_knowledge TEXT,  -- JSON array
                comments TEXT  -- JSON array
            )
        ''')
        
        # Search index table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                item_id,
                title,
                content,
                tags,
                content_type
            )
        ''')
        
        # User ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                comment TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, item_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_initial_content(self):
        """Load initial knowledge base content"""
        self._create_initial_knowledge_items()
        self._create_initial_case_studies()
        self._create_initial_best_practices()
    
    def _create_initial_knowledge_items(self):
        """Create initial knowledge articles"""
        
        # Common troubleshooting article
        troubleshooting_article = KnowledgeItem(
            id="kb_001",
            title="Common RUL System Issues and Solutions",
            content_type=ContentType.TROUBLESHOOTING_GUIDE,
            content="""
# Common RUL System Issues and Solutions

## Issue 1: API Returns 503 Service Unavailable

**Symptoms:**
- All prediction requests return HTTP 503
- Health check shows model_ready: false
- System appears to be running but not responding

**Root Cause:**
- Models failed to load during startup
- Insufficient memory allocation
- Corrupted model files

**Solution:**
1. Check system memory usage: `docker stats rul-api`
2. Restart the service: `docker-compose restart rul-api`
3. Monitor startup logs: `docker-compose logs -f rul-api`
4. If models are corrupted, restore from backup

**Prevention:**
- Allocate sufficient memory (minimum 8GB)
- Implement model file integrity checks
- Regular backup procedures

## Issue 2: High False Positive Rate

**Symptoms:**
- Excessive anomaly alerts for healthy equipment
- User complaints about alert fatigue
- FPR metrics above 5%

**Root Cause:**
- Anomaly detection thresholds too sensitive
- Model trained on limited data
- Data distribution changes

**Solution:**
1. Adjust anomaly thresholds in configuration
2. Retrain models with recent data
3. Implement adaptive thresholding
4. Review data quality and preprocessing

**Prevention:**
- Regular model performance monitoring
- Continuous data quality checks
- Scheduled model retraining
            """,
            tags=["troubleshooting", "api", "false-positives", "common-issues"],
            author="System Administrator",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            status=Status.PUBLISHED,
            priority=Priority.HIGH
        )
        
        self.add_knowledge_item(troubleshooting_article)
        
        # FAQ article
        faq_article = KnowledgeItem(
            id="kb_002",
            title="Frequently Asked Questions - RUL Predictions",
            content_type=ContentType.FAQ_ITEM,
            content="""
# Frequently Asked Questions - RUL Predictions

## Q: How accurate are the RUL predictions?

**A:** The system achieves:
- RMSE: 5.2 cycles (typical)
- MAE: 3.8 cycles (typical)
- R² Score: 0.92 (excellent correlation)
- False Positive Rate: <3%

Accuracy depends on data quality and equipment condition.

## Q: What should I do when I get an anomaly alert?

**A:** Follow this procedure:
1. Verify the alert details and timestamp
2. Check equipment visually if safe to do so
3. Review recent maintenance history
4. If anomaly score >0.8, consider immediate inspection
5. Document findings and actions taken

## Q: How often should models be retrained?

**A:** Retrain models when:
- Accuracy drops below 80% (R² < 0.8)
- FPR exceeds 5%
- New equipment types are introduced
- Monthly scheduled retraining (recommended)

## Q: Can I customize the alert thresholds?

**A:** Yes, thresholds can be adjusted in the configuration:
- Anomaly detection threshold (default: 0.5)
- RUL warning levels (default: 25 cycles)
- Critical alerts (default: 10 cycles)

Contact your system administrator for threshold adjustments.
            """,
            tags=["faq", "accuracy", "alerts", "configuration"],
            author="Technical Writer",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            status=Status.PUBLISHED,
            priority=Priority.MEDIUM
        )
        
        self.add_knowledge_item(faq_article)
    
    def _create_initial_case_studies(self):
        """Create initial case studies"""
        
        case_study_1 = CaseStudy(
            id="cs_001",
            title="Power Supply Capacitor Failure Prevention",
            equipment_type="Power Supply Unit",
            problem_description="Critical power supply unit showing early signs of capacitor degradation in manufacturing facility",
            symptoms=[
                "Voltage fluctuations during peak load",
                "Increased operating temperature",
                "Slight humming noise from unit"
            ],
            root_cause="Electrolytic capacitor degradation due to high ambient temperature and continuous operation",
            solution="Implemented RUL monitoring with 2-week prediction horizon. Scheduled maintenance during planned downtime when RUL reached 15 cycles.",
            outcome="Prevented unplanned downtime, saved $50,000 in lost production. Maintenance completed successfully with no equipment failure.",
            lessons_learned=[
                "Early RUL monitoring enables proactive maintenance",
                "Conservative RUL thresholds prevent emergency situations",
                "Environmental factors significantly impact degradation rates",
                "Planned maintenance is 70% less expensive than emergency repairs"
            ],
            author="Maintenance Manager",
            date_created=datetime.now(),
            tags=["power-supply", "capacitor", "preventive-maintenance", "cost-savings"]
        )
        
        self.add_case_study(case_study_1)
    
    def _create_initial_best_practices(self):
        """Create initial best practices"""
        
        best_practice_1 = BestPractice(
            id="bp_001",
            title="RUL-Based Maintenance Scheduling",
            category="Maintenance Planning",
            description="Optimize maintenance schedules using RUL predictions to minimize downtime and costs",
            implementation_steps=[
                "Set up RUL monitoring for critical equipment",
                "Define maintenance trigger points (e.g., RUL < 25 cycles)",
                "Integrate RUL data with CMMS system",
                "Train maintenance staff on RUL interpretation",
                "Establish parts inventory based on RUL forecasts",
                "Monitor and adjust thresholds based on experience"
            ],
            benefits=[
                "Reduce unplanned downtime by 60-80%",
                "Lower maintenance costs by 20-30%",
                "Improve equipment reliability",
                "Better resource allocation and planning",
                "Reduced inventory carrying costs"
            ],
            prerequisites=[
                "RUL prediction system installed and operational",
                "CMMS system with API integration capability",
                "Trained maintenance personnel",
                "Management support for predictive maintenance"
            ],
            success_metrics=[
                "Unplanned downtime reduction",
                "Maintenance cost per equipment unit",
                "Mean time between failures (MTBF)",
                "Maintenance schedule adherence",
                "Parts inventory turnover"
            ],
            author="Maintenance Excellence Team",
            date_created=datetime.now(),
            validated=True,
            validation_date=datetime.now()
        )
        
        self.add_best_practice(best_practice_1)
    
    def add_knowledge_item(self, item: KnowledgeItem) -> str:
        """Add knowledge item to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_items 
            (id, title, content_type, content, tags, author, created_date, 
             last_modified, status, priority, views, rating, rating_count, 
             related_items, attachments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id, item.title, item.content_type.value, item.content,
            json.dumps(item.tags), item.author, item.created_date,
            item.last_modified, item.status.value, item.priority.value,
            item.views, item.rating, item.rating_count,
            json.dumps(item.related_items), json.dumps(item.attachments)
        ))
        
        # Update search index
        cursor.execute('''
            INSERT OR REPLACE INTO search_index 
            (item_id, title, content, tags, content_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            item.id, item.title, item.content,
            ' '.join(item.tags), item.content_type.value
        ))
        
        conn.commit()
        conn.close()
        
        return item.id
    
    def add_case_study(self, case_study: CaseStudy) -> str:
        """Add case study to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO case_studies 
            (id, title, equipment_type, problem_description, symptoms, 
             root_cause, solution, outcome, lessons_learned, author, 
             date_created, anonymized, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case_study.id, case_study.title, case_study.equipment_type,
            case_study.problem_description, json.dumps(case_study.symptoms),
            case_study.root_cause, case_study.solution, case_study.outcome,
            json.dumps(case_study.lessons_learned), case_study.author,
            case_study.date_created, case_study.anonymized,
            json.dumps(case_study.tags)
        ))
        
        conn.commit()
        conn.close()
        
        return case_study.id
    
    def add_best_practice(self, best_practice: BestPractice) -> str:
        """Add best practice to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO best_practices 
            (id, title, category, description, implementation_steps, 
             benefits, prerequisites, success_metrics, author, 
             date_created, validated, validation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            best_practice.id, best_practice.title, best_practice.category,
            best_practice.description, json.dumps(best_practice.implementation_steps),
            json.dumps(best_practice.benefits), json.dumps(best_practice.prerequisites),
            json.dumps(best_practice.success_metrics), best_practice.author,
            best_practice.date_created, best_practice.validated,
            best_practice.validation_date
        ))
        
        conn.commit()
        conn.close()
        
        return best_practice.id
    
    def search_knowledge_base(self, query: str, content_type: Optional[ContentType] = None,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """Search knowledge base using full-text search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build search query
        search_query = query
        if content_type:
            search_query += f" AND content_type:{content_type.value}"
        
        cursor.execute('''
            SELECT item_id, title, content, tags, content_type,
                   rank
            FROM search_index 
            WHERE search_index MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (search_query, limit))
        
        results = cursor.fetchall()
        
        # Get additional details for each result
        search_results = []
        for row in results:
            item_id = row[0]
            
            # Get full item details
            cursor.execute('''
                SELECT title, content_type, author, created_date, 
                       views, rating, status, priority
                FROM knowledge_items 
                WHERE id = ?
            ''', (item_id,))
            
            item_details = cursor.fetchone()
            if item_details:
                search_results.append({
                    'id': item_id,
                    'title': item_details[0],
                    'content_type': item_details[1],
                    'author': item_details[2],
                    'created_date': item_details[3],
                    'views': item_details[4],
                    'rating': item_details[5],
                    'status': item_details[6],
                    'priority': item_details[7],
                    'snippet': self._extract_snippet(row[2], query),
                    'tags': json.loads(row[3]) if row[3] else []
                })
        
        conn.close()
        return search_results
    
    def _extract_snippet(self, content: str, query: str, snippet_length: int = 200) -> str:
        """Extract relevant snippet from content based on query"""
        # Find the first occurrence of any query term
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        best_position = 0
        for term in query_terms:
            position = content_lower.find(term)
            if position != -1:
                best_position = max(0, position - snippet_length // 2)
                break
        
        # Extract snippet
        snippet = content[best_position:best_position + snippet_length]
        if best_position > 0:
            snippet = "..." + snippet
        if best_position + snippet_length < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM knowledge_items WHERE id = ?
        ''', (item_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return KnowledgeItem(
            id=row[0],
            title=row[1],
            content_type=ContentType(row[2]),
            content=row[3],
            tags=json.loads(row[4]) if row[4] else [],
            author=row[5],
            created_date=datetime.fromisoformat(row[6]),
            last_modified=datetime.fromisoformat(row[7]),
            status=Status(row[8]),
            priority=Priority(row[9]),
            views=row[10],
            rating=row[11],
            rating_count=row[12],
            related_items=json.loads(row[13]) if row[13] else [],
            attachments=json.loads(row[14]) if row[14] else []
        )
    
    def increment_views(self, item_id: str):
        """Increment view count for knowledge item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE knowledge_items 
            SET views = views + 1 
            WHERE id = ?
        ''', (item_id,))
        
        conn.commit()
        conn.close()
    
    def rate_item(self, user_id: str, item_id: str, rating: int, comment: str = "") -> bool:
        """Rate a knowledge item"""
        if not (1 <= rating <= 5):
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add or update user rating
            cursor.execute('''
                INSERT OR REPLACE INTO user_ratings 
                (user_id, item_id, rating, comment)
                VALUES (?, ?, ?, ?)
            ''', (user_id, item_id, rating, comment))
            
            # Recalculate average rating
            cursor.execute('''
                SELECT AVG(rating), COUNT(rating) 
                FROM user_ratings 
                WHERE item_id = ?
            ''', (item_id,))
            
            avg_rating, rating_count = cursor.fetchone()
            
            # Update knowledge item
            cursor.execute('''
                UPDATE knowledge_items 
                SET rating = ?, rating_count = ?
                WHERE id = ?
            ''', (avg_rating or 0, rating_count or 0, item_id))
            
            conn.commit()
            return True
            
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def create_problem_ticket(self, title: str, description: str, category: str,
                            priority: Priority, reporter: str) -> str:
        """Create a new problem ticket"""
        ticket_id = self._generate_ticket_id()
        
        ticket = ProblemTicket(
            id=ticket_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            status="Open",
            reporter=reporter,
            assignee=None,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO problem_tickets 
            (id, title, description, category, priority, status, 
             reporter, assignee, created_date, last_updated, 
             resolution, resolution_date, related_knowledge, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticket.id, ticket.title, ticket.description, ticket.category,
            ticket.priority.value, ticket.status, ticket.reporter,
            ticket.assignee, ticket.created_date, ticket.last_updated,
            ticket.resolution, ticket.resolution_date,
            json.dumps(ticket.related_knowledge), json.dumps(ticket.comments)
        ))
        
        conn.commit()
        conn.close()
        
        return ticket_id
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"TICKET-{timestamp}"
    
    def get_popular_content(self, content_type: Optional[ContentType] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular content by views and ratings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, title, content_type, author, views, rating, rating_count
            FROM knowledge_items 
            WHERE status = 'published'
        '''
        params = []
        
        if content_type:
            query += ' AND content_type = ?'
            params.append(content_type.value)
        
        query += ' ORDER BY (views * 0.7 + rating * rating_count * 0.3) DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'title': row[1],
                'content_type': row[2],
                'author': row[3],
                'views': row[4],
                'rating': row[5],
                'rating_count': row[6]
            }
            for row in results
        ]
    
    def get_recent_content(self, days: int = 30, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently added or updated content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, content_type, author, created_date, last_modified
            FROM knowledge_items 
            WHERE status = 'published' 
            AND last_modified > datetime('now', '-{} days')
            ORDER BY last_modified DESC 
            LIMIT ?
        '''.format(days), (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'title': row[1],
                'content_type': row[2],
                'author': row[3],
                'created_date': row[4],
                'last_modified': row[5]
            }
            for row in results
        ]
    
    def generate_knowledge_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge base report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Content statistics
        cursor.execute('''
            SELECT content_type, COUNT(*), AVG(views), AVG(rating)
            FROM knowledge_items 
            WHERE status = 'published'
            GROUP BY content_type
        ''')
        content_stats = cursor.fetchall()
        
        # Most viewed items
        cursor.execute('''
            SELECT title, views, rating 
            FROM knowledge_items 
            WHERE status = 'published'
            ORDER BY views DESC 
            LIMIT 5
        ''')
        most_viewed = cursor.fetchall()
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) 
            FROM knowledge_items 
            WHERE last_modified > datetime('now', '-7 days')
        ''')
        recent_updates = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'content_statistics': [
                {
                    'content_type': stat[0],
                    'count': stat[1],
                    'avg_views': stat[2] or 0,
                    'avg_rating': stat[3] or 0
                }
                for stat in content_stats
            ],
            'most_viewed_items': [
                {
                    'title': item[0],
                    'views': item[1],
                    'rating': item[2]
                }
                for item in most_viewed
            ],
            'recent_activity': {
                'updates_last_7_days': recent_updates
            },
            'generated_at': datetime.now().isoformat()
        }

# Global knowledge management system instance
knowledge_system = KnowledgeManagementSystem()

def get_knowledge_system() -> KnowledgeManagementSystem:
    """Get the global knowledge management system instance"""
    return knowledge_system