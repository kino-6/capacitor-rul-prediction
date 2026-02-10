"""
Collaborative Problem-Solving Platform
Enables team collaboration, discussion forums, expert consultation,
and community-driven problem resolution for RUL system issues
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from datetime import datetime, timedelta
import hashlib

class DiscussionType(Enum):
    QUESTION = "question"
    PROBLEM_REPORT = "problem_report"
    FEATURE_REQUEST = "feature_request"
    BEST_PRACTICE = "best_practice"
    GENERAL_DISCUSSION = "general_discussion"

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Status(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class UserRole(Enum):
    USER = "user"
    EXPERT = "expert"
    MODERATOR = "moderator"
    ADMINISTRATOR = "administrator"

@dataclass
class User:
    id: str
    username: str
    email: str
    role: UserRole
    expertise_areas: List[str]
    reputation_score: int = 0
    join_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    profile_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Discussion:
    id: str
    title: str
    description: str
    discussion_type: DiscussionType
    priority: Priority
    status: Status
    author_id: str
    created_date: datetime
    last_updated: datetime
    tags: List[str] = field(default_factory=list)
    views: int = 0
    upvotes: int = 0
    downvotes: int = 0
    expert_assigned: Optional[str] = None
    resolution_summary: Optional[str] = None
    related_discussions: List[str] = field(default_factory=list)

@dataclass
class Comment:
    id: str
    discussion_id: str
    author_id: str
    content: str
    created_date: datetime
    last_edited: Optional[datetime] = None
    upvotes: int = 0
    downvotes: int = 0
    is_solution: bool = False
    parent_comment_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)

@dataclass
class ExpertConsultation:
    id: str
    title: str
    description: str
    requester_id: str
    expert_id: Optional[str]
    priority: Priority
    status: Status
    created_date: datetime
    scheduled_date: Optional[datetime] = None
    consultation_notes: Optional[str] = None
    follow_up_required: bool = False
    estimated_duration: int = 60  # minutes

class CollaborationPlatform:
    """Collaborative problem-solving platform"""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self._init_database()
        self._create_default_users()
    
    def _init_database(self):
        """Initialize collaboration database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                expertise_areas TEXT,  -- JSON array
                reputation_score INTEGER DEFAULT 0,
                join_date DATETIME NOT NULL,
                last_active DATETIME NOT NULL,
                profile_data TEXT  -- JSON object
            )
        ''')
        
        # Discussions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discussions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                discussion_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                author_id TEXT NOT NULL,
                created_date DATETIME NOT NULL,
                last_updated DATETIME NOT NULL,
                tags TEXT,  -- JSON array
                views INTEGER DEFAULT 0,
                upvotes INTEGER DEFAULT 0,
                downvotes INTEGER DEFAULT 0,
                expert_assigned TEXT,
                resolution_summary TEXT,
                related_discussions TEXT,  -- JSON array
                FOREIGN KEY (author_id) REFERENCES users (id),
                FOREIGN KEY (expert_assigned) REFERENCES users (id)
            )
        ''')
        
        # Comments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                discussion_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_date DATETIME NOT NULL,
                last_edited DATETIME,
                upvotes INTEGER DEFAULT 0,
                downvotes INTEGER DEFAULT 0,
                is_solution BOOLEAN DEFAULT FALSE,
                parent_comment_id TEXT,
                attachments TEXT,  -- JSON array
                FOREIGN KEY (discussion_id) REFERENCES discussions (id),
                FOREIGN KEY (author_id) REFERENCES users (id),
                FOREIGN KEY (parent_comment_id) REFERENCES comments (id)
            )
        ''')
        
        # Expert consultations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expert_consultations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                requester_id TEXT NOT NULL,
                expert_id TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_date DATETIME NOT NULL,
                scheduled_date DATETIME,
                consultation_notes TEXT,
                follow_up_required BOOLEAN DEFAULT FALSE,
                estimated_duration INTEGER DEFAULT 60,
                FOREIGN KEY (requester_id) REFERENCES users (id),
                FOREIGN KEY (expert_id) REFERENCES users (id)
            )
        ''')
        
        # User votes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                target_type TEXT NOT NULL,  -- 'discussion' or 'comment'
                target_id TEXT NOT NULL,
                vote_type TEXT NOT NULL,  -- 'upvote' or 'downvote'
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, target_type, target_id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                related_id TEXT,  -- ID of related discussion/comment/etc
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                read_date DATETIME,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_default_users(self):
        """Create default system users"""
        # Check if users already exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]
        conn.close()
        
        if user_count == 0:
            # Create system expert
            system_expert = User(
                id="expert_001",
                username="system_expert",
                email="expert@rul-system.com",
                role=UserRole.EXPERT,
                expertise_areas=["system_administration", "troubleshooting", "model_optimization"],
                reputation_score=1000
            )
            self.create_user(system_expert)
            
            # Create moderator
            moderator = User(
                id="mod_001",
                username="community_moderator",
                email="moderator@rul-system.com",
                role=UserRole.MODERATOR,
                expertise_areas=["community_management", "content_moderation"],
                reputation_score=500
            )
            self.create_user(moderator)
    
    def create_user(self, user: User) -> str:
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users 
            (id, username, email, role, expertise_areas, reputation_score, 
             join_date, last_active, profile_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user.id, user.username, user.email, user.role.value,
            json.dumps(user.expertise_areas), user.reputation_score,
            user.join_date, user.last_active, json.dumps(user.profile_data)
        ))
        
        conn.commit()
        conn.close()
        
        return user.id
    
    def create_discussion(self, title: str, description: str, discussion_type: DiscussionType,
                         priority: Priority, author_id: str, tags: List[str] = None) -> str:
        """Create a new discussion"""
        discussion_id = self._generate_discussion_id()
        
        discussion = Discussion(
            id=discussion_id,
            title=title,
            description=description,
            discussion_type=discussion_type,
            priority=priority,
            status=Status.OPEN,
            author_id=author_id,
            created_date=datetime.now(),
            last_updated=datetime.now(),
            tags=tags or []
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO discussions 
            (id, title, description, discussion_type, priority, status, 
             author_id, created_date, last_updated, tags, views, upvotes, 
             downvotes, expert_assigned, resolution_summary, related_discussions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            discussion.id, discussion.title, discussion.description,
            discussion.discussion_type.value, discussion.priority.value,
            discussion.status.value, discussion.author_id,
            discussion.created_date, discussion.last_updated,
            json.dumps(discussion.tags), discussion.views,
            discussion.upvotes, discussion.downvotes,
            discussion.expert_assigned, discussion.resolution_summary,
            json.dumps(discussion.related_discussions)
        ))
        
        conn.commit()
        conn.close()
        
        # Auto-assign expert for high priority issues
        if priority in [Priority.HIGH, Priority.URGENT]:
            self._auto_assign_expert(discussion_id, discussion_type)
        
        return discussion_id
    
    def _generate_discussion_id(self) -> str:
        """Generate unique discussion ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"DISC-{timestamp}"
    
    def _auto_assign_expert(self, discussion_id: str, discussion_type: DiscussionType):
        """Auto-assign expert based on discussion type and availability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find available experts with relevant expertise
        expertise_map = {
            DiscussionType.PROBLEM_REPORT: "troubleshooting",
            DiscussionType.FEATURE_REQUEST: "system_administration",
            DiscussionType.QUESTION: "general_support"
        }
        
        required_expertise = expertise_map.get(discussion_type, "general_support")
        
        cursor.execute('''
            SELECT id FROM users 
            WHERE role IN ('expert', 'administrator')
            AND expertise_areas LIKE ?
            ORDER BY reputation_score DESC, last_active DESC
            LIMIT 1
        ''', (f'%{required_expertise}%',))
        
        expert = cursor.fetchone()
        if expert:
            cursor.execute('''
                UPDATE discussions 
                SET expert_assigned = ?, last_updated = ?
                WHERE id = ?
            ''', (expert[0], datetime.now(), discussion_id))
            
            # Create notification for expert
            self._create_notification(
                expert[0],
                "New Discussion Assignment",
                f"You have been assigned to discussion: {discussion_id}",
                "assignment",
                discussion_id
            )
        
        conn.commit()
        conn.close()
    
    def add_comment(self, discussion_id: str, author_id: str, content: str,
                   parent_comment_id: Optional[str] = None) -> str:
        """Add comment to discussion"""
        comment_id = self._generate_comment_id()
        
        comment = Comment(
            id=comment_id,
            discussion_id=discussion_id,
            author_id=author_id,
            content=content,
            created_date=datetime.now(),
            parent_comment_id=parent_comment_id
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO comments 
            (id, discussion_id, author_id, content, created_date, 
             last_edited, upvotes, downvotes, is_solution, 
             parent_comment_id, attachments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comment.id, comment.discussion_id, comment.author_id,
            comment.content, comment.created_date, comment.last_edited,
            comment.upvotes, comment.downvotes, comment.is_solution,
            comment.parent_comment_id, json.dumps(comment.attachments)
        ))
        
        # Update discussion last_updated
        cursor.execute('''
            UPDATE discussions 
            SET last_updated = ?
            WHERE id = ?
        ''', (datetime.now(), discussion_id))
        
        conn.commit()
        conn.close()
        
        # Notify discussion participants
        self._notify_discussion_participants(discussion_id, author_id, "new_comment")
        
        return comment_id
    
    def _generate_comment_id(self) -> str:
        """Generate unique comment ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"COMM-{timestamp}"
    
    def vote_on_content(self, user_id: str, target_type: str, target_id: str, 
                       vote_type: str) -> bool:
        """Vote on discussion or comment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Remove existing vote if any
            cursor.execute('''
                DELETE FROM user_votes 
                WHERE user_id = ? AND target_type = ? AND target_id = ?
            ''', (user_id, target_type, target_id))
            
            # Add new vote
            cursor.execute('''
                INSERT INTO user_votes (user_id, target_type, target_id, vote_type)
                VALUES (?, ?, ?, ?)
            ''', (user_id, target_type, target_id, vote_type))
            
            # Update vote counts
            if target_type == "discussion":
                if vote_type == "upvote":
                    cursor.execute('UPDATE discussions SET upvotes = upvotes + 1 WHERE id = ?', (target_id,))
                else:
                    cursor.execute('UPDATE discussions SET downvotes = downvotes + 1 WHERE id = ?', (target_id,))
            elif target_type == "comment":
                if vote_type == "upvote":
                    cursor.execute('UPDATE comments SET upvotes = upvotes + 1 WHERE id = ?', (target_id,))
                else:
                    cursor.execute('UPDATE comments SET downvotes = downvotes + 1 WHERE id = ?', (target_id,))
            
            conn.commit()
            return True
            
        except sqlite3.Error:
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def mark_as_solution(self, comment_id: str, user_id: str) -> bool:
        """Mark comment as solution (only by discussion author or expert)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verify user can mark as solution
        cursor.execute('''
            SELECT d.author_id, d.expert_assigned, u.role
            FROM discussions d
            JOIN comments c ON d.id = c.discussion_id
            JOIN users u ON u.id = ?
            WHERE c.id = ?
        ''', (user_id, comment_id))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
        
        author_id, expert_id, user_role = result
        
        # Check permissions
        if user_id not in [author_id, expert_id] and user_role not in ['moderator', 'administrator']:
            conn.close()
            return False
        
        # Mark as solution
        cursor.execute('''
            UPDATE comments 
            SET is_solution = TRUE 
            WHERE id = ?
        ''', (comment_id,))
        
        # Update discussion status
        cursor.execute('''
            UPDATE discussions 
            SET status = 'resolved', last_updated = ?
            WHERE id = (SELECT discussion_id FROM comments WHERE id = ?)
        ''', (datetime.now(), comment_id))
        
        conn.commit()
        conn.close()
        
        return True
    
    def request_expert_consultation(self, title: str, description: str, 
                                  requester_id: str, priority: Priority,
                                  preferred_expert_id: Optional[str] = None) -> str:
        """Request expert consultation"""
        consultation_id = self._generate_consultation_id()
        
        consultation = ExpertConsultation(
            id=consultation_id,
            title=title,
            description=description,
            requester_id=requester_id,
            expert_id=preferred_expert_id,
            priority=priority,
            status=Status.OPEN,
            created_date=datetime.now()
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO expert_consultations 
            (id, title, description, requester_id, expert_id, priority, 
             status, created_date, scheduled_date, consultation_notes, 
             follow_up_required, estimated_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            consultation.id, consultation.title, consultation.description,
            consultation.requester_id, consultation.expert_id,
            consultation.priority.value, consultation.status.value,
            consultation.created_date, consultation.scheduled_date,
            consultation.consultation_notes, consultation.follow_up_required,
            consultation.estimated_duration
        ))
        
        conn.commit()
        conn.close()
        
        # Auto-assign expert if not specified
        if not preferred_expert_id:
            self._auto_assign_consultation_expert(consultation_id)
        
        return consultation_id
    
    def _generate_consultation_id(self) -> str:
        """Generate unique consultation ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CONSULT-{timestamp}"
    
    def _auto_assign_consultation_expert(self, consultation_id: str):
        """Auto-assign expert for consultation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find available expert with highest reputation
        cursor.execute('''
            SELECT id FROM users 
            WHERE role IN ('expert', 'administrator')
            ORDER BY reputation_score DESC, last_active DESC
            LIMIT 1
        ''')
        
        expert = cursor.fetchone()
        if expert:
            cursor.execute('''
                UPDATE expert_consultations 
                SET expert_id = ?
                WHERE id = ?
            ''', (expert[0], consultation_id))
            
            # Create notification
            self._create_notification(
                expert[0],
                "New Consultation Request",
                f"You have been assigned consultation: {consultation_id}",
                "consultation",
                consultation_id
            )
        
        conn.commit()
        conn.close()
    
    def _create_notification(self, user_id: str, title: str, message: str,
                           notification_type: str, related_id: Optional[str] = None):
        """Create notification for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notifications 
            (user_id, title, message, notification_type, related_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, title, message, notification_type, related_id))
        
        conn.commit()
        conn.close()
    
    def _notify_discussion_participants(self, discussion_id: str, author_id: str, 
                                      notification_type: str):
        """Notify all participants in a discussion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all participants (discussion author + commenters + assigned expert)
        cursor.execute('''
            SELECT DISTINCT author_id FROM 
            (SELECT author_id FROM discussions WHERE id = ?
             UNION
             SELECT author_id FROM comments WHERE discussion_id = ?
             UNION
             SELECT expert_assigned FROM discussions WHERE id = ? AND expert_assigned IS NOT NULL)
        ''', (discussion_id, discussion_id, discussion_id))
        
        participants = cursor.fetchall()
        
        for participant in participants:
            participant_id = participant[0]
            if participant_id != author_id:  # Don't notify the person who made the comment
                self._create_notification(
                    participant_id,
                    "Discussion Update",
                    f"New activity in discussion: {discussion_id}",
                    notification_type,
                    discussion_id
                )
        
        conn.close()
    
    def get_discussions(self, status: Optional[Status] = None, 
                       discussion_type: Optional[DiscussionType] = None,
                       limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get discussions with filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT d.*, u.username as author_name
            FROM discussions d
            JOIN users u ON d.author_id = u.id
            WHERE 1=1
        '''
        params = []
        
        if status:
            query += ' AND d.status = ?'
            params.append(status.value)
        
        if discussion_type:
            query += ' AND d.discussion_type = ?'
            params.append(discussion_type.value)
        
        query += ' ORDER BY d.last_updated DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        discussions = []
        for row in results:
            discussions.append({
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'discussion_type': row[3],
                'priority': row[4],
                'status': row[5],
                'author_id': row[6],
                'author_name': row[16],
                'created_date': row[7],
                'last_updated': row[8],
                'tags': json.loads(row[9]) if row[9] else [],
                'views': row[10],
                'upvotes': row[11],
                'downvotes': row[12],
                'expert_assigned': row[13]
            })
        
        return discussions
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, title, message, notification_type, related_id, 
                   created_date, read_date
            FROM notifications 
            WHERE user_id = ?
        '''
        params = [user_id]
        
        if unread_only:
            query += ' AND read_date IS NULL'
        
        query += ' ORDER BY created_date DESC LIMIT 50'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'title': row[1],
                'message': row[2],
                'notification_type': row[3],
                'related_id': row[4],
                'created_date': row[5],
                'read_date': row[6],
                'is_read': row[6] is not None
            }
            for row in results
        ]
    
    def generate_collaboration_report(self) -> Dict[str, Any]:
        """Generate collaboration platform activity report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Discussion statistics
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM discussions 
            GROUP BY status
        ''')
        discussion_stats = dict(cursor.fetchall())
        
        # Most active users
        cursor.execute('''
            SELECT u.username, COUNT(d.id) as discussions, COUNT(c.id) as comments
            FROM users u
            LEFT JOIN discussions d ON u.id = d.author_id
            LEFT JOIN comments c ON u.id = c.author_id
            WHERE u.last_active > datetime('now', '-30 days')
            GROUP BY u.id, u.username
            ORDER BY (COUNT(d.id) + COUNT(c.id)) DESC
            LIMIT 10
        ''')
        active_users = cursor.fetchall()
        
        # Expert consultation stats
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM expert_consultations 
            GROUP BY status
        ''')
        consultation_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'discussion_statistics': discussion_stats,
            'most_active_users': [
                {
                    'username': user[0],
                    'discussions': user[1],
                    'comments': user[2],
                    'total_activity': user[1] + user[2]
                }
                for user in active_users
            ],
            'consultation_statistics': consultation_stats,
            'generated_at': datetime.now().isoformat()
        }

# Global collaboration platform instance
collaboration_platform = CollaborationPlatform()

def get_collaboration_platform() -> CollaborationPlatform:
    """Get the global collaboration platform instance"""
    return collaboration_platform