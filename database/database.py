# database.py - Updated version
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

class DatabaseManager:
    def __init__(self, host="ep-solitary-shadow-a1pohjxz.ap-southeast-1.pg.koyeb.app", database="attendance_db", 
                 user="admin", password="npg_48RFqybgkTad", port=5432):
        """Initialize database connection"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = True
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print(f"   Host: {self.connection_params['host']}")
            print(f"   Database: {self.connection_params['database']}")
            print(f"   User: {self.connection_params['user']}")
            self.conn = None
            return False
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        if self.conn is None:
            return False
    
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")
    
    def save_face_template(self, person_id: str, person_name: str,
                       embedding: np.ndarray, metadata: Optional[dict] = None,
                       cursor: Optional[Any] = None) -> Optional[int]:
        """
        Save face template to database.

        If `cursor` is provided, this function will use it (and won't commit/rollback/close).
        If `cursor` is None, the function will create its own cursor and commit/rollback as needed.
        Uses ON CONFLICT to upsert by person_id to avoid duplicate-key errors.
        """
        # Prepare embedding and metadata
        embedding_list = embedding.tolist()
        embedding_dim = len(embedding_list)
        metadata_json = json.dumps(metadata or {})

        own_cursor = False
        local_cursor = cursor
        if local_cursor is None:
            # Standalone usage: ensure connection is active and create a cursor
            if not self.is_connected():
                print("❌ Database connection is not active")
                return None
            local_cursor = self.conn.cursor()
            own_cursor = True

        try:
            local_cursor.execute("""
                INSERT INTO face_templates (person_id, person_name, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (person_id) DO UPDATE SET
                    person_name = EXCLUDED.person_name,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                RETURNING id
            """, (person_id, person_name, embedding_list, metadata_json))

            face_template_id = local_cursor.fetchone()[0]
            # Print information for logging
            print(f"✅ Face template saved with ID: {face_template_id}")
            print(f"   Embedding dimension: {embedding_dim}")

            if own_cursor:
                # commit only when this function opened the cursor/owns the transaction
                try:
                    self.conn.commit()
                except Exception:
                    # If commit fails, ensure rollback
                    self.conn.rollback()
                    raise

            return face_template_id

        except Exception as e:
            # If we own the cursor, rollback the standalone transaction and close cursor as usual.
            if own_cursor:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                local_cursor.close()
            # If called inside a parent transaction we should re-raise so parent can rollback.
            print(f"❌ Error saving face template: {e}")
            # re-raise to allow outer transaction handler to manage rollback
            raise

        finally:
            # Close the cursor only if we opened it here
            if own_cursor and not local_cursor.closed:
                try:
                    local_cursor.close()
                except Exception:
                    pass

    
    def register_user(self, user_data: Dict[str, Any], face_templates_data: List[Dict[str, Any]]) -> bool:
        """Register a new user with face templates (atomic transaction)."""
        if not self.is_connected():
            return False

        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN")

            # 1️⃣ SAVE FACE TEMPLATES FIRST (use same cursor so single transaction)
            face_template_ids = []
            for template in face_templates_data:
                # Use the same cursor to avoid nested transactions and connection poisoning
                template_id = self.save_face_template(
                    template['person_id'],
                    template['person_name'],
                    np.array(template['embedding']),
                    template.get('metadata', {}),
                    cursor=cursor
                )
                if template_id:
                    face_template_ids.append(template_id)

            if not face_template_ids:
                cursor.execute("ROLLBACK")
                print("❌ No face templates saved")
                return False

            # 2️⃣ SAVE USER AFTER FACE EXISTS (CHILD TABLE)
            cursor.execute("""
                INSERT INTO users (
                    user_id, person_id, full_name, user_type, 
                    email, phone, date_of_birth, department
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    person_id = EXCLUDED.person_id,
                    full_name = EXCLUDED.full_name,
                    user_type = EXCLUDED.user_type,
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    date_of_birth = EXCLUDED.date_of_birth,
                    department = EXCLUDED.department
            """, (
                user_data['user_id'],
                user_data['person_id'],
                user_data['full_name'],
                user_data['user_type'],
                user_data.get('email'),
                user_data.get('phone'),
                user_data.get('date_of_birth'),
                user_data.get('department')
            ))

            # 3️⃣ ROLE-SPECIFIC DATA (STILL inside same transaction)
            user_type = user_data['user_type']

            if user_type == 'student' and 'student' in user_data:
                student_data = user_data['student']
                cursor.execute("""
                    INSERT INTO students (
                        student_id, user_id, full_name, enrollment_number,
                        semester, program, batch_year, email, phone, date_of_birth
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (student_id) DO UPDATE SET
                        full_name = EXCLUDED.full_name,
                        enrollment_number = EXCLUDED.enrollment_number,
                        semester = EXCLUDED.semester,
                        program = EXCLUDED.program,
                        batch_year = EXCLUDED.batch_year,
                        email = EXCLUDED.email,
                        phone = EXCLUDED.phone,
                        date_of_birth = EXCLUDED.date_of_birth
                """, (
                    student_data['student_id'],
                    student_data['user_id'],
                    student_data['full_name'],
                    student_data.get('enrollment_number'),
                    student_data.get('semester'),
                    student_data.get('program'),
                    student_data.get('batch_year'),
                    student_data.get('email'),
                    student_data.get('phone'),
                    student_data.get('date_of_birth')
                ))

            elif user_type == 'faculty' and 'faculty' in user_data:
                faculty_data = user_data['faculty']
                cursor.execute("""
                    INSERT INTO faculty (
                        faculty_id, user_id, designation, qualification
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (faculty_id) DO NOTHING
                """, (
                    faculty_data['faculty_id'],
                    faculty_data['user_id'],
                    faculty_data.get('designation'),
                    faculty_data.get('qualification')
                ))

            # Commit the single transaction
            cursor.execute("COMMIT")
            print(f"✅ User {user_data['full_name']} registered successfully")
            return True

        except Exception as e:
            # Rollback the transaction on any error
            if cursor:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
            print(f"❌ Error registering user: {e}")
            return False

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass


    
    # ==================== TIMETABLE MANAGEMENT ====================
    
    def create_time_slot(self, slot_data: Dict[str, Any]) -> Optional[int]:
        """Create a predefined time slot"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO time_slots (slot_name, slot_type, start_time, end_time, day_of_week)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING slot_id
            """, (
                slot_data['slot_name'],
                slot_data['slot_type'],
                slot_data['start_time'],
                slot_data['end_time'],
                slot_data.get('day_of_week')
            ))
            slot_id = cursor.fetchone()[0]
            print(f"✅ Time slot '{slot_data['slot_name']}' created with ID: {slot_id}")
            return slot_id
        except Exception as e:
            print(f"❌ Error creating time slot: {e}")
            return None
        finally:
            if cursor: cursor.close()

    def add_timetable_entry(self, entry_data: Dict[str, Any]) -> Optional[int]:
        """Add an entry to the recurring weekly timetable"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO recurring_timetable (
                    day_of_week, time_slot_id, subject_name, subject_code,
                    session_type, faculty_id, faculty_name, batch_name,
                    expected_attendees, room_number
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING timetable_id
            """, (
                entry_data['day_of_week'],
                entry_data['time_slot_id'],
                entry_data['subject_name'],
                entry_data.get('subject_code'),
                entry_data.get('session_type', 'lecture'),
                entry_data.get('faculty_id'),
                entry_data.get('faculty_name'),
                entry_data.get('batch_name'),
                entry_data.get('expected_attendees'),
                entry_data.get('room_number')
            ))
            timetable_id = cursor.fetchone()[0]
            print(f"✅ Timetable entry for '{entry_data['subject_name']}' added with ID: {timetable_id}")
            return timetable_id
        except Exception as e:
            print(f"❌ Error adding timetable entry: {e}")
            return None
        finally:
            if cursor: cursor.close()

    def get_today_timetable(self) -> List[Dict[str, Any]]:
        """Fetch all timetable entries for the current day"""
        if not self.is_connected():
            return []
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            # PostgreSQL day_of_week: 1 is Mon, 2 is Tue... 7 is Sun
            # Python datetime.weekday(): 0 is Mon... 6 is Sun
            day_of_week = datetime.now().weekday() + 1
            
            cursor.execute("""
                SELECT rt.*, ts.start_time, ts.end_time, ts.slot_name
                FROM recurring_timetable rt
                JOIN time_slots ts ON rt.time_slot_id = ts.slot_id
                WHERE rt.day_of_week = %s AND rt.is_active = TRUE
                ORDER BY ts.start_time ASC
            """, (day_of_week,))
            return cursor.fetchall()
        except Exception as e:
            print(f"❌ Error fetching today's timetable: {e}")
            return []
        finally:
            if cursor: cursor.close()

    def auto_create_daily_sessions(self) -> int:
        """Create attendance sessions for today based on the recurring timetable"""
        entries = self.get_today_timetable()
        if not entries:
            print("📅 No timetable entries found for today")
            return 0
        
        created_count = 0
        today = datetime.now().date()
        
        for entry in entries:
            # Combine today's date with slot times
            start_ts = datetime.combine(today, entry['start_time'])
            end_ts = datetime.combine(today, entry['end_time'])
            
            session_data = {
                'session_name': f"{entry['subject_name']} - {entry['slot_name']}",
                'session_type': entry['session_type'],
                'time_slot_id': entry['time_slot_id'],
                'recurring_timetable_id': entry['timetable_id'],
                'scheduled_start': start_ts,
                'scheduled_end': end_ts,
                'subject_name': entry['subject_name'],
                'faculty_in_charge': entry['faculty_id'],
                'room_number': entry['room_number'],
                'expected_attendees': entry['expected_attendees']
            }
            
            # Check if session already exists for this slot today to avoid duplicates
            if self.session_exists(entry['timetable_id'], today):
                continue
                
            if self.create_attendance_session(session_data):
                created_count += 1
                
        print(f"🚀 Created {created_count} attendance sessions based on today's timetable")
        return created_count

    def session_exists(self, timetable_id: int, date) -> bool:
        """Check if a session for a specific timetable entry and date already exists"""
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 1 FROM attendance_sessions 
                WHERE recurring_timetable_id = %s 
                AND DATE(scheduled_start) = %s
            """, (timetable_id, date))
            return cursor.fetchone() is not None
        except Exception:
            return False
        finally:
            if cursor: cursor.close()

    def record_student_attendance(self, attendance_data: Dict[str, Any]) -> Optional[int]:
        """Record student attendance in database"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Get current timestamp for attendance_time
            attendance_time = attendance_data.get('attendance_time', datetime.now())
            
            cursor.execute("""
                INSERT INTO student_attendance (
                    student_id, user_id, person_id,
                    attendance_date, attendance_time, attendance_type,
                    attendance_status, confidence_score, face_template_id,
                    authentication_method
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING attendance_id
            """, (
                attendance_data['student_id'],
                attendance_data['user_id'],
                attendance_data['person_id'],
                attendance_time.date(),
                attendance_time,
                attendance_data.get('attendance_type', 'class'),
                attendance_data.get('attendance_status', 'present'),
                attendance_data.get('confidence_score'),
                attendance_data.get('face_template_id'),
                attendance_data.get('authentication_method', 'face')
            ))
            
            attendance_id = cursor.fetchone()[0]
            print(f"✅ Student attendance recorded with ID: {attendance_id}")
            
            return attendance_id
            
        except Exception as e:
            print(f"❌ Error recording student attendance: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def show_today_student_attendance(self, student_id: Optional[str] = None):
        """Show today's student attendance - now properly defined"""
        if not self.is_connected():
            print("❌ Database not connected")
            return
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            if student_id:
                cursor.execute("""
                    SELECT sa.*, s.full_name, u.department
                    FROM student_attendance sa
                    JOIN students s ON sa.student_id = s.student_id
                    JOIN users u ON sa.user_id = u.user_id
                    WHERE sa.student_id = %s 
                    AND DATE(sa.attendance_date) = CURRENT_DATE
                    ORDER BY sa.attendance_time DESC
                """, (student_id,))
            else:
                cursor.execute("""
                    SELECT sa.*, s.full_name, u.department
                    FROM student_attendance sa
                    JOIN students s ON sa.student_id = s.student_id
                    JOIN users u ON sa.user_id = u.user_id
                    WHERE DATE(sa.attendance_date) = CURRENT_DATE
                    ORDER BY sa.attendance_time DESC
                    LIMIT 10
                """)
            
            records = cursor.fetchall()
            
            if records:
                print("\n📊 TODAY'S STUDENT ATTENDANCE:")
                print("-" * 80)
                for record in records:
                    time_str = record['attendance_time'].strftime('%H:%M:%S') if record['attendance_time'] else "N/A"
                    print(f"👤 {record['full_name']} (Student ID: {record['student_id']})")
                    print(f"   Type: {record['attendance_type']} at {time_str}")
                    print(f"   Status: {record['attendance_status']} | Confidence: {record.get('confidence_score', 0):.2f}")
                    print("-" * 40)
            else:
                print("📭 No student attendance records for today")
                
        except Exception as e:
            print(f"⚠️ Error fetching student attendance: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def find_user_by_face(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Find user by face embedding"""
        if not self.is_connected():
            print("❌ Database connection is not active")
            return None
    
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            embedding_list = embedding.tolist()
            embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'
        
            # Use PostgreSQL vector similarity search
            cursor.execute("""
                WITH face_matches AS (
                    SELECT 
                        ft.id as face_template_id,
                        ft.person_id,
                        ft.person_name,
                        ft.embedding <=> %s as similarity,
                        u.user_id,
                        u.full_name,
                        u.user_type,
                        u.email,
                        u.department
                    FROM face_templates ft
                    JOIN users u ON ft.person_id = u.person_id
                    WHERE ft.embedding <=> %s < %s
                    ORDER BY similarity ASC
                )
                SELECT DISTINCT ON (person_id) *
                FROM face_matches
                ORDER BY person_id, similarity ASC
                LIMIT 1
            """, (embedding_str, embedding_str, 1 - threshold))
        
            result = cursor.fetchone()
        
            if result:
                similarity = 1 - result['similarity']
                result['confidence'] = similarity
                
                # Get additional user info based on user_type
                if result['user_type'] == 'student':
                    cursor.execute("""
                        SELECT s.student_id, s.enrollment_number, s.semester, s.program
                        FROM students s
                        WHERE s.user_id = %s
                    """, (result['user_id'],))
                    student_info = cursor.fetchone()
                    if student_info:
                        result.update(dict(student_info))
                elif result['user_type'] == 'faculty':
                    cursor.execute("""
                        SELECT f.faculty_id, f.designation
                        FROM faculty f
                        WHERE f.user_id = %s
                    """, (result['user_id'],))
                    faculty_info = cursor.fetchone()
                    if faculty_info:
                        result.update(dict(faculty_info))
                
                print(f"✅ Face match found: {result['full_name']} ({result['user_type']})")
                print(f"   Confidence: {similarity:.3f}")
                return dict(result)
        
            return None
        
        except Exception as e:
            print(f"❌ Error in face search: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user details by user_id"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT u.*, 
                       CASE 
                           WHEN u.user_type = 'student' THEN s.student_id
                           WHEN u.user_type = 'faculty' THEN f.faculty_id
                           ELSE NULL
                       END as role_specific_id
                FROM users u
                LEFT JOIN students s ON u.user_id = s.user_id AND u.user_type = 'student'
                LEFT JOIN faculty f ON u.user_id = f.user_id AND u.user_type = 'faculty'
                WHERE u.user_id = %s
            """, (user_id,))
            
            user = cursor.fetchone()
            
            if user:
                user_dict = dict(user)
                
                # Add role-specific details
                if user_dict['user_type'] == 'student':
                    cursor.execute("""
                        SELECT s.enrollment_number, s.semester, s.program, s.batch_year
                        FROM students s
                        WHERE s.user_id = %s
                    """, (user_id,))
                    student_details = cursor.fetchone()
                    if student_details:
                        user_dict.update(dict(student_details))
                
                elif user_dict['user_type'] == 'faculty':
                    cursor.execute("""
                        SELECT f.designation, f.qualification
                        FROM faculty f
                        WHERE f.user_id = %s
                    """, (user_id,))
                    faculty_details = cursor.fetchone()
                    if faculty_details:
                        user_dict.update(dict(faculty_details))
                
                return user_dict
            return None
            
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all registered users"""
        if not self.is_connected():
            return []
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT u.user_id, u.full_name, u.user_type, 
                       u.email, u.department, u.is_active,
                       CASE 
                           WHEN u.user_type = 'student' THEN s.enrollment_number
                           WHEN u.user_type = 'faculty' THEN 'Faculty'
                           ELSE 'Staff'
                       END as role_identifier,
                       COUNT(ft.id) as face_templates_count
                FROM users u
                LEFT JOIN students s ON u.user_id = s.user_id AND u.user_type = 'student'
                LEFT JOIN faculty f ON u.user_id = f.user_id AND u.user_type = 'faculty'
                LEFT JOIN face_templates ft ON u.person_id = ft.person_id
                GROUP BY u.user_id, u.full_name, u.user_type, u.email, 
                         u.department, u.is_active, s.enrollment_number
                ORDER BY u.user_type, u.full_name
            """)
            
            users = cursor.fetchall()
            
            return [dict(user) for user in users]
            
        except Exception as e:
            print(f"❌ Error getting users: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and associated data (cascade)"""
        if not self.is_connected():
            return False
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Get user info before deletion
            user_info = self.get_user_by_id(user_id)
            if not user_info:
                print(f"❌ User {user_id} not found")
                return False
            
            cursor.execute("BEGIN")
            
            # Delete user (cascade will delete related records)
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            
            cursor.execute("COMMIT")
            print(f"✅ User {user_info['full_name']} (ID: {user_id}) deleted successfully")
            return True
            
        except Exception as e:
            if cursor:
                cursor.execute("ROLLBACK")
            print(f"❌ Error deleting user: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student details by student_id"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT s.*, u.person_id, u.user_type, u.email as user_email
                FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.student_id = %s
            """, (student_id,))
            
            student = cursor.fetchone()
            
            if student:
                return dict(student)
            return None
            
        except Exception as e:
            print(f"❌ Error getting student: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_student_attendance_summary(self, student_id: str, start_date: Optional[str] = None, 
                                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get attendance summary for a student"""
        if not self.is_connected():
            return {}
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query with optional date filters
            query = """
                SELECT 
                    COUNT(*) as total_days,
                    COUNT(CASE WHEN attendance_status = 'present' THEN 1 END) as present_days,
                    COUNT(CASE WHEN attendance_status = 'absent' THEN 1 END) as absent_days,
                    COUNT(CASE WHEN attendance_status = 'late' THEN 1 END) as late_days,
                    AVG(confidence_score) as avg_confidence
                FROM student_attendance
                WHERE student_id = %s
            """
            
            params = [student_id]
            
            if start_date:
                query += " AND attendance_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND attendance_date <= %s"
                params.append(end_date)
            
            cursor.execute(query, tuple(params))
            summary = cursor.fetchone()
            
            if summary:
                result = dict(summary)
                if result['total_days'] > 0:
                    result['attendance_percentage'] = (result['present_days'] / result['total_days']) * 100
                else:
                    result['attendance_percentage'] = 0
                return result
            return {}
            
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def create_attendance_session(self, session_data: Dict[str, Any]) -> Optional[int]:
        """Create a new attendance session with enhanced real-time fields"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO attendance_sessions (
                    session_name, session_type, location_id, room_number,
                    scheduled_start, scheduled_end, course_id, subject_name,
                    faculty_in_charge, session_status, expected_duration_minutes, 
                    notes, time_slot_id, recurring_timetable_id, expected_attendees,
                    is_active
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING session_id
            """, (
                session_data.get('session_name'),
                session_data.get('session_type', 'lecture'),
                session_data.get('location_id'),
                session_data.get('room_number'),
                session_data.get('scheduled_start'),
                session_data.get('scheduled_end'),
                session_data.get('course_id'),
                session_data.get('subject_name'),
                session_data.get('faculty_in_charge'),
                session_data.get('session_status', 'scheduled'),
                session_data.get('expected_duration_minutes'),
                session_data.get('notes'),
                session_data.get('time_slot_id'),
                session_data.get('recurring_timetable_id'),
                session_data.get('expected_attendees'),
                session_data.get('is_active', False)
            ))
            
            session_id = cursor.fetchone()[0]
            print(f"✅ Attendance session created with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"❌ Error creating attendance session: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    # ==================== REAL-TIME ATTENDANCE OPERATIONS ====================

    def get_current_active_session(self) -> Optional[Dict[str, Any]]:
        """Get the session that is currently marked as active and ongoing"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM v_active_sessions LIMIT 1")
            return cursor.fetchone()
        except Exception as e:
            print(f"❌ Error getting active session: {e}")
            return None
        finally:
            if cursor: cursor.close()

    def mark_realtime_attendance(self, attendance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record attendance in the log with duplicate prevention and anti-spoofing data.
        """
        if not self.is_connected():
            return {'status': 'error', 'message': 'Database not connected'}
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # 1. Get current active session if not provided
            session_id = attendance_data.get('session_id')
            if not session_id:
                active_session = self.get_current_active_session()
                if not active_session:
                    return {'status': 'ignored', 'message': 'No active session found'}
                session_id = active_session['session_id']

            # 2. Insert into attendance log (Unique Index handles duplicate check)
            try:
                cursor.execute("""
                    INSERT INTO attendance_log (
                        session_id, person_id, user_id, 
                        attendance_status, face_template_id, confidence_score,
                        liveness_score, texture_analysis_score, optical_flow_score,
                        anti_spoof_result, device_id, location, notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING log_id
                """, (
                    session_id,
                    attendance_data['person_id'],
                    attendance_data['user_id'],
                    attendance_data.get('attendance_status', 'present'),
                    attendance_data.get('face_template_id'),
                    attendance_data.get('confidence_score'),
                    attendance_data.get('liveness_score'),
                    attendance_data.get('texture_analysis_score'),
                    attendance_data.get('optical_flow_score'),
                    attendance_data.get('anti_spoof_result'),
                    attendance_data.get('device_id'),
                    attendance_data.get('location'),
                    attendance_data.get('notes')
                ))
                log_id = cursor.fetchone()[0]
                return {
                    'status': 'success', 
                    'log_id': log_id, 
                    'session_id': session_id,
                    'message': 'Attendance marked successfully'
                }
            except psycopg2.errors.UniqueViolation:
                if self.conn: self.conn.rollback() 
                return {
                    'status': 'duplicate', 
                    'message': 'Attendance already marked for this session today'
                }
                
        except Exception as e:
            print(f"❌ Error marking realtime attendance: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            if cursor: cursor.close()

    def get_live_dashboard_stats(self, session_id: int) -> Dict[str, Any]:
        """Fetch live statistics for a specific session"""
        if not self.is_connected():
            return {}
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM v_active_sessions WHERE session_id = %s", (session_id,))
            session_info = cursor.fetchone()
            cursor.execute("SELECT * FROM v_live_attendance_stats WHERE session_id = %s", (session_id,))
            live_stats = cursor.fetchone()
            result = dict(session_info) if session_info else {}
            if live_stats:
                result.update(dict(live_stats))
            return result
        except Exception as e:
            print(f"❌ Error fetching live stats: {e}")
            return {}
        finally:
            if cursor: cursor.close()
