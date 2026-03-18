import sqlite3
import datetime
import os
from typing import List, Dict, Optional, Tuple


class DatabaseManager:
    """
    数据库管理器
    负责违规记录的存储、查询和统计
    """

    def __init__(self, db_name="traffic_violations.db"):
        """
        初始化数据库连接
        Args:
            db_name: 数据库文件名
        """
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """创建数据库表"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                car_id TEXT,
                plate_number TEXT,
                vehicle_type TEXT,
                location TEXT,
                duration REAL,
                image_path TEXT,
                video_clip_path TEXT,
                status TEXT DEFAULT '未处理',
                notes TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_time TEXT NOT NULL,
                log_level TEXT,
                module TEXT,
                message TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS roi_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                points TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)

        self._migrate_table()

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON violations(timestamp)
        """)

        try:
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate_number 
                ON violations(plate_number)
            """)
        except sqlite3.OperationalError:
            pass

        self.conn.commit()

    def _migrate_table(self):
        """迁移旧表结构，添加缺失的列"""
        self.cursor.execute("PRAGMA table_info(violations)")
        columns = [col[1] for col in self.cursor.fetchall()]

        migrations = {
            'plate_number': 'TEXT',
            'vehicle_type': 'TEXT',
            'duration': 'REAL',
            'video_clip_path': 'TEXT',
            'status': "TEXT DEFAULT '未处理'",
            'notes': 'TEXT'
        }

        for col_name, col_type in migrations.items():
            if col_name not in columns:
                try:
                    self.cursor.execute(f"ALTER TABLE violations ADD COLUMN {col_name} {col_type}")
                    print(f"[DB] 已添加列: {col_name}")
                except sqlite3.OperationalError as e:
                    print(f"[DB] 添加列 {col_name} 失败: {e}")

        self.conn.commit()

    def insert_violation(self, car_id: str, image_path: str,
                         plate_number: str = None, vehicle_type: str = None,
                         location: str = "禁停区A", duration: float = 0,
                         video_clip_path: str = None) -> str:
        """
        插入一条违规记录
        Args:
            car_id: 车辆追踪ID
            image_path: 截图路径
            plate_number: 车牌号码
            vehicle_type: 车辆类型
            location: 违停位置
            duration: 停留时长(秒)
            video_clip_path: 视频片段路径
        Returns:
            timestamp: 记录时间戳
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute("""
            INSERT INTO violations 
            (timestamp, car_id, plate_number, vehicle_type, location, duration, image_path, video_clip_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_time, car_id, plate_number, vehicle_type, location, duration, image_path, video_clip_path))

        self.conn.commit()
        print(f"[DB] 记录已保存: {car_id} | {plate_number} | {current_time}")
        return current_time

    def get_all_violations(self) -> List[Tuple]:
        """获取所有违规记录"""
        self.cursor.execute("""
            SELECT id, timestamp, car_id, plate_number, vehicle_type, 
                   location, duration, image_path, status
            FROM violations 
            ORDER BY id DESC
        """)
        return self.cursor.fetchall()

    def get_violations_by_date(self, start_date: str, end_date: str) -> List[Tuple]:
        """
        按日期范围获取违规记录
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.cursor.execute("""
            SELECT id, timestamp, car_id, plate_number, vehicle_type, 
                   location, duration, image_path, status
            FROM violations 
            WHERE date(timestamp) BETWEEN date(?) AND date(?)
            ORDER BY timestamp DESC
        """, (start_date, end_date))
        return self.cursor.fetchall()

    def get_violation_by_id(self, violation_id: int) -> Optional[Tuple]:
        """根据ID获取违规记录"""
        self.cursor.execute("""
            SELECT * FROM violations WHERE id = ?
        """, (violation_id,))
        return self.cursor.fetchone()

    def update_violation_status(self, violation_id: int, status: str, notes: str = None):
        """更新违规记录状态"""
        if notes:
            self.cursor.execute("""
                UPDATE violations SET status = ?, notes = ? WHERE id = ?
            """, (status, notes, violation_id))
        else:
            self.cursor.execute("""
                UPDATE violations SET status = ? WHERE id = ?
            """, (status, violation_id))
        self.conn.commit()

    def delete_violation(self, violation_id: int):
        """删除违规记录"""
        self.cursor.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
        self.conn.commit()

    def get_statistics(self) -> Dict:
        """获取统计数据"""
        stats = {}

        self.cursor.execute("SELECT COUNT(*) FROM violations")
        stats['total_violations'] = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT COUNT(DISTINCT plate_number) FROM violations 
            WHERE plate_number IS NOT NULL AND plate_number != ''
        """)
        stats['unique_vehicles'] = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT vehicle_type, COUNT(*) as count 
            FROM violations 
            GROUP BY vehicle_type
        """)
        stats['by_type'] = dict(self.cursor.fetchall())

        self.cursor.execute("""
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM violations
            GROUP BY hour
            ORDER BY hour
        """)
        stats['by_hour'] = dict(self.cursor.fetchall())

        self.cursor.execute("""
            SELECT date(timestamp) as date, COUNT(*) as count
            FROM violations
            GROUP BY date
            ORDER BY date DESC
            LIMIT 7
        """)
        stats['daily_recent'] = dict(self.cursor.fetchall())

        self.cursor.execute("""
            SELECT AVG(duration) FROM violations WHERE duration > 0
        """)
        result = self.cursor.fetchone()[0]
        stats['avg_duration'] = round(result, 1) if result else 0

        return stats

    def get_daily_statistics(self, days: int = 30) -> List[Dict]:
        """获取每日统计数据"""
        self.cursor.execute("""
            SELECT date(timestamp) as date, 
                   COUNT(*) as count,
                   COUNT(DISTINCT plate_number) as unique_vehicles
            FROM violations
            WHERE date(timestamp) >= date('now', ?)
            GROUP BY date
            ORDER BY date
        """, (f'-{days} days',))

        return [
            {'date': row[0], 'count': row[1], 'unique_vehicles': row[2]}
            for row in self.cursor.fetchall()
        ]

    def get_hourly_statistics(self) -> List[Dict]:
        """获取时段统计数据"""
        self.cursor.execute("""
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM violations
            GROUP BY hour
            ORDER BY hour
        """)

        return [
            {'hour': int(row[0]), 'count': row[1]}
            for row in self.cursor.fetchall()
        ]

    def get_type_statistics(self) -> List[Dict]:
        """获取车辆类型统计"""
        self.cursor.execute("""
            SELECT vehicle_type, COUNT(*) as count
            FROM violations
            GROUP BY vehicle_type
            ORDER BY count DESC
        """)

        return [
            {'type': row[0] or '未知', 'count': row[1]}
            for row in self.cursor.fetchall()
        ]

    def add_log(self, level: str, module: str, message: str):
        """添加系统日志"""
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            INSERT INTO system_logs (log_time, log_level, module, message)
            VALUES (?, ?, ?, ?)
        """, (log_time, level, module, message))
        self.conn.commit()

    def get_recent_logs(self, limit: int = 100) -> List[Tuple]:
        """获取最近的日志"""
        self.cursor.execute("""
            SELECT log_time, log_level, module, message
            FROM system_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return self.cursor.fetchall()

    def save_roi_config(self, name: str, points: List[Tuple]):
        """保存ROI配置"""
        points_str = ','.join([f"{x},{y}" for x, y in points])
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute("""
            INSERT INTO roi_config (name, points, created_at)
            VALUES (?, ?, ?)
        """, (name, points_str, created_at))
        self.conn.commit()

    def load_roi_config(self, name: str = None) -> Optional[List[Tuple]]:
        """加载ROI配置"""
        if name:
            self.cursor.execute("""
                SELECT points FROM roi_config 
                WHERE name = ? AND is_active = 1
                ORDER BY id DESC LIMIT 1
            """, (name,))
        else:
            self.cursor.execute("""
                SELECT points FROM roi_config 
                WHERE is_active = 1
                ORDER BY id DESC LIMIT 1
            """)

        result = self.cursor.fetchone()
        if result:
            points_str = result[0]
            points = []
            for pair in points_str.split(','):
                if pair:
                    x, y = pair.split(',')
                    points.append((int(x), int(y)))
            return points
        return None

    def export_to_dict(self) -> List[Dict]:
        """导出所有记录为字典列表"""
        records = self.get_all_violations()
        return [
            {
                'id': r[0],
                'timestamp': r[1],
                'car_id': r[2],
                'plate_number': r[3],
                'vehicle_type': r[4],
                'location': r[5],
                'duration': r[6],
                'image_path': r[7],
                'status': r[8]
            }
            for r in records
        ]

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("[DB] 数据库连接已关闭")


if __name__ == "__main__":
    print("数据库模块测试...")

    test_db = DatabaseManager("test_violations.db")

    test_db.insert_violation(
        car_id="Car_1",
        image_path="test.jpg",
        plate_number="京A12345",
        vehicle_type="car",
        duration=65.5
    )

    stats = test_db.get_statistics()
    print(f"统计数据: {stats}")

    all_records = test_db.get_all_violations()
    print(f"总记录数: {len(all_records)}")

    os.remove("test_violations.db")
    print("数据库模块测试完成!")
