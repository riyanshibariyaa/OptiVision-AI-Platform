# models.py
from pymongo import MongoClient
from app_settings import cls_app_settings as settings
import bcrypt
from datetime import datetime

class cls_app_db:
    db_conn = MongoClient(settings.MONGO_URI)
    db = db_conn.optiview

    # Users collection
    coll_fw_units = db.units
    coll_fw_menus = db.menus
    coll_fw_users = db.users
    coll_fw_events = db.events

    # Sample Data
    sample_units = [
        {"_id": 1, "name": "Unit1", "location": "NY", "address": "123 Finance St",  "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 2, "name": "Unit2", "location": "LA", "address": "456 HR Ave",      "is_deleted": False, "modified": "2024/01/01 12:00"}
    ]
    
    sample_menus = [
        {"_id": 1, "name": "Dashboard", "title": "Main Dashboard",  "icon": "dashboard-icon.png",       "is_group": False,  "parent": None,   "view": "/dashboard", "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 2, "name": "Masters",   "title": "Masters",         "icon": "user-reports-icon.png",    "is_group": True,   "parent": None,   "view": "",           "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 3, "name": "Units",     "title": "Manage Units",    "icon": "user-reports-icon1.png",   "is_group": False,  "parent": 2,      "view": "/untis",     "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 4, "name": "Menus",     "title": "Manage Menus",    "icon": "user-reports-icon2.png",   "is_group": False,  "parent": 2,      "viek": "/menus",     "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 5, "name": "Users",     "title": "Manage Users",    "icon": "user-reports-icon3.png",   "is_group": False,  "parent": 2,      "view": "/users",     "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 6, "name": "Settings",  "title": "Manage Settings", "icon": "user-reports-icon4.png",   "is_group": False,  "parent": 2,      "view": "/settings",  "is_deleted": False, "modified": "2024/01/01 12:00"}
    ]

    sample_users = [
        {
            "_id": 1, "name": "Admin", "user_id": "admin", "designation": "Adminstrator", "email": "admin@fw.com", "photo": "admin.jpg", "address": "123 Maple St", 
            "dob": datetime(1990, 1, 1), "password": bcrypt.hashpw("admin@2024".encode('utf-8'), bcrypt.gensalt()), 
            "accessible_units": [
                {"unit_id": 1, "menus": [1, 2, 3, 4, 5, 6]},
                {"unit_id": 2, "menus": [1, 2, 3, 4, 5, 6]}
            ],
            "is_deleted": False, "modified": "2024/01/01 12:00"
        },
        {
            "_id": 2, "name": "Guest", "user_id": "guest", "designation": "Guest",        "email": "guest@fw.com", "photo": "guest.jpg", "address": "456 Oak St", 
            "dob": datetime(2000, 1, 1), "password": bcrypt.hashpw("g".encode('utf-8'), bcrypt.gensalt()), 
            "accessible_units": [
                {"unit_id": 2, "menus": [1, 2, 3]}
            ],
            "is_deleted": False, "modified": "2024/01/01 12:00"
        }
    ]

    sample_events = [
        {"_id": 1, "date": "2024/01/01 12:00", "type": "Login",     "brief": "Login attempt succeeded", "details": "",  "is_deleted": False, "modified": "2024/01/01 12:00"},
        {"_id": 2, "date": "2024/01/01 12:00", "type": "Update",    "brief": "Update on table Users",   "details": "",  "is_deleted": False, "modified": "2024/01/01 12:00"}
    ]

    def __init__(self):

        # Add sample data to db Collections if they are empty
        if self.coll_fw_units.count_documents({}) == 0:
            self.coll_fw_units.insert_many(self.sample_units)

        if self.coll_fw_menus.count_documents({}) == 0:
            self.coll_fw_menus.insert_many(self.sample_menus)

        if self.coll_fw_users.count_documents({}) == 0:
            self.coll_fw_users.insert_many(self.sample_users)

        if self.coll_fw_events.count_documents({}) == 0:
            self.coll_fw_events.insert_many(self.sample_events)


cls_app_db()