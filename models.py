from pymongo import MongoClient

# MongoDB connection (local, change URI for Atlas if needed)
MONGO_URI = "mongodb+srv://ramyabhupathi7_db_user:lCJdnDv0u0R73dXU@cluster0.mzcov3g.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)

# Database and collection
db = client["croptracker"]
users_collection = db["users"]
