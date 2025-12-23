from pymongo import MongoClient

MONGO_URI = "mongodb+srv://ramyabhupathi7_db_user:lCJdnDv0u0R73dXU@cluster0.mzcov3g.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["croptracker"]

for user in db.users.find():
    print(user)
