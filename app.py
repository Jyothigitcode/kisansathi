from flask import Flask, render_template, request, jsonify, session, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
import os
import json
import joblib
import pandas as pd
from time import time
import secrets
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ---------------- Flask Setup ----------------

app = Flask(__name__, template_folder="templates")
CORS(app)

# Load secrets from .env
app.secret_key = os.getenv("SECRET_KEY")

# Upload config
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- MongoDB Setup ----------------
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["farmer_db"]

# Collections
users_collection = db["users"]
crops_collection = db["crops"]
disease_collection = db["disease_reports"]
products_collection = db["products"]
messages_collection = db["messages"]

# ---------------- Home ----------------
@app.route("/")
def home():
    return render_template("start.html")
@app.route("/veg-price")
def veg_price_page():
    return render_template("index1.html")

@app.route("/login")
def login_page():
    return render_template("login.html")
@app.route("/news")
def news_page():
    return render_template("news.html")
@app.route("/learning")
def learning_page():
    return render_template("learningmaterial.html")
@app.route("/soilrecommendation")
def soil_page():
    return render_template("soilrecommendation.html")
@app.route("/experts")
def experts_page():
    return render_template("experts.html")
@app.route("/expert1")
def expert1_page():
    return render_template("expert1.html")
@app.route("/expert2")
def expert2_page():
    return render_template("expert2.html")          
@app.route("/expert3")
def expert3_page():
    return render_template("expert3.html")
@app.route("/expert4")
def expert4_page():
    return render_template("expert4.html")
@app.route("/expert5")
def expert5_page():
    return render_template("expert5.html")

# =========================
# Veg Price Prediction Model
# =========================
veg_price_model = joblib.load('veg_price_model.pkl')
training_columns = pd.read_csv(
    'training_columns.csv', header=None
).iloc[:, 0].tolist()

# ---------------- Mock Login ----------------
@app.route("/login/<username>")
def login_mock(username):
    user = users_collection.find_one({"username": username})
    if not user:
        user_id = users_collection.insert_one({"username": username}).inserted_id
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    session["user_id"] = str(user["_id"])
    session["username"] = user["username"]
    return redirect("/mainpage")
def preprocess_veg_input(df):
    df_encoded = pd.get_dummies(df)

    # Add missing columns
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep column order same as training
    df_encoded = df_encoded[training_columns]
    return df_encoded
@app.route('/predict-veg-price', methods=['POST'])
def predict_veg_price():
    try:
        input_df = pd.DataFrame({
            'year': [int(request.form['year'])],
            'month': [int(request.form['month'])],
            'day': [int(request.form['day'])],
            'State': [request.form['state']],
            'District': [request.form['district']],
            'Market': [request.form['market']],
            'Commodity': [request.form['commodity']],
            'Variety': [request.form['variety']],
            'Grade': [request.form['grade']]
        })

        processed_input = preprocess_veg_input(input_df)
        prediction = veg_price_model.predict(processed_input)[0]

        return render_template(
            'index1.html',
            veg_prediction=round(prediction, 2)
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ---------------- Auth: Signup ----------------
@app.route("/auth/signup", methods=["POST"])
def signup():
    try:
        # Accept JSON or Form data safely
        data = request.get_json(silent=True) or request.form

        fullName = data.get("fullName")
        email = data.get("email")
        password = data.get("password")

        if not fullName or not email or not password:
            return jsonify({"message": "All fields required"}), 400

        if users_collection.find_one({"email": email}):
            return jsonify({"message": "User already exists"}), 409

        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        users_collection.insert_one({
            "fullName": fullName,
            "email": email,
            "password": hashed
        })

        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print("Signup error:", e)
        return jsonify({"message": "Server error during signup"}), 500



# ---------------- Auth: Login ----------------
@app.route("/auth/login", methods=["POST"])
def login():
    try:
        # Accept JSON or Form data safely
        data = request.get_json(silent=True) or request.form

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"message": "Email and password required"}), 400

        user = users_collection.find_one({"email": email})

        if not user:
            return jsonify({"message": "Invalid email or password"}), 401

        if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            return jsonify({"message": "Invalid email or password"}), 401

        session["user_id"] = str(user["_id"])
        session["username"] = user.get("fullName")

        return jsonify({
            "message": "Login successful",
            "redirect": "/mainpage"
        }), 200

    except Exception as e:
        print("Login error:", e)
        return jsonify({"message": "Server error during login"}), 500

# ---------------- Profile ----------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect("/login")
    user = users_collection.find_one({"_id": ObjectId(session["user_id"])})

    if request.method == "POST":
        users_collection.update_one(
            {"_id": ObjectId(session["user_id"])},
            {"$set": {
                "land_size": request.form.get("landSize", ""),
                "experience": request.form.get("experience", ""),
                "city": request.form.get("city", ""),
                "mobile": request.form.get("mobile", "")
            }}
        )
        return redirect("/profile")

    return render_template("profile.html", user=user)

# ---------------- Reset Password ----------------
@app.route("/auth/reset", methods=["POST"])
def reset_password():
    data = request.json
    email = data.get("email")
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found"}), 400
    return jsonify({"message": "Password reset link sent (simulation)"})

# ---------------- Diseases ----------------
@app.route("/diseases")
def diseases_page():
    return render_template("diseases.html")

@app.route("/upload_disease", methods=["POST"])
def upload_disease():
    try:
        crop = request.form["crop"]
        disease = request.form["disease"]
        symptoms = request.form.get("symptoms", "")
        lat = float(request.form["lat"])
        lng = float(request.form["lng"])
        image = request.files["image"]

        filename = f"{int(time())}_{secure_filename(image.filename)}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(path)

        report = {
            "crop": crop,
            "disease": disease,
            "symptoms": symptoms,
            "location": {"lat": lat, "lng": lng},
            "image_url": f"/static/uploads/{filename}"
        }

        disease_collection.insert_one(report)
        return jsonify({"message": "Disease report uploaded successfully!"})
    except Exception as e:
        print("Upload error:", e)
        return jsonify({"message": "Error uploading report"}), 500

@app.route("/get_diseases")
def get_diseases():
    try:
        reports = list(disease_collection.find({}, {"_id": 0}))
        return jsonify(reports)
    except Exception as e:
        print("Fetch error:", e)
        return jsonify([])

# ---------------- Marketplace ----------------
@app.route("/sellandbuy")
def marketplace():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("sellandbuy.html", username=session["username"], user_id=session["user_id"])

@app.route("/post_product", methods=["POST"])
def post_product():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.form
    product = {
        "name": data.get("name"),
        "quantity": data.get("quantity"),
        "price": data.get("price"),
        "location": data.get("location"),
        "harvest_date": data.get("harvest_date"),
        "contact": data.get("contact"),
        "description": data.get("description"),
        "user_id": session["user_id"],
        "username": session["username"]
    }
    if "image" in request.files:
        image = request.files["image"]
        filename = f"static/uploads/{image.filename}"
        os.makedirs("static/uploads", exist_ok=True)
        image.save(filename)
        product["image"] = "/" + filename

    products_collection.insert_one(product)
    return jsonify({"message": "Product posted successfully!"})

@app.route("/products")
def get_products():
    products = list(products_collection.find({}, {"_id": 0}))
    return jsonify(products)

@app.route("/my_products")
def get_my_products():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    products = list(products_collection.find({"user_id": session["user_id"]}, {"_id": 0}))
    return jsonify(products)

# ---------------- Status ----------------
@app.route("/status")
def status():
    return jsonify({"message": "Crop Advisor Backend with MongoDB is running!"})

# ---------------- Chat ----------------
@app.route("/chat")
def chat():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("community.html", username=session["username"])

@app.route("/get_messages")
def get_messages():
    messages = list(messages_collection.find({}, {"username":1, "text":1, "timestamp":1}))
    for m in messages:
        m["_id"] = str(m["_id"])
    return jsonify(messages)

@app.route("/send_message", methods=["POST"])
def send_message():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "Message is empty"}), 400
    msg = {
        "username": session["username"],
        "text": text,
        "timestamp": datetime.utcnow()
    }
    inserted = messages_collection.insert_one(msg)
    msg["_id"] = str(inserted.inserted_id)
    return jsonify(msg)

@app.route("/edit_message/<msg_id>", methods=["POST"])
def edit_message(msg_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    msg = messages_collection.find_one({"_id": ObjectId(msg_id)})
    if not msg:
        return jsonify({"error": "Message not found"}), 404
    if msg["username"] != session["username"]:
        return jsonify({"error": "You can only edit your own messages"}), 403
    new_text = request.json.get("text")
    messages_collection.update_one({"_id": ObjectId(msg_id)}, {"$set": {"text": new_text}})
    return jsonify({"message": "Message updated"})

@app.route("/delete_message/<msg_id>", methods=["DELETE"])
def delete_message(msg_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    msg = messages_collection.find_one({"_id": ObjectId(msg_id)})
    if not msg:
        return jsonify({"error": "Message not found"}), 404
    if msg["username"] != session["username"]:
        return jsonify({"error": "You can only delete your own messages"}), 403
    messages_collection.delete_one({"_id": ObjectId(msg_id)})
    return jsonify({"message": "Message deleted"})

# ---------------- Weather ----------------
@app.route("/weather")
def weather_page():
    if "user_id" not in session:
        return redirect("/login")
    user = users_collection.find_one({"_id": ObjectId(session["user_id"])})
    city = user.get("city", "Delhi")
    return render_template("weather.html", city=city)

# ---------------- Crop Tracker ----------------
@app.route("/investmentandloss")
def investment_and_loss():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("investmentandloss.html")

@app.route("/add_crop", methods=["POST"])
def add_crop():
    data = request.json
    user_id = data.get("userId")
    crop_name = data.get("name")
    sow_date = data.get("sowDate")
    season_no = data.get("seasonNo")  # <-- new field from frontend
    
    crop = {
        "userId": user_id,
        "name": crop_name,
        "sowDate": sow_date,
        "seasonNo": season_no,  # <-- added column
        "expenses": [],
        "totalInvestment": 0,
        "harvestDate": "",
        "harvestQty": 0,
        "income": None,
        "finished": False
    }
    crops_collection.insert_one(crop)
    return jsonify({"message": "Crop added successfully"})

@app.route("/calender")
def calendar_page():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("calender.html")  # New template

@app.route("/get_crops/<user_id>")
def get_crops(user_id):
    crops = list(crops_collection.find({"userId": user_id}, {"_id": 0}))
    return jsonify(crops)

@app.route("/update_crop/<crop_name>", methods=["POST"])
def update_crop(crop_name):
    data = request.json
    user_id = data.get("userId")
    crop = crops_collection.find_one({"userId": user_id, "name": crop_name})
    if not crop:
        return jsonify({"message": "Crop not found"}), 404
    if "expenses" in data:
        crop["expenses"] = data["expenses"]
        crop["totalInvestment"] = sum(e["amount"] for e in data["expenses"])
    if "harvestDate" in data:
        crop["harvestDate"] = data["harvestDate"]
        crop["harvestQty"] = data["harvestQty"]
        crop["income"] = data["income"]
        crop["finished"] = True
    crops_collection.update_one({"userId": user_id, "name": crop_name}, {"$set": crop})
    return jsonify({"message": "Crop updated successfully"})

# ---------------- Main Page ----------------
@app.route("/mainpage")
def mainpage():
    if "user_id" not in session:
        return redirect("/login")

    user = users_collection.find_one({"_id": ObjectId(session["user_id"])})
    city = user.get("city", "Delhi")  # fallback if profile city not set
    username = user.get("username", "Guest")

    return render_template("mainpage.html", city=city, username=username)

# ---------------- Crop Calendar & Notifications ----------------
from datetime import timedelta

# MongoDB setup for crop calendar tasks
calendar_client = MongoClient("MONGO_URI")
calendar_db = calendar_client.cropwise_calendar
tasks_collection = calendar_db.tasks

# Load JSON dataset for crops/tasks
with open("final_merged_crops.json", "r", encoding="utf-8") as f:
    crop_data = json.load(f)

# Helper to fetch crop info
def get_crop_info(crop_name):
    for state in crop_data["states"]:
        for district in state["districts"]:
            for crop in district["crops"]:
                if crop["name"].lower() == crop_name.lower():
                    return crop
    return None

@app.route('/generate', methods=['POST'])
def generate():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    crop_name = data.get('crop')
    sowing_date_str = data.get('sowing_date')

    if not crop_name or not sowing_date_str:
        return jsonify({'error': 'Missing crop or sowing_date'}), 400

    try:
        sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    crop_info = get_crop_info(crop_name)
    if not crop_info:
        return jsonify({'error': f'Crop {crop_name} not found'}), 404

    crop_name = crop_info["name"]
    tasks = []
    harvest_date = sowing_date + timedelta(days=90)

    for t in crop_info.get("tasks", []):
        task = {
            "user_id": session["user_id"],
            "crop": crop_name,
            "stage": t["stage"],
            "task": t["task"]
        }
        if "days_after_sowing" in t:
            task["date"] = (sowing_date + timedelta(days=t["days_after_sowing"])).strftime("%Y-%m-%d")
        elif "days_before_harvest" in t:
            task["date"] = (harvest_date - timedelta(days=t["days_before_harvest"])).strftime("%Y-%m-%d")
        elif "days_after_harvest" in t:
            task["date"] = (harvest_date + timedelta(days=t["days_after_harvest"])).strftime("%Y-%m-%d")
        else:
            task["date"] = sowing_date_str
        tasks.append(task)

    # Save tasks in MongoDB (per user)
    for task in tasks:
        tasks_collection.update_one(
            {"user_id": session["user_id"], "crop": crop_name, "task": task["task"], "date": task["date"]},
            {"$set": task},
            upsert=True
        )

    # Notifications
    notifications = crop_info.get("advisories", [])
    fertilizers = crop_info.get("fertilizers", [])
    pests = crop_info.get("pests_diseases", [])
    seasons = [crop_info.get("season", "Unknown")]

    today_str = datetime.now().strftime("%Y-%m-%d")
    for t in tasks:
        if t["date"] == today_str:
            notifications.append(f"Today: {t['task']} ({t['stage']})")
        elif datetime.strptime(t["date"], "%Y-%m-%d") > datetime.now():
            notifications.append(f"Upcoming ({t['date']}): {t['task']} ({t['stage']})")

    tasks_header = f"{crop_name} - Sowing Date: {sowing_date_str}"

    return jsonify({
        "tasks_header": tasks_header,
        "tasks": tasks,
        "seasons": seasons,
        "notifications": notifications,
        "fertilizers": fertilizers,
        "pests_diseases": pests
    })

@app.route('/all-tasks', methods=['GET'])
def all_tasks():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    tasks = list(tasks_collection.find({"user_id": session["user_id"]}, {"_id": 0}))
    return jsonify(tasks)

@app.route('/add-task', methods=['POST'])
def add_task():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    if not all(k in data for k in ("crop", "task", "stage", "date")):
        return jsonify({"error": "Missing fields"}), 400
    data["user_id"] = session["user_id"]
    tasks_collection.insert_one(data)
    return jsonify({"message": "Task added"}), 201

@app.route('/update-task', methods=['POST'])
def update_task():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    crop = data.get("crop")
    task_name = data.get("task")
    date = data.get("date")
    new_date = data.get("new_date")

    if not all([crop, task_name, date, new_date]):
        return jsonify({"error": "Missing data"}), 400

    result = tasks_collection.update_one(
        {"user_id": session["user_id"], "crop": crop, "task": task_name, "date": date},
        {"$set": {"date": new_date}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Task not found or unchanged"}), 404

    return jsonify({"message": "Task updated"})

@app.route('/delete-task', methods=['POST'])
def delete_task():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    crop = data.get("crop")
    task_name = data.get("task")
    date = data.get("date")

    if not all([crop, task_name, date]):
        return jsonify({"error": "Missing data"}), 400

    result = tasks_collection.delete_one(
        {"user_id": session["user_id"], "crop": crop, "task": task_name, "date": date}
    )

    if result.deleted_count == 0:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({"message": "Task deleted"})
@app.route('/notifications', methods=['GET'])
def get_notifications():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    # Fetch all tasks for the user
    tasks = list(tasks_collection.find({"user_id": session["user_id"]}, {"_id": 0}))

    today = datetime.now().date()
    time_labels = ["Morning", "Afternoon", "Evening"]
    notifications = []

    for t in tasks:
        task_date = datetime.strptime(t["date"], "%Y-%m-%d").date()
        days_diff = (task_date - today).days

        if days_diff in [0, 1, 2]:  # today, 1 day ahead, 2 days ahead
            day_label = "Today" if days_diff == 0 else f"In {days_diff} day(s)"
            for time_label in time_labels:
                notifications.append(f"🔔 {day_label} ({time_label}): {t['task']} ({t['stage']})")


    # Optional: sort notifications by date
    notifications.sort()

    return jsonify({"notifications": notifications})
@app.route('/notifications-page')
def notifications_page():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("notifications.html")
@app.route('/cropRotation')
def rotation_page():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("cropRotation.html")

if __name__ == "__main__":
    app.run()
