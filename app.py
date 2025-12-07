from flask import Flask, render_template, request, jsonify
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)  # templates/index.html di folder yang sama

# =======================
# STATE DUMMY (SENSOR & IRIGASI)
# =======================
sensor_state = {
    "soil_moisture": 55.0,
    "temperature": 27.0,
    "humidity": 65.0,
    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

irrigation_state = {
    "is_active": False,
    "end_time": None
}

# =======================
# RIWAYAT PENYIRAMAN / PERHITUNGAN
# =======================
history = []  # list of dict {timestamp, soil_moisture, temperature, humidity, duration, category}
MAX_HISTORY = 200  # batasi jumlah data riwayat agar tidak membengkak

# =======================
# FUNGSI MEMBERSHIP & FUZZY
# =======================

def trap(x, a, b, c, d):
    x = float(x)
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0

def tri(x, a, b, c):
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    if x == b:
        return 1.0
    return 0.0

# Soil moisture (0-100)
def soil_dry(x):
    return trap(x, 0, 0, 25, 45)

def soil_normal(x):
    return tri(x, 30, 50, 70)

def soil_wet(x):
    return trap(x, 55, 75, 100, 100)

# Temperature (°C)
def temp_cold(x):
    return trap(x, 0, 0, 12, 20)

def temp_normal(x):
    return tri(x, 15, 25, 33)

def temp_hot(x):
    return trap(x, 28, 34, 50, 50)

# Air humidity (0-100)
def hum_low(x):
    return trap(x, 0, 0, 25, 45)

def hum_normal(x):
    return tri(x, 30, 50, 70)

def hum_high(x):
    return trap(x, 55, 75, 100, 100)

# Output duration membership functions (0-60 minutes)
def dur_short(x):
    return trap(x, 0, 0, 10, 20)

def dur_medium(x):
    return tri(x, 15, 30, 45)

def dur_long(x):
    return trap(x, 35, 45, 60, 60)

def evaluate_rules(soil_val, temp_val, hum_val):
    s_dry  = soil_dry(soil_val)
    s_norm = soil_normal(soil_val)
    s_wet  = soil_wet(soil_val)

    t_cold = temp_cold(temp_val)
    t_norm = temp_normal(temp_val)
    t_hot  = temp_hot(temp_val)

    h_low  = hum_low(hum_val)
    h_norm = hum_normal(hum_val)
    h_high = hum_high(hum_val)

    x_out = np.linspace(0, 60, 601)
    aggregated = np.zeros_like(x_out)

    def apply_rule(strength, consequent_fn):
        if strength <= 0:
            return np.zeros_like(x_out)
        vals = np.array([consequent_fn(x) for x in x_out])
        return np.minimum(vals, strength)

    # RULES
    r1 = apply_rule(s_dry, dur_long)
    aggregated = np.maximum(aggregated, r1)

    r2 = apply_rule(s_norm, dur_medium)
    aggregated = np.maximum(aggregated, r2)

    r3 = apply_rule(s_wet, dur_short)
    aggregated = np.maximum(aggregated, r3)

    r4_strength = min(t_hot, s_norm)
    r4 = apply_rule(r4_strength, dur_long)
    aggregated = np.maximum(aggregated, r4)

    r5_strength = min(h_low, s_norm)
    r5 = apply_rule(r5_strength, dur_long)
    aggregated = np.maximum(aggregated, r5)

    r6_strength = min(t_cold, s_norm)
    r6 = apply_rule(r6_strength, dur_short)
    aggregated = np.maximum(aggregated, r6)

    r7_strength = min(h_high, s_norm)
    r7 = apply_rule(r7_strength, dur_short)
    aggregated = np.maximum(aggregated, r7)

    r8_strength = min(s_dry, t_hot, h_low)
    r8 = apply_rule(r8_strength, dur_long)
    aggregated = np.maximum(aggregated, r8)

    r9_strength = min(s_wet, max(t_cold, h_high))
    r9 = apply_rule(r9_strength, dur_short)
    aggregated = np.maximum(aggregated, r9)

    return x_out, aggregated

def defuzzify_centroid(x, mu):
    numerator = np.trapz(x * mu, x)
    denominator = np.trapz(mu, x)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def duration_category(d):
    if d <= 0.01:
        return "Tidak perlu disiram (OFF)"
    if d <= 20:
        return "Siram Sedikit (ON)"
    if d <= 40:
        return "Siram Sedang (ON)"
    return "Siram Banyak (ON)"

# =======================
# ROUTE HALAMAN UTAMA
# =======================

@app.route('/')
def index():
    return render_template('index.html')

# =======================
# API UNTUK JAVASCRIPT (index.html)
# =======================

@app.route('/api/sensor-data', methods=['GET'])
def api_sensor_data():
    # kirim nilai sensor_state ke frontend
    return jsonify({
        "soil_moisture": float(sensor_state["soil_moisture"]),
        "temperature": float(sensor_state["temperature"]),
        "humidity": float(sensor_state["humidity"]),
        "last_update": sensor_state["last_update"]
    })

@app.route('/api/calculate-fuzzy', methods=['POST'])
def api_calculate_fuzzy():
    data = request.get_json() or {}

    soil = float(data.get('soil_moisture', 50))
    temp = float(data.get('temperature', 25))
    hum  = float(data.get('humidity', 50))

    # sinkronkan input manual ke "sensor"
    sensor_state["soil_moisture"] = soil
    sensor_state["temperature"] = temp
    sensor_state["humidity"] = hum
    sensor_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Hitung fuzzy
    x_out, aggregated = evaluate_rules(soil, temp, hum)
    duration = defuzzify_centroid(x_out, aggregated)
    duration = float(np.round(duration, 2))

    category = duration_category(duration)
    recommendation = f"Rekomendasi: {category} dengan durasi sekitar {duration} menit."

    # ====== SIMPAN KE RIWAYAT ======
    record = {
        "timestamp": sensor_state["last_update"],
        "soil_moisture": soil,
        "temperature": temp,
        "humidity": hum,
        "duration": duration,
        "category": category
    }
    history.append(record)
    # batasi max panjang riwayat
    if len(history) > MAX_HISTORY:
        history.pop(0)
    # ===============================

    return jsonify({
        "duration": duration,
        "category": category,
        "recommendation": recommendation
    })

@app.route('/api/start-irrigation', methods=['POST'])
def api_start_irrigation():
    data = request.get_json() or {}
    duration = float(data.get('duration', 0))

    if duration <= 0:
        return jsonify({"message": "Durasi tidak valid!"}), 400

    irrigation_state["is_active"] = True
    irrigation_state["end_time"] = datetime.now() + timedelta(minutes=duration)

    return jsonify({"message": f"Irigasi dimulai selama {duration} menit."})

@app.route('/api/stop-irrigation', methods=['POST'])
def api_stop_irrigation():
    irrigation_state["is_active"] = False
    irrigation_state["end_time"] = None
    return jsonify({"message": "Irigasi dihentikan."})

@app.route('/api/irrigation-status', methods=['GET'])
def api_irrigation_status():
    is_active = irrigation_state["is_active"]
    remaining_time = 0.0

    if is_active and irrigation_state["end_time"] is not None:
        now = datetime.now()
        diff = (irrigation_state["end_time"] - now).total_seconds() / 60.0
        if diff <= 0:
            irrigation_state["is_active"] = False
            irrigation_state["end_time"] = None
            is_active = False
            remaining_time = 0.0
        else:
            remaining_time = diff

    return jsonify({
        "is_active": is_active,
        "remaining_time": remaining_time
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    """
    Mengirim seluruh riwayat perhitungan fuzzy / penyiraman
    """
    return jsonify(history)

# =======================
# MAIN
# =======================
if __name__ == '__main__':
    app.run(debug=True)
