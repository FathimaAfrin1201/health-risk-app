import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load machine learning models with caching and error handling
@st.cache_resource
def load_ml_assets():
    try:
        with open('models/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return model, encoders, target_encoder
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file missing! Please ensure the 'models/' directory contains 'xgb_model.pkl', 'encoders.pkl', and 'target_encoder.pkl'.\n\nError details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the machine learning models: {e}")
        st.stop()

model, encoders, target_encoder = load_ml_assets()

# Stylesheet Injection
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(circle at 10% -10%, #f1f4ff 0%, #f6f8fc 40%, #fbfcfe 80%, #f7f9fc 100%) !important;
    }

    /* Hide Streamlit default components */
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container { 
        padding-top: 1.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1280px !important; 
    }

    /* Custom styled container wrapper for border containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.02) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
    }

    /* Form headers styling */
    .form-header {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* Beautiful Hero Header */
    .brand-hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 24px;
        padding: 2.2rem 2.5rem;
        color: white;
        box-shadow: 0 20px 40px -15px rgba(49, 46, 129, 0.3);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }

    .brand-hero::after {
        content: '';
        position: absolute;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        top: -50px;
        right: -50px;
        border-radius: 50%;
        pointer-events: none;
    }

    .brand-hero h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        margin: 0 !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
    }

    .brand-hero p {
        font-size: 1.02rem !important;
        font-weight: 400 !important;
        opacity: 0.88 !important;
        margin: 0.5rem 0 1rem 0 !important;
        max-width: 650px;
        line-height: 1.4 !important;
        color: #e0e7ff !important;
    }

    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .glow-badge {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e0e7ff !important;
        padding: 0.3rem 0.8rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        backdrop-filter: blur(4px);
    }

    /* Predict Button styling */
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
        height: 3.2rem !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    }

    [data-testid="stBaseButton-primary"]:active {
        transform: translateY(1px) !important;
    }

    /* Streamlit label styling */
    div[data-testid="stSlider"] > label, 
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Result cards animations */
    .result-card-container {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-bottom: 1.2rem;
    }

    .metric-item {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(226, 232, 240, 0.7);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.01);
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .metric-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }

    .metric-details h4 {
        margin: 0 !important;
        font-size: 0.75rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-details p {
        margin: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Status backgrounds */
    .bg-fit { background: rgba(16, 185, 129, 0.1) !important; color: #059669 !important; }
    .bg-risk { background: rgba(245, 158, 11, 0.1) !important; color: #d97706 !important; }
    .bg-unhealthy { background: rgba(239, 68, 68, 0.1) !important; color: #dc2626 !important; }
    .bg-info { background: rgba(59, 130, 246, 0.1) !important; color: #2563eb !important; }

    /* Large Diagnosis Card */
    .diag-card {
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }

    .diag-card-fit {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 15px 30px rgba(16, 185, 129, 0.25);
    }

    .diag-card-risk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 15px 30px rgba(245, 158, 11, 0.25);
    }

    .diag-card-unhealthy {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 15px 30px rgba(239, 68, 68, 0.25);
    }

    .diag-card h2 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        margin: 0 0 0.4rem 0 !important;
        color: white !important;
    }

    .diag-card p {
        font-size: 1rem !important;
        opacity: 0.9 !important;
        margin: 0 !important;
        color: white !important;
    }

    .diag-badge {
        position: absolute;
        top: 1.2rem;
        right: 1.2rem;
        font-size: 2.2rem;
        opacity: 0.85;
    }

    /* Recommendations list */
    .rec-card {
        background: white;
        border-radius: 18px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.01);
        padding: 1.2rem;
    }

    .rec-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: #0f172a !important;
        margin-bottom: 0.8rem !important;
    }

    .rec-item {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #6366f1;
        background: #f8fafc;
        display: flex;
        gap: 0.7rem;
    }

    .rec-item-orange { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.02); }
    .rec-item-red { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.02); }
    .rec-item-blue { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.02); }
    .rec-item-green { border-left-color: #10b981; background: rgba(16, 185, 129, 0.02); }

    .rec-icon {
        font-size: 1.1rem;
        line-height: 1.2;
    }

    .rec-content h5 {
        margin: 0 0 0.1rem 0 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        color: #1e293b !important;
    }

    .rec-content p {
        margin: 0 !important;
        font-size: 0.82rem !important;
        color: #475569 !important;
        line-height: 1.35 !important;
    }

    /* Empty State Styling */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(148, 163, 184, 0.3);
        border-radius: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 380px;
    }

    .pulse-circle {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        color: #4338ca;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.3);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.5);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 12px rgba(99, 102, 241, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(99, 102, 241, 0);
        }
    }

    .empty-state h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        margin: 0 0 0.4rem 0 !important;
    }

    .empty-state p {
        font-size: 0.88rem !important;
        color: #64748b !important;
        max-width: 300px !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }

    /* Sandbox Label Header */
    .sandbox-container {
        background: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
    }

    .label-small {
        font-size: 0.75rem;
        font-weight: 700;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .footnote { 
        text-align: center; 
        color: #94a3b8; 
        font-size: 0.78rem; 
        margin-top: 2.5rem; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Patient presets mock data definitions
presets = {
    "None (Use Custom Inputs)": {},
    "🏃 The Healthy Athlete": {
        "bmi": 21.5, "heart_rate": 62, "gender": "female",
        "step_count": 12500, "exercise_duration": 60, "physical_activity_level": "active",
        "calorie_expenditure": 2600, "water_intake": 2.8,
        "sleep_duration": 8.0, "sleep_quality": "good", "stress_level": "low",
        "diet_type": "balanced", "smoking_alcohol": "no"
    },
    "📚 The Stressed Student": {
        "bmi": 24.0, "heart_rate": 88, "gender": "male",
        "step_count": 4200, "exercise_duration": 10, "physical_activity_level": "sedentary",
        "calorie_expenditure": 1800, "water_intake": 1.2,
        "sleep_duration": 5.0, "sleep_quality": "poor", "stress_level": "high",
        "diet_type": "non-veg", "smoking_alcohol": "occasional"
    },
    "🛋️ The Sedentary Gamer": {
        "bmi": 29.5, "heart_rate": 82, "gender": "other",
        "step_count": 2500, "exercise_duration": 0, "physical_activity_level": "sedentary",
        "calorie_expenditure": 1600, "water_intake": 0.8,
        "sleep_duration": 6.0, "sleep_quality": "average", "stress_level": "medium",
        "diet_type": "non-veg", "smoking_alcohol": "yes"
    }
}

# Initializing Session States
default_values = {
    "bmi": 22.0, "heart_rate": 75, "gender": "male",
    "step_count": 8000, "exercise_duration": 30, "physical_activity_level": "moderate",
    "calorie_expenditure": 2200, "water_intake": 2.0,
    "sleep_duration": 7.0, "sleep_quality": "average", "stress_level": "medium",
    "diet_type": "balanced", "smoking_alcohol": "no"
}

for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'selected_preset' not in st.session_state:
    st.session_state['selected_preset'] = "None (Use Custom Inputs)"

# Update session state on selectbox change
def handle_preset_change():
    preset_name = st.session_state['preset_select']
    st.session_state['selected_preset'] = preset_name
    if preset_name in presets and presets[preset_name]:
        for k, v in presets[preset_name].items():
            st.session_state[k] = v

# Header
st.markdown(
    """
    <div class="brand-hero">
        <h1>🩺 Student Health Risk Predictor</h1>
        <p>Analyze vitals, physical activity metrics, sleep quality, and dietary choices to predict risk profiles using clinical machine learning algorithms.</p>
        <div class="badge-row">
            <span class="glow-badge">⚡ XGBoost Engine</span>
            <span class="glow-badge">🎯 96.6% Validation Accuracy</span>
            <span class="glow-badge">🛡️ Clinical Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sandbox Presets selector
with st.container():
    st.markdown('<div class="label-small">⚡ Interactive Sandbox Presets</div>', unsafe_allow_html=True)
    st.selectbox(
        "Load student lifestyle presets to see how parameters shape risk trends:",
        options=list(presets.keys()),
        key="preset_select",
        index=list(presets.keys()).index(st.session_state['selected_preset']),
        on_change=handle_preset_change,
        label_visibility="collapsed"
    )

st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

# Main layout split
col_input, col_output = st.columns([1.1, 0.9])

with col_input:
    # 📏 Body & Vitals
    with st.container(border=True):
        st.markdown('<div class="form-header">📏 Body & Vitals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.slider("BMI", 10.0, 45.0, key="bmi", step=0.5)
        with c2:
            heart_rate = st.slider("Heart rate (bpm)", 40, 150, key="heart_rate")
        with c3:
            gender = st.selectbox("Gender", ['male', 'female', 'other'], key="gender")

    # 🏃 Activity & Energy
    with st.container(border=True):
        st.markdown('<div class="form-header">🏃 Activity & Energy</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            step_count = st.number_input("Daily step count", min_value=0, max_value=30000, key="step_count", step=100)
        with c2:
            exercise_duration = st.slider("Exercise duration (mins)", 0, 180, key="exercise_duration")
        with c3:
            physical_activity_level = st.selectbox("Activity level", ['sedentary', 'moderate', 'active'], key="physical_activity_level")
        c4, c5 = st.columns(2)
        with c4:
            calorie_expenditure = st.number_input("Calorie expenditure (kcal)", min_value=500, max_value=5000, key="calorie_expenditure", step=50)
        with c5:
            water_intake = st.slider("Water intake (litres)", 0.0, 5.0, key="water_intake", step=0.1)

    # 😴 Sleep & Wellbeing
    with st.container(border=True):
        st.markdown('<div class="form-header">😴 Sleep & Wellbeing</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep_duration = st.slider("Sleep duration (hours)", 0.0, 12.0, key="sleep_duration", step=0.5)
        with c2:
            sleep_quality = st.selectbox("Sleep quality", ['poor', 'average', 'good'], key="sleep_quality")
        with c3:
            stress_level = st.selectbox("Stress level", ['low', 'medium', 'high'], key="stress_level")

    # 🍽️ Diet & Habits
    with st.container(border=True):
        st.markdown('<div class="form-header">🍽️ Diet & Habits</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            diet_type = st.selectbox("Diet type", ['veg', 'non-veg', 'balanced'], key="diet_type")
        with c2:
            smoking_alcohol = st.selectbox("Smoking/alcohol usage", ['no', 'occasional', 'yes'], key="smoking_alcohol")

    predict_clicked = st.button("✨ Predict Health Risk Profile", type="primary", use_container_width=True)

with col_output:
    if not predict_clicked:
        st.markdown(
            """
            <div class="empty-state">
                <div class="pulse-circle">🩺</div>
                <h3>Diagnostic System Idle</h3>
                <p>Configure vitals and metrics in the input panels, then click the prediction button to compile health risks.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Prepare data frame for prediction
        input_data = pd.DataFrame({
            'sleep_duration': [sleep_duration],
            'heart_rate': [heart_rate],
            'bmi': [bmi],
            'calorie_expenditure': [calorie_expenditure],
            'step_count': [step_count],
            'exercise_duration': [exercise_duration],
            'water_intake': [water_intake],
            'diet_type': [encoders['diet_type'].transform([diet_type])[0]],
            'stress_level': [encoders['stress_level'].transform([stress_level])[0]],
            'sleep_quality': [encoders['sleep_quality'].transform([sleep_quality])[0]],
            'physical_activity_level': [encoders['physical_activity_level'].transform([physical_activity_level])[0]],
            'smoking_alcohol': [encoders['smoking_alcohol'].transform([smoking_alcohol])[0]],
            'gender': [encoders['gender'].transform([gender])[0]],
        })

        probs = model.predict_proba(input_data)[0]
        pred_class = probs.argmax()
        pred_label = target_encoder.inverse_transform([pred_class])[0]
        confidence = probs[pred_class]

        # Diagnosis mapping
        style_map = {
            "fit": ("diag-card-fit", "🟢", "Fit / Healthy", "Your inputs suggest excellent physical conditioning, balanced sleep, and robust activity levels. Keep up these routines!", "bg-fit"),
            "at-risk": ("diag-card-risk", "🟡", "At-Risk Profile", "Some of your lifestyle indicators suggest potential strain. Minor wellness corrections can yield significant positive shifts.", "bg-risk"),
            "unhealthy": ("diag-card-unhealthy", "🔴", "Unhealthy Profile", "Several indicators display clinical strain. We strongly suggest seeking professional guidance to outline positive changes.", "bg-unhealthy"),
        }
        css_class, emoji, title, blurb, metric_class = style_map.get(pred_label, ("diag-card-risk", "🟡", pred_label.title(), "", "bg-risk"))

        st.markdown(f'<div class="result-card-container">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="diag-card {css_class}">
                <div class="diag-badge">{emoji}</div>
                <h2>{title}</h2>
                <p>Confidence rating: <b>{confidence*100:.1f}%</b></p>
                <div style="margin-top: 0.8rem; font-size: 0.9rem; line-height: 1.4; opacity: 0.95;">{blurb}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Mini Dashboard Indicators
        # Water: >=2L is Fit/Good
        water_status = "Hydrated" if water_intake >= 2.0 else "Dehydrated"
        water_bg = "bg-fit" if water_intake >= 2.0 else "bg-unhealthy"
        
        # Steps: >=8000 is Fit, >=5000 is Mod/At-Risk, <5000 is Sed/Unhealthy
        if step_count >= 8000:
            step_status, step_bg = "Highly Active", "bg-fit"
        elif step_count >= 5000:
            step_status, step_bg = "Moderately Active", "bg-risk"
        else:
            step_status, step_bg = "Sedentary", "bg-unhealthy"
            
        # Sleep: >=7h is Good, >=6h is Average, <6h is Poor
        if sleep_duration >= 7.0:
            sleep_status, sleep_bg = "Rested", "bg-fit"
        elif sleep_duration >= 6.0:
            sleep_status, sleep_bg = "Fatigued", "bg-risk"
        else:
            sleep_status, sleep_bg = "Sleep Deprived", "bg-unhealthy"
            
        # Stress indicators
        stress_bg = "bg-fit" if stress_level == "low" else ("bg-risk" if stress_level == "medium" else "bg-unhealthy")
        stress_status = f"{stress_level.title()} Stress"

        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-icon-box {water_bg}">💧</div>
                    <div class="metric-details">
                        <h4>Hydration</h4>
                        <p>{water_intake:.1f}L &middot; {water_status}</p>
                    </div>
                </div>
                <div class="metric-item">
                    <div class="metric-icon-box {step_bg}">🏃</div>
                    <div class="metric-details">
                        <h4>Movement</h4>
                        <p>{step_count:,} steps</p>
                    </div>
                </div>
                <div class="metric-item">
                    <div class="metric-icon-box {sleep_bg}">😴</div>
                    <div class="metric-details">
                        <h4>Sleep</h4>
                        <p>{sleep_duration:.1f}h &middot; {sleep_status}</p>
                    </div>
                </div>
                <div class="metric-item">
                    <div class="metric-icon-box {stress_bg}">🧘</div>
                    <div class="metric-details">
                        <h4>Wellness</h4>
                        <p>{stress_status}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Plotly probability bar chart styled beautifully
        prob_df = pd.DataFrame({
            "Class": [c.title() for c in target_encoder.classes_],
            "Probability": probs,
        }).sort_values("Probability", ascending=True)

        color_lookup = {"Fit": "#10b981", "At-Risk": "#f59e0b", "Unhealthy": "#ef4444"}
        bar_colors = [color_lookup.get(c, "#6366f1") for c in prob_df["Class"]]

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"], y=prob_df["Class"], orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(255,255,255,0.6)', width=1.5)
            ),
            text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
            textposition="outside",
            width=0.45
        ))
        
        fig.update_layout(
            height=160, 
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=False, showticklabels=False),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color="#1e293b", size=12, family="Plus Jakarta Sans, sans-serif")
            ),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#475569"),
        )
        
        st.markdown('<div class="label-small">📊 Prediction Probability Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Dynamically compile clinical/lifestyle advice
        recommendations = []
        
        if bmi > 25.0:
            recommendations.append({
                "class": "rec-item-orange",
                "icon": "⚖️",
                "title": "Weight Management Guidance",
                "text": f"Your current BMI of {bmi} falls outside the healthy baseline. Incorporate physical activity and review nutritional structure."
            })
        elif bmi < 18.5:
            recommendations.append({
                "class": "rec-item-orange",
                "icon": "⚖️",
                "title": "Nutritional Support",
                "text": f"Your current BMI of {bmi} is lower than typical health standards. Prioritize calorie-dense balanced meals."
            })

        if step_count < 6000:
            recommendations.append({
                "class": "rec-item-red",
                "icon": "🚶",
                "title": "Low Daily Step Goal",
                "text": f"You logged {step_count:,} steps. Aim for at least 8,000 steps daily. Take active micro-breaks during study blocks."
            })

        if water_intake < 2.0:
            recommendations.append({
                "class": "rec-item-blue",
                "icon": "💧",
                "title": "Increase Daily Hydration",
                "text": f"You logged {water_intake}L. Aim to consume at least 2.0L of water daily to support physiological functions."
            })

        if sleep_duration < 6.5:
            recommendations.append({
                "class": "rec-item-red",
                "icon": "🌙",
                "title": "Optimize Sleep Schedule",
                "text": f"Getting {sleep_duration} hours of sleep limits recovery. Target 7-8 hours nightly and shut off screens 30 mins before sleep."
            })

        if stress_level == "high":
            recommendations.append({
                "class": "rec-item-orange",
                "icon": "🧘",
                "title": "Stress & Wellness Support",
                "text": "Reported stress is high. Integrate 5-minute breathing intervals and short physical activity breaks."
            })

        if not recommendations:
            recommendations.append({
                "class": "rec-item-green",
                "icon": "🏆",
                "title": "All Metrics Optimal",
                "text": "Excellent! All of your input variables fall within optimal lifestyle boundaries. Keep up these routines!"
            })

        # Print recommendations
        rec_html = "".join([
            f'<div class="rec-item {r["class"]}">'
            f'<div class="rec-icon">{r["icon"]}</div>'
            f'<div class="rec-content">'
            f'<h5>{r["title"]}</h5>'
            f'<p>{r["text"]}</p>'
            f'</div>'
            f'</div>'
            for r in recommendations
        ])

        st.markdown(
            f'<div class="rec-card">'
            f'<div class="rec-title">📋 Personalized Wellness Suggestions</div>'
            f'{rec_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(f'</div>', unsafe_allow_html=True) # End of result container

st.markdown(
    '<div class="footnote">🩺 Student Wellness Diagnostic Engine &middot; This tool estimates statistical risk and does not constitute medical advice.</div>',
    unsafe_allow_html=True,
)
