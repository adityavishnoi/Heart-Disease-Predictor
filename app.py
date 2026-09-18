import streamlit as st
import pandas as pd
import pickle

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(239, 68, 68, 0.14), transparent 25%),
        radial-gradient(circle at 90% 10%, rgba(59, 130, 246, 0.12), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(16, 185, 129, 0.10), transparent 30%),
        #070b14;
    color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ================= HERO ================= */

.hero {
    text-align: center;
    padding: 42px 20px;
    border-radius: 28px;

    background: linear-gradient(
        135deg,
        rgba(239,68,68,0.18),
        rgba(59,130,246,0.13),
        rgba(16,185,129,0.10)
    );

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35);

    margin-bottom: 30px;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #f87171,
        #60a5fa,
        #34d399
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
    margin-top: 8px;
}

/* ================= SECTIONS ================= */

.section-title {
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 4px;
}

.section-description {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* ================= INPUT HELP ================= */

.input-help {
    color: #94a3b8;
    font-size: 0.78rem;
    line-height: 1.45;
    margin-top: -8px;
    margin-bottom: 10px;
}

/* ================= RESULT ================= */

.result-box {
    margin-top: 30px;
    padding: 42px 20px;
    text-align: center;

    border-radius: 25px;

    background: linear-gradient(
        135deg,
        rgba(239,68,68,0.20),
        rgba(59,130,246,0.14),
        rgba(16,185,129,0.12)
    );

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 15px 50px rgba(0,0,0,0.30);
}

.result-title {
    color: #cbd5e1;
    font-size: 1rem;
}

.result-value {
    font-size: 2.7rem;
    font-weight: 800;
    margin-top: 8px;
}

/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;
    padding: 15px;

    border-radius: 15px;
    border: none;

    color: white;

    font-size: 1rem;
    font-weight: 700;

    background: linear-gradient(
        90deg,
        #dc2626,
        #2563eb,
        #059669
    );

    box-shadow:
        0 10px 30px rgba(37,99,235,0.25);

    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 15px 40px rgba(37,99,235,0.4);
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b1020,
        #111827
    );

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 45px;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_models():

    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return knn, scaler


knn, scaler = load_models()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='text-align:center;'>❤️ Heart Health</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='color:#94a3b8; text-align:center; line-height:1.6;'>
        Enter the patient's information and use the
        trained KNN model to generate a prediction.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model Information")

    st.write("**Algorithm:** KNN Classification")
    st.write("**Scaling:** StandardScaler")
    st.write("**Features Used:** 9")

    st.markdown("---")

    st.markdown("### 📌 Model Features")

    st.markdown("""
    - Age
    - Sex
    - Chest pain type
    - Maximum heart rate
    - Exercise-induced angina
    - Oldpeak
    - Slope
    - Major vessels
    - Thalassemia result
    """)

    st.markdown("---")

    st.warning(
        "⚠️ This application is for educational purposes "
        "and is not a medical diagnosis."
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        ❤️ Heart Disease Predictor
    </div>
    <div class="hero-subtitle">
        Machine Learning powered prediction using KNN Classification
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    "<div class='section-title'>📝 Patient Assessment</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='section-description'>
    Enter the patient's information below. For clinical measurements,
    use values from an actual medical report when available.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SECTION 1 — BASIC INFORMATION
# =========================================================

st.markdown(
    "<div class='section-title'>👤 Basic Information</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='section-description'>Basic patient characteristics.</div>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    st.markdown(
        """
        <div class='input-help'>
        The patient's age in years.
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
            "Female (0)" if x == 0 else "Male (1)"
    )

    st.markdown(
        """
        <div class='input-help'>
        Sex is represented using the dataset's numerical coding:
        0 = Female, 1 = Male.
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Typical Angina",
            1: "1 — Atypical Angina",
            2: "2 — Non-anginal Pain",
            3: "3 — Asymptomatic"
        }[x]
    )

    st.markdown(
        """
        <div class='input-help'>
        Describes the type of chest pain recorded for the patient.
        The model uses the numerical code shown above.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SECTION 2 — EXERCISE / HEART RATE
# =========================================================

st.markdown(
    "<div class='section-title'>🏃 Exercise & Heart Rate</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='section-description'>
    Information related to exercise testing and heart rate.
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    st.markdown(
        """
        <div class='input-help'>
        The highest heart rate achieved during an exercise test,
        measured in beats per minute (bpm).
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x:
            "0 — No" if x == 0 else "1 — Yes"
    )

    st.markdown(
        """
        <div class='input-help'>
        Indicates whether the patient experiences angina
        (chest discomfort) during exercise.
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    st.markdown(
        """
        <div class='input-help'>
        Oldpeak measures ST-segment depression during exercise
        compared with rest. It is normally obtained from an
        exercise ECG test.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SECTION 3 — TEST RESULTS
# =========================================================

st.markdown(
    "<div class='section-title'>🫀 Cardiac Test Information</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='section-description'>
    These values are usually obtained from cardiovascular
    testing rather than personal estimation.
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    slope = st.selectbox(
        "Peak Exercise ST Segment Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Upsloping",
            1: "1 — Flat",
            2: "2 — Downsloping"
        }[x]
    )

    st.markdown(
        """
        <div class='input-help'>
        Describes the direction of the ST segment during peak
        exercise on an ECG. It is a coded medical test result.
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    ca = st.selectbox(
        "Major Vessels (ca)",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — None",
            1: "1 — One",
            2: "2 — Two",
            3: "3 — Three",
            4: "4 — Four"
        }[x]
    )

    st.markdown(
        """
        <div class='input-help'>
        Number of major blood vessels observed through
        fluoroscopy. This should come from the relevant medical test.
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Category 0",
            1: "1 — Normal",
            2: "2 — Fixed Defect",
            3: "3 — Reversible Defect"
        }[x]
    )

    st.markdown(
        """
        <div class='input-help'>
        A coded feature related to the thalassemia / thallium
        stress-test result in the dataset. Use the recorded value.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FEATURE SUMMARY
# =========================================================

with st.expander("🔍 View Entered Values"):

    preview_data = pd.DataFrame({
        "Feature": [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Maximum Heart Rate",
            "Exercise-Induced Angina",
            "Oldpeak",
            "ST Segment Slope",
            "Major Vessels",
            "Thalassemia"
        ],
        "Value": [
            age,
            sex,
            cp,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]
    })

    st.dataframe(
        preview_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔮 PREDICT HEART DISEASE"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE INPUT DATA
        # EXACT SAME FEATURES USED TO TRAIN THE MODEL
        # -------------------------------------------------

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        # -------------------------------------------------
        # EXACT FEATURE ORDER USED DURING TRAINING
        # -------------------------------------------------

        expected_columns = [
            "age",
            "sex",
            "cp",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]

        input_data = input_data[expected_columns]

        # -------------------------------------------------
        # SCALE THE INPUT
        # -------------------------------------------------

        scaled_input = scaler.transform(input_data)

        # -------------------------------------------------
        # KNN PREDICTION
        # -------------------------------------------------

        prediction = knn.predict(scaled_input)

        result = prediction[0]

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        if result == 1:

            st.markdown(
                """
                <div class="result-box">
                    <div class="result-title">
                        🫀 Prediction Result
                    </div>
                    <div class="result-value">
                        ⚠️ Heart Disease Detected
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "The KNN model predicted class 1. "
                "This result is a machine-learning prediction "
                "and does not constitute a medical diagnosis."
            )

        else:

            st.markdown(
                """
                <div class="result-box">
                    <div class="result-title">
                        🫀 Prediction Result
                    </div>
                    <div class="result-value">
                        ✅ No Heart Disease Detected
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "The KNN model predicted class 0. "
                "This result is a machine-learning prediction "
                "and does not constitute a medical diagnosis."
            )


        # =================================================
        # PROBABILITIES
        # =================================================

        if hasattr(knn, "predict_proba"):

            probabilities = knn.predict_proba(
                scaled_input
            )[0]

            # Handle binary classification safely
            classes = list(knn.classes_)

            probability_data = []

            for class_value, probability in zip(
                classes,
                probabilities
            ):

                if class_value == 0:
                    label = "No Heart Disease"
                else:
                    label = "Heart Disease"

                probability_data.append({
                    "Prediction": label,
                    "Probability": probability
                })

            probability_df = pd.DataFrame(
                probability_data
            )

            st.markdown(
                "### 📊 Prediction Probability"
            )

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Heart disease probability
            if 1 in classes:

                heart_index = classes.index(1)

                heart_probability = probabilities[
                    heart_index
                ]

                st.progress(
                    float(heart_probability),
                    text=(
                        f"Estimated Heart Disease Probability: "
                        f"{heart_probability:.1%}"
                    )
                )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.markdown(
            """
            Please make sure that:

            - `knn_model.pkl` was created from your KNN model.
            - `scaler.pkl` was fitted on these exact 9 features.
            - The feature order matches the training data.
            """,
        )

        st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """<div class="footer">
        ❤️ Heart Disease Prediction
        <br>
        KNN Classification • StandardScaler • Streamlit
        <br><br>
        ⚠️ For educational purposes only.
        This tool does not replace professional medical
        advice, diagnosis, or treatment.
    </div>
    """,
    unsafe_allow_html=True
)
