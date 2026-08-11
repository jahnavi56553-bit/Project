import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# IMPORTANT: set_page_config MUST BE CALLED ONLY ONCE
# ============================================================

st.set_page_config(
    page_title="SkillSync - Employee Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ============================================================
# VIDEO BACKGROUND
# ============================================================

if os.path.exists("video.mp4"):

    with open("video.mp4", "rb") as video_file:
        video_bytes = video_file.read()

    video_base64 = base64.b64encode(video_bytes).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background: transparent;
        }}

        #background-video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -2;
            object-fit: cover;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.35);
            z-index: -1;
        }}

        .main,
        [data-testid="stAppViewContainer"] {{
            position: relative;
            z-index: 1;
            background: transparent !important;
        }}

        </style>

        <video autoplay muted loop playsinline id="background-video">
            <source
                src="data:video/mp4;base64,{video_base64}"
                type="video/mp4"
            >
        </video>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# GENERAL CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .main {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #1e293b 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .block-container {
        padding-top: 2rem;
    }

    .stButton > button {
        border-radius: 12px;
        border: none;
        min-height: 45px;
        font-weight: 600;
        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    .card {
        background: rgba(255,255,255,0.94);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 5px 25px rgba(0,0,0,0.10);
        margin-bottom: 20px;
    }

    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 22px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.10);
        text-align: center;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-sub {
        color: #16a34a;
        font-size: 13px;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: white;
    }

    .subtitle {
        color: #e2e8f0;
        font-size: 16px;
    }

    .login-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    local_file = "employee_salary_prediction_2500_updated.csv"

    if os.path.exists(local_file):
        return pd.read_csv(local_file)

    # Fallback for users who still have the older dataset filename.
    fallback_file = "employee_salary_15_25_lpa.csv"
    if os.path.exists(fallback_file):
        return pd.read_csv(fallback_file)

    url = (
        "https://raw.githubusercontent.com/"
        "jahnavi56553-bit/Project/main/"
        "employee_salary_15_25_lpa.csv"
    )
    return pd.read_csv(url)




# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Age",
    "Experience_Years",
    "Education",
    "Department",
    "Job_Level",
    "Performance_Rating",
    "Projects_Completed",
    "Certifications",
    "Weekly_Work_Hours",
    "Remote_Work",
    "City_Tier",
    "Job_Satisfaction"
]


CATEGORICAL_FEATURES = [
    "Education",
    "Department",
    "Remote_Work",
    "City_Tier"
]


NUMERIC_FEATURES = [
    "Age",
    "Experience_Years",
    "Job_Level",
    "Performance_Rating",
    "Projects_Completed",
    "Certifications",
    "Weekly_Work_Hours",
    "Job_Satisfaction"
]

df = load_data()

# Validate that the dataset contains everything required by the app.
REQUIRED_COLUMNS = set(FEATURES) | {"Salary_LPA", "Employee_ID"}
missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Ensure salary is numeric and keep only valid salary records.
df["Salary_LPA"] = pd.to_numeric(df["Salary_LPA"], errors="coerce")
df = df.dropna(subset=["Salary_LPA"]).copy()

if df.empty:
    st.error("No valid Salary_LPA values were found in the dataset.")
    st.stop()



# ============================================================
# CREATE SALARY STATUS FROM SALARY_LPA
# ============================================================

# This is important.
# If your dataset contains "High" for every employee,
# we do NOT use that column directly.
#
# Instead:
#
# Salary >= median salary  -> High
# Salary < median salary   -> Low

median_salary = df["Salary_LPA"].median()

df["Salary_Status"] = np.where(
    df["Salary_LPA"] >= median_salary,
    "High",
    "Low"
)


# ============================================================
# TRAIN CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    work = data[
        FEATURES + ["Salary_Status"]
    ].copy()

    work = work.dropna()

    X = work[FEATURES]

    y = work["Salary_Status"]

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            )
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "High"]
    )

    return (
        pipeline,
        accuracy,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        cm
    )


(
    model,
    accuracy,
    X_train,
    X_test,
    y_train,
    y_test,
    predictions,
    cm
) = train_model(df)


# ============================================================
# TRAIN SALARY REGRESSION MODEL
# ============================================================

@st.cache_resource
def train_salary_model(data):

    work = data[
        FEATURES + ["Salary_LPA"]
    ].copy()

    work = work.dropna(
        subset=["Salary_LPA"]
    )

    X = work[FEATURES]

    y = work["Salary_LPA"]

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            )
        ],
        remainder="drop"
    )

    salary_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                salary_model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        pipeline,
        mae,
        r2,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions
    )


(
    salary_model,
    salary_mae,
    salary_r2,
    salary_X_train,
    salary_X_test,
    salary_y_train,
    salary_y_test,
    salary_predictions
) = train_salary_model(df)


# ============================================================
# PROFILE CSV
# ============================================================

PROFILE_FILE = "employee_profiles.csv"


def save_profile_to_csv(
    name,
    email,
    project,
    employee_id,
    department,
    experience,
    education
):

    new_profile = pd.DataFrame({
        "Employee_ID": [employee_id],
        "Name": [name],
        "Email": [email],
        "Project": [project],
        "Department": [department],
        "Experience_Years": [experience],
        "Education": [education]
    })

    if os.path.exists(PROFILE_FILE):

        try:

            existing_profiles = pd.read_csv(
                PROFILE_FILE
            )

            updated_profiles = pd.concat(
                [
                    existing_profiles,
                    new_profile
                ],
                ignore_index=True
            )

        except Exception:

            updated_profiles = new_profile

    else:

        updated_profiles = new_profile

    updated_profiles.to_csv(
        PROFILE_FILE,
        index=False
    )


def load_saved_profiles():

    if os.path.exists(PROFILE_FILE):

        try:

            return pd.read_csv(
                PROFILE_FILE
            )

        except Exception:

            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    left, right = st.columns(
        [1.1, 1]
    )

    with left:

        st.markdown(
            """
            <div style="
                padding:70px 30px;
                text-align:center;
            ">

            <div style="font-size:100px;">
            💼
            </div>

            <h1 style="
                font-size:52px;
                font-weight:900;
                color:white;
            ">
            SkillSync
            </h1>

            <h3 style="
                color:white;
            ">
            Employee Salary Intelligence
            </h3>

            <p style="
                color:white;
                font-size:17px;
                line-height:1.8;
            ">
            Analyze employee profiles,<br>
            predict salary status,<br>
            and understand workforce trends.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="login-title">
            Welcome Back 👋
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Sign in to access your employee analytics dashboard."
        )

        email = st.text_input(
            "📧 Email",
            placeholder="Enter your email"
        )

        password = st.text_input(
            "🔐 Password",
            type="password",
            placeholder="Enter your password"
        )

        st.checkbox(
            "Remember me"
        )

        if st.button(
            "🚀 Login",
            use_container_width=True
        ):

            if email and password:

                st.session_state.logged_in = True

                st.session_state.page = "Dashboard"

                st.rerun()

            else:

                st.error(
                    "Please enter email and password."
                )

        st.markdown(
            """
            <center>
            <small style="color:white;">
            SkillSync Employee Intelligence Portal
            </small>
            </center>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:20px;
            ">

            <div style="font-size:50px;">
            💼
            </div>

            <h2>
            SkillSync
            </h2>

            <p style="
                color:#94a3b8;
            ">
            Employee Intelligence
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        navigation = {
            "🏠 Dashboard": "Dashboard",
            "💰 Salary Prediction": "Prediction",
            "📊 Analytics": "Analytics",
            "🤖 Model Performance": "Model",
            "👥 Employee Data": "Employees",
            "👤 Profile": "Profile",
            "📂 Saved Profiles": "Saved Profiles",
            "⚙️ Settings": "Settings"
        }

        for label, page in navigation.items():

            if st.button(
                label,
                use_container_width=True
            ):

                st.session_state.page = page

                st.rerun()

        st.markdown("---")

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.page = "Dashboard"

            st.rerun()


# ============================================================
# HEADER
# ============================================================

def header():

    st.markdown(
        f"""
        <div class="main-title">
        {st.session_state.page}
        </div>

        <div class="subtitle">
        Employee Salary Intelligence Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="
            color:#DA70D6;
            font-weight:700;
        ">
        Welcome to SkillSync 👋
        </h2>

        <p style="color:purple;">
        Here's an overview of your employee salary
        dataset and machine learning model.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(df)

    avg_salary = df[
        "Salary_LPA"
    ].mean()

    high_count = (
        df["Salary_Status"] == "High"
    ).sum()

    high_percentage = (
        high_count / total
    ) * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Total Employees
            </div>

            <div class="metric-value">
            {total:,}
            </div>

            <div class="metric-sub">
            Dataset records
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Average Salary
            </div>

            <div class="metric-value">
            ₹{avg_salary:.2f}
            </div>

            <div class="metric-sub">
            LPA
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            High Salary Employees
            </div>

            <div class="metric-value">
            {high_count:,}
            </div>

            <div class="metric-sub">
            {high_percentage:.1f}% of dataset
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Model Accuracy
            </div>

            <div class="metric-value">
            {accuracy * 100:.1f}%
            </div>

            <div class="metric-sub">
            Random Forest
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="card">

            <h3 style="color:red;">
            💰 Salary Distribution
            </h3>
            """,
            unsafe_allow_html=True
        )

        fig = px.histogram(
            df,
            x="Salary_LPA",
            nbins=30
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

            <h3 style="color:red;">
            📌 Salary Status
            </h3>
            """,
            unsafe_allow_html=True
        )

        status = (
            df["Salary_Status"]
            .value_counts()
            .reset_index()
        )

        status.columns = [
            "Status",
            "Count"
        ]

        fig = px.pie(
            status,
            names="Status",
            values="Count",
            hole=0.45
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    department_salary = (
        df.groupby("Department")[
            "Salary_LPA"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        department_salary,
        x="Department",
        y="Salary_LPA",
        title="Average Salary by Department",
        text_auto=".2f"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SALARY PREDICTION
# ============================================================

def prediction():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:purple;">
        💰 Employee Salary Prediction
        </h2>

        <p style="color:#64748b;">
        Enter employee details to predict salary in LPA
        and determine whether the salary status is High or Low.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30,
            step=1
        )

        experience = st.number_input(
            "Experience Years",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )

        education = st.selectbox(
            "Education",
            sorted(
                df["Education"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        department = st.selectbox(
            "Department",
            sorted(
                df["Department"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with c2:

        job_level = st.number_input(
            "Job Level",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        performance = st.number_input(
            "Performance Rating",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1
        )

        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=50,
            value=8,
            step=1
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=30,
            value=3,
            step=1
        )

    with c3:

        weekly_hours = st.number_input(
            "Weekly Work Hours",
            min_value=20.0,
            max_value=80.0,
            value=40.0,
            step=0.1
        )

        remote = st.selectbox(
            "Remote Work",
            sorted(
                df["Remote_Work"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        city = st.selectbox(
            "City Tier",
            sorted(
                df["City_Tier"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        satisfaction = st.number_input(
            "Job Satisfaction",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    if st.button(
        "🔮 Predict Salary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Age": [int(age)],

            "Experience_Years": [
                int(experience)
            ],

            "Education": [
                str(education)
            ],

            "Department": [
                str(department)
            ],

            "Job_Level": [
                int(job_level)
            ],

            "Performance_Rating": [
                float(performance)
            ],

            "Projects_Completed": [
                int(projects)
            ],

            "Certifications": [
                int(certifications)
            ],

            "Weekly_Work_Hours": [
                float(weekly_hours)
            ],

            "Remote_Work": [
                str(remote)
            ],

            "City_Tier": [
                str(city)
            ],

            "Job_Satisfaction": [
                int(satisfaction)
            ]
        })

        try:

            # =================================================
            # PREDICT SALARY AMOUNT
            # =================================================

            predicted_salary = salary_model.predict(
                input_data
            )[0]

            predicted_salary = float(
                np.clip(
                    predicted_salary,
                    5,
                    50
                )
            )

            # =================================================
            # PREDICT SALARY STATUS
            # =================================================

            status_result = model.predict(
                input_data
            )[0]

            status_probability = model.predict_proba(
                input_data
            )[0]

            classes = list(
                model.classes_
            )

            result_index = classes.index(
                status_result
            )

            confidence = (
                float(
                    status_probability[
                        result_index
                    ]
                ) * 100
            )

            # =================================================
            # HIGH / LOW PROBABILITIES
            # =================================================

            low_probability = 0.0

            high_probability = 0.0

            for class_name, class_probability in zip(
                classes,
                status_probability
            ):

                class_name = (
                    str(class_name)
                    .strip()
                    .title()
                )

                if class_name == "High":

                    high_probability = (
                        float(class_probability)
                        * 100
                    )

                elif class_name == "Low":

                    low_probability = (
                        float(class_probability)
                        * 100
                    )

            # =================================================
            # DISPLAY SALARY
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card">

                <h2 style="color:green;">
                💰 Predicted Salary
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                f"💰 Predicted Salary: "
                f"₹{predicted_salary:.2f} LPA"
            )

            salary_rupees = (
                predicted_salary * 100000
            )

            st.info(
                f"💵 Approximate annual salary: "
                f"₹{salary_rupees:,.0f}"
            )

            # =================================================
            # STATUS
            # =================================================

            st.markdown(
                "### 📊 Salary Status"
            )

            if (
                str(status_result)
                .strip()
                .title()
                == "High"
            ):

                st.success(
                    "🟢 This employee is predicted "
                    "to have a HIGH salary status."
                )

            else:

                st.warning(
                    "🔴 This employee is predicted "
                    "to have a LOW salary status."
                )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.markdown(
                "### 📈 Salary Status Probability"
            )

            p1, p2 = st.columns(2)

            with p1:

                st.metric(
                    "🔴 Low Probability",
                    f"{low_probability:.1f}%"
                )

            with p2:

                st.metric(
                    "🟢 High Probability",
                    f"{high_probability:.1f}%"
                )

            st.progress(
                min(
                    max(
                        confidence / 100,
                        0.0
                    ),
                    1.0
                )
            )

            st.caption(
                f"Prediction confidence: "
                f"{confidence:.1f}%"
            )

        except Exception as error:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(error)


# ============================================================
# ANALYTICS
# ============================================================

def analytics():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:green;">
        📊 Employee Analytics
        </h2>

        <p style="
            color:#4CAF50;
            font-size:20px;
        ">
        Explore relationships between employee attributes and salary.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    fig = px.scatter(
        df,
        x="Experience_Years",
        y="Salary_LPA",
        color="Salary_Status",
        hover_data=[
            "Department",
            "Education",
            "Job_Level"
        ],
        title="Experience vs Salary"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.box(
            df,
            x="Department",
            y="Salary_LPA",
            color="Department",
            title="Salary by Department"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.scatter(
            df,
            x="Performance_Rating",
            y="Salary_LPA",
            color="Salary_Status",
            title="Performance vs Salary"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def model_performance():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:orchid;">
        🤖 Random Forest Model Performance
        </h2>

        <p style="color:green;">
        The model is trained using an 80/20 train-test split.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Training Samples",
            len(X_train)
        )

    with c2:

        st.metric(
            "Testing Samples",
            len(X_test)
        )

    with c3:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # REGRESSION PERFORMANCE
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Salary MAE",
            f"{salary_mae:.2f} LPA"
        )

    with c2:

        st.metric(
            "Salary R² Score",
            f"{salary_r2:.3f}"
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h3 style="color:green;">
        Confusion Matrix
        </h3>

        </div>
        """,
        unsafe_allow_html=True
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Low",
            "Actual High"
        ],
        columns=[
            "Predicted Low",
            "Predicted High"
        ]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title=""
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    comparison = (
        result_df
        .value_counts()
        .reset_index(name="Count")
    )

    fig = px.bar(
        comparison,
        x="Actual",
        y="Count",
        color="Predicted",
        barmode="group",
        title="Actual vs Predicted Salary Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EMPLOYEE DATA
# ============================================================

def employees():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:violet;">
        👥 Employee Dataset
        </h2>

        <p style="color:#64748b;">
        Browse and filter employee records from the dataset.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style="color:#1565C0;">
        🔍 Search Employee ID
        </h3>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Search Employee ID"
    )

    filtered = df.copy()

    if search:

        try:

            employee_id = int(search)

            filtered = filtered[
                filtered["Employee_ID"]
                == employee_id
            ]

        except ValueError:

            st.warning(
                "Please enter a valid Employee ID."
            )

    department_filter = st.multiselect(
        "Filter by Department",
        sorted(
            df["Department"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if department_filter:

        filtered = filtered[
            filtered["Department"]
            .isin(department_filter)
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )


# ============================================================
# PROFILE
# ============================================================

def profile():

    header()

    c1, c2 = st.columns(
        [1, 2]
    )

    with c1:

        st.markdown(
            """
            <div class="card"
            style="text-align:center;">

            <div style="font-size:100px;">
            👩‍💻
            </div>

            <h2 style="color:purple;">
            Employee Profile
            </h2>

            <p style="color:#64748b;">
            Create and save employee profiles.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

            <h2 style="color:orchid;">
            👤 Profile Information
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

        employee_id = st.number_input(
            "Employee ID",
            min_value=1,
            step=1,
            value=1
        )

        name = st.text_input(
            "Name",
            placeholder="Enter employee name"
        )

        email = st.text_input(
            "Email",
            placeholder="Enter email address"
        )

        project = st.text_input(
            "Project",
            placeholder="Enter project name"
        )

        department = st.selectbox(
            "Department",
            sorted(
                df["Department"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        education = st.selectbox(
            "Education",
            sorted(
                df["Education"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        experience = st.number_input(
            "Experience Years",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

        if st.button(
            "💾 Save Profile",
            use_container_width=True
        ):

            if (
                name.strip()
                and email.strip()
                and project.strip()
            ):

                save_profile_to_csv(
                    name=name.strip(),
                    email=email.strip(),
                    project=project.strip(),
                    employee_id=int(employee_id),
                    department=department,
                    experience=int(experience),
                    education=education
                )

                st.success(
                    "✅ Profile saved successfully!"
                )

                st.info(
                    "The profile has been added to "
                    "employee_profiles.csv."
                )

            else:

                st.warning(
                    "⚠️ Please fill Name, Email and Project."
                )


# ============================================================
# SAVED PROFILES
# ============================================================

def saved_profiles():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:purple;">
        📂 Saved Profiles
        </h2>

        <p style="color:#64748b;">
        All employee profiles saved in the CSV file
        are displayed here.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    profiles = load_saved_profiles()

    if profiles.empty:

        st.info(
            "📭 No saved profiles found. "
            "Go to Profile and save an employee profile first."
        )

        return

    st.success(
        f"✅ {len(profiles)} saved profile(s) found."
    )

    st.dataframe(
        profiles,
        use_container_width=True,
        hide_index=True,
        height=450
    )

    st.markdown(
        "### 🔎 View Individual Profile"
    )

    selected_id = st.selectbox(
        "Select Employee ID",
        profiles["Employee_ID"].tolist()
    )

    selected_profile = profiles[
        profiles["Employee_ID"]
        == selected_id
    ]

    if not selected_profile.empty:

        person = selected_profile.iloc[0]

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="card">

                <h2 style="color:purple;">
                👤 {person["Name"]}
                </h2>

                <p>
                <b>Employee ID:</b>
                {person["Employee_ID"]}
                </p>

                <p>
                <b>Email:</b>
                {person["Email"]}
                </p>

                <p>
                <b>Project:</b>
                {person["Project"]}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="card">

                <h3 style="color:orchid;">
                💼 Employee Details
                </h3>

                <p>
                <b>Department:</b>
                {person["Department"]}
                </p>

                <p>
                <b>Education:</b>
                {person["Education"]}
                </p>

                <p>
                <b>Experience:</b>
                {person["Experience_Years"]} years
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    csv_data = profiles.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Saved Profiles CSV",
        data=csv_data,
        file_name="employee_profiles.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# SETTINGS
# ============================================================

def settings():

    header()

    st.markdown(
        """
        <div class="card">

        <h2 style="color:darkgray;">
        ⚙️ Settings
        </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.toggle(
        "Enable Notifications",
        True
    )

    st.toggle(
        "Show Prediction Confidence",
        True
    )

    st.selectbox(
        "Language",
        [
            "English",
            "Telugu",
            "Hindi"
        ]
    )

    if st.button(
        "Save Settings"
    ):

        st.success(
            "Settings updated successfully!"
        )


# ============================================================
# APPLICATION
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    sidebar()

    if st.session_state.page == "Dashboard":

        dashboard()

    elif st.session_state.page == "Prediction":

        prediction()

    elif st.session_state.page == "Analytics":

        analytics()

    elif st.session_state.page == "Model":

        model_performance()

    elif st.session_state.page == "Employees":

        employees()

    elif st.session_state.page == "Profile":

        profile()

    elif st.session_state.page == "Saved Profiles":

        saved_profiles()

    elif st.session_state.page == "Settings":

        settings()