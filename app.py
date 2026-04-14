import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Loan Risk Dashboard", layout="wide")


# ------------------ CUSTOM STYLING ------------------
st.markdown("""
            
<style>
[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 1rem !important;
}

.stApp {
    background: linear-gradient(180deg, #020817 0%, #061326 100%);
}
</style>


<style>
/* Whole app */
.stApp {
    background: linear-gradient(180deg, #020817 0%, #061326 100%);
    color: #f8fafc;
}

/* Remove top toolbar completely */
[data-testid="stHeader"] {
    display: none;
}

/* Remove hamburger menu spacing */
[data-testid="collapsedControl"] {
    display: none;
}


/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031025 0%, #071a35 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

/* Titles and text */
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label {
    color: #cbd5e1 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.1rem;
    font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
    color: white !important;
}

/* Text input / number input / select / multiselect containers */
div[data-baseweb="input"],
div[data-baseweb="select"] {
    background-color: transparent !important;
}

/* Actual input field */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

/* Make typed numbers visible */
input, textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Number input stepper buttons */
button[kind="secondary"] {
    color: #ffffff !important;
}

/* Dropdown selected text */
span[data-baseweb="tag"] {
    background-color: #ef4444 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Placeholder text */
input::placeholder,
textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    overflow: hidden;
}

/* Horizontal rule */
hr {
    border-color: rgba(148, 163, 184, 0.14);
}
</style>
""", unsafe_allow_html=True)


# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_loan_data.csv")


# ------------------ CLEAN INTEREST RATE ------------------
def clean_interest_rate_column(df):
    if "int_rate" in df.columns:
        if df["int_rate"].dtype == object:
            df["int_rate"] = (
                df["int_rate"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    return df


# ------------------ TRAIN MODEL ------------------
@st.cache_resource
def train_model(df):
    required_cols = ["loan_amnt", "annual_inc", "int_rate", "grade", "term", "risk_category"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return None, None, None, missing_cols

    model_df = df[required_cols].dropna().copy()

    X = model_df.drop("risk_category", axis=1)
    y = model_df["risk_category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    numeric_features = ["loan_amnt", "annual_inc", "int_rate"]
    categorical_features = ["grade", "term"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, label_encoder, accuracy, None


# ------------------ PDF REPORT ------------------
def create_pdf_report(filtered_df, accuracy):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter

    y = height - 50
    pdf.setTitle("Loan Risk Dashboard Report")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Loan Risk Dashboard Report")

    y -= 30
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Total Loans: {len(filtered_df):,}")

    if "loan_amnt" in filtered_df.columns:
        y -= 20
        pdf.drawString(50, y, f"Average Loan Amount: ${filtered_df['loan_amnt'].mean():,.2f}")

    if "annual_inc" in filtered_df.columns:
        y -= 20
        pdf.drawString(50, y, f"Average Annual Income: ${filtered_df['annual_inc'].mean():,.2f}")

    if "int_rate" in filtered_df.columns:
        y -= 20
        pdf.drawString(50, y, f"Average Interest Rate: {filtered_df['int_rate'].mean():.2f}%")

    if accuracy is not None:
        y -= 20
        pdf.drawString(50, y, f"Model Accuracy: {accuracy:.2%}")

    if "risk_category" in filtered_df.columns:
        y -= 35
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Risk Category Counts")

        y -= 20
        pdf.setFont("Helvetica", 11)
        risk_counts = filtered_df["risk_category"].value_counts()
        for category, count in risk_counts.items():
            pdf.drawString(60, y, f"{category}: {count}")
            y -= 18

    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Dataset Preview")

    y -= 20
    pdf.setFont("Helvetica", 9)
    preview_df = filtered_df.head(10)

    for _, row in preview_df.iterrows():
        row_text = " | ".join([str(v)[:20] for v in row.values[:5]])
        pdf.drawString(50, y, row_text)
        y -= 15
        if y < 50:
            pdf.showPage()
            y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer


# ------------------ PLOT STYLE ------------------
def style_plotly(fig):
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        title_font=dict(color="#f8fafc", size=18),
        xaxis=dict(showgrid=False, color="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.18)", color="#cbd5e1"),
        legend=dict(font=dict(color="#f8fafc")),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


# ------------------ MAIN ------------------
df = load_data()
df = clean_interest_rate_column(df)

st.title("Banking Customer Risk Loan Analysis")
st.write("A professional dashboard for exploring loan patterns, customer risk segments, and ML-based risk prediction.")
st.markdown("---")


# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("🔎 Filters")

selected_grade = []
selected_term = []

if "grade" in df.columns:
    grade_options = sorted(df["grade"].dropna().unique())
    selected_grade = st.sidebar.multiselect(
        "Select Grade",
        options=grade_options,
        default=grade_options
    )

if "term" in df.columns:
    term_options = sorted(df["term"].dropna().unique())
    selected_term = st.sidebar.multiselect(
        "Loan Term",
        options=term_options,
        default=term_options
    )

filtered_df = df.copy()

if "grade" in df.columns and selected_grade:
    filtered_df = filtered_df[filtered_df["grade"].isin(selected_grade)]

if "term" in df.columns and selected_term:
    filtered_df = filtered_df[filtered_df["term"].isin(selected_term)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# ------------------ MODEL ------------------
model, label_encoder, accuracy, missing_cols = train_model(df)

if missing_cols:
    st.error(f"Model could not be trained because these columns are missing: {missing_cols}")


# ------------------ KPI SECTION ------------------
st.subheader("Key Metrics")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("💰 Total Loans", f"{len(filtered_df):,}")

with k2:
    avg_loan_value = f"${filtered_df['loan_amnt'].mean():,.0f}" if "loan_amnt" in filtered_df.columns else "N/A"
    st.metric("💵 Avg Loan Amount", avg_loan_value)

with k3:
    avg_rate_value = f"{filtered_df['int_rate'].mean():.2f}%" if "int_rate" in filtered_df.columns else "N/A"
    st.metric("📈 Avg Interest Rate", avg_rate_value)

with k4:
    acc_value = f"{accuracy:.2%}" if accuracy is not None else "N/A"
    st.metric("🤖 ML Accuracy", acc_value)

st.markdown("")


# ------------------ CHARTS ------------------
st.subheader("Insights")

col1, col2 = st.columns(2)

with col1:
    if "loan_amnt" in filtered_df.columns and "risk_category" in filtered_df.columns:
        fig1 = px.histogram(
            filtered_df,
            x="loan_amnt",
            color="risk_category",
            title="Loan Amount Distribution",
            barmode="overlay"
        )
        fig1 = style_plotly(fig1)
        st.plotly_chart(fig1, width="stretch")

with col2:
    if "risk_category" in filtered_df.columns:
        risk_df = filtered_df["risk_category"].value_counts().reset_index()
        risk_df.columns = ["risk_category", "count"]

        fig2 = px.bar(
            risk_df,
            x="risk_category",
            y="count",
            color="risk_category",
            title="Risk Category Distribution"
        )
        fig2 = style_plotly(fig2)
        st.plotly_chart(fig2, width="stretch")

col3, col4 = st.columns(2)

with col3:
    if "risk_category" in filtered_df.columns and "loan_amnt" in filtered_df.columns:
        avg_loan = filtered_df.groupby("risk_category", as_index=False)["loan_amnt"].mean()

        fig3 = px.bar(
            avg_loan,
            x="risk_category",
            y="loan_amnt",
            color="risk_category",
            title="Average Loan Amount by Risk"
        )
        fig3 = style_plotly(fig3)
        st.plotly_chart(fig3, width="stretch")

with col4:
    if "risk_category" in filtered_df.columns and "annual_inc" in filtered_df.columns:
        avg_income = filtered_df.groupby("risk_category", as_index=False)["annual_inc"].mean()

        fig4 = px.bar(
            avg_income,
            x="risk_category",
            y="annual_inc",
            color="risk_category",
            title="Average Income by Risk"
        )
        fig4 = style_plotly(fig4)
        st.plotly_chart(fig4, width="stretch")


# ------------------ DATA TABLE ------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), width="stretch")


# ------------------ PREDICTION SECTION ------------------
st.subheader("AI Risk Prediction")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    loan = st.number_input(
        "Loan Amount",
        min_value=1000,
        max_value=50000,
        value=15000,
        step=500
    )
    income = st.number_input(
        "Annual Income",
        min_value=10000,
        max_value=200000,
        value=60000,
        step=1000
    )

with pred_col2:
    if "grade" in df.columns:
        input_grade = st.selectbox("Grade", sorted(df["grade"].dropna().unique()))
    else:
        input_grade = None

    if "term" in df.columns:
        input_term = st.selectbox("Loan Term", sorted(df["term"].dropna().unique()))
    else:
        input_term = None

with pred_col3:
    int_rate_input = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=12.5,
        step=0.1
    )

if st.button("Predict Risk"):
    if model is not None and label_encoder is not None:
        input_df = pd.DataFrame([{
            "loan_amnt": loan,
            "annual_inc": income,
            "int_rate": int_rate_input,
            "grade": input_grade,
            "term": input_term
        }])

        prediction_encoded = model.predict(input_df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.markdown("### Prediction Result")
        if str(prediction_label).lower() == "low risk":
            st.success(f"Predicted Risk Category: {prediction_label}")
        elif str(prediction_label).lower() == "medium risk":
            st.warning(f"Predicted Risk Category: {prediction_label}")
        else:
            st.error(f"Predicted Risk Category: {prediction_label}")
    else:
        st.error("Model is not available because required columns are missing.")


# ------------------ DOWNLOAD REPORT ------------------
st.subheader("Download Report")
pdf_buffer = create_pdf_report(filtered_df, accuracy)

st.download_button(
    label="Download PDF Report",
    data=pdf_buffer,
    file_name="loan_risk_dashboard_report.pdf",
    mime="application/pdf"
)


# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Built with Streamlit | Loan Risk Analysis Dashboard")