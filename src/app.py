"""
╔══════════════════════════════════════════════════════════════╗
║   Student Dropout Early Warning System — Streamlit App       ║
║   GWE 2026 Data Science Challenge                            ║
╚══════════════════════════════════════════════════════════════╝

Redundancy modes:
  • Mode A — Saved Model   : loads lgbm_dropout_model.pkl directly
  • Mode B — Raw Dataset   : uploads OULAD CSVs, reruns feature
                             engineering, feeds same predict path
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Early Warning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1117;
    color: #E8EAF0;
}
[data-testid="stSidebar"] {
    background-color: #1C1F2E;
    border-right: 1px solid #2A2D3E;
}

/* Cards */
.card {
    background: #1C1F2E;
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-red   { border-left: 4px solid #E74C3C; }
.card-orange{ border-left: 4px solid #F39C12; }
.card-green { border-left: 4px solid #2ECC71; }
.card-blue  { border-left: 4px solid #4F8EF7; }

/* Metric row */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-box {
    background: #1C1F2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1; min-width: 130px;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.metric-lbl { font-size: 0.72rem; color: #8B90A0; text-transform: uppercase;
               letter-spacing: 0.08em; margin-top: 0.2rem; }

/* Risk badge */
.badge {
    display: inline-block; padding: 0.25rem 0.9rem;
    border-radius: 99px; font-size: 0.8rem;
    font-weight: 700; letter-spacing: 0.05em;
}
.badge-high   { background:#E74C3C22; color:#E74C3C; border:1px solid #E74C3C55; }
.badge-medium { background:#F39C1222; color:#F39C12; border:1px solid #F39C1255; }
.badge-low    { background:#2ECC7122; color:#2ECC71; border:1px solid #2ECC7155; }

/* Eyebrow labels */
.eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4F8EF7; margin-bottom: 0.3rem;
}

/* Mode banner */
.mode-banner {
    background: linear-gradient(90deg, #1C1F2E, #252837);
    border: 1px solid #2A2D3E; border-radius: 10px;
    padding: 0.7rem 1.2rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.85rem; color: #8B90A0;
}

/* Pulse gauge container */
.gauge-wrap { text-align: center; padding: 1rem 0; }

/* Intervention chip */
.chip {
    display: inline-block; background: #252837;
    border: 1px solid #2A2D3E; border-radius: 8px;
    padding: 0.4rem 0.8rem; margin: 0.3rem 0.2rem;
    font-size: 0.82rem; color: #E8EAF0;
}

/* Sidebar section label */
.sidebar-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8B90A0;
    margin: 1rem 0 0.3rem;
}

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    'gender_encoded','disability_encoded','age_band_encoded',
    'imd_band_encoded','education_encoded','num_of_prev_attempts',
    'mean_score','std_score','miss_ratio','score_trend',
    'num_assessments','avg_submission_delay',
    'total_clicks','active_days','avg_daily_clicks',
    'engagement_rate','click_trend','material_diversity','activity_span',
    'early_unregistration','days_until_unreg',
]
THRESHOLDS = {'medium': 0.40, 'high': 0.70}

# app.py lives in src/, so models/ is one level up at the repo root.
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(_SRC_DIR)
MODEL_PATH      = os.path.join(_BASE_DIR, 'models', 'lgbm_dropout_model.pkl')
FEATURES_PATH   = os.path.join(_BASE_DIR, 'models', 'feature_list.pkl')
THRESHOLDS_PATH = os.path.join(_BASE_DIR, 'models', 'thresholds.pkl')

IMD_ORDER = ['0-10%','10-20%','20-30%','30-40%','40-50%',
             '50-60%','60-70%','70-80%','80-90%','90-100%']
AGE_ORDER = ['0-35','35-55','55<=']
EDU_ORDER = ['No Formal quals','Lower Than A Level','A Level or Equivalent',
             'HE Qualification','Post Graduate Qualification']


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        mdl = pickle.load(f)

    # Load saved thresholds if available (overrides the hardcoded defaults)
    global THRESHOLDS
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'medium' in loaded and 'high' in loaded:
                THRESHOLDS = loaded

    # Load saved feature list if available (overrides the hardcoded FEATURES)
    global FEATURES
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, list) and len(loaded) > 0:
                FEATURES = loaded

    return mdl


def get_risk_tier(prob):
    if prob >= THRESHOLDS['high']:   return 'HIGH'
    if prob >= THRESHOLDS['medium']: return 'MEDIUM'
    return 'LOW'


def recommend_intervention(row):
    risk = get_risk_tier(row.get('dropout_prob', 0))
    if risk == 'LOW':
        return ['✅ No immediate action needed — continue regular monitoring.']
    interventions = []
    if row.get('imd_band_encoded', 5) >= 7:
        interventions.append('💰 Financial Aid / Scholarship Referral')
    if row.get('miss_ratio', 0) > 0.3 or row.get('mean_score', 100) < 40:
        interventions.append('📚 Academic Tutoring & Study Support')
    if row.get('click_trend', 0) < 0 or row.get('total_clicks', 999) < 50:
        interventions.append('📱 Peer Mentoring & Re-engagement Program')
    if row.get('score_trend', 0) < -10:
        interventions.append('🧠 One-on-one Counseling with Academic Advisor')
    if row.get('num_of_prev_attempts', 0) > 1:
        interventions.append('🔄 Learning Strategy Workshop (repeated attempts)')
    if not interventions:
        interventions.append('🚨 Urgent welfare check — contact student immediately' if risk == 'HIGH'
                             else '📅 Schedule check-in with Academic Advisor')
    return interventions


def make_gauge(prob, risk):
    color = {'HIGH': '#E74C3C', 'MEDIUM': '#F39C12', 'LOW': '#2ECC71'}[risk]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'color': color, 'size': 48}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B90A0',
                     'tickfont': {'color': '#8B90A0', 'size': 11}},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': '#1C1F2E',
            'bordercolor': '#2A2D3E',
            'steps': [
                {'range': [0,  40],  'color': 'rgba(46,204,113,0.09)'},
                {'range': [40, 70],  'color': 'rgba(243,156,18,0.09)'},
                {'range': [70, 100], 'color': 'rgba(231,76,60,0.09)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8, 'value': prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=260, margin=dict(t=20, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#E8EAF0'},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (Mode B — Raw Dataset path)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def engineer_features(
    student_info_bytes, student_registration_bytes,
    student_assessment_bytes, assessments_bytes,
    vle_bytes, courses_bytes, student_vle_bytes,
    sample_size=5000, seed=42
):
    """Full feature engineering pipeline from raw OULAD CSVs."""

    # Load
    student_info         = pd.read_csv(io.BytesIO(student_info_bytes))
    student_registration = pd.read_csv(io.BytesIO(student_registration_bytes))
    student_assessment   = pd.read_csv(io.BytesIO(student_assessment_bytes))
    assessments          = pd.read_csv(io.BytesIO(assessments_bytes))
    vle                  = pd.read_csv(io.BytesIO(vle_bytes))
    courses              = pd.read_csv(io.BytesIO(courses_bytes))
    student_vle          = pd.read_csv(io.BytesIO(student_vle_bytes))

    # ── Target & stratified sample ─────────────────────────────────────────
    student_info['dropout'] = (student_info['final_result'] == 'Withdrawn').astype(int)
    n = min(sample_size, len(student_info))
    sampled_ids = (
        student_info.groupby('dropout', group_keys=False)
        .apply(lambda x: x.sample(frac=n/len(student_info), random_state=seed))
        ['id_student'].unique()
    )
    student_info  = student_info[student_info['id_student'].isin(sampled_ids)].copy()
    student_vle   = student_vle[student_vle['id_student'].isin(sampled_ids)].copy()

    # ── Clean ──────────────────────────────────────────────────────────────
    student_info['imd_band'].fillna(student_info['imd_band'].mode()[0], inplace=True)

    # ── Encode ────────────────────────────────────────────────────────────
    student_info['imd_band_encoded']   = pd.Categorical(student_info['imd_band'],
                                            categories=IMD_ORDER, ordered=True).codes
    student_info['age_band_encoded']   = pd.Categorical(student_info['age_band'],
                                            categories=AGE_ORDER, ordered=True).codes
    student_info['education_encoded']  = pd.Categorical(student_info['highest_education'],
                                            categories=EDU_ORDER, ordered=True).codes
    student_info['gender_encoded']     = (student_info['gender'] == 'M').astype(int)
    student_info['disability_encoded'] = (student_info['disability'] == 'Y').astype(int)

    # ── Academic features ─────────────────────────────────────────────────
    assess_full = student_assessment.merge(
        assessments[['id_assessment','code_module','code_presentation','date']],
        on='id_assessment', how='left'
    )
    def score_trend(g):
        s = g.dropna()
        if len(s) < 2: return 0
        mid = len(s) // 2
        return s.iloc[mid:].mean() - s.iloc[:mid].mean()

    af = assess_full.groupby(['id_student','code_module','code_presentation']).agg(
        mean_score          =('score','mean'),
        std_score           =('score','std'),
        num_assessments     =('score','count'),
        num_missed          =('score', lambda x: x.isnull().sum()),
        avg_submission_delay=('date_submitted',
                              lambda x: (assess_full.loc[x.index,'date_submitted'] -
                                         assess_full.loc[x.index,'date']).mean()),
    ).reset_index()
    tr = assess_full.groupby(['id_student','code_module','code_presentation'])['score']\
         .apply(score_trend).reset_index()
    tr.columns = ['id_student','code_module','code_presentation','score_trend']
    af = af.merge(tr, on=['id_student','code_module','code_presentation'])
    af['miss_ratio']            = af['num_missed'] / (af['num_assessments'] + af['num_missed'])
    af['std_score']             = af['std_score'].fillna(0)
    af['avg_submission_delay']  = af['avg_submission_delay'].fillna(0)

    # ── VLE features ──────────────────────────────────────────────────────
    vs = student_vle.groupby(['id_student','code_module','code_presentation']).agg(
        total_clicks      =('sum_click','sum'),
        active_days       =('date','nunique'),
        avg_daily_clicks  =('sum_click','mean'),
        last_active_day   =('date','max'),
        first_active_day  =('date','min'),
    ).reset_index()
    vs['activity_span']   = vs['last_active_day'] - vs['first_active_day']
    vs['engagement_rate'] = vs['active_days'] / (vs['activity_span'] + 1)

    course_length = courses.set_index(['code_module','code_presentation'])['module_presentation_length']
    def click_trend_fn(group, mod, pres):
        key    = (mod, pres)
        length = course_length.get(key, group['date'].max())
        mid    = length / 2
        return group[group['date'] > mid]['sum_click'].sum() - \
               group[group['date'] <= mid]['sum_click'].sum()

    ct_rows = []
    for (sid, mod, pres), grp in student_vle.groupby(['id_student','code_module','code_presentation']):
        ct_rows.append({
            'id_student': sid,
            'code_module': mod,
            'code_presentation': pres,
            'click_trend': click_trend_fn(grp, mod, pres),
        })
    ct = pd.DataFrame(ct_rows)
    vs = vs.merge(ct, on=['id_student','code_module','code_presentation'])

    md = student_vle.merge(vle[['id_site','activity_type']], on='id_site', how='left')\
         .groupby(['id_student','code_module','code_presentation'])['activity_type']\
         .nunique().reset_index()
    md.columns = ['id_student','code_module','code_presentation','material_diversity']
    vs = vs.merge(md, on=['id_student','code_module','code_presentation'])

    # ── Registration features ─────────────────────────────────────────────
    reg = student_registration.copy()
    reg['early_unregistration'] = reg['date_unregistration'].notnull().astype(int)
    reg['days_until_unreg']     = reg['date_unregistration'].fillna(999)

    # ── Master merge ──────────────────────────────────────────────────────
    master = student_info.copy()
    master = master.merge(af, on=['id_student','code_module','code_presentation'], how='left')
    master = master.merge(vs, on=['id_student','code_module','code_presentation'], how='left')
    master = master.merge(
        reg[['id_student','code_module','code_presentation','early_unregistration','days_until_unreg']],
        on=['id_student','code_module','code_presentation'], how='left'
    )
    for c in ['total_clicks','active_days','avg_daily_clicks','click_trend',
              'engagement_rate','material_diversity','activity_span']:
        master[c] = master[c].fillna(0)
    for c in ['mean_score','std_score','num_assessments','miss_ratio','score_trend']:
        master[c] = master[c].fillna(master[c].median())

    return master, student_vle


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Dropout Warning")
    st.markdown("---")

    st.markdown('<p class="sidebar-label">Data Source</p>', unsafe_allow_html=True)
    mode = st.radio(
        label="mode",
        options=["🤖 Saved Model", "📂 Raw Dataset"],
        label_visibility="collapsed",
    )

    st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        label="page",
        options=["🏠 Home", "📊 EDA Dashboard", "🔍 Predict", "📋 About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem;color:#8B90A0;">GWE 2026 · Data Science Challenge</p>',
        unsafe_allow_html=True
    )

# Shared state
model      = load_model()
master_df  = None
svle_df    = None
mode_label = mode.split(" ", 1)[1]

# ── Mode B: file uploaders ─────────────────────────────────────────────────
if mode == "📂 Raw Dataset":
    with st.sidebar:
        st.markdown('<p class="sidebar-label">Upload OULAD CSVs</p>', unsafe_allow_html=True)
        up_info  = st.file_uploader("studentInfo.csv",         type="csv", key="info")
        up_reg   = st.file_uploader("studentRegistration.csv", type="csv", key="reg")
        up_asses = st.file_uploader("studentAssessment.csv",   type="csv", key="asses")
        up_asmeta= st.file_uploader("assessments.csv",         type="csv", key="asmeta")
        up_vle   = st.file_uploader("vle.csv",                 type="csv", key="vle")
        up_cours = st.file_uploader("courses.csv",             type="csv", key="cours")
        up_svle  = st.file_uploader("studentVle.csv",          type="csv", key="svle")

        all_uploaded = all([up_info, up_reg, up_asses, up_asmeta,
                            up_vle, up_cours, up_svle])
        if all_uploaded:
            with st.spinner("Running feature engineering…"):
                master_df, svle_df = engineer_features(
                    up_info.read(),  up_reg.read(),   up_asses.read(),
                    up_asmeta.read(), up_vle.read(),  up_cours.read(),
                    up_svle.read(),
                )
            st.success(f"✅ {len(master_df):,} students loaded")

            # Retrain a quick model on uploaded data
            from sklearn.model_selection import train_test_split
            from imblearn.over_sampling import SMOTE
            import lightgbm as lgb

            mdf = master_df[FEATURES + ['dropout']].dropna(subset=['mean_score','total_clicks'])
            mdf = mdf.fillna(mdf.median(numeric_only=True))
            X, y = mdf[FEATURES], mdf['dropout']
            X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                                 random_state=42, stratify=y)
            X_tr_r, y_tr_r = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            raw_model = lgb.LGBMClassifier(random_state=42, n_estimators=200, verbose=-1)
            raw_model.fit(X_tr_r, y_tr_r)
            st.session_state['raw_model']  = raw_model
            st.session_state['master_df']  = master_df
            st.session_state['svle_df']    = svle_df
        else:
            missing = 7 - sum([bool(f) for f in
                [up_info,up_reg,up_asses,up_asmeta,up_vle,up_cours,up_svle]])
            st.info(f"Upload all 7 CSVs ({missing} remaining)")

# Retrieve from session state if already processed
if 'master_df' in st.session_state:
    master_df = st.session_state['master_df']
if 'svle_df'   in st.session_state:
    svle_df   = st.session_state['svle_df']

# Active model pointer
active_model = st.session_state.get('raw_model') if mode == "📂 Raw Dataset" else model


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎓 Student Dropout Early Warning System")
    st.markdown("##### Identify at-risk students early. Intervene before it's too late.")
    st.markdown("---")

    # Mode banner
    icon  = "🤖" if mode == "🤖 Saved Model" else "📂"
    color = "#4F8EF7" if mode == "🤖 Saved Model" else "#2ECC71"
    ready = (model is not None) if mode == "🤖 Saved Model" else (master_df is not None)
    status = "Ready" if ready else ("Model file not found — check models/" if mode == "🤖 Saved Model"
                                     else "Upload all 7 CSVs in the sidebar")
    st.markdown(f"""
    <div class="mode-banner">
        <span style="font-size:1.3rem">{icon}</span>
        <span>Active mode: <strong style="color:{color}">{mode_label}</strong>
        &nbsp;·&nbsp; {status}</span>
    </div>
    """, unsafe_allow_html=True)

    # Stats row (show if data available)
    if master_df is not None:
        total   = len(master_df)
        n_drop  = master_df['dropout'].sum()
        rate    = master_df['dropout'].mean() * 100
        courses = master_df['code_module'].nunique()
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val" style="color:#4F8EF7">{total:,}</div>
                <div class="metric-lbl">Students</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#E74C3C">{n_drop:,}</div>
                <div class="metric-lbl">Withdrawn</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#F39C12">{rate:.1f}%</div>
                <div class="metric-lbl">Dropout Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#2ECC71">{courses}</div>
                <div class="metric-lbl">Courses</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card card-blue">
            <div class="eyebrow">The Problem</div>
            <h4 style="margin:0.3rem 0">Dropouts leave silent signals</h4>
            <p style="color:#8B90A0;font-size:0.9rem;margin:0">
            By the time a student officially withdraws, the warning signs were
            already present two semesters earlier — in missed assessments,
            declining clicks, and falling scores. This system reads those signals
            before it's too late.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-green">
            <div class="eyebrow">Intervention Layer</div>
            <h4 style="margin:0.3rem 0">Not just a classifier</h4>
            <p style="color:#8B90A0;font-size:0.9rem;margin:0">
            Every at-risk student receives a <strong>specific recommended action</strong>
            — financial aid referral, tutoring, peer mentoring, or counseling —
            based on which signals triggered the alert.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card card-red">
            <div class="eyebrow">Two Data Modes</div>
            <h4 style="margin:0.3rem 0">Flexible by design</h4>
            <p style="color:#8B90A0;font-size:0.9rem;margin:0">
            <strong>🤖 Saved Model</strong> — loads the pre-trained LightGBM instantly.
            No data needed.<br><br>
            <strong>📂 Raw Dataset</strong> — upload the OULAD CSVs and re-run the full
            feature engineering pipeline for fresh results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-orange">
            <div class="eyebrow">Optimal Window</div>
            <h4 style="margin:0.3rem 0">Week 8 is the sweet spot</h4>
            <p style="color:#8B90A0;font-size:0.9rem;margin:0">
            Our analysis shows Week 8 of the module achieves ~82% ROC-AUC
            with enough time remaining for meaningful intervention.
            Waiting for full data gains accuracy but loses the window to act.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="eyebrow">How to use this app</div>
    """, unsafe_allow_html=True)
    steps = [
        ("1", "Choose your mode", "Select **Saved Model** for instant use, or **Raw Dataset** to upload CSVs."),
        ("2", "Explore the data", "Visit **EDA Dashboard** to understand dropout patterns across the cohort."),
        ("3", "Predict risk",     "Use **Predict** to assess a single student or batch-upload a class roster."),
        ("4", "Act on insights",  "Each prediction comes with a specific **intervention recommendation**."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="min-height:130px">
                <div style="font-size:1.6rem;font-weight:800;color:#4F8EF7">{num}</div>
                <div style="font-weight:600;margin:0.3rem 0">{title}</div>
                <div style="color:#8B90A0;font-size:0.82rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown("# 📊 EDA Dashboard")
    st.markdown("Explore dropout patterns across the student cohort.")

    if master_df is None:
        st.warning("No dataset available. Switch to **Raw Dataset** mode and upload CSVs, "
                   "or switch to **Saved Model** mode and the dashboard will use cached data.")
        st.stop()

    df = master_df.copy()

    # ── Row 1: Overview ──────────────────────────────────────────────────────
    st.markdown('<div class="eyebrow">Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])

    with c1:
        counts = df['final_result'].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={
                'Pass':'#2ECC71','Distinction':'#4F8EF7',
                'Withdrawn':'#E74C3C','Fail':'#F39C12'
            },
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E8EAF0', legend_font_size=11,
            margin=dict(t=10,b=10,l=10,r=10), height=280,
            title=dict(text="Result Distribution", font_color='#8B90A0', font_size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dr = df.groupby('code_module')['dropout'].mean().mul(100).reset_index()
        dr.columns = ['Course','Dropout Rate (%)']
        dr = dr.sort_values('Dropout Rate (%)', ascending=True)
        fig2 = px.bar(dr, x='Dropout Rate (%)', y='Course', orientation='h',
                      color='Dropout Rate (%)',
                      color_continuous_scale=['#2ECC71','#F39C12','#E74C3C'])
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E8EAF0', coloraxis_showscale=False,
            margin=dict(t=30,b=10,l=10,r=10), height=280,
            title=dict(text="Dropout Rate by Course (%)", font_color='#8B90A0', font_size=12),
            yaxis=dict(color='#8B90A0'), xaxis=dict(color='#8B90A0'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Learning Pulse ─────────────────────────────────────────────────
    if svle_df is not None:
        st.markdown('<div class="eyebrow" style="margin-top:1rem">Learning Pulse</div>',
                    unsafe_allow_html=True)
        vle_labeled = svle_df.merge(
            df[['id_student','code_module','code_presentation','final_result']],
            on=['id_student','code_module','code_presentation']
        )
        vle_labeled['week'] = (vle_labeled['date'] // 7).astype(int)
        pulse = vle_labeled.groupby(['week','final_result'])['sum_click'].mean().reset_index()
        pulse = pulse[pulse['week'].between(0, 30)]

        color_map = {'Pass':'#2ECC71','Distinction':'#4F8EF7',
                     'Withdrawn':'#E74C3C','Fail':'#F39C12'}
        fig3 = px.line(pulse, x='week', y='sum_click', color='final_result',
                       color_discrete_map=color_map, markers=True)
        fig3.add_vline(x=8, line_dash='dash', line_color='#8B90A0',
                       annotation_text='Week 8', annotation_font_color='#8B90A0')
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E8EAF0', height=320,
            title=dict(text="📈 Learning Pulse — Average Weekly Clicks by Outcome",
                       font_color='#8B90A0', font_size=12),
            xaxis=dict(title='Week', color='#8B90A0', gridcolor='#2A2D3E'),
            yaxis=dict(title='Avg Clicks / Week', color='#8B90A0', gridcolor='#2A2D3E'),
            legend_title='Result',
            margin=dict(t=40,b=10,l=10,r=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 3: Demographics ───────────────────────────────────────────────────
    st.markdown('<div class="eyebrow" style="margin-top:1rem">Demographics</div>',
                unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        gd = df.groupby('gender')['dropout'].mean().mul(100).reset_index()
        gd.columns = ['Gender','Dropout Rate (%)']
        fig = px.bar(gd, x='Gender', y='Dropout Rate (%)',
                     color='Dropout Rate (%)',
                     color_continuous_scale=['#2ECC71','#E74C3C'],
                     text_auto='.1f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#E8EAF0', coloraxis_showscale=False,
                          title=dict(text="By Gender", font_color='#8B90A0', font_size=11),
                          margin=dict(t=30,b=5,l=5,r=5), height=220,
                          xaxis=dict(color='#8B90A0'), yaxis=dict(color='#8B90A0'))
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        ad = df.groupby('age_band')['dropout'].mean().mul(100).reset_index()
        ad.columns = ['Age Band','Dropout Rate (%)']
        fig = px.bar(ad, x='Age Band', y='Dropout Rate (%)',
                     color='Dropout Rate (%)',
                     color_continuous_scale=['#2ECC71','#E74C3C'], text_auto='.1f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#E8EAF0', coloraxis_showscale=False,
                          title=dict(text="By Age Band", font_color='#8B90A0', font_size=11),
                          margin=dict(t=30,b=5,l=5,r=5), height=220,
                          xaxis=dict(color='#8B90A0'), yaxis=dict(color='#8B90A0'))
        st.plotly_chart(fig, use_container_width=True)

    with d3:
        dd = df.groupby('disability')['dropout'].mean().mul(100).reset_index()
        dd.columns = ['Disability','Dropout Rate (%)']
        fig = px.bar(dd, x='Disability', y='Dropout Rate (%)',
                     color='Dropout Rate (%)',
                     color_continuous_scale=['#2ECC71','#E74C3C'], text_auto='.1f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#E8EAF0', coloraxis_showscale=False,
                          title=dict(text="By Disability Status", font_color='#8B90A0', font_size=11),
                          margin=dict(t=30,b=5,l=5,r=5), height=220,
                          xaxis=dict(color='#8B90A0'), yaxis=dict(color='#8B90A0'))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Score & Engagement ─────────────────────────────────────────────
    st.markdown('<div class="eyebrow" style="margin-top:1rem">Academic & Engagement Signals</div>',
                unsafe_allow_html=True)
    e1, e2 = st.columns(2)

    with e1:
        fig = go.Figure()
        for label, val, color in [('Non-Withdrawn', 0, '#2ECC71'), ('Withdrawn', 1, '#E74C3C')]:
            data = df[df['dropout'] == val]['mean_score'].dropna()
            fig.add_trace(go.Violin(x=[label]*len(data), y=data, name=label,
                                    fillcolor=color, line_color=color,
                                    opacity=0.7, box_visible=True, meanline_visible=True))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#E8EAF0', showlegend=False,
                          title=dict(text="Score Distribution by Status",
                                     font_color='#8B90A0', font_size=11),
                          yaxis=dict(color='#8B90A0', gridcolor='#2A2D3E'),
                          xaxis=dict(color='#8B90A0'),
                          margin=dict(t=30,b=10,l=10,r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with e2:
        fig = go.Figure()
        for label, val, color in [('Non-Withdrawn', 0, '#2ECC71'), ('Withdrawn', 1, '#E74C3C')]:
            data = df[df['dropout'] == val]['miss_ratio'].dropna()
            fig.add_trace(go.Box(x=[label]*len(data), y=data, name=label,
                                  marker_color=color, line_color=color))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#E8EAF0', showlegend=False,
                          title=dict(text="Assessment Miss Ratio by Status",
                                     font_color='#8B90A0', font_size=11),
                          yaxis=dict(title='Miss Ratio', color='#8B90A0', gridcolor='#2A2D3E'),
                          xaxis=dict(color='#8B90A0'),
                          margin=dict(t=30,b=10,l=10,r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown("# 🔍 Student Risk Predictor")

    if active_model is None:
        st.error("No model available. "
                 "For Saved Model mode, ensure `GWE 2026 Data Science Challenge/models/lgbm_dropout_model.pkl` exists. "
                 "For Raw Dataset mode, upload all 7 CSVs first.")
        st.stop()

    predict_tab, batch_tab = st.tabs(["👤 Single Student", "📋 Batch Upload"])

    # ── Single Student ────────────────────────────────────────────────────────
    with predict_tab:
        st.markdown('<div class="eyebrow">Student Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographic**")
            gender   = st.selectbox("Gender", ["Male", "Female"])
            age_band = st.selectbox("Age Band", AGE_ORDER)
            edu      = st.selectbox("Highest Education", EDU_ORDER)
            imd      = st.selectbox("Deprivation Band (IMD)", IMD_ORDER, index=4)
            disability = st.checkbox("Has Disability")
            prev_attempts = st.number_input("Previous Attempts", 0, 10, 0)

        with c2:
            st.markdown("**Academic**")
            mean_score  = st.slider("Mean Assessment Score", 0.0, 100.0, 60.0, 0.5)
            std_score   = st.slider("Score Std Dev",         0.0, 40.0,  10.0, 0.5)
            miss_ratio  = st.slider("Miss Ratio (0=none, 1=all)", 0.0, 1.0, 0.1, 0.01)
            score_trend = st.slider("Score Trend (late − early)", -50.0, 50.0, 0.0, 0.5)
            num_assess  = st.number_input("Assessments Taken", 0, 30, 5)
            sub_delay   = st.number_input("Avg Submission Delay (days)", -30, 30, 0)

        with c3:
            st.markdown("**Engagement (VLE)**")
            total_clicks     = st.number_input("Total Clicks",       0, 50000, 500)
            active_days      = st.number_input("Active Days",        0, 365,   40)
            avg_daily_clicks = st.number_input("Avg Daily Clicks",   0, 500,   12)
            click_trend      = st.number_input("Click Trend (late−early)", -5000, 5000, 0)
            engagement_rate  = st.slider("Engagement Rate",          0.0, 1.0, 0.5, 0.01)
            material_div     = st.number_input("Material Diversity (types)", 0, 15, 5)
            activity_span    = st.number_input("Activity Span (days)", 0, 365, 120)
            early_unreg      = st.checkbox("Early Unregistration")
            days_unreg       = st.number_input("Days Until Unreg (999=N/A)", 0, 999, 999)

        st.markdown("---")

        if st.button("⚡ Predict Dropout Risk", type="primary", use_container_width=True):
            row = {
                'gender_encoded':      1 if gender == "Male" else 0,
                'disability_encoded':  int(disability),
                'age_band_encoded':    AGE_ORDER.index(age_band),
                'imd_band_encoded':    IMD_ORDER.index(imd),
                'education_encoded':   EDU_ORDER.index(edu),
                'num_of_prev_attempts': prev_attempts,
                'mean_score':          mean_score,
                'std_score':           std_score,
                'miss_ratio':          miss_ratio,
                'score_trend':         score_trend,
                'num_assessments':     num_assess,
                'avg_submission_delay': sub_delay,
                'total_clicks':        total_clicks,
                'active_days':         active_days,
                'avg_daily_clicks':    avg_daily_clicks,
                'engagement_rate':     engagement_rate,
                'click_trend':         click_trend,
                'material_diversity':  material_div,
                'activity_span':       activity_span,
                'early_unregistration': int(early_unreg),
                'days_until_unreg':    days_unreg,
            }
            X_input = pd.DataFrame([row])[FEATURES]
            prob    = float(active_model.predict_proba(X_input)[0, 1])
            risk    = get_risk_tier(prob)
            row['dropout_prob'] = prob

            # Results layout
            g1, g2 = st.columns([1, 2])
            with g1:
                st.plotly_chart(make_gauge(prob, risk), use_container_width=True)
                badge_cls = f"badge-{risk.lower()}"
                st.markdown(f"""
                <div style="text-align:center;margin-top:-1rem">
                    <span class="badge {badge_cls}">{risk} RISK</span>
                </div>
                """, unsafe_allow_html=True)

            with g2:
                color_map = {'HIGH':'#E74C3C','MEDIUM':'#F39C12','LOW':'#2ECC71'}
                card_cls  = {'HIGH':'card-red','MEDIUM':'card-orange','LOW':'card-green'}[risk]
                st.markdown(f"""
                <div class="card {card_cls}" style="margin-top:1rem">
                    <div class="eyebrow">Risk Assessment</div>
                    <h3 style="color:{color_map[risk]};margin:0.3rem 0">
                        {prob*100:.1f}% Dropout Probability
                    </h3>
                    <p style="color:#8B90A0;font-size:0.85rem;margin:0">
                        {'This student shows strong risk signals. Immediate action recommended.'
                         if risk=='HIGH' else
                         'Moderate risk detected. Proactive monitoring advised.'
                         if risk=='MEDIUM' else
                         'Low risk. Continue regular engagement.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                interventions = recommend_intervention(row)
                st.markdown('<div class="eyebrow" style="margin-top:1rem">Recommended Interventions</div>',
                            unsafe_allow_html=True)
                chips = "".join([f'<span class="chip">{i}</span>' for i in interventions])
                st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

            # Key signals
            st.markdown('<div class="eyebrow" style="margin-top:1.5rem">Key Signals</div>',
                        unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            signals = [
                ("Mean Score",    f"{mean_score:.0f}/100", mean_score < 50),
                ("Miss Ratio",    f"{miss_ratio*100:.0f}%",  miss_ratio > 0.3),
                ("Total Clicks",  f"{total_clicks:,}",     total_clicks < 100),
                ("Score Trend",   f"{score_trend:+.1f}",   score_trend < -5),
            ]
            for col, (lbl, val, warn) in zip([s1,s2,s3,s4], signals):
                color = "#E74C3C" if warn else "#2ECC71"
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-val" style="color:{color};font-size:1.5rem">{val}</div>
                        <div class="metric-lbl">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Batch Upload ──────────────────────────────────────────────────────────
    with batch_tab:
        st.markdown('<div class="eyebrow">Batch Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown(
            "Upload a CSV with columns matching the feature list below. "
            "The app will score every student and return a risk report you can download."
        )

        with st.expander("📋 Required CSV columns"):
            st.code(", ".join(FEATURES))

        batch_file = st.file_uploader("Upload student roster CSV", type="csv", key="batch")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            missing_cols = [c for c in FEATURES if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                X_batch  = batch_df[FEATURES].fillna(0)
                probs    = active_model.predict_proba(X_batch)[:, 1]
                batch_df['dropout_prob'] = probs
                batch_df['risk_tier']    = [get_risk_tier(p) for p in probs]
                batch_df['intervention'] = batch_df.apply(
                    lambda r: ' | '.join(recommend_intervention(r.to_dict())), axis=1
                )

                # Summary
                tier_counts = batch_df['risk_tier'].value_counts()
                st.markdown(f"""
                <div class="metric-row" style="margin-top:1rem">
                    <div class="metric-box">
                        <div class="metric-val" style="color:#4F8EF7">{len(batch_df):,}</div>
                        <div class="metric-lbl">Students Scored</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" style="color:#E74C3C">{tier_counts.get('HIGH',0)}</div>
                        <div class="metric-lbl">High Risk</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" style="color:#F39C12">{tier_counts.get('MEDIUM',0)}</div>
                        <div class="metric-lbl">Medium Risk</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" style="color:#2ECC71">{tier_counts.get('LOW',0)}</div>
                        <div class="metric-lbl">Low Risk</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Table — show high risk first
                show_cols = ['dropout_prob','risk_tier','intervention'] + \
                            [c for c in ['id_student','mean_score','miss_ratio','total_clicks']
                             if c in batch_df.columns]
                st.dataframe(
                    batch_df[show_cols].sort_values('dropout_prob', ascending=False),
                    use_container_width=True,
                    height=360,
                )

                # Download
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Full Risk Report (CSV)",
                    csv_out, "dropout_risk_report.csv", "text/csv",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 About":
    st.markdown("# 📋 About This Project")
    st.markdown("---")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <div class="card card-blue">
            <div class="eyebrow">The Model</div>
            <h4 style="margin:0.3rem 0">LightGBM Classifier</h4>
            <p style="color:#8B90A0;font-size:0.85rem">
            Gradient boosted trees trained on OULAD with stratified sampling.
            Class imbalance handled via SMOTE. Hyperparameters: 300 estimators,
            learning rate 0.05, 63 leaves, balanced class weight.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-green">
            <div class="eyebrow">Dataset</div>
            <h4 style="margin:0.3rem 0">OULAD — Open University Learning Analytics</h4>
            <p style="color:#8B90A0;font-size:0.85rem">
            ~32,000 students · 7 courses · UK Open University<br>
            Features span demographics, assessment scores,
            and 10M+ rows of VLE clickstream data.
            <a href="https://analyse.kmi.open.ac.uk/open_dataset"
               style="color:#4F8EF7">Official source ↗</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="card card-orange">
            <div class="eyebrow">Performance (estimated)</div>
            <h4 style="margin:0.3rem 0">Model Metrics</h4>
        """, unsafe_allow_html=True)

        metrics = {
            'ROC-AUC': ('~0.91', '#4F8EF7'),
            'Recall':  ('~0.85', '#2ECC71'),
            'F1 Score':('~0.83', '#F39C12'),
            'Precision':('~0.81','#9B59B6'),
        }
        cols = st.columns(4)
        for col, (k, (v, c)) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"""
                <div style="text-align:center;padding:0.5rem 0">
                    <div style="font-size:1.4rem;font-weight:700;color:{c}">{v}</div>
                    <div style="font-size:0.7rem;color:#8B90A0;text-transform:uppercase;
                                letter-spacing:0.08em">{k}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-red" style="margin-top:1rem">
            <div class="eyebrow">Features Used</div>
            <h4 style="margin:0.3rem 0">21 Features across 4 categories</h4>
            <p style="color:#8B90A0;font-size:0.85rem">
            <strong>Demographic</strong>: gender, age, education, deprivation index, disability<br>
            <strong>Academic</strong>: scores, miss ratio, trend, submission delay<br>
            <strong>Behavioral</strong>: clicks, active days, engagement rate, click trend<br>
            <strong>Registration</strong>: early unregistration, days until withdrawal
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="eyebrow">Redundancy Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p style="color:#8B90A0;font-size:0.85rem;margin:0">
    This app implements a dual-mode data strategy so it remains usable in any environment:
    <br><br>
    <strong style="color:#4F8EF7">🤖 Saved Model Mode</strong> — Loads the pre-trained
    <code>lgbm_dropout_model.pkl</code> directly. No data upload required.
    Predictions are instant. Best for production demos and advisors who don't
    have access to raw student data files.
    <br><br>
    <strong style="color:#2ECC71">📂 Raw Dataset Mode</strong> — Accepts all 7 OULAD CSVs,
    runs the full stratified-sampling + feature-engineering pipeline, then trains a fresh
    LightGBM model on the uploaded cohort. Best for researchers or institutions who want
    results specific to their own student population. Both modes converge on the same
    prediction and intervention logic downstream.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="eyebrow">Built for</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-blue">
        <strong>GWE 2026 Data Science Challenge</strong> · Grow With EDM Gen 7<br>
        <span style="color:#8B90A0;font-size:0.85rem">
        Theme: Leveraging Data for Societal and Industrial Impact<br>
        Sub-theme: Education · Risk Prediction
        </span>
    </div>
    """, unsafe_allow_html=True)