"""
K-평균 군집화를 활용한 수학 학습자 유형 분석
Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════
st.set_page_config(
    page_title="수학 학습자 유형 분석",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════
# 상수 정의
# ══════════════════════════════════════════
FACTOR_ITEMS = {
    "math_anxiety_mean": [
        "A1. 수학 시간에 어려운 문제가 나오면 긴장된다.",
        "A2. 수학 시험을 생각하면 걱정이 앞선다.",
        "A3. 수학 문제를 풀다가 막히면 당황해서 더 생각하기 어렵다.",
        "A4. 친구들 앞에서 수학 문제를 풀거나 발표하는 것이 부담스럽다.",
        "A5. 수학 성적이 잘 나오지 않을까 봐 불안하다.",
        "A6. 수학 시간에 선생님이 질문할까 봐 긴장될 때가 있다.",
        "A7. 수학 문제를 보면 시작하기 전부터 어렵게 느껴진다.",
        "A8. 수학 시험 중 시간이 부족할 것 같으면 머리가 하얘진다.",
    ],
    "math_self_efficacy_mean": [
        "E1. 나는 노력하면 수학 실력을 향상시킬 수 있다고 생각한다.",
        "E2. 처음에는 어려운 수학 문제도 차근차근 생각하면 해결할 수 있다.",
        "E3. 수학 문제를 틀려도 다시 도전할 수 있다.",
        "E4. 나는 수학 수업 내용을 이해할 수 있다는 자신감이 있다.",
        "E5. 새로운 수학 개념도 설명을 들으면 이해할 수 있다고 생각한다.",
        "E6. 수학 과제를 끝까지 해낼 수 있다.",
        "E7. 시험에서 모르는 문제가 나와도 아는 내용을 활용해 보려고 한다.",
        "E8. 나는 수학 공부 방법을 스스로 조절할 수 있다.",
    ],
    "math_interest_mean": [
        "I1. 수학 문제를 해결했을 때 성취감을 느낀다.",
        "I2. 수학 시간에 배우는 내용이 흥미롭다고 느낄 때가 있다.",
        "I3. 새로운 수학 개념을 배우는 것이 재미있다.",
        "I4. 수학이 실생활이나 다른 분야와 연결된다는 점이 흥미롭다.",
        "I5. 어려운 문제를 고민해 보는 과정이 의미 있다고 생각한다.",
        "I6. 수학 관련 활동이나 탐구 과제에 참여해 보고 싶다.",
        "I7. 수학을 잘하면 앞으로 도움이 될 것이라고 생각한다.",
        "I8. 수학 시간에 적극적으로 참여하고 싶다.",
    ],
    "learning_attitude_mean": [
        "T1. 나는 수학 공부 계획을 세우고 실천하는 편이다.",
        "T2. 수학 숙제나 과제를 성실히 하는 편이다.",
        "T3. 모르는 수학 문제가 있으면 질문하거나 찾아보는 편이다.",
        "T4. 수학 공부를 할 때 집중이 잘 되는 편이다.",
        "T5. 수학 수업에서 친구들과 함께 문제를 해결하는 것이 도움이 된다.",
        "T6. 수학을 포기하고 싶다고 느낄 때가 있다.",  # 역채점
    ],
}
NEGATIVE_ITEM = "T6. 수학을 포기하고 싶다고 느낄 때가 있다."

FACTOR_LABELS = {
    "math_anxiety_mean":       "수학불안",
    "math_self_efficacy_mean": "자기효능감",
    "math_interest_mean":      "수학흥미",
    "learning_attitude_mean":  "학습태도",
}

# 따뜻한 색상 팔레트
C_COLORS = ["#E07B54", "#5B8DB8", "#6BAA75"]   # 테라코타·스틸블루·세이지
C_NAMES  = ["취약형", "중간형", "강점형"]
C_DESCS  = [
    "수학불안이 높고 자기효능감·흥미·학습태도가 낮은 집단 — 심리적 지원과 기초 강화가 필요합니다.",
    "모든 요인이 평균 수준인 집단 — 동기 유발과 개인별 잠재력 발견이 중요합니다.",
    "수학불안이 낮고 자기효능감·흥미·학습태도가 높은 집단 — 심화 도전 기회를 제공합니다.",
]
STRATEGIES = [
    {
        "title": "심리 안정 + 기초 강화",
        "items": [
            "경쟁보다 협력 분위기 조성, 실수 허용 문화 정착",
            "개별화 피드백으로 작은 성취도 칭찬·격려",
            "수학 불안 관리 전략 교육 (심호흡, 긍정적 자기 대화)",
            "쉬운 문제부터 단계적 난이도 향상, 성공 경험 누적",
            "놀이·게임·실생활 연계로 수학에 대한 흥미 유발",
        ],
    },
    {
        "title": "동기 유발 + 잠재력 발굴",
        "items": [
            "학습 스타일·선호도 파악 후 맞춤 경험 제공",
            "수학-실생활 연결로 학습 의미 탐색",
            "토론·발표 기회 확대로 참여 의욕 높이기",
            "또래 멘토링·소그룹으로 편안한 학습 환경 조성",
            "학습 일지·면담으로 자기 성찰 기회 제공",
        ],
    },
    {
        "title": "심화 도전 + 창의적 확장",
        "items": [
            "현 수준을 넘는 심화 문제·탐구 과제 제공",
            "경시 대회·동아리·멘토링 참여 기회 연결",
            "자기 주도 학습 목표 설정 및 실행 독려",
            "다양한 풀이 전략 탐색, 비판적 사고 훈련",
            "동료 설명·리더십 역할로 깊은 이해 도모",
        ],
    },
]

FACTOR_META = {
    "math_anxiety_mean":       ("A", "#E07B54", "#fdf2ee"),
    "math_self_efficacy_mean": ("E", "#5B8DB8", "#eef3f9"),
    "math_interest_mean":      ("I", "#6BAA75", "#eef6f0"),
    "learning_attitude_mean":  ("T", "#B8975B", "#f9f5ee"),
}

# ══════════════════════════════════════════
# CSS  (따뜻하고 심플한 톤)
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background: #faf9f7; }
.block-container { padding-top: 28px !important; max-width: 1080px !important; }
[data-testid="stSidebar"] { background: #fff; }

/* ── 탭 스타일 ── */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #e8e3dc; }
[data-baseweb="tab"] {
    font-size: 14px; font-weight: 500; color: #9e9387;
    padding: 10px 18px; border-radius: 8px 8px 0 0;
    background: transparent; border: none;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #3d2b1f; border-bottom: 2px solid #E07B54;
    background: #fff8f5;
}

/* ── 히어로 ── */
.hero {
    background: linear-gradient(135deg, #fff8f5 0%, #f5f0ea 100%);
    border: 1px solid #e8e0d5; border-radius: 16px;
    padding: 42px 48px 36px; margin-bottom: 28px;
}
.hero-kicker {
    font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #E07B54; margin-bottom: 10px;
}
.hero h1 {
    font-size: 24px; font-weight: 700; color: #2d1f14;
    letter-spacing: -.3px; margin: 0 0 8px;
}
.hero .sub { font-size: 15px; color: #6b5c52; line-height: 1.7; margin: 0 0 28px; }
.hero-line { height: 1px; background: #e8e0d5; margin: 0 0 24px; }
.feat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.feat {
    background: #fff; border: 1px solid #e8e0d5; border-radius: 12px;
    padding: 16px; text-align: center;
}
.feat .fi { font-size: 22px; margin-bottom: 7px; }
.feat .ft { font-size: 13px; font-weight: 600; color: #2d1f14; margin-bottom: 4px; }
.feat .fd { font-size: 12px; color: #9e9387; line-height: 1.5; }

/* ── 공통 카드 ── */
.card {
    background: #fff; border: 1px solid #e8e0d5;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.card-warm {
    background: #fff8f5; border: 1px solid #f0e4da;
    border-radius: 12px; padding: 16px 20px; margin-bottom: 14px;
}

/* ── 섹션 헤더 ── */
.sec-tag { font-size: 11px; font-weight: 700; letter-spacing: 2px;
           text-transform: uppercase; color: #E07B54; margin-bottom: 2px; }
.sec-h { font-size: 18px; font-weight: 700; color: #2d1f14; margin-bottom: 4px; }
.sec-d { font-size: 13px; color: #6b5c52; margin-bottom: 18px; line-height: 1.6; }

/* ── 메트릭 박스 ── */
.mbox {
    background: #fff; border: 1px solid #e8e0d5; border-radius: 12px;
    padding: 18px 20px; text-align: center;
}
.mbox-val { font-size: 30px; font-weight: 700; color: #2d1f14; line-height: 1; }
.mbox-lbl { font-size: 11px; color: #9e9387; margin-top: 5px;
            text-transform: uppercase; letter-spacing: 1px; }

/* ── 문항 블록 ── */
.iblock { border: 1px solid #e8e0d5; border-radius: 10px;
          overflow: hidden; margin-bottom: 14px; }
.ihead { display: flex; align-items: center; gap: 10px; padding: 10px 16px;
         border-bottom: 1px solid #e8e0d5; }
.ibadge { display: inline-flex; align-items: center; justify-content: center;
          width: 26px; height: 26px; border-radius: 6px;
          font-size: 12px; font-weight: 700; }
.ihtit { font-size: 13px; font-weight: 600; color: #2d1f14; }
.ihsub { font-size: 12px; color: #9e9387; margin-left: auto; }
.irow { display: flex; gap: 10px; padding: 8px 16px;
        border-bottom: 1px solid #f5f0ea; background: #fff; font-size: 13px; }
.irow:last-child { border-bottom: none; }
.irow:hover { background: #faf9f7; }
.inum { font-family: monospace; font-size: 11px; color: #9e9387;
        min-width: 26px; padding-top: 2px; }
.itxt { color: #4e3d35; flex: 1; line-height: 1.5; }
.irev { font-size: 10px; color: #E07B54; background: #fdf2ee;
        padding: 2px 6px; border-radius: 3px; white-space: nowrap; align-self: flex-start; margin-top: 1px; }

/* ── 인식 결과 ── */
.rok { background: #eef6f0; border-radius: 7px; padding: 8px 14px;
       font-size: 13px; color: #3a7a49; display: flex; align-items: center;
       gap: 8px; margin-bottom: 5px; }
.rfail { background: #fdf2ee; border-radius: 7px; padding: 8px 14px;
         font-size: 13px; color: #c05c3a; display: flex; align-items: center;
         gap: 8px; margin-bottom: 5px; }

/* ── 군집 카드 ── */
.ccard {
    background: #fff; border: 1px solid #e8e0d5; border-radius: 12px;
    padding: 18px 20px; border-top: 4px solid;
}
.ccard-lbl { font-size: 10px; font-weight: 700; letter-spacing: 2px;
             text-transform: uppercase; margin-bottom: 4px; }
.ccard-name { font-size: 17px; font-weight: 700; color: #2d1f14; margin-bottom: 5px; }
.ccard-desc { font-size: 12px; color: #6b5c52; line-height: 1.5; margin-bottom: 12px; }
.cstat { display: flex; justify-content: space-between; font-size: 12px;
         padding: 4px 0; border-bottom: 1px solid #f5f0ea; }
.cstat:last-child { border-bottom: none; }

/* ── 전략 카드 ── */
.stcard {
    background: #fff; border: 1px solid #e8e0d5; border-radius: 12px;
    padding: 20px; border-left: 4px solid; height: 100%;
}
.stlbl { font-size: 10px; font-weight: 700; letter-spacing: 2px;
         text-transform: uppercase; margin-bottom: 4px; }
.sttit { font-size: 14px; font-weight: 700; color: #2d1f14; margin-bottom: 12px; }
.stli li { font-size: 12px; color: #6b5c52; line-height: 1.8; margin-bottom: 2px; }

/* ── 인포 박스 ── */
.infobox {
    background: #fff8f5; border: 1px solid #f0e4da;
    border-radius: 8px; padding: 13px 18px; margin-bottom: 16px;
}
.infobox .ilbl { font-size: 10px; font-weight: 700; letter-spacing: 2px;
                 text-transform: uppercase; color: #E07B54; margin-bottom: 4px; }
.infobox p { font-size: 13px; color: #6b5c52; line-height: 1.7; margin: 0; }

/* ── 결론 배너 ── */
.cbanner {
    background: linear-gradient(135deg, #3d2b1f 0%, #5a3e2b 100%);
    border-radius: 14px; padding: 28px 32px; color: #fff;
    margin-bottom: 24px; display: flex; align-items: center; gap: 24px;
}
.cbn { font-size: 52px; font-weight: 700; opacity: .1;
       font-family: monospace; line-height: 1; }
.cbh { font-size: 19px; font-weight: 700; margin: 0 0 6px; }
.cbp { font-size: 13px; opacity: .8; line-height: 1.7; margin: 0; }

/* ── 업로드 ── */
.uzone {
    background: #fff; border: 2px dashed #d5c9bc; border-radius: 12px;
    padding: 28px; text-align: center; margin-bottom: 18px;
}
.uzt { font-size: 15px; font-weight: 600; color: #2d1f14; margin-bottom: 5px; }
.uzs { font-size: 13px; color: #9e9387; }

/* ── 설정 패널 ── */
.setpanel {
    background: #fff8f5; border: 1px solid #e8e0d5;
    border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;
}
.setpanel-t { font-size: 13px; font-weight: 600; color: #2d1f14;
              margin-bottom: 14px; padding-bottom: 10px;
              border-bottom: 1px solid #e8e0d5; }

/* ── 구분선 ── */
.hdiv { height: 1px; background: #e8e0d5; margin: 22px 0; }

/* ── 상태바 ── */
.statusbar {
    font-size: 11px; color: #9e9387; text-align: right;
    margin-bottom: -10px; padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 데이터 처리
# ══════════════════════════════════════════
def _compute_factors(raw_df: pd.DataFrame):
    recognized = {}
    df_f = pd.DataFrame(index=raw_df.index)
    for fk, items in FACTOR_ITEMS.items():
        found = [c for c in items if c in raw_df.columns]
        recognized[fk] = found
        if found:
            tmp = raw_df[found].copy()
            for c in found:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            if fk == "learning_attitude_mean" and NEGATIVE_ITEM in tmp.columns:
                tmp[NEGATIVE_ITEM] = 6 - tmp[NEGATIVE_ITEM]
            df_f[fk] = tmp.mean(axis=1)
        else:
            df_f[fk] = np.nan
    avail = [v for v in FACTOR_ITEMS if df_f[v].notna().sum() > 0]
    return df_f, recognized, avail


def load_file(file_bytes: bytes, fname: str):
    try:
        engine = "xlrd" if fname.lower().endswith(".xls") else "openpyxl"
        raw = pd.read_excel(io.BytesIO(file_bytes), engine=engine)
    except Exception:
        raw = pd.read_excel(io.BytesIO(file_bytes))
    raw.columns = raw.columns.astype(str).str.strip()
    df_f, recognized, avail = _compute_factors(raw)
    return raw, df_f, recognized, avail


def make_demo():
    np.random.seed(42)
    n = 73
    i1, i2, i3 = np.arange(0, 21), np.arange(21, 54), np.arange(54, 73)
    demo = {}
    for col in FACTOR_ITEMS["math_anxiety_mean"]:
        a = np.zeros(n, int)
        a[i1] = np.random.choice([4,5], len(i1), p=[.4,.6])
        a[i2] = np.random.choice([2,3,4], len(i2), p=[.3,.4,.3])
        a[i3] = np.random.choice([1,2], len(i3), p=[.5,.5])
        demo[col] = a
    for fk in ["math_self_efficacy_mean","math_interest_mean","learning_attitude_mean"]:
        for col in FACTOR_ITEMS[fk]:
            a = np.zeros(n, int)
            a[i1] = np.random.choice([1,2], len(i1), p=[.5,.5])
            a[i2] = np.random.choice([2,3,4], len(i2), p=[.3,.4,.3])
            a[i3] = np.random.choice([4,5], len(i3), p=[.4,.6])
            demo[col] = a
    raw = pd.DataFrame(demo)
    raw.columns = raw.columns.astype(str).str.strip()
    df_f, recognized, avail = _compute_factors(raw)
    return raw, df_f, recognized, avail


@st.cache_data(show_spinner=False)
def run_kmeans(fp: bytes, sel_key: str, k: int):
    df = pd.read_parquet(io.BytesIO(fp))
    sel = sel_key.split("|")
    X = df[sel].dropna()
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    df_res = df.loc[X.index, sel].copy()
    df_res["cluster"] = labels + 1
    df_z = pd.DataFrame(X_sc, columns=sel, index=X.index)
    df_z["cluster"] = labels + 1
    pr = df_res.groupby("cluster")[sel].mean()
    pz = df_z.groupby("cluster")[sel].mean()
    return df_res, pr, pz, X_sc


@st.cache_data(show_spinner=False)
def calc_elbow(fp: bytes, sel_key: str, k_min: int, k_max: int):
    df = pd.read_parquet(io.BytesIO(fp))
    sel = sel_key.split("|")
    X = df[sel].dropna()
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    ks, vals = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        ks.append(k); vals.append(round(km.inertia_, 2))
    return ks, vals


# ══════════════════════════════════════════
# UI 헬퍼
# ══════════════════════════════════════════
def sec(tag, h, d=""):
    html = f'<div class="sec-tag">{tag}</div><div class="sec-h">{h}</div>'
    if d:
        html += f'<div class="sec-d">{d}</div>'
    st.markdown(html, unsafe_allow_html=True)


def infobox(lbl, body):
    st.markdown(
        f'<div class="infobox"><div class="ilbl">{lbl}</div><p>{body}</p></div>',
        unsafe_allow_html=True)


def hdiv():
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)


def clabel(i, k):
    return f"C{i+1} {C_NAMES[i] if i < len(C_NAMES) else ''}"


# ══════════════════════════════════════════
# 차트 함수
# ══════════════════════════════════════════
PLOT_FONT = dict(family="Noto Sans KR, sans-serif", size=12, color="#4e3d35")

def _layout(**kw):
    base = dict(
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=PLOT_FONT,
        margin=dict(t=14, b=44, l=56, r=20),
    )
    base.update(kw)
    return base


def chart_elbow(ks, inertias, k_sel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias, mode="lines+markers+text",
        line=dict(color="#E07B54", width=2.5),
        marker=dict(
            size=[11 if k == k_sel else 7 for k in ks],
            color=["#c05c3a" if k == k_sel else "#E07B54" for k in ks],
            line=dict(width=2, color="#fff"),
        ),
        text=[str(round(v,1)) for v in inertias],
        textposition="top center", textfont=dict(size=11),
    ))
    if k_sel in ks:
        idx = ks.index(k_sel)
        fig.add_annotation(x=k_sel, y=inertias[idx],
                           text=f"  ← K={k_sel}",
                           showarrow=False, font=dict(color="#c05c3a", size=12),
                           xanchor="left")
        fig.add_vline(x=k_sel, line_dash="dot",
                      line_color="rgba(192,92,58,0.3)", line_width=1.5)
    fig.update_layout(
        **_layout(height=320),
        xaxis=dict(title="클러스터 수 K", tickvals=ks, gridcolor="#f5f0ea",
                   tickfont=dict(size=12)),
        yaxis=dict(title="Inertia", gridcolor="#f5f0ea"),
        showlegend=False,
    )
    return fig


def chart_bar_count(df_res, k):
    counts = df_res["cluster"].value_counts().sort_index()
    labels = [clabel(i, k) for i in range(k)]
    vals   = [counts.get(i+1, 0) for i in range(k)]
    colors = C_COLORS[:k]
    mc = [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18)"
          for c in colors]
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=mc, marker_line_color=colors, marker_line_width=1.5,
        text=vals, textposition="outside",
        textfont=dict(size=14, color=colors),
    ))
    fig.update_layout(
        **_layout(height=300),
        yaxis=dict(title="응답자 수 (명)", gridcolor="#f5f0ea"),
        showlegend=False,
    )
    return fig


def chart_heatmap(profile: pd.DataFrame, is_z: bool):
    z = profile.values
    yl = [clabel(i, len(profile)) for i in range(len(profile))]
    xl = [FACTOR_LABELS.get(c, c) for c in profile.columns]
    cs = (
        [[0,"#fdf2ee"],[0.5,"#faf9f7"],[1,"#eef6f0"]] if is_z
        else [[0,"#eef3f9"],[1,"#2d5f8a"]]
    )
    anns = []
    for i in range(len(z)):
        for j in range(len(z[0])):
            v = z[i][j]
            anns.append(dict(
                x=j, y=i,
                text=f"{'+'if is_z and v>0 else ''}{v:.2f}",
                showarrow=False,
                font=dict(size=13, color="#2d1f14", family="monospace"),
            ))
    fig = go.Figure(go.Heatmap(
        z=z, x=xl, y=yl, colorscale=cs, showscale=True, hoverongaps=False,
    ))
    fig.update_traces(
        colorbar=dict(thickness=11, len=0.8),
        annotations=anns,
    )
    fig.update_layout(
        **dict(plot_bgcolor="#fff", paper_bgcolor="#fff",
               font=PLOT_FONT, margin=dict(t=46,b=12,l=110,r=65)),
        xaxis=dict(side="top"),
        height=190 + len(profile)*32,
    )
    return fig


def chart_radar(profile_z: pd.DataFrame, show_list: list):
    cats = [FACTOR_LABELS.get(c, c) for c in profile_z.columns]
    cats_c = cats + [cats[0]]
    fig = go.Figure()
    for i, (_, row) in enumerate(profile_z.iterrows()):
        if (i+1) not in show_list:
            continue
        vals = list(row.values) + [row.values[0]]
        r, g, b = int(C_COLORS[i][1:3],16), int(C_COLORS[i][3:5],16), int(C_COLORS[i][5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats_c, fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color=C_COLORS[i], width=2.5),
            name=clabel(i, len(profile_z)),
            marker=dict(size=7, color=C_COLORS[i]),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2.5,2.5],
                            tickfont=dict(size=10), gridcolor="#e8e0d5"),
            angularaxis=dict(tickfont=dict(size=13)),
            bgcolor="#fff",
        ),
        paper_bgcolor="#fff", font=PLOT_FONT,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(t=20,b=60,l=60,r=60), height=440,
    )
    return fig


def chart_centroid(fp: bytes, sel_vars: list, k: int):
    df = pd.read_parquet(io.BytesIO(fp))
    X = df[sel_vars].dropna()
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sc)

    rng = np.random.RandomState(42)
    cur = X2[rng.choice(len(X2), k, replace=False)].copy()
    steps = 6
    fl, fc = [], []
    for _ in range(steps):
        d = np.linalg.norm(X2[:,None]-cur[None,:], axis=2)
        lbl = d.argmin(axis=1)
        fl.append(lbl.copy())
        nc = np.array([X2[lbl==ci].mean(axis=0) if (lbl==ci).any()
                       else cur[ci] for ci in range(k)])
        fc.append(nc.copy()); cur = nc

    titles = ["초기 배정"] + [f"반복 {i}회" for i in range(1,5)] + ["수렴(최종)"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                        horizontal_spacing=.07, vertical_spacing=.15)
    for fi in range(steps):
        r, c = fi//3+1, fi%3+1
        for ci in range(k):
            mask = fl[fi]==ci
            ri, gi, bi = int(C_COLORS[ci][1:3],16), int(C_COLORS[ci][3:5],16), int(C_COLORS[ci][5:7],16)
            fig.add_trace(go.Scatter(
                x=X2[mask,0], y=X2[mask,1], mode="markers",
                marker=dict(color=f"rgba({ri},{gi},{bi},0.7)", size=5),
                name=clabel(ci,k) if fi==0 else None,
                showlegend=(fi==0), legendgroup=f"C{ci+1}",
            ), row=r, col=c)
        for ci in range(k):
            cx_, cy_ = fc[fi][ci]
            fig.add_trace(go.Scatter(
                x=[cx_], y=[cy_], mode="markers",
                marker=dict(symbol="x", size=13, color=C_COLORS[ci],
                            line=dict(width=2.5, color="#2d1f14")),
                showlegend=False,
            ), row=r, col=c)
    fig.update_layout(
        height=500, paper_bgcolor="#fff", font=PLOT_FONT,
        legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center"),
        margin=dict(t=40,b=40,l=18,r=18),
    )
    for ax in fig.layout:
        if ax.startswith(("xaxis","yaxis")):
            fig.layout[ax].update(showgrid=True, gridcolor="#f5f0ea", zeroline=False)
    return fig


def chart_scatter(df_res, df_f, x_var, y_var, k):
    merged = df_res[["cluster"]].join(df_f[[x_var,y_var]], how="inner").dropna()
    fig = go.Figure()
    for ci in range(k):
        sub = merged[merged["cluster"]==ci+1]
        ri,gi,bi = int(C_COLORS[ci][1:3],16), int(C_COLORS[ci][3:5],16), int(C_COLORS[ci][5:7],16)
        fig.add_trace(go.Scatter(
            x=sub[x_var], y=sub[y_var], mode="markers",
            marker=dict(color=f"rgba({ri},{gi},{bi},0.75)", size=8,
                        line=dict(width=0.5, color="#fff")),
            name=clabel(ci, k),
        ))
    cents = merged.groupby("cluster")[[x_var,y_var]].mean()
    for ci in range(k):
        if (ci+1) in cents.index:
            fig.add_trace(go.Scatter(
                x=[cents.loc[ci+1,x_var]], y=[cents.loc[ci+1,y_var]], mode="markers",
                marker=dict(symbol="x", size=15, color=C_COLORS[ci],
                            line=dict(width=3, color="#2d1f14")),
                showlegend=False,
            ))
    fig.update_layout(
        **_layout(height=380),
        xaxis=dict(title=FACTOR_LABELS.get(x_var,x_var), gridcolor="#f5f0ea", zeroline=False),
        yaxis=dict(title=FACTOR_LABELS.get(y_var,y_var), gridcolor="#f5f0ea", zeroline=False),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"),
    )
    return fig


def chart_pca(X_sc, df_res, k):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sc)
    ev = pca.explained_variance_ratio_*100
    lbl = df_res["cluster"].values
    fig = go.Figure()
    for ci in range(k):
        mask = lbl==ci+1
        if not mask.any(): continue
        ri,gi,bi = int(C_COLORS[ci][1:3],16), int(C_COLORS[ci][3:5],16), int(C_COLORS[ci][5:7],16)
        fig.add_trace(go.Scatter(
            x=X2[mask,0], y=X2[mask,1], mode="markers",
            marker=dict(color=f"rgba({ri},{gi},{bi},0.72)", size=7,
                        line=dict(width=0.5, color="#fff")),
            name=clabel(ci,k),
        ))
        cx_, cy_ = X2[mask].mean(axis=0)
        fig.add_trace(go.Scatter(
            x=[cx_], y=[cy_], mode="markers",
            marker=dict(symbol="x", size=15, color=C_COLORS[ci],
                        line=dict(width=3, color="#2d1f14")),
            showlegend=False,
        ))
    fig.update_layout(
        **_layout(height=400),
        xaxis=dict(title=f"PC1 ({ev[0]:.1f}%)", gridcolor="#f5f0ea",
                   zeroline=True, zerolinecolor="#e8e0d5"),
        yaxis=dict(title=f"PC2 ({ev[1]:.1f}%)", gridcolor="#f5f0ea",
                   zeroline=True, zerolinecolor="#e8e0d5"),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"),
    )
    return fig


# ══════════════════════════════════════════
# 탭 1 : 소개
# ══════════════════════════════════════════
def tab_intro():
    # ── 히어로 ──
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">K-Means Clustering · Math Learner Analysis</div>
      <h1>K-평균 군집화를 활용한 수학 학습자 유형 분석</h1>
      <p class="sub">
        수학불안 · 자기효능감 · 흥미 · 학습태도 설문 데이터를 업로드하면<br>
        학생 집단을 자동으로 유형화하고 시각화합니다.
      </p>
      <div class="hero-line"></div>
      <div style="font-size:11px;font-weight:700;color:#9e9387;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:12px;">이 웹앱으로 확인할 수 있는 것</div>
      <div class="feat-grid">
        <div class="feat">
          <div class="fi">📊</div>
          <div class="ft">학습자 유형 자동 분류</div>
          <div class="fd">K-평균 알고리즘으로 학생을 취약형·중간형·강점형으로 군집화</div>
        </div>
        <div class="feat">
          <div class="fi">🔍</div>
          <div class="ft">어려움 원인 진단</div>
          <div class="fd">불안·자신감·흥미·태도 4요인 중 학습을 방해하는 원인 파악</div>
        </div>
        <div class="feat">
          <div class="fi">📈</div>
          <div class="ft">인터랙티브 시각화</div>
          <div class="fd">히트맵·레이더·산점도로 군집 특성을 직관적으로 탐색</div>
        </div>
        <div class="feat">
          <div class="fi">🎯</div>
          <div class="ft">맞춤형 교수 전략</div>
          <div class="fd">군집별 특성에 맞는 구체적인 수학 수업 지도 방안 제시</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 파일 업로드 ──
    sec("Step 1", "분석 파일 업로드",
        "수학 설문 응답이 담긴 엑셀 파일(.xlsx / .xls)을 업로드하면 분석이 시작됩니다.")

    up = st.file_uploader(
        "엑셀 파일 업로드",
        type=["xlsx","xls"],
        label_visibility="collapsed",
        key="uploader",
    )
    if up is not None:
        st.session_state["file_data"] = up.read()
        st.session_state["file_name"] = up.name
        st.success(f"✅ **{up.name}** 업로드 완료 — '데이터' 탭에서 확인하세요.")
    elif "file_name" in st.session_state:
        st.info(f"📎 현재 파일: **{st.session_state['file_name']}** — 새 파일을 업로드하면 교체됩니다.")
    else:
        st.markdown("""
        <div class="uzone">
          <div class="uzt">📁 파일을 업로드하세요</div>
          <div class="uzs">지원 형식: .xlsx · .xls &nbsp;|&nbsp; 파일이 없으면 데모 데이터(73명)로 진행됩니다.</div>
        </div>""", unsafe_allow_html=True)

    hdiv()
    sec("Step 2", "파일 형식 안내",
        "아래 컬럼명이 정확히 포함되어 있어야 문항이 인식됩니다.")

    guides = [
        ("수학불안 A1–A8",  "#E07B54", "#fdf2ee",
         "A1. 수학 시간에 어려운 문제가 나오면 긴장된다."),
        ("자기효능감 E1–E8", "#5B8DB8", "#eef3f9",
         "E1. 나는 노력하면 수학 실력을 향상시킬 수 있다고 생각한다."),
        ("수학흥미 I1–I8",   "#6BAA75", "#eef6f0",
         "I1. 수학 문제를 해결했을 때 성취감을 느낀다."),
        ("학습태도 T1–T6",   "#B8975B", "#f9f5ee",
         "T1. 나는 수학 공부 계획을 세우고 실천하는 편이다."),
    ]
    cols = st.columns(4)
    for (lbl, color, bg, ex), col in zip(guides, cols):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color}30;border-radius:10px;
                        padding:14px;font-size:12px;">
              <div style="font-weight:700;color:{color};margin-bottom:6px;">{lbl}</div>
              <div style="color:#6b5c52;line-height:1.5;">
                컬럼명 예시:<br>
                <code style="font-size:10px;color:#2d1f14;">{ex}</code><br>
                … 동일 형식 반복
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    infobox("참고",
            "파일 없이 실행하면 <b>데모 데이터(73명)</b>로 모든 분석이 자동 진행됩니다. "
            "'데이터분석 및 시각화' 탭에서 바로 결과를 확인하세요.")


# ══════════════════════════════════════════
# 탭 2 : 데이터
# ══════════════════════════════════════════
def tab_data(raw, df_f, recognized, avail, is_demo):
    if is_demo:
        infobox("데모 데이터 사용 중",
                "'소개' 탭에서 엑셀 파일을 업로드하면 실제 데이터로 분석됩니다.")

    # ── 요약 메트릭 ──
    n_total   = len(raw)
    n_cols    = len(raw.columns)
    n_avail   = len(avail)
    n_found   = sum(len(v) for v in recognized.values())
    n_all_q   = sum(len(v) for v in FACTOR_ITEMS.values())
    n_missing = df_f[avail].isna().any(axis=1).sum()

    sec("Overview", "데이터 요약")
    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1,c2,c3,c4],
        [f"{n_total}명", f"{n_cols}개", f"{n_avail}개 하위요인", f"{n_found}/{n_all_q}개"],
        ["전체 응답 수","전체 컬럼 수","분석 가능 변수","인식된 문항"],
    ):
        with col:
            st.markdown(f"""
            <div class="mbox">
              <div class="mbox-val">{val}</div>
              <div class="mbox-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    if n_missing > 0:
        st.warning(f"⚠️ 결측치 포함 행 {n_missing}개 — 분석 시 자동 제외됩니다.")

    hdiv()

    # ── 원자료 미리보기 (토글) ──
    sec("Raw Data", "원자료 미리보기")
    with st.expander("📋 원자료 테이블 펼치기 / 접기", expanded=False):
        n_p = st.slider("표시 행 수", 5, min(50, n_total), 10, key="prev_n")
        st.dataframe(raw.head(n_p), use_container_width=True, height=300)
        st.caption(f"전체 {n_total}행 × {n_cols}열 중 앞 {n_p}행 표시")

    hdiv()

    # ── 문항 인식 결과 확인 버튼 ──
    sec("Item Recognition", "문항 인식 결과 확인",
        "업로드 파일에서 각 하위요인 문항이 몇 개 인식되었는지 확인합니다.")

    if st.button("🔍 문항 인식 결과 확인", type="primary", key="btn_recog"):
        st.session_state["show_recog"] = True

    if st.session_state.get("show_recog", False):
        all_ok = True
        for fk in FACTOR_ITEMS:
            letter, color, bg = FACTOR_META[fk]
            found = recognized.get(fk, [])
            total = FACTOR_ITEMS[fk]
            lbl   = FACTOR_LABELS[fk]
            ok    = len(found) == len(total)
            if not ok: all_ok = False

            st.markdown(f"""
            <div class="iblock">
              <div class="ihead" style="background:{bg};">
                <span class="ibadge" style="background:{color}20;color:{color};">{letter}</span>
                <span class="ihtit">{lbl}</span>
                <span class="ihsub" style="color:{'#3a7a49' if ok else '#c05c3a'};">
                  {'✅' if ok else '⚠️'} {len(found)}/{len(total)} 인식
                </span>
              </div>""", unsafe_allow_html=True)

            rows_html = ""
            for item in total:
                num  = item.split(".")[0]
                text = ".".join(item.split(".")[1:]).strip()
                is_f = item in found
                rev  = '<span class="irev">역채점</span>' if item == NEGATIVE_ITEM else ""
                rows_html += (f'<div class="irow">'
                              f'<div class="inum">{num}</div>'
                              f'<div class="itxt">{"✅" if is_f else "❌"} {text}</div>'
                              f'{rev}</div>')
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if all_ok:
            st.success("✅ 모든 문항이 정상적으로 인식되었습니다.")
        else:
            st.warning("⚠️ 일부 문항을 찾을 수 없습니다. 컬럼명을 확인해 주세요.")

    hdiv()

    # ── 하위요인 평균 미리보기 ──
    sec("Factor Scores", "하위요인 평균 점수 미리보기")
    with st.expander("📊 하위요인 평균 데이터 펼치기", expanded=False):
        disp = df_f[avail].rename(columns=FACTOR_LABELS)
        st.dataframe(disp.round(2).head(20), use_container_width=True)
        st.caption(f"전체 {len(disp)}행 중 앞 20행 | 결측치 포함 행은 분석에서 제외됩니다.")


# ══════════════════════════════════════════
# 탭 3 : 데이터분석 및 시각화
# ══════════════════════════════════════════
def tab_analysis(df_f, avail, fp):
    # ── 분석 설정 ──
    st.markdown('<div class="setpanel"><div class="setpanel-t">⚙️ 분석 설정</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        sel_lbls = st.multiselect(
            "군집화에 사용할 변수",
            options=[FACTOR_LABELS[v] for v in avail],
            default=[FACTOR_LABELS[v] for v in avail],
            key="sel_vars",
        )
        sel_vars = [v for v in avail if FACTOR_LABELS[v] in sel_lbls]
        if len(sel_vars) < 2:
            st.warning("최소 2개 이상 선택하세요.")
            sel_vars = avail[:2]
    with c2:
        k = int(st.number_input("클러스터 수 K", 2, 8, 3, 1, key="k_val"))
    with c3:
        k_min = int(st.number_input("Elbow K 최솟값", 2, 6, 2, 1, key="kmin"))
        k_max = int(st.number_input("Elbow K 최댓값", 3, 12, 8, 1, key="kmax"))
    st.markdown("</div>", unsafe_allow_html=True)

    if k_min >= k_max: k_max = k_min + 1
    sel_key = "|".join(sel_vars)

    with st.spinner("군집화 실행 중..."):
        df_res, pr, pz, X_sc = run_kmeans(fp, sel_key, k)

    # 결론 탭 공유
    st.session_state["result"] = (df_res, pr, pz, k)

    # ── 서브탭 ──
    s = st.tabs([
        "📉 Elbow 분석", "👥 군집 인원",
        "🗺️ 히트맵", "🕸️ 레이더 차트",
        "⚙️ Centroid 변화", "🔵 산점도", "📝 설문 문항",
    ])

    # ─ Elbow ─
    with s[0]:
        sec("Elbow", "최적 K 탐색",
            "Inertia 감소폭이 완만해지는 '팔꿈치' 지점이 최적 K 후보입니다.")
        infobox("해석 안내",
                "K가 증가할수록 Inertia(군집 내 분산)는 감소합니다. "
                "감소폭이 급격히 줄어드는 지점 이후가 팔꿈치입니다. "
                "설정 패널에서 K와 범위를 바꾸면 즉시 반영됩니다.")
        ks, ins = calc_elbow(fp, sel_key, k_min, k_max)
        st.plotly_chart(chart_elbow(ks, ins, k), use_container_width=True)
        infobox("현재 선택",
                f"<b>K={k}</b> 이후 감소폭이 완만해집니다. "
                "데이터 특성과 해석 목적을 함께 고려해 최종 K를 선택하세요.")

    # ─ 군집 인원 ─
    with s[1]:
        sec("Distribution", "군집별 응답자 수",
            f"K={k}로 분류된 각 군집의 학생 수 분포입니다.")
        counts = df_res["cluster"].value_counts().sort_index()
        n_tot  = len(df_res)
        cols   = st.columns(k)
        for i, col in enumerate(cols):
            if i >= k: break
            n = counts.get(i+1, 0)
            with col:
                st.metric(clabel(i, k), f"{n}명", f"{n/n_tot*100:.1f}%")
        st.plotly_chart(chart_bar_count(df_res, k), use_container_width=True)

    # ─ 히트맵 ─
    with s[2]:
        sec("Heatmap", "군집 프로파일 히트맵")
        h1, h2 = st.tabs(["원점수 평균", "표준화 평균 (Z-score)"])
        with h1:
            infobox("해석 안내",
                    "1~5점 척도 기준 군집별 실제 평균 점수입니다. "
                    "숫자가 클수록 해당 특성이 강합니다.")
            _pr = pr.copy()
            _pr.index = [clabel(i, k) for i in range(len(_pr))]
            st.plotly_chart(chart_heatmap(_pr, False), use_container_width=True)
        with h2:
            infobox("해석 안내",
                    "Z-score 0보다 크면 전체 평균 이상(주황), 작으면 이하(파랑). "
                    "군집 간 상대적 차이를 파악하는 데 유용합니다.")
            _pz = pz.copy()
            _pz.index = [clabel(i, k) for i in range(len(_pz))]
            st.plotly_chart(chart_heatmap(_pz, True), use_container_width=True)
            st.dataframe(
                _pz.rename(columns=FACTOR_LABELS).round(2)
                   .style.background_gradient(cmap="RdYlGn", axis=None)
                   .format("{:+.2f}"),
                use_container_width=True,
            )

    # ─ 레이더 ─
    with s[3]:
        sec("Radar", "표준화 군집 프로파일 레이더 차트",
            "4개 변수 축으로 군집의 특성 패턴을 한눈에 비교합니다.")
        show = []
        rcols = st.columns(k)
        for i, col in enumerate(rcols):
            if i >= k: break
            with col:
                if st.checkbox(clabel(i, k), value=True, key=f"rc_{i}"):
                    show.append(i+1)
        if show:
            st.plotly_chart(chart_radar(pz, show), use_container_width=True)
        else:
            st.info("하나 이상의 군집을 선택하세요.")

    # ─ Centroid ─
    with s[4]:
        sec("Process", "K-평균 군집화 과정 — Centroid 변화",
            "알고리즘 반복에 따라 군집 중심(✕)이 이동하고 소속이 재배정됩니다. "
            "PCA로 다차원 데이터를 2차원으로 축소하여 시각화합니다.")
        infobox("알고리즘 흐름",
                "① 랜덤 초기 중심 배정 → "
                "② 각 점을 가장 가까운 중심에 배정 → "
                "③ 중심 재계산 → "
                "②~③ 반복 → "
                "④ 변화 없으면 수렴")
        with st.spinner("시각화 생성 중..."):
            st.plotly_chart(
                chart_centroid(fp, sel_vars, k),
                use_container_width=True)

    # ─ 산점도 ─
    with s[5]:
        sec("Scatter", "군집 분포 산점도")
        sp1, sp2 = st.tabs(["PCA 2차원 산점도", "변수 선택 산점도"])
        with sp1:
            infobox("PCA",
                    "선택 변수를 PCA로 2차원 축소 후 군집 소속을 색으로 표시합니다. "
                    "✕는 군집 중심(centroid)입니다.")
            st.plotly_chart(chart_pca(X_sc, df_res, k), use_container_width=True)
        with sp2:
            infobox("변수 선택",
                    "X·Y축 변수를 직접 선택해 변수 간 관계와 군집 분포를 탐색합니다.")
            vopts = {FACTOR_LABELS[v]: v for v in sel_vars}
            xa, ya = st.columns(2)
            xl = xa.selectbox("X축 변수", list(vopts.keys()), index=0)
            yl = ya.selectbox("Y축 변수", list(vopts.keys()),
                              index=min(1, len(vopts)-1))
            st.plotly_chart(
                chart_scatter(df_res, df_f, vopts[xl], vopts[yl], k),
                use_container_width=True)

    # ─ 설문 문항 ─
    with s[6]:
        sec("Survey", "설문 문항 구성",
            "모든 문항은 5점 리커트 척도(1=전혀 그렇지 않다 ~ 5=매우 그렇다)입니다.")
        for fk in FACTOR_ITEMS:
            letter, color, bg = FACTOR_META[fk]
            lbl   = FACTOR_LABELS[fk]
            items = FACTOR_ITEMS[fk]
            st.markdown(f"""
            <div class="iblock">
              <div class="ihead" style="background:{bg};">
                <span class="ibadge" style="background:{color}20;color:{color};">{letter}</span>
                <span class="ihtit">{lbl}</span>
                <span class="ihsub">{len(items)}문항</span>
              </div>""", unsafe_allow_html=True)
            rows_html = ""
            for item in items:
                num  = item.split(".")[0]
                text = ".".join(item.split(".")[1:]).strip()
                rev  = '<span class="irev">역채점</span>' if item == NEGATIVE_ITEM else ""
                rows_html += (f'<div class="irow">'
                              f'<div class="inum">{num}</div>'
                              f'<div class="itxt">{text}</div>{rev}</div>')
            st.markdown(rows_html + "</div><br>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 탭 4 : 결론
# ══════════════════════════════════════════
def tab_conclusion(df_res, pr, pz, k):
    # ── 결론 배너 ──
    st.markdown(f"""
    <div class="cbanner">
      <div class="cbn">K{k}</div>
      <div>
        <div class="cbh">최적 클러스터 수: K = {k}</div>
        <p class="cbp">
          Elbow Method 분석 결과 K={k}에서 Inertia 감소폭이 완만해졌습니다.<br>
          학생들은 <b>취약형 · 중간형 · 강점형</b> 3가지 유형으로 분류됩니다.
        </p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 군집별 특징 ──
    sec("Characteristics", "군집별 특징")
    cols = st.columns(k)
    for i, col in enumerate(cols):
        if i >= k or i >= len(pz): break
        row   = pz.iloc[i]
        color = C_COLORS[i]
        name  = C_NAMES[i] if i < len(C_NAMES) else f"군집{i+1}"
        desc  = C_DESCS[i] if i < len(C_DESCS) else ""
        n_cnt = int((df_res["cluster"]==i+1).sum())
        stats = "".join(
            f'<div class="cstat">'
            f'<span style="color:#9e9387;">{FACTOR_LABELS.get(v,v)}</span>'
            f'<span style="color:{color};font-weight:700;font-family:monospace;">'
            f'{"+" if row[v]>0 else ""}{row[v]:.2f}σ</span></div>'
            for v in pz.columns
        )
        with col:
            st.markdown(f"""
            <div class="ccard" style="border-top-color:{color};margin-bottom:18px;">
              <div class="ccard-lbl" style="color:{color};">군집 {i+1} · {n_cnt}명</div>
              <div class="ccard-name">{name}</div>
              <div class="ccard-desc">{desc}</div>
              {stats}
            </div>""", unsafe_allow_html=True)

    hdiv()

    # ── 종합 해석 ──
    sec("Interpretation", "종합 해석")
    st.markdown("""
    <div class="card" style="font-size:14px;color:#4e3d35;line-height:1.9;">
      <p>본 분석은 <b>수학불안·자기효능감·수학흥미·학습태도</b> 4가지 하위요인을 기준으로
      학생들을 유형화하여, 각 유형의 특징을 이해하고 맞춤형 지원 방안을 모색하는 것을 목표로 하였습니다.</p>
      <p style="margin-top:12px;">
        K-평균 군집화 분석을 통해 학생들을 총 <b>3가지 유형</b>으로 분류할 수 있었습니다.
        단순히 성적만으로 학생들을 평가하기보다, 정서적·행동적 요인을 함께 고려해
        학생을 이해해야 함을 보여 줍니다.
      </p>
      <p style="margin-top:12px;">
        PCA 2차원 공간에서도 군집이 대체로 뚜렷하게 구분되어 군집화의 타당성을 뒷받침합니다.
        이 결과는 학교 수학 수업에서 집단별 맞춤형 교수 전략을 수립하는 데 중요한 기초 자료가 됩니다.
      </p>
    </div>""", unsafe_allow_html=True)

    hdiv()

    # ── 맞춤 지도 방안 ──
    sec("Strategies", "군집별 맞춤형 지도 방안")
    s1, s2, s3 = st.columns(3)
    for i, col in enumerate([s1, s2, s3]):
        if i >= min(k, 3): break
        color = C_COLORS[i]
        name  = C_NAMES[i] if i < len(C_NAMES) else f"군집{i+1}"
        strat = STRATEGIES[i] if i < len(STRATEGIES) else {"title":"","items":[]}
        li_html = "".join(f"<li>{it}</li>" for it in strat["items"])
        with col:
            st.markdown(f"""
            <div class="stcard" style="border-left-color:{color};">
              <div class="stlbl" style="color:{color};">군집 {i+1} · {name}</div>
              <div class="sttit">{strat['title']}</div>
              <ul class="stli">{li_html}</ul>
            </div>""", unsafe_allow_html=True)

    hdiv()

    # ── 시사점 ──
    infobox("종합 시사점",
            "세 군집은 성적이 아닌 <b>정서·동기·행동</b> 요인으로 분류됩니다. "
            "동일 성적대라도 군집이 다를 수 있어 유형별 맞춤 접근이 필요합니다.<br>"
            "<b>취약형</b>은 심리적 돌봄 우선, "
            "<b>중간형</b>은 동기 유발·잠재력 발굴, "
            "<b>강점형</b>은 심화 도전 기회 제공이 핵심입니다.")

    hdiv()

    # ── 다운로드 ──
    sec("Export", "분석 결과 다운로드")
    c1, c2 = st.columns(2)
    with c1:
        _pz = pz.copy()
        _pz.index = [clabel(i,k) for i in range(len(_pz))]
        _pz.columns = [FACTOR_LABELS.get(c,c) for c in _pz.columns]
        st.download_button(
            "📥 군집 프로파일 (Z-score) CSV",
            data=_pz.round(3).to_csv(encoding="utf-8-sig"),
            file_name="cluster_profile_zscore.csv", mime="text/csv",
        )
    with c2:
        out = df_res.rename(columns=FACTOR_LABELS)
        st.download_button(
            "📥 학생별 군집 배정 결과 CSV",
            data=out.to_csv(index=False, encoding="utf-8-sig"),
            file_name="student_cluster_result.csv", mime="text/csv",
        )


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    # session state 초기화
    for k, v in [("show_recog", False), ("result", None),
                 ("file_data", None), ("file_name", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 데이터 로드 ──
    is_demo = False
    if st.session_state["file_data"]:
        try:
            raw, df_f, recognized, avail = load_file(
                st.session_state["file_data"],
                st.session_state["file_name"],
            )
        except Exception as e:
            st.error(f"파일 처리 오류: {e}")
            raw, df_f, recognized, avail = make_demo()
            is_demo = True
    else:
        raw, df_f, recognized, avail = make_demo()
        is_demo = True

    # parquet 직렬화 (캐시 키)
    buf = io.BytesIO()
    df_f.to_parquet(buf)
    fp = buf.getvalue()

    # ── 상태바 ──
    src = "🔵 데모 데이터 (73명)" if is_demo \
          else f"📁 {st.session_state.get('file_name','')}"
    st.markdown(
        f'<div class="statusbar">현재 데이터: '
        f'<b style="color:#E07B54;">{src}</b></div>',
        unsafe_allow_html=True)

    # ── 메인 탭 ──
    t1, t2, t3, t4 = st.tabs([
        "🏠 소개",
        "📋 데이터",
        "📊 데이터분석 및 시각화",
        "🎯 결론",
    ])

    with t1:
        tab_intro()

    with t2:
        tab_data(raw, df_f, recognized, avail, is_demo)

    with t3:
        if not avail:
            st.warning("분석 가능한 변수가 없습니다. 파일 형식을 확인해 주세요.")
        else:
            tab_analysis(df_f, avail, fp)

    with t4:
        result = st.session_state.get("result")
        if result:
            df_res, pr, pz, k = result
            tab_conclusion(df_res, pr, pz, k)
        else:
            infobox("이용 순서",
                    "① <b>소개</b> 탭에서 파일 업로드 → "
                    "② <b>데이터</b> 탭에서 문항 인식 확인 → "
                    "③ <b>데이터분석 및 시각화</b> 탭 클릭(자동 실행) → "
                    "④ <b>결론</b> 탭에서 최종 결과 확인")
            st.info("먼저 '데이터분석 및 시각화' 탭을 열어 분석을 실행하면 결론이 표시됩니다.")


if __name__ == "__main__":
    main()
