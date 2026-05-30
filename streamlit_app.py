import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="수학 학습자 유형 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
# 전역 상수
# ══════════════════════════════════════════════════════════
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
        "T6. 수학을 포기하고 싶다고 느낄 때가 있다.",
    ],
}
NEGATIVE_ITEM = "T6. 수학을 포기하고 싶다고 느낄 때가 있다."

FACTOR_LABELS = {
    "math_anxiety_mean":       "수학불안",
    "math_self_efficacy_mean": "자기효능감",
    "math_interest_mean":      "수학흥미",
    "learning_attitude_mean":  "학습태도",
}
FACTOR_BADGE = {
    "math_anxiety_mean":       ("A", "#E03131", "#fff0f0"),
    "math_self_efficacy_mean": ("E", "#1B64DA", "#eef3fd"),
    "math_interest_mean":      ("I", "#0D9E75", "#e6f7f1"),
    "learning_attitude_mean":  ("T", "#C47D0E", "#fff8e6"),
}

CLUSTER_COLORS = ["#E03131", "#0D9E75", "#C47D0E"]
CLUSTER_NAMES  = ["취약형", "우수형", "보통형"]
CLUSTER_DESCS  = [
    "수학불안이 매우 높고 자기효능감·흥미·학습태도가 전반적으로 낮은 집단",
    "수학불안이 낮고 자기효능감·흥미·학습태도가 모두 평균 이상인 집단",
    "모든 요인이 평균에 근접한 중간 집단",
]
STRATEGY_ITEMS = [
    [
        "협력 분위기 조성, 실수 허용 문화 정착",
        "개별화 피드백으로 작은 성취도 칭찬·격려",
        "수학 불안 관리 전략 교육 (긍정적 자기 대화)",
        "쉬운 문제부터 단계적 난이도 향상 경험",
        "놀이·게임·실생활 연계로 흥미 유발",
        "탐구 활동으로 수학의 유용성 직접 발견",
    ],
    [
        "현 수준 초과 심화 문제·탐구 과제 제공",
        "수학 경시대회·동아리·멘토링 기회 연결",
        "자기 주도 학습 목표 설정 및 실행 독려",
        "다양한 풀이 전략 탐색·비판적 사고 훈련",
        "생성형 AI 활용 수학적 아이디어 발전",
        "동료 설명·리더십 역할로 깊은 이해 도모",
    ],
    [
        "학습 스타일·선호도 파악 후 맞춤 경험 제공",
        "수학-실생활 연결로 학습 의미 탐색",
        "토론·발표 기회 확대로 적극성 북돋기",
        "또래 멘토링·소그룹으로 편안한 환경 조성",
        "학습 일지·면담으로 자기 성찰 지원",
        "작은 성공 경험 축적으로 자기효능감 증진",
    ],
]

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

.stApp { background:#f7f8fa; font-family:'Noto Sans KR',sans-serif; }
[data-testid="stSidebar"]{ background:#fff; border-right:1px solid #e8eaed; }

/* ── 히어로 섹션 ── */
.hero{background:#fff;border:1px solid #e8eaed;border-radius:16px;padding:40px 44px;margin-bottom:28px;}
.hero .kicker{font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#1b64da;margin-bottom:10px;}
.hero h1{font-size:28px;font-weight:700;color:#191f28;letter-spacing:-0.3px;margin:0 0 8px;}
.hero .subtitle{font-size:15px;color:#4e5968;margin:0 0 28px;line-height:1.6;}
.hero .divider{height:1px;background:#e8eaed;margin:24px 0;}

/* ── 기능 카드 ── */
.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:4px;}
.feature-card{background:#f7f8fa;border:1px solid #e8eaed;border-radius:12px;padding:16px 18px;}
.feature-card .fc-icon{font-size:22px;margin-bottom:8px;}
.feature-card .fc-title{font-size:13px;font-weight:700;color:#191f28;margin-bottom:4px;}
.feature-card .fc-desc{font-size:12px;color:#8b95a1;line-height:1.5;}

/* ── 업로드 존 ── */
.upload-zone{background:#fff;border:2px dashed #d0d8e8;border-radius:12px;padding:32px;text-align:center;margin-bottom:24px;}
.upload-zone .uz-title{font-size:15px;font-weight:600;color:#191f28;margin-bottom:6px;}
.upload-zone .uz-sub{font-size:13px;color:#8b95a1;}

/* ── 섹션 헤더 ── */
.sec-kicker{font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1b64da;margin-bottom:3px;}
.sec-title{font-size:19px;font-weight:700;color:#191f28;margin-bottom:4px;}
.sec-desc{font-size:13px;color:#4e5968;margin-bottom:20px;line-height:1.6;}

/* ── 카드 ── */
.card{background:#fff;border:1px solid #e8eaed;border-radius:12px;padding:20px 24px;margin-bottom:16px;}
.card-sm{background:#fff;border:1px solid #e8eaed;border-radius:10px;padding:16px 18px;}
.card-title{font-size:12px;font-weight:600;color:#8b95a1;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}

/* ── 메트릭 ── */
.metric-row{display:flex;gap:14px;margin-bottom:20px;}
.metric-box{flex:1;background:#fff;border:1px solid #e8eaed;border-radius:10px;padding:16px 20px;text-align:center;}
.metric-box .mb-val{font-size:28px;font-weight:700;color:#191f28;line-height:1;}
.metric-box .mb-label{font-size:11px;color:#8b95a1;margin-top:5px;text-transform:uppercase;letter-spacing:1px;}

/* ── 문항표 ── */
.item-block{border:1px solid #e8eaed;border-radius:10px;overflow:hidden;margin-bottom:16px;}
.item-head{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#f7f8fa;border-bottom:1px solid #e8eaed;}
.item-badge{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:6px;font-size:12px;font-weight:700;}
.item-head-title{font-size:13px;font-weight:600;color:#191f28;}
.item-head-sub{font-size:12px;color:#8b95a1;margin-left:auto;}
.item-row{display:flex;gap:12px;padding:8px 16px;border-bottom:1px solid #f0f1f3;background:#fff;}
.item-row:last-child{border-bottom:none;}
.item-row:hover{background:#fafbfc;}
.item-num{font-family:monospace;font-size:11px;color:#8b95a1;min-width:26px;padding-top:2px;}
.item-text{font-size:13px;color:#4e5968;flex:1;line-height:1.5;}
.item-rev{font-size:10px;color:#e03131;background:#fff0f0;padding:2px 6px;border-radius:3px;white-space:nowrap;align-self:flex-start;margin-top:2px;}

/* ── 인식결과 ── */
.recog-item{display:flex;align-items:center;gap:10px;padding:8px 14px;border-radius:8px;margin-bottom:6px;font-size:13px;}
.recog-ok{background:#e6f7f1;color:#0d9e75;}
.recog-fail{background:#fff0f0;color:#e03131;}
.recog-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}

/* ── 분석설정 패널 ── */
.settings-panel{background:#fff;border:1px solid #e8eaed;border-radius:12px;padding:20px 24px;margin-bottom:24px;}
.settings-panel .sp-title{font-size:13px;font-weight:600;color:#191f28;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #f0f1f3;}

/* ── 클러스터 카드 ── */
.cluster-card{background:#fff;border:1px solid #e8eaed;border-radius:12px;padding:18px 20px;border-top:4px solid;}
.cluster-card .cc-label{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;}
.cluster-card .cc-name{font-size:17px;font-weight:700;color:#191f28;margin-bottom:5px;}
.cluster-card .cc-desc{font-size:12px;color:#4e5968;line-height:1.5;margin-bottom:12px;}
.cluster-card .cc-stats{display:flex;flex-direction:column;gap:5px;}
.cc-stat-row{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:4px 0;border-bottom:1px solid #f0f1f3;}
.cc-stat-row:last-child{border-bottom:none;}

/* ── 전략 카드 ── */
.strat-card{background:#fff;border:1px solid #e8eaed;border-radius:12px;padding:20px;border-left:4px solid;height:100%;}
.strat-label{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.strat-title{font-size:15px;font-weight:700;color:#191f28;margin-bottom:14px;}
.strat-list li{font-size:12px;color:#4e5968;line-height:1.7;margin-bottom:2px;}

/* ── 인포박스 ── */
.info-box{background:#eef3fd;border:1px solid #dbe8fb;border-radius:8px;padding:14px 18px;margin-bottom:18px;}
.info-box .ib-label{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1b64da;margin-bottom:5px;}
.info-box p{font-size:13px;color:#4e5968;line-height:1.7;margin:0;}

/* ── 결론 배너 ── */
.conclusion-banner{background:#1b64da;border-radius:12px;padding:28px 32px;color:#fff;margin-bottom:24px;display:flex;align-items:center;gap:24px;}
.conclusion-banner .cb-num{font-size:52px;font-weight:700;line-height:1;opacity:.15;font-family:monospace;}
.conclusion-banner h3{font-size:20px;font-weight:700;margin:0 0 6px;}
.conclusion-banner p{font-size:13px;opacity:.85;line-height:1.6;margin:0;}

/* ── 구분선 ── */
.divider{height:1px;background:#e8eaed;margin:24px 0;}

/* Streamlit 기본 여백 줄이기 */
.block-container{padding-top:24px!important;padding-bottom:40px!important;max-width:1100px!important;}
[data-testid="stMetricValue"]{font-size:24px!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 데이터 처리
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_and_process(file_bytes: bytes, file_name: str):
    """파일 바이트 → raw_df + df_factors 반환"""
    try:
        engine = "xlrd" if file_name.endswith(".xls") else "openpyxl"
        raw_df = pd.read_excel(file_bytes, engine=engine)
    except Exception:
        raw_df = pd.read_excel(file_bytes)
    raw_df.columns = raw_df.columns.astype(str).str.strip()

    # 하위요인 평균 계산
    factors = pd.DataFrame(index=raw_df.index)
    recognized = {}
    for fk, items in FACTOR_ITEMS.items():
        found = [c for c in items if c in raw_df.columns]
        recognized[fk] = found
        if found:
            tmp = raw_df[found].copy()
            for c in found:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            if fk == "learning_attitude_mean" and NEGATIVE_ITEM in tmp.columns:
                tmp[NEGATIVE_ITEM] = 6 - tmp[NEGATIVE_ITEM]
            factors[fk] = tmp.mean(axis=1)
        else:
            factors[fk] = np.nan

    avail = [v for v in FACTOR_ITEMS if factors[v].notna().sum() > 0]
    return raw_df, factors, recognized, avail


@st.cache_data
def run_kmeans(factor_bytes, sel_vars_key: str, k: int):
    """캐시를 위해 직렬화된 데이터 사용"""
    import io
    df_factors = pd.read_parquet(io.BytesIO(factor_bytes))
    sel_vars = sel_vars_key.split("|")

    X = df_factors[sel_vars].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)

    df_res = df_factors.loc[X.index, sel_vars].copy()
    df_res["cluster"] = labels + 1
    df_sc = pd.DataFrame(X_sc, columns=sel_vars, index=X.index)
    df_sc["cluster"] = labels + 1

    profile_raw = df_res.groupby("cluster")[sel_vars].mean()
    profile_z   = df_sc.groupby("cluster")[sel_vars].mean()
    return df_res, df_sc, profile_raw, profile_z, X_sc


@st.cache_data
def calc_elbow(factor_bytes, sel_vars_key: str, k_min: int, k_max: int):
    import io
    df_factors = pd.read_parquet(io.BytesIO(factor_bytes))
    sel_vars = sel_vars_key.split("|")
    X = df_factors[sel_vars].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    ks, vals = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        ks.append(k); vals.append(km.inertia_)
    return ks, vals


# ══════════════════════════════════════════════════════════
# 공통 UI
# ══════════════════════════════════════════════════════════
def sec(kicker, title, desc=""):
    html = f'<div class="sec-kicker">{kicker}</div><div class="sec-title">{title}</div>'
    if desc:
        html += f'<div class="sec-desc">{desc}</div>'
    st.markdown(html, unsafe_allow_html=True)


def infobox(label, body):
    st.markdown(f"""<div class="info-box"><div class="ib-label">{label}</div><p>{body}</p></div>""",
                unsafe_allow_html=True)


def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════
def plot_elbow(ks, inertias, k_sel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias, mode="lines+markers+text",
        line=dict(color="#1B64DA", width=2.5),
        marker=dict(
            size=[12 if k == k_sel else 7 for k in ks],
            color=["#E03131" if k == k_sel else "#1B64DA" for k in ks],
            line=dict(width=2, color="white"),
        ),
        text=[str(round(v, 1)) for v in inertias],
        textposition="top center", textfont=dict(size=11),
    ))
    if k_sel in ks:
        idx = ks.index(k_sel)
        fig.add_annotation(x=k_sel, y=inertias[idx],
                           text=f"  ← K={k_sel} 선택",
                           showarrow=False, font=dict(color="#E03131", size=12),
                           xanchor="left")
        fig.add_vline(x=k_sel, line_dash="dot", line_color="rgba(224,49,49,0.3)", line_width=1.5)
    fig.update_layout(
        xaxis=dict(title="클러스터 수 (K)", tickvals=ks, gridcolor="#f0f1f3"),
        yaxis=dict(title="Inertia", gridcolor="#f0f1f3"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=40, l=50, r=20), height=320, showlegend=False,
    )
    return fig


def plot_counts(df_res, k):
    counts = df_res["cluster"].value_counts().sort_index()
    labels = [f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}" for i in range(k)]
    vals   = [counts.get(i+1, 0) for i in range(k)]
    colors = CLUSTER_COLORS[:k]
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=["rgba({},{},{},0.15)".format(int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)) for c in colors],
        marker_line_color=colors, marker_line_width=1.5,
        text=vals, textposition="outside",
        textfont=dict(size=14, color=colors),
    ))
    fig.update_layout(
        yaxis=dict(title="참가자 수 (명)", gridcolor="#f0f1f3"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=40, l=50, r=20), height=300, showlegend=False,
    )
    return fig


def plot_heatmap(profile, is_z):
    z = profile.values
    yl = [f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}" for i in range(len(profile))]
    xl = [FACTOR_LABELS.get(c, c) for c in profile.columns]
    cs = ([[0,"#FFF0F0"],[0.5,"#F7F8FA"],[1,"#E6F7F1"]] if is_z
          else [[0,"#EEF3FD"],[1,"#1B64DA"]])
    anns = []
    for i in range(len(z)):
        for j in range(len(z[0])):
            v = z[i][j]
            anns.append(dict(x=j, y=i, text=f"{'+'if is_z and v>0 else ''}{v:.2f}",
                             showarrow=False, font=dict(size=13, color="#191f28", family="monospace")))
    fig = go.Figure(go.Heatmap(z=z, x=xl, y=yl, colorscale=cs, showscale=True,
                               hoverongaps=False))
    fig.update_traces(colorbar=dict(thickness=12, len=0.8), annotations=anns)
    fig.update_layout(
        xaxis=dict(side="top"), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=10, l=110, r=70), height=190 + len(profile) * 28,
    )
    return fig


def plot_radar(profile_z, show_list):
    cats = [FACTOR_LABELS.get(c, c) for c in profile_z.columns]
    cats_c = cats + [cats[0]]
    fig = go.Figure()
    for i, (_, row) in enumerate(profile_z.iterrows()):
        if (i+1) not in show_list:
            continue
        vals = list(row.values) + [row.values[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats_c, fill="toself",
            fillcolor="rgba({},{},{},0.13)".format(*[int(CLUSTER_COLORS[i][j:j+2],16) for j in (1,3,5)]),
            line=dict(color=CLUSTER_COLORS[i], width=2.5),
            name=f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}",
            marker=dict(size=7, color=CLUSTER_COLORS[i]),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2.5, 2.5], tickfont=dict(size=10), gridcolor="#e8eaed"),
            angularaxis=dict(tickfont=dict(size=13)), bgcolor="white",
        ),
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(t=20, b=60, l=60, r=60), height=440,
    )
    return fig


def plot_centroid(factor_bytes, sel_vars, k):
    import io
    df_factors = pd.read_parquet(io.BytesIO(factor_bytes))
    X = df_factors[sel_vars].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sc)

    rng = np.random.RandomState(42)
    centers = X2[rng.choice(len(X2), k, replace=False)]
    n_steps, frames_l, frames_c = 6, [], []
    cur = centers.copy()
    for _ in range(n_steps):
        dists = np.linalg.norm(X2[:, None] - cur[None, :], axis=2)
        labels = dists.argmin(axis=1)
        frames_l.append(labels.copy())
        new_c = np.array([X2[labels==ci].mean(axis=0) if (labels==ci).any() else cur[ci] for ci in range(k)])
        frames_c.append(new_c.copy())
        cur = new_c

    titles = ["초기 랜덤 배정"] + [f"반복 {i}회 후" for i in range(1, 5)] + ["수렴 (최종)"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.06, vertical_spacing=0.14)
    for fi in range(n_steps):
        r, c = fi // 3 + 1, fi % 3 + 1
        for ci in range(k):
            mask = frames_l[fi] == ci
            fig.add_trace(go.Scatter(
                x=X2[mask, 0], y=X2[mask, 1], mode="markers",
                marker=dict(color=CLUSTER_COLORS[ci], size=5, opacity=0.7),
                name=f"C{ci+1}" if fi == 0 else None,
                showlegend=(fi == 0), legendgroup=f"C{ci+1}",
            ), row=r, col=c)
        for ci in range(k):
            cx_, cy_ = frames_c[fi][ci]
            fig.add_trace(go.Scatter(
                x=[cx_], y=[cy_], mode="markers",
                marker=dict(symbol="x", size=13, color=CLUSTER_COLORS[ci],
                            line=dict(width=2.5, color="#191f28")),
                showlegend=False,
            ), row=r, col=c)
    fig.update_layout(height=500, paper_bgcolor="white",
                      legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center"),
                      margin=dict(t=40, b=40, l=20, r=20))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=True, gridcolor="#f0f1f3", zeroline=False)
    return fig


def plot_scatter_var(df_res, df_factors, x_var, y_var, k):
    merged = df_res[["cluster"]].join(df_factors[[x_var, y_var]], how="inner").dropna()
    fig = go.Figure()
    for ci in range(k):
        sub = merged[merged["cluster"] == ci+1]
        fig.add_trace(go.Scatter(
            x=sub[x_var], y=sub[y_var], mode="markers",
            marker=dict(color=CLUSTER_COLORS[ci], size=7, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            name=f"C{ci+1} {CLUSTER_NAMES[ci] if ci < len(CLUSTER_NAMES) else ''}",
        ))
    cents = merged.groupby("cluster")[[x_var, y_var]].mean()
    for ci in range(k):
        if (ci+1) in cents.index:
            fig.add_trace(go.Scatter(
                x=[cents.loc[ci+1, x_var]], y=[cents.loc[ci+1, y_var]], mode="markers",
                marker=dict(symbol="x", size=15, color=CLUSTER_COLORS[ci],
                            line=dict(width=3, color="#191f28")),
                showlegend=False,
            ))
    fig.update_layout(
        xaxis=dict(title=FACTOR_LABELS.get(x_var, x_var), gridcolor="#f0f1f3", zeroline=False),
        yaxis=dict(title=FACTOR_LABELS.get(y_var, y_var), gridcolor="#f0f1f3", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        margin=dict(t=10, b=60, l=60, r=20), height=380,
    )
    return fig


def plot_pca(X_sc, df_res, k):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sc)
    ev = pca.explained_variance_ratio_ * 100
    fig = go.Figure()
    lbl_arr = df_res["cluster"].values
    for ci in range(k):
        mask = lbl_arr == ci+1
        if mask.any():
            fig.add_trace(go.Scatter(
                x=X2[mask, 0], y=X2[mask, 1], mode="markers",
                marker=dict(color=CLUSTER_COLORS[ci], size=7, opacity=0.72,
                            line=dict(width=0.5, color="white")),
                name=f"C{ci+1} {CLUSTER_NAMES[ci] if ci < len(CLUSTER_NAMES) else ''}",
            ))
            cx_, cy_ = X2[mask].mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[cx_], y=[cy_], mode="markers",
                marker=dict(symbol="x", size=15, color=CLUSTER_COLORS[ci],
                            line=dict(width=3, color="#191f28")),
                showlegend=False,
            ))
    fig.update_layout(
        xaxis=dict(title=f"PC1 ({ev[0]:.1f}%)", gridcolor="#f0f1f3", zeroline=True, zerolinecolor="#d0d4db"),
        yaxis=dict(title=f"PC2 ({ev[1]:.1f}%)", gridcolor="#f0f1f3", zeroline=True, zerolinecolor="#d0d4db"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        margin=dict(t=10, b=60, l=60, r=20), height=400,
    )
    return fig


# ══════════════════════════════════════════════════════════
# 탭 1 : 소개
# ══════════════════════════════════════════════════════════
def tab_intro(uploaded):
    # ── 히어로 ──
    st.markdown("""
    <div class="hero">
        <div class="kicker">K-Means Clustering Analysis</div>
        <h1>K-평균 군집화를 활용한 수학 학습자 유형 분석</h1>
        <p class="subtitle">
            수학불안 · 자기효능감 · 흥미 · 학습태도 설문 데이터를 업로드하면<br>
            학생 집단을 자동으로 유형화하고 시각화합니다.
        </p>
        <div class="divider"></div>
        <div style="font-size:12px;font-weight:600;color:#8b95a1;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">이 웹앱으로 확인할 수 있는 것</div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="fc-icon">📊</div>
                <div class="fc-title">학습자 유형 자동 분류</div>
                <div class="fc-desc">K-평균 알고리즘으로 학생들을 취약형·우수형·보통형으로 자동 군집화</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🔍</div>
                <div class="fc-title">어려움 원인 규명</div>
                <div class="fc-desc">불안·자신감·흥미·태도 4가지 요인 중 어느 것이 학습 방해 요인인지 파악</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">📈</div>
                <div class="fc-title">인터랙티브 시각화</div>
                <div class="fc-desc">히트맵·레이더·산점도·Elbow 그래프로 군집 특성을 다각도로 탐색</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🎯</div>
                <div class="fc-title">맞춤형 교수 전략</div>
                <div class="fc-desc">군집별 특성에 맞는 구체적인 수학 수업 지도 방안을 제시</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 파일 업로드 ──
    sec("Step 1", "분석 파일 업로드",
        "수학 설문 응답이 담긴 엑셀 파일(.xlsx / .xls)을 업로드하면 분석이 시작됩니다.")

    uploaded_file = st.file_uploader(
        "엑셀 파일을 여기에 드래그하거나 클릭하여 선택하세요",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
        key="main_uploader",
    )

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.success(f"✅ **{uploaded_file.name}** 업로드 완료 — '데이터' 탭에서 내용을 확인하세요.")
    elif "uploaded_file" in st.session_state:
        st.info("📎 이전에 업로드한 파일이 사용됩니다. 새 파일을 업로드하면 교체됩니다.")
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="uz-title">📁 파일을 업로드하세요</div>
            <div class="uz-sub">지원 형식: .xlsx · .xls &nbsp;|&nbsp; 파일이 없으면 데모 데이터로 분석됩니다</div>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # ── 파일 형식 안내 ──
    sec("Step 2", "파일 형식 안내",
        "아래 컬럼명이 정확히 포함되어 있어야 문항이 인식됩니다.")

    col_guide = {
        "수학불안 (A1~A8)": "A1. 수학 시간에 어려운 문제가 나오면 긴장된다.",
        "자기효능감 (E1~E8)": "E1. 나는 노력하면 수학 실력을 향상시킬 수 있다고 생각한다.",
        "수학흥미 (I1~I8)": "I1. 수학 문제를 해결했을 때 성취감을 느낀다.",
        "학습태도 (T1~T6)": "T1. 나는 수학 공부 계획을 세우고 실천하는 편이다.",
    }
    col_colors = ["#E03131", "#1B64DA", "#0D9E75", "#C47D0E"]
    cols = st.columns(4)
    for (lbl, ex), color, col in zip(col_guide.items(), col_colors, cols):
        with col:
            st.markdown(f"""
            <div class="card-sm">
                <div style="font-size:12px;font-weight:700;color:{color};margin-bottom:6px;">{lbl}</div>
                <div style="font-size:11px;color:#8b95a1;line-height:1.5;">컬럼명 예시:<br><code style="font-size:10px;color:#4e5968;">{ex}</code><br>… 동일 형식 반복</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    infobox("참고", "파일 없이 실행하면 <b>데모 데이터(67명)</b>로 모든 분석이 자동 진행됩니다. '데이터분석' 탭에서 바로 결과를 확인할 수 있습니다.")


# ══════════════════════════════════════════════════════════
# 탭 2 : 데이터
# ══════════════════════════════════════════════════════════
def tab_data(raw_df, df_factors, recognized, avail_vars):
    sec("Data Overview", "데이터 확인")

    # ── 요약 메트릭 ──
    n_total = len(raw_df)
    n_cols  = len(raw_df.columns)
    n_avail = len(avail_vars)
    n_items_found = sum(len(v) for v in recognized.values())
    n_items_total = sum(len(v) for v in FACTOR_ITEMS.values())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("전체 응답 수", f"{n_total}명")
    with c2:
        st.metric("전체 컬럼 수", f"{n_cols}개")
    with c3:
        st.metric("분석 가능 변수", f"{n_avail}개 하위요인")
    with c4:
        st.metric("인식된 문항 수", f"{n_items_found} / {n_items_total}개")

    divider()

    # ── 원자료 미리보기 (토글) ──
    sec("Raw Data", "원자료 미리보기")
    with st.expander("📋 원자료 테이블 펼치기 / 접기", expanded=False):
        n_preview = st.slider("표시할 행 수", 5, min(50, n_total), 10, key="preview_rows")
        st.dataframe(raw_df.head(n_preview), use_container_width=True, height=320)
        st.caption(f"전체 {n_total}행 × {n_cols}열 중 앞 {n_preview}행 표시")

    divider()

    # ── 문항 인식 결과 ──
    sec("Item Recognition", "문항 인식 결과 확인",
        "업로드된 파일에서 각 하위요인 문항이 몇 개나 인식되었는지 확인합니다.")

    if st.button("🔍 문항 인식 결과 확인", type="primary", use_container_width=False):
        st.session_state["show_recog"] = True

    if st.session_state.get("show_recog", False):
        for fk in FACTOR_ITEMS:
            letter, color, bg = FACTOR_BADGE[fk]
            found  = recognized.get(fk, [])
            total  = FACTOR_ITEMS[fk]
            label  = FACTOR_LABELS[fk]
            ok     = len(found) == len(total)

            st.markdown(f"""
            <div class="item-block">
                <div class="item-head">
                    <span class="item-badge" style="background:{bg};color:{color};">{letter}</span>
                    <span class="item-head-title">{label}</span>
                    <span class="item-head-sub" style="color:{'#0d9e75' if ok else '#e03131'};">
                        {'✅' if ok else '⚠️'} {len(found)}/{len(total)} 인식
                    </span>
                </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for item in total:
                num  = item.split(".")[0]
                text = ".".join(item.split(".")[1:]).strip()
                is_found = item in found
                rev  = '<span class="item-rev">역채점</span>' if item == NEGATIVE_ITEM else ""
                icon = "✅" if is_found else "❌"
                rows_html += f'<div class="item-row"><div class="item-num">{num}</div><div class="item-text">{icon} {text}</div>{rev}</div>'

            st.markdown(rows_html + "</div>", unsafe_allow_html=True)

        if n_items_found < n_items_total:
            st.warning(f"⚠️ {n_items_total - n_items_found}개 문항을 찾을 수 없습니다. 컬럼명이 정확한지 확인해 주세요.")
        else:
            st.success("✅ 모든 문항이 정상적으로 인식되었습니다.")

    divider()

    # ── 하위요인 평균 미리보기 ──
    sec("Factor Scores", "하위요인 평균 점수 미리보기")
    with st.expander("📊 하위요인 평균 데이터 펼치기", expanded=False):
        disp = df_factors[avail_vars].copy()
        disp.columns = [FACTOR_LABELS.get(c, c) for c in disp.columns]
        st.dataframe(disp.round(2).head(20), use_container_width=True)
        st.caption(f"전체 {len(disp)}행 중 앞 20행 | 결측치 포함 행은 분석에서 제외됩니다.")


# ══════════════════════════════════════════════════════════
# 탭 3 : 데이터 분석
# ══════════════════════════════════════════════════════════
def tab_analysis(df_factors, avail_vars, factor_bytes):

    # ── 분석 설정 패널 ──
    with st.container():
        st.markdown('<div class="settings-panel"><div class="sp-title">⚙️ 분석 설정</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            sel_labels = st.multiselect(
                "군집화에 사용할 변수",
                options=[FACTOR_LABELS[v] for v in avail_vars],
                default=[FACTOR_LABELS[v] for v in avail_vars],
                key="sel_vars_ms",
            )
            sel_vars = [v for v in avail_vars if FACTOR_LABELS[v] in sel_labels]
            if len(sel_vars) < 2:
                st.warning("최소 2개 이상의 변수를 선택하세요.")
                sel_vars = avail_vars[:2]
        with c2:
            k = st.number_input("클러스터 수 (K)", 2, 8, 3, 1, key="k_val")
        with c3:
            k_min = st.number_input("Elbow K 최솟값", 2, 6, 2, 1)
            k_max = st.number_input("Elbow K 최댓값", 3, 12, 8, 1)
        st.markdown("</div>", unsafe_allow_html=True)

    if k_min >= k_max:
        k_max = k_min + 1

    sel_key = "|".join(sel_vars)
    k = int(k)

    # ── K-Means 실행 ──
    with st.spinner("K-Means 군집화 실행 중..."):
        df_res, df_sc, profile_raw, profile_z, X_sc = run_kmeans(factor_bytes, sel_key, k)

    # ── 서브탭 ──
    sub = st.tabs([
        "① Elbow 분석",
        "② 군집 인원",
        "③ 히트맵",
        "④ 레이더 차트",
        "⑤ Centroid 변화",
        "⑥ 산점도",
        "⑦ 설문 문항",
    ])

    # ① Elbow
    with sub[0]:
        sec("Elbow Method", "최적 K 탐색",
            "변수와 K 범위는 위 분석 설정에서 조절합니다. Inertia 감소폭이 완만해지는 지점이 최적 K입니다.")
        infobox("해석 안내",
                "K가 증가할수록 Inertia(군집 내 분산)는 감소합니다. 감소폭이 급격히 줄어드는 <b>'팔꿈치(Elbow)'</b> 지점이 최적 K 후보입니다.")
        ks, inertias = calc_elbow(factor_bytes, sel_key, int(k_min), int(k_max))
        st.plotly_chart(plot_elbow(ks, inertias, k), use_container_width=True)
        infobox("결과 해석",
                f"현재 선택된 <b>K={k}</b> 이후 Inertia 감소폭이 완만해집니다. "
                "위 설정 패널의 K 값을 바꾸면 그래프가 즉시 업데이트됩니다.")

    # ② 인원
    with sub[1]:
        sec("Distribution", "군집별 참가자 수",
            f"K={k}으로 분류된 각 군집의 학생 수 분포입니다.")
        counts = df_res["cluster"].value_counts().sort_index()
        n_total = len(df_res)
        cols = st.columns(k)
        for i, col in enumerate(cols):
            if i >= k: break
            n = counts.get(i+1, 0)
            name = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"군집{i+1}"
            with col:
                st.metric(f"C{i+1} {name}", f"{n}명", f"{n/n_total*100:.1f}%")
        st.plotly_chart(plot_counts(df_res, k), use_container_width=True)

    # ③ 히트맵
    with sub[2]:
        sec("Heatmap", "군집 프로파일 히트맵")
        ht1, ht2 = st.tabs(["원점수 평균", "표준화 평균 (Z-score)"])
        with ht1:
            infobox("해석 안내", "1~5점 척도 기준 군집별 평균 점수. 색이 진할수록 점수가 높습니다.")
            pr = profile_raw.copy()
            pr.index = [f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}" for i in range(len(pr))]
            st.plotly_chart(plot_heatmap(pr, False), use_container_width=True)
        with ht2:
            infobox("해석 안내",
                    "Z-score 0보다 크면 전체 평균 이상(초록), 0보다 작으면 이하(빨강). 군집 간 상대적 차이를 파악합니다.")
            pz = profile_z.copy()
            pz.index = [f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}" for i in range(len(pz))]
            st.plotly_chart(plot_heatmap(pz, True), use_container_width=True)
            st.dataframe(
                pz.round(2).style.background_gradient(cmap="RdYlGn", axis=None).format("{:+.2f}"),
                use_container_width=True,
            )

    # ④ 레이더
    with sub[3]:
        sec("Radar Chart", "표준화 군집 프로파일 레이더 차트",
            "4개 변수 축으로 각 군집의 특성 패턴을 한눈에 비교합니다.")
        show = []
        rcols = st.columns(k)
        for i, col in enumerate(rcols):
            if i >= k: break
            nm = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"군집{i+1}"
            with col:
                if st.checkbox(f"C{i+1} {nm}", value=True, key=f"rc_{i}"):
                    show.append(i+1)
        if show:
            st.plotly_chart(plot_radar(profile_z, show), use_container_width=True)
        else:
            st.info("하나 이상의 군집을 선택하세요.")

    # ⑤ Centroid
    with sub[4]:
        sec("K-Means Process", "Centroid 변화 과정",
            "알고리즘 반복에 따라 군집 중심(✕)이 이동하고 포인트 소속이 재배정됩니다. PCA로 4차원→2차원 축소.")
        infobox("알고리즘 흐름",
                "① 랜덤 초기 중심 배정 → ② 각 점을 가장 가까운 중심에 배정 → "
                "③ 중심 재계산 → ②~③ 반복 → ④ 변화 없으면 수렴")
        with st.spinner("시각화 생성 중..."):
            st.plotly_chart(plot_centroid(factor_bytes, sel_vars, k), use_container_width=True)

    # ⑥ 산점도
    with sub[5]:
        sec("Scatter Plot", "군집 분포 산점도")
        sp1, sp2 = st.tabs(["PCA 2차원 산점도", "변수 선택 산점도"])
        with sp1:
            infobox("PCA 분석",
                    "변수를 주성분 분석(PCA)으로 2차원으로 축소 후 군집 소속을 색으로 표시합니다. ✕는 군집 중심(centroid).")
            st.plotly_chart(plot_pca(X_sc, df_res, k), use_container_width=True)
        with sp2:
            infobox("변수 선택", "X축·Y축에 표시할 변수를 선택해 변수 간 관계와 군집 분포를 탐색합니다.")
            var_opts = {FACTOR_LABELS[v]: v for v in sel_vars}
            xa_col, ya_col = st.columns(2)
            x_lbl = xa_col.selectbox("X축 변수", list(var_opts.keys()), index=0)
            y_lbl = ya_col.selectbox("Y축 변수", list(var_opts.keys()),
                                     index=min(1, len(var_opts)-1))
            st.plotly_chart(
                plot_scatter_var(df_res, df_factors, var_opts[x_lbl], var_opts[y_lbl], k),
                use_container_width=True,
            )

    # ⑦ 설문 문항
    with sub[6]:
        sec("Survey Items", "설문 문항 구성",
            "모든 문항은 5점 리커트 척도(1=전혀 그렇지 않다 ~ 5=매우 그렇다)입니다.")
        for fk in FACTOR_ITEMS:
            letter, color, bg = FACTOR_BADGE[fk]
            label = FACTOR_LABELS[fk]
            items = FACTOR_ITEMS[fk]
            st.markdown(f"""
            <div class="item-block">
                <div class="item-head">
                    <span class="item-badge" style="background:{bg};color:{color};">{letter}</span>
                    <span class="item-head-title">{label}</span>
                    <span class="item-head-sub">{len(items)}문항</span>
                </div>
            """, unsafe_allow_html=True)
            rows_html = ""
            for item in items:
                num  = item.split(".")[0]
                text = ".".join(item.split(".")[1:]).strip()
                rev  = '<span class="item-rev">역채점</span>' if item == NEGATIVE_ITEM else ""
                rows_html += (f'<div class="item-row">'
                              f'<div class="item-num">{num}</div>'
                              f'<div class="item-text">{text}</div>{rev}</div>')
            st.markdown(rows_html + "</div><br>", unsafe_allow_html=True)

    return df_res, profile_z, k


# ══════════════════════════════════════════════════════════
# 탭 4 : 결론
# ══════════════════════════════════════════════════════════
def tab_conclusion(profile_z, k):
    # 결론 배너
    st.markdown(f"""
    <div class="conclusion-banner">
        <div class="cb-num">K{k}</div>
        <div>
            <h3>최적 클러스터 수: K = {k}</h3>
            <p>Elbow Method 분석 결과 K={k}에서 Inertia 감소폭이 완만해졌습니다.<br>
            학생들은 <b>취약형 · 우수형 · 보통형</b>의 3가지 뚜렷한 유형으로 분류됩니다.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 군집 특징 카드
    sec("Cluster Characteristics", "군집별 특징")
    cols = st.columns(k)
    for i, col in enumerate(cols):
        if i >= k or i >= len(profile_z): break
        row   = profile_z.iloc[i]
        color = CLUSTER_COLORS[i]
        name  = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"군집{i+1}"
        desc  = CLUSTER_DESCS[i] if i < len(CLUSTER_DESCS) else ""
        stats = "".join(
            f'<div class="cc-stat-row">'
            f'<span style="color:#8b95a1;">{FACTOR_LABELS.get(v,v)}</span>'
            f'<span style="color:{color};font-weight:700;font-family:monospace;">{"+"if row[v]>0 else ""}{row[v]:.2f}σ</span>'
            f'</div>'
            for v in profile_z.columns
        )
        with col:
            st.markdown(f"""
            <div class="cluster-card" style="border-top-color:{color};margin-bottom:20px;">
                <div class="cc-label" style="color:{color};">군집 {i+1} · {name}</div>
                <div class="cc-name">{name}</div>
                <div class="cc-desc">{desc}</div>
                <div class="cc-stats">{stats}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # 맞춤 전략
    sec("Teaching Strategies", "군집별 맞춤형 지도 방안")
    s1, s2, s3 = st.columns(3)
    strategy_cols = [s1, s2, s3]
    strategy_titles = ["심리적 안정 + 기초 강화", "심화 도전 + 창의성 확장", "동기 유발 + 잠재력 개발"]

    for i in range(min(k, 3)):
        color = CLUSTER_COLORS[i]
        name  = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"군집{i+1}"
        title = strategy_titles[i] if i < len(strategy_titles) else ""
        items = STRATEGY_ITEMS[i] if i < len(STRATEGY_ITEMS) else []
        li_html = "".join(f"<li>{it}</li>" for it in items)
        with strategy_cols[i]:
            st.markdown(f"""
            <div class="strat-card" style="border-left-color:{color};">
                <div class="strat-label" style="color:{color};">군집 {i+1} · {name}</div>
                <div class="strat-title">{title}</div>
                <ul class="strat-list">{li_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # 종합
    infobox("종합 시사점",
            "세 군집은 성적이 아닌 <b>정서·동기·행동</b> 요인으로 분류됩니다. "
            "동일 성적대라도 군집이 다를 수 있어 유형별 맞춤 접근이 필요합니다.<br>"
            "취약형은 <b>심리적 돌봄</b> 우선, 우수형은 <b>지적 도전</b>으로 잠재력 극대화, "
            "보통형은 <b>내적 동기 발굴</b>이 핵심 전략입니다.")

    # 분석 결과 다운로드
    divider()
    sec("Export", "분석 결과 다운로드")
    if profile_z is not None and len(profile_z) > 0:
        result_df = profile_z.copy()
        result_df.index = [f"C{i+1} {CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else ''}"
                           for i in range(len(result_df))]
        result_df.columns = [FACTOR_LABELS.get(c, c) for c in result_df.columns]
        csv = result_df.round(3).to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 군집 프로파일 CSV 다운로드",
            data=csv,
            file_name="cluster_profile_zscore.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    import io

    # session state 초기화
    if "show_recog" not in st.session_state:
        st.session_state["show_recog"] = False

    # ── 데이터 로드 ──
    uf = st.session_state.get("uploaded_file", None)
    if uf is not None:
        file_bytes = uf.read()
        uf.seek(0)
        raw_df, df_factors, recognized, avail_vars = load_and_process(file_bytes, uf.name)
        data_source = f"📁 {uf.name}"
    else:
        # 데모 데이터
        np.random.seed(42)
        n = 67
        demo = {}
        # 군집이 뚜렷하도록 그룹별 생성
        idx_c1 = np.arange(0, 20)   # 취약형
        idx_c2 = np.arange(20, 45)  # 우수형
        idx_c3 = np.arange(45, 67)  # 보통형
        for col in FACTOR_ITEMS["math_anxiety_mean"]:
            arr = np.zeros(n, dtype=int)
            arr[idx_c1] = np.random.choice([4,5], len(idx_c1), p=[0.4,0.6])
            arr[idx_c2] = np.random.choice([1,2], len(idx_c2), p=[0.5,0.5])
            arr[idx_c3] = np.random.choice([2,3,4], len(idx_c3), p=[0.3,0.4,0.3])
            demo[col] = arr
        for col in FACTOR_ITEMS["math_self_efficacy_mean"]:
            arr = np.zeros(n, dtype=int)
            arr[idx_c1] = np.random.choice([1,2], len(idx_c1), p=[0.5,0.5])
            arr[idx_c2] = np.random.choice([4,5], len(idx_c2), p=[0.4,0.6])
            arr[idx_c3] = np.random.choice([2,3,4], len(idx_c3), p=[0.3,0.4,0.3])
            demo[col] = arr
        for col in FACTOR_ITEMS["math_interest_mean"]:
            arr = np.zeros(n, dtype=int)
            arr[idx_c1] = np.random.choice([1,2], len(idx_c1), p=[0.5,0.5])
            arr[idx_c2] = np.random.choice([4,5], len(idx_c2), p=[0.4,0.6])
            arr[idx_c3] = np.random.choice([2,3,4], len(idx_c3), p=[0.3,0.4,0.3])
            demo[col] = arr
        for col in FACTOR_ITEMS["learning_attitude_mean"]:
            arr = np.zeros(n, dtype=int)
            arr[idx_c1] = np.random.choice([1,2], len(idx_c1), p=[0.5,0.5])
            arr[idx_c2] = np.random.choice([4,5], len(idx_c2), p=[0.4,0.6])
            arr[idx_c3] = np.random.choice([2,3,4], len(idx_c3), p=[0.3,0.4,0.3])
            demo[col] = arr
        raw_df = pd.DataFrame(demo)
        raw_df.columns = raw_df.columns.astype(str).str.strip()
        # 데모 데이터 직접 처리
        recognized = {}
        df_factors = pd.DataFrame(index=raw_df.index)
        for fk, items in FACTOR_ITEMS.items():
            found = [c for c in items if c in raw_df.columns]
            recognized[fk] = found
            if found:
                tmp = raw_df[found].copy()
                for c in found:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                if fk == "learning_attitude_mean" and NEGATIVE_ITEM in tmp.columns:
                    tmp[NEGATIVE_ITEM] = 6 - tmp[NEGATIVE_ITEM]
                df_factors[fk] = tmp.mean(axis=1)
            else:
                df_factors[fk] = np.nan
        avail_vars = [v for v in FACTOR_ITEMS if df_factors[v].notna().sum() > 0]
        data_source = "🔵 데모 데이터 (67명)"
        data_source = "🔵 데모 데이터 (67명)"

    # parquet 직렬화 (캐시 키 용도)
    buf = io.BytesIO()
    df_factors.to_parquet(buf)
    factor_bytes = buf.getvalue()

    # ── 상단 상태바 ──
    st.markdown(
        f'<div style="font-size:11px;color:#8b95a1;text-align:right;margin-bottom:-16px;">'
        f'현재 데이터: <b style="color:#1b64da;">{data_source}</b></div>',
        unsafe_allow_html=True,
    )

    # ── 메인 탭 ──
    t1, t2, t3, t4 = st.tabs([
        "🏠  소개",
        "📋  데이터",
        "📊  데이터분석",
        "🎯  결론",
    ])

    with t1:
        tab_intro(uf)

    with t2:
        tab_data(raw_df, df_factors, recognized, avail_vars)

    with t3:
        result = tab_analysis(df_factors, avail_vars, factor_bytes)
        # 결론 탭에서 쓸 결과 저장
        if result:
            st.session_state["analysis_result"] = result

    with t4:
        result = st.session_state.get("analysis_result", None)
        if result:
            df_res, profile_z, k = result
            tab_conclusion(profile_z, k)
        else:
            st.info("먼저 '데이터분석' 탭에서 분석을 실행하세요.")
            st.markdown("""
            <div class="info-box">
                <div class="ib-label">안내</div>
                <p>① <b>소개</b> 탭에서 파일을 업로드하고<br>
                ② <b>데이터</b> 탭에서 문항 인식 결과를 확인한 후<br>
                ③ <b>데이터분석</b> 탭으로 이동하면 결론이 자동으로 여기에 표시됩니다.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
