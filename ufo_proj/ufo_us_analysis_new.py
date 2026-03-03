import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="アメリカのUFO目撃データ分析", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }
.stApp { background-color: #0f1117; color: #e2e8f0; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
h1 { font-size: 1.7rem !important; font-weight: 700 !important; color: #f1f5f9 !important;
     padding-bottom: 0.6rem; border-bottom: 1px solid #1e293b; margin-bottom: 0.3rem !important; }
h2 { font-size: 0.72rem !important; font-weight: 500 !important; color: #475569 !important;
     letter-spacing: 0.12em; text-transform: uppercase;
     margin-top: 2.5rem !important; margin-bottom: 1rem !important; }
h3 { font-size: 0.9rem !important; font-weight: 500 !important;
     color: #94a3b8 !important; margin-bottom: 0.3rem !important; }
p  { color: #64748b !important; font-size: 0.84rem !important; line-height: 1.9 !important; }
hr { border-color: #1e293b !important; margin: 2rem 0 !important; }
code { background-color: #1e293b !important; color: #7dd3fc !important;
       padding: 1px 6px !important; border-radius: 4px !important;
       font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }
section[data-testid="stSidebar"] { background-color: #080b11 !important; border-right: 1px solid #1a2133; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #64748b !important; font-size: 0.8rem !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #141b27; border-radius: 8px; padding: 3px; gap: 2px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #475569 !important;
     border-radius: 6px; font-size: 0.8rem; padding: 5px 16px; }
.stTabs [aria-selected="true"] { background-color: #1e3a5f !important; color: #93c5fd !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🛸 アメリカのUFO目撃データ分析")
st.markdown("""<p style='margin-top:-0.2rem; color:#334155 !important; font-size:0.78rem !important;'>
データ出典：National UFO Reporting Center (NUFORC)（約66,000件）&nbsp;·&nbsp;
<a href='https://www.kaggle.com/datasets/NUFORC/ufo-sightings/data' target='_blank'
style='color:#3b82f6 !important; text-decoration:none;'>Kaggle でデータを見る →</a>
</p>""", unsafe_allow_html=True)

BG, GRID, TEXT = "#0f1117", "#1e293b", "#64748b"
PL = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Noto Sans JP", size=11),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT)),
    legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color="#94a3b8")),
    margin=dict(t=36, b=36, l=48, r=24),
)

# ─── State mapping ─────────────────────────────────────────────
STATE_MAP = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "ny": "New York",
    "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma",
    "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming", "dc": "Washington D.C.",
}

# ─── データ読み込み ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ufo_us.csv"))
    df = df.drop(columns=["Unnamed: 0", "country"], errors="ignore")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"]  = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"]  = df["datetime"].dt.hour
    df["state_full"] = df["state"].map(STATE_MAP).fillna(df["state"].str.upper())
    df = df[df["year"].between(1940, 2014)]
    return df

df = load_data()

# ─── サイドバー フィルター ───────────────────────────────────────
st.sidebar.header("フィルター")

year_range = st.sidebar.slider("年の範囲",
    int(df["year"].min()), int(df["year"].max()), (1990, 2014))

top_n = st.sidebar.slider("上位N形状", 3, 15, 8)

# ── 形状フィルター（Allオプション付き）──────────────────────────
all_shapes = sorted(df["shape"].dropna().value_counts().head(15).index.tolist())
shape_options = ["All"] + all_shapes

sel_shapes_raw = st.sidebar.multiselect(
    "形状",
    options=shape_options,
    default=["All"]
)

# "All" が選択されている場合は全形状を使用
if "All" in sel_shapes_raw:
    sel_shapes = all_shapes
else:
    sel_shapes = sel_shapes_raw

# ── 州フィルター（Allオプション付き）────────────────────────────
all_states = sorted(df["state_full"].dropna().unique().tolist())
state_options = ["All"] + all_states

sel_states_raw = st.sidebar.multiselect(
    "州",
    options=state_options,
    default=["All"]
)

# "All" が選択されている場合は全州を使用
if "All" in sel_states_raw:
    sel_states = all_states
else:
    sel_states = sel_states_raw

# ─── フィルター適用 ────────────────────────────────────────────
dff = df[df["year"].between(year_range[0], year_range[1])]
if sel_shapes:
    dff = dff[dff["shape"].isin(sel_shapes)]
if sel_states:
    dff = dff[dff["state_full"].isin(sel_states)]

st.sidebar.markdown(f"**{len(dff):,}** 件選択中")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 1. 1変数の分布
# ══════════════════════════════════════════════════════════════
st.header("1. 1変数の分布")

col1, col2 = st.columns(2)

with col1:
    st.subheader("年別:目撃件数のヒストグラム")
    yearly_hist = dff.groupby("year").size().reset_index(name="count")
    fig1 = px.bar(yearly_hist, x="year", y="count",
                  labels={"year": "年", "count": "目撃件数"},
                  color_discrete_sequence=["#3b82f6"])
    fig1.update_layout(**PL)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
このグラフは、年ごとのUFO目撃件数の変化を観察するために使用する。
横軸は年、縦軸は目撃件数を表している。

全体的な傾向として、目撃件数は時代とともに増加する傾向が見られる。
ただし、この増加はUFOが実際に増えたことを意味するとは限らず、
インターネットの普及によって報告しやすい環境が整ったことも
大きく影響していると考えられる。
フィルターを使って特定の期間・州・形状に絞ることで、
より詳細な傾向を確認することができる。
""")

with col2:
    st.subheader("時間帯別:目撃件数のヒストグラム")
    fig2 = px.histogram(dff.dropna(subset=["hour"]), x="hour", nbins=24,
                        labels={"hour": "時刻（0〜23時）", "count": "目撃件数"},
                        color_discrete_sequence=["#f97316"])
    fig2.update_layout(**PL)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
このグラフは、UFOの目撃が1日のどの時間帯に集中しているかを確認するために使用する。
横軸は時刻（0〜23時）、縦軸は件数を表している。

全体的な傾向として、夜間の目撃件数が昼間より多い傾向が見られる。
これは、暗い空のほうが光を認識しやすいことや、
夜間に屋外で過ごす人の行動パターンが影響していると考えられる。
つまり、目撃件数の多さはUFOの出現頻度だけでなく、
観測条件や人間の行動とも深く関係している可能性がある。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 2. グループ間の分布比較
# ══════════════════════════════════════════════════════════════
st.header("2. グループ間の分布比較")

top_shapes = dff["shape"].value_counts().head(top_n).index.tolist()
dff_shapes = dff[dff["shape"].isin(top_shapes)]
shape_year = dff_shapes.groupby(["shape","year"]).size().reset_index(name="count")

tab1, tab2 = st.tabs(["ボックスプロット", "バイオリンプロット"])

with tab1:
    st.subheader("形状別:年間目撃件数のボックスプロット")
    fig3 = px.box(shape_year, x="shape", y="count", color="shape",
                  labels={"shape": "形状", "count": "年間目撃件数"})
    fig3.update_layout(**PL, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
このグラフは、UFOの形状ごとに年間目撃件数のばらつきを比較するために使用する。
横軸はUFOの形状、縦軸は1年あたりの目撃件数を表している。
箱の中央線が中央値、箱の上下端が第3・第1四分位数、
箱の外側の点が外れ値（outlier）を示している。

全体的な傾向として、形状によって件数のばらつき方が異なることが読み取れる。
ばらつきが大きい形状は年によって件数が不安定であり、
ばらつきが小さい形状は比較的安定した目撃傾向を持つといえる。
""")

with tab2:
    st.subheader("形状別:年間目撃件数のバイオリンプロット")
    fig4 = px.violin(shape_year, x="shape", y="count", color="shape", box=True,
                     labels={"shape": "形状", "count": "年間目撃件数"})
    fig4.update_layout(**PL, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
このグラフは、ボックスプロットと同じデータをバイオリンプロットで表示したものである。
ボックスプロットに加えて分布の形状（密度）も視覚的に確認できる点が特徴で、
図形の幅が広い部分ほどその値のデータが多いことを意味する。

ボックスプロットと並べて読むことで、中央値やばらつきだけでなく、
データがどの値に偏って分布しているかも把握することができる。
形状によって分布の形が異なるため、単純な件数の比較では見えにくい
特徴を発見するのに役立つグラフである。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 3. 2変数間の関係
# ══════════════════════════════════════════════════════════════
st.header("3. 2変数間の関係")

col3, col4 = st.columns(2)

with col3:
    st.subheader("形状別:目撃地点の散布図マップ")
    scatter_df = dff_shapes.dropna(subset=["latitude","longitude"])
    scatter_df = scatter_df.sample(min(3000, len(scatter_df)), random_state=42)
    fig5 = px.scatter_mapbox(
        scatter_df, lat="latitude", lon="longitude", color="shape",
        hover_name="city",
        hover_data={"state_full": True, "year": True, "shape": True,
                    "latitude": False, "longitude": False},
        zoom=2, height=420,
        mapbox_style="carto-darkmatter", opacity=0.6,
        labels={"state_full": "州", "shape": "形状", "year": "年"}
    )
    fig5.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0),
        mapbox={
            "layers": [{
                "sourcetype": "geojson",
                "source": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
                "type": "line",
                "color": "#334155",
                "line": {"width": 1}
            }]
        }
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
このグラフは、UFOの目撃地点と形状の関係を地図上で視覚的に確認するために使用する。
各点は1件の目撃報告を示し、色はUFOの形状に対応している。

全体的な傾向として、目撃報告は人口密度の高い地域に集まりやすい。
これは報告バイアス（reporting bias）の影響であり、
人口が多いほど報告者も多くなるためと考えられる。
形状ごとの地理的な偏りを確認したい場合は、
サイドバーで特定の形状を選択して絞り込むことができる。
""")

with col4:
    st.subheader("形状×月:ヒートマップ")
    hm = dff_shapes.groupby(["shape","month"]).size().unstack(fill_value=0)
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    hm.columns = [month_map[c] for c in hm.columns if c in month_map]
    fig6 = px.imshow(hm, color_continuous_scale="Blues", aspect="auto",
                     labels=dict(x="月", y="形状", color="件数"))
    fig6.update_layout(**PL)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
このグラフは、UFOの形状と目撃された月の組み合わせによる件数の傾向を
一覧で比較するために使用する。色が濃いほど件数が多いことを示している。

全体的な傾向として、夏季に件数が多くなる季節性のパターンが見られる。
これは屋外活動の増加や観測機会の増加が影響していると考えられる。
また、形状によって特定の月に集中するパターンがあるかどうかも
このグラフで確認することができる。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 4. 構成比
# ══════════════════════════════════════════════════════════════
st.header("4. 構成比")

shape_counts = dff["shape"].value_counts().head(top_n).reset_index()
shape_counts.columns = ["shape", "count"]

col5, col6 = st.columns(2)

with col5:
    st.subheader("UFO形状の構成比:ドーナツチャート")
    fig7 = px.pie(shape_counts, values="count", names="shape", hole=0.5)
    fig7.update_traces(textposition="outside", textinfo="label+percent")
    fig7.update_layout(paper_bgcolor=BG, showlegend=False, font=dict(color=TEXT, family="Noto Sans JP"), margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""
このグラフは、目撃されたUFOの形状ごとの構成比を確認するために使用する。

全体的な傾向として、「光」として報告されるケースが最も多く、
次いで三角形・円形などの具体的な形状が続く。
フィルターで期間や州を変更すると構成比も変化するため、
条件によって報告される形状の傾向が異なるかどうかを比較するのに役立つ。
""")

with col6:
    st.subheader("UFO形状の件数:ツリーマップ")
    fig8 = px.treemap(shape_counts, path=["shape"], values="count",
                      color="count", color_continuous_scale="Blues")
    fig8.update_layout(paper_bgcolor=BG, coloraxis_showscale=False, font=dict(color="#cbd5e1", family="Noto Sans JP"), margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown("""
このグラフは、ドーナツチャートと同じデータを別の角度から確認するために使用する。
各形状の件数に比例した面積で表示されるため、形状間の大小関係を直感的に比較しやすい。

ドーナツチャートと合わせて読むことで、件数の順位と割合の両方を
同時に把握することができる。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 5. 時系列の傾向
# ══════════════════════════════════════════════════════════════
st.header("5. 時系列の傾向")

tab3, tab4 = st.tabs(["年別", "月別"])

with tab3:
    st.subheader("年別:目撃件数の折れ線グラフ")
    yearly = dff.groupby("year").size().reset_index(name="count")
    fig9 = px.line(yearly, x="year", y="count", markers=True,
                   labels={"year": "年", "count": "目撃件数"},
                   color_discrete_sequence=["#3b82f6"])
    fig9.update_traces(line_width=2, marker=dict(size=4))
    fig9.update_layout(**PL)
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown("""
このグラフは、UFO目撃件数の年ごとの変化を連続的に追うために使用する。
横軸は年、縦軸は目撃件数を表している。

全体的な傾向として、件数は時代とともに増加する傾向にある。
この増加がUFOの実際の増加を意味するのか、
それとも報告環境の変化によるものかを考察する際に有効なグラフである。
""")

with tab4:
    st.subheader("月別:目撃件数の棒グラフ")
    month_label = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = dff.groupby("month").size().reset_index(name="count")
    monthly["month_name"] = monthly["month"].map(month_label)
    fig10 = px.bar(monthly, x="month_name", y="count",
                   labels={"month_name": "月", "count": "目撃件数"},
                   color_discrete_sequence=["#3b82f6"])
    fig10.update_layout(**PL)
    st.plotly_chart(fig10, use_container_width=True)
    st.markdown("""
このグラフは、月ごとの目撃件数の季節的な傾向を確認するために使用する。
横軸は月、縦軸は件数を表している。

全体的な傾向として、夏季に件数が多くなる季節性のパターンが見られる。
このパターンがUFOの出現頻度によるものか、
それとも屋外活動の増加など人間側の要因によるものかを
考察する際に有効なグラフである。
ヒートマップと合わせて読むと、この傾向をより詳しく確認することができる。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 6. 州別の分布
# ══════════════════════════════════════════════════════════════
st.header("6. 州別の分布")

state_counts = dff.groupby(["state","state_full"]).size().reset_index(name="count")
state_counts = state_counts.sort_values("count", ascending=False)

tab5, tab6 = st.tabs(["棒グラフ", "コロプレスマップ"])

with tab5:
    st.subheader("目撃件数:上位20州の棒グラフ")
    fig11 = px.bar(state_counts.head(20), x="state_full", y="count",
                   labels={"state_full": "州", "count": "目撃件数"},
                   color="count", color_continuous_scale="Blues")
    fig11.update_layout(**PL, coloraxis_showscale=False, xaxis_tickangle=-35)
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown("""
このグラフは、州ごとの目撃件数を比較するために使用する。
横軸は州名、縦軸は件数を表している。

全体的な傾向として、人口の多い州や面積の広い州で件数が多くなりやすい。
ただし、件数の絶対値が多い州が必ずしも「UFOが出やすい州」とは言えず、
人口規模が影響している可能性がある。
人口比を考慮した分析を行うと、異なる傾向が見えてくる可能性がある。
""")

with tab6:
    st.subheader("州別:目撃件数のコロプレスマップ")
    fig12 = px.choropleth(state_counts,
                          locations="state", locationmode="USA-states",
                          color="count", scope="usa",
                          color_continuous_scale="Blues",
                          hover_name="state_full",
                          labels={"count": "目撃件数"})
    fig12.update_layout(paper_bgcolor=BG, geo=dict(bgcolor=BG, lakecolor=BG, landcolor="#141b27", subunitcolor="#1e293b"), font=dict(color=TEXT, family="Noto Sans JP"), margin=dict(t=20,b=20,l=0,r=0))
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown("""
このグラフは、州ごとの目撃件数の地理的な分布パターンを把握するために使用する。
色が濃いほど件数が多いことを示している。

棒グラフと組み合わせて読むことで、件数の順位だけでなく
地理的な偏りのパターンも同時に確認することができる。
全体的な傾向として、人口分布や都市化の度合いと
似たパターンが見られるかどうかを考察する際に有効なグラフである。
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 7. 地理的分布
# ══════════════════════════════════════════════════════════════
st.header("7. 地理的分布")

st.subheader("UFO目撃地点の散布図マップ")

map_df = dff_shapes.dropna(subset=["latitude","longitude"])
map_df = map_df.sample(min(5000, len(map_df)), random_state=42)

fig13 = px.scatter_mapbox(
    map_df, lat="latitude", lon="longitude",
    color="shape", hover_name="city",
    hover_data={"state_full": True, "year": True, "shape": True,
                "latitude": False, "longitude": False},
    zoom=3, height=520,
    mapbox_style="carto-darkmatter", opacity=0.55,
    labels={"state_full": "州", "shape": "形状", "year": "年"}
)
fig13.update_layout(
    paper_bgcolor=BG,
    legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color="#94a3b8")),
    margin=dict(l=0,r=0,t=0,b=0),
    mapbox={
        "layers": [{
            "sourcetype": "geojson",
            "source": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
            "type": "line",
            "color": "#334155",
            "line": {"width": 1}
        }]
    }
)
st.plotly_chart(fig13, use_container_width=True)
st.markdown("""
このグラフは、UFOの目撃地点の地理的な分布を詳しく把握するために使用する。
各点は1件の目撃報告を示し、色はUFOの形状に対応している。
州境界線を重ねて表示しているため、どの州のどの地域に目撃が
集中しているかを視覚的に確認することができる。

全体的な傾向として、目撃報告は人口密集地域に集まりやすい。
サイドバーで形状・州・期間を絞り込むことで、
特定の条件下での地理的な分布の変化を確認することができる。
各データ点にカーソルを合わせると、都市名・州名・年・形状などの
詳細情報を確認することができる。
""")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#1e293b; font-size:0.74rem;'>UFO目撃データ分析 · National UFO Reporting Center (NUFORC) · Streamlit & Plotly</p>", unsafe_allow_html=True)
