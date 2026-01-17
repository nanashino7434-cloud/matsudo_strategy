import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
import math
import re

# ページ設定
st.set_page_config(layout="wide", page_title="Matsudo Strategy Map (Final Fixed)", page_icon="⚔️")

# ---------------------------------------------------------
# 0. ヘルパー関数
# ---------------------------------------------------------
def normalize_town_name(name):
    if not isinstance(name, str): return str(name)
    name = name.replace('ヶ', 'ケ').replace('ヵ', 'ケ')
    name = name.split('（')[0].split('(')[0]
    table = str.maketrans('０１２３４５６７８９', '0123456789')
    name = name.translate(table)
    kanji_nums = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
    for k, v in kanji_nums.items():
        if f"{k}丁目" in name: name = name.replace(f"{k}丁目", f"{v}丁目")
    return name

def calculate_bins_diverging(df, col_name):
    """
    平均を基準にしたビン計算 (青-黄-赤)
    """
    data = df[col_name].dropna()
    if len(data) == 0: return None
    
    mean_val = data.mean()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # [最小, -1σ, -0.3σ, +0.3σ, +1σ, 最大]
    bins = [
        min_val,
        mean_val - 1.0 * std_val,
        mean_val - 0.3 * std_val,
        mean_val + 0.3 * std_val,
        mean_val + 1.0 * std_val,
        max_val
    ]
    
    # 範囲補正
    if bins[0] > min_val: bins[0] = min_val
    if bins[-1] < max_val: bins[-1] = max_val
    
    # 凡例文字の重なり防止（丸め）
    new_bins = [bins[0]]
    for b in bins[1:-1]:
        if max_val > 100: new_bins.append(math.floor(b))
        else: new_bins.append(round(b, 1))
    new_bins.append(bins[-1])
    
    bins = sorted(list(set(new_bins)))
    
    # エラー回避の微小バッファ
    bins[0] -= 0.01
    bins[-1] += 0.01
    
    if len(bins) < 3:
        bins = np.linspace(min_val - 0.1, max_val + 0.1, 6).tolist()
        
    return bins

# ---------------------------------------------------------
# 1. データ読み込み & 前処理
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    try:
        df_pop = pd.read_csv('population.csv', encoding='utf-8')
    except:
        df_pop = pd.read_csv('population.csv', encoding='cp932')
        
    df_prices = pd.read_csv('prices.csv')
    df_stations = pd.read_csv('stations.csv')
    df_votes = pd.read_csv('votes.csv')
    
    town_col = None
    for c in ['town-name', '町丁字名', '町名', 'Name', 'name']:
        if c in df_pop.columns:
            town_col = c
            break
    if not town_col: st.error("人口データに町名がありません"); st.stop()
    
    # 総数行除外
    df_pop = df_pop[~df_pop[town_col].astype(str).str.contains('総数|合計|Total')].copy()
    
    for col in df_pop.columns:
        if col != town_col:
            if df_pop[col].dtype == object:
                df_pop[col] = df_pop[col].astype(str).str.replace(',', '').str.replace('-', '0').str.replace('－', '0')
            df_pop[col] = pd.to_numeric(df_pop[col], errors='coerce').fillna(0)

    # 年齢階級集計
    df_pop['pop_0_14'] = 0
    df_pop['pop_20_34'] = 0
    df_pop['pop_30_49'] = 0
    df_pop['pop_65_over'] = 0
    df_pop['pop_working_all'] = 0
    
    for col in df_pop.columns:
        if col == town_col: continue
        nums = re.findall(r'\d+', col)
        if not nums: continue
        age = int(nums[0])
        val = df_pop[col]
        
        if age < 15: df_pop['pop_0_14'] += val
        if 20 <= age <= 34: df_pop['pop_20_34'] += val
        if 30 <= age <= 49: df_pop['pop_30_49'] += val
        if age >= 65: df_pop['pop_65_over'] += val
        if 15 <= age <= 64: df_pop['pop_working_all'] += val

    # 総人口
    total_col = None
    hh_col = None
    for c in ['pop-total', '総数', 'Total']:
        if c in df_pop.columns: total_col = c; break
    for c in ['pop_households', '世帯数']:
        if c in df_pop.columns: hh_col = c; break
    
    if total_col: df_pop['total_pop'] = df_pop[total_col]
    else: df_pop['total_pop'] = df_pop['pop_working_all'] + df_pop['pop_0_14'] + df_pop['pop_65_over']
    
    if hh_col: df_pop['households'] = df_pop[hh_col]
    else: df_pop['households'] = 0

    df_pop['town_key'] = df_pop[town_col].apply(normalize_town_name)
    
    # ★人口データの重複排除
    agg_cols = {
        'total_pop': 'sum', 'households': 'sum',
        'pop_0_14': 'sum', 'pop_20_34': 'sum', 'pop_30_49': 'sum',
        'pop_65_over': 'sum', 'pop_working_all': 'sum'
    }
    df_pop_grouped = df_pop.groupby('town_key')[list(agg_cols.keys())].sum().reset_index()
    df_pop_names = df_pop.groupby('town_key')[town_col].first().reset_index()
    df_pop = df_pop_grouped.merge(df_pop_names, on='town_key')
    
    df_pop['avg_members'] = df_pop.apply(lambda x: x['total_pop'] / x['households'] if x['households'] > 0 else 2.0, axis=1)

    # ターゲット基礎数
    df_pop['tgt_family_base'] = df_pop['pop_30_49'] + df_pop['pop_0_14']
    df_pop['tgt_single_base'] = df_pop['pop_20_34']

    # 地図データ
    try:
        gdf = gpd.read_file('r2ka12207.shp', encoding='cp932')
        if gdf.crs is None or gdf.crs.to_string() != 'EPSG:4326': gdf = gdf.to_crs(epsg=4326)
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0005, preserve_topology=True)
        gdf['town_key'] = gdf['S_NAME'].apply(normalize_town_name)
        # ★地図データの重複排除
        gdf = gdf.dissolve(by='town_key', as_index=False)
    except: st.error("地図データ読込エラー"); st.stop()

    # マッチング
    df_prices['town_key'] = df_prices['name'].apply(normalize_town_name)
    df_prices_agg = df_prices.groupby('town_key')['Price/m^2'].mean().reset_index()

    # 結合
    gdf_merged = gdf.merge(df_pop, on='town_key', how='left').fillna(0)
    gdf_merged = gdf_merged.merge(df_prices_agg, on='town_key', how='left')
    gdf_merged['land_price'] = gdf_merged['Price/m^2'].fillna(0)

    # 率計算
    def calc_rate(row, num_col):
        return (row[num_col] / row['total_pop'] * 100) if row['total_pop'] > 0 else 0

    gdf_merged['rate_family'] = gdf_merged.apply(lambda r: calc_rate(r, 'tgt_family_base'), axis=1)
    gdf_merged['rate_single'] = gdf_merged.apply(lambda r: calc_rate(r, 'tgt_single_base'), axis=1)
    gdf_merged['rate_senior'] = gdf_merged.apply(lambda r: calc_rate(r, 'pop_65_over'), axis=1)

    # 地価ポイント
    price_locations = []
    for index, row in df_prices.iterrows():
        matched = gdf[gdf['town_key'] == row['town_key']]
        if not matched.empty:
            c = matched.geometry.centroid.iloc[0]
            price_locations.append({'name': row['name'], 'price': row['Price/m^2'], 'lat': c.y, 'lon': c.x})
    df_prices_geocoded = pd.DataFrame(price_locations)

    return df_pop, df_stations, df_votes, gdf_merged, df_prices_geocoded

try:
    df_pop, df_stations, df_votes, gdf_merged, df_prices_geocoded = load_and_process_data()
except Exception as e:
    st.error(f"システムエラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. UI設定
# ---------------------------------------------------------
st.sidebar.title("🗳️ 松戸市 戦略マップ Final")
st.sidebar.caption("平均基準(青⇔赤) & ロジック解説版")

app_mode = st.sidebar.radio("モード切替", ["🗺️ 地図分析", "📊 データリスト"])

heatmap_options = {
    "⚔️ [戦略] ターゲット優勢区分 (Dominant)": "dominant_type",
    "🏠 [Target 1] ファミリー (親+子)": "tgt_family_base",
    "💼 [Target 2] 独身・手取り (若手)": "tgt_single_base",
    "👥 全人口 (Total)": "total_pop",
    "👴 高齢者 (65歳以上)": "pop_65_over",
    "🏢 現役世代全体 (15-64歳)": "pop_working_all"
}
selected_layer_label = st.sidebar.selectbox("背景ヒートマップ", list(heatmap_options.keys()))
selected_col_base = heatmap_options[selected_layer_label]

view_type = st.sidebar.radio("表示基準", ("実数 (人)", "構成比 (%)"), horizontal=True)
if view_type == "実数 (人)":
    map_col = selected_col_base
    legend_title = f"{selected_layer_label} [人数]"
else:
    if selected_col_base == "total_pop": map_col = "total_pop"
    elif "family" in selected_col_base: map_col = "rate_family"
    elif "single" in selected_col_base: map_col = "rate_single"
    elif "senior" in selected_col_base: map_col = "rate_senior"
    else: map_col = selected_col_base 
    legend_title = f"{selected_layer_label} [%]"

# ---------------------------------------------------------
# 3. 戦略スコア計算 (詳細ロジック保持)
# ---------------------------------------------------------
with st.sidebar.expander("🛠️ パラメータ調整", expanded=False):
    catchment_radius = st.slider("商圏半径 (m)", 300, 2000, 750, step=50)
    w_commuter = st.slider("駅近係数 (単身)", 1.0, 2.0, 1.3)
    w_pain = st.slider("地価係数 (痛み)", 1.0, 2.0, 1.1)
    thresh_turnout = st.slider("攻略投票率ライン", 30.0, 60.0, float(df_votes['turnout'].mean()))

# 空間集計
gdf_votes = gpd.GeoDataFrame(df_votes, geometry=gpd.points_from_xy(df_votes.lon, df_votes.lat), crs="EPSG:4326")
votes_proj = gdf_votes.to_crs(epsg=6677)
towns_proj = gdf_merged.to_crs(epsg=6677)
towns_proj['town_centroid'] = towns_proj.geometry.centroid

votes_proj['buffer'] = votes_proj.geometry.buffer(catchment_radius)
joined = gpd.sjoin(votes_proj.set_geometry('buffer'), towns_proj.set_geometry('town_centroid'), how='left', predicate='contains')

groupby_col = 'name_left' if 'name_left' in joined.columns else 'name'

# 集計
agg = joined.groupby(groupby_col).agg({
    'tgt_family_base': 'sum',
    'tgt_single_base': 'sum',
    'pop_0_14': 'sum',
    'total_pop': 'sum',
    'avg_members': 'mean',
    'land_price': 'mean'
}).reset_index()

df_analysis = gdf_votes.merge(agg, left_on='name', right_on=groupby_col, how='left').fillna(0)

# 係数計算
def get_station_dist(row, stations):
    dists = []
    for _, s in stations.iterrows():
        d_lat = (row['lat'] - s['lat']) * 111
        d_lon = (row['lon'] - s['lon']) * 91
        dists.append(math.sqrt(d_lat**2 + d_lon**2))
    return min(dists) if dists else 999
df_analysis['dist_km'] = df_analysis.apply(lambda x: get_station_dist(x, df_stations), axis=1)

df_analysis['val_commuter'] = df_analysis['dist_km'].apply(lambda d: w_commuter if d <= 1.0 else 0.9)

mean_p = df_analysis[df_analysis['land_price'] > 0]['land_price'].mean()
if mean_p == 0: mean_p = 1
df_analysis['val_pain'] = df_analysis['land_price'].apply(
    lambda x: 1.0 + ((x / mean_p - 1) * (w_pain - 1.0)) if x > 0 else 1.0
)

df_analysis['val_potential'] = df_analysis['turnout'].apply(
    lambda x: 1.0 + (thresh_turnout - x) * 0.01  # 1%低いごとに0.01ポイント加算など
).clip(0.8, 1.5)

# ★スコア計算
mean_fam_base = df_analysis['tgt_family_base'].mean()
mean_single_base = df_analysis['tgt_single_base'].mean()

def calc_normalized_score(row):
    # 1. Family層の基礎点（世帯人数補正）
    fam_hh_boost = 1.2 if row['avg_members'] >= 2.0 else 0.9
    raw_fam = row['tgt_family_base'] * fam_hh_boost
    
    # 2. Single層の基礎点（駅近補正）
    # ※val_commuterを使っても良いですが、ロジック維持のためこのままでOK
    station_boost = w_commuter if row['dist_km'] <= 1.0 else 0.9
    raw_single = row['tgt_single_base'] * station_boost
    
    # 正規化（ターゲット判定用）
    norm_fam = raw_fam / mean_fam_base if mean_fam_base > 0 else 0
    norm_single = raw_single / mean_single_base if mean_single_base > 0 else 0
    
    # 3. 総合スコア計算
    # 【修正】ここで再計算せず、スライダー値が反映された既存の係数(val_pain)を使う
    price_boost = row['val_pain']
    
    # 合計スコア = (ターゲット基礎力) × 地価係数 × 投票率係数
    total_score = (raw_fam + raw_single) * price_boost * row['val_potential']
    
    # ドミナント（優勢）タイプの判定
    if norm_fam > norm_single * 1.1: dom = "Family"
    elif norm_single > norm_fam * 1.1: dom = "Single"
    else: dom = "Mix"
        
    return pd.Series([total_score, dom])
    # この1行を追加してください（関数の定義が終わった直後）
df_analysis[['muto_score', 'dom_type']] = df_analysis.apply(calc_normalized_score, axis=1)

# ▼▼▼▼▼▼ 追加：ポテンシャル票数（埋蔵票）の計算 ▼▼▼▼▼▼
# 目標投票率（例えば60%）までの差分を有権者数に掛けて「眠っている票数」を出す
target_turnout = 60.0 # 目標投票率（%）
df_analysis['potential_votes'] = df_analysis.apply(
    lambda x: int(x['eligible_voters'] * (target_turnout - x['turnout']) / 100) 
    if x['turnout'] < target_turnout else 0, 
    axis=1
)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ★4象限 & 理由説明ロジック
med_score = df_analysis['muto_score'].median()

def classify_quadrant(row):
    high_s = row['muto_score'] >= med_score
    high_t = row['turnout'] >= thresh_turnout
    
    # 判定理由の言語化
    reasons = []
    if row['muto_score'] >= med_score:
        reasons.append("ターゲット濃度が高く")
    else:
        reasons.append("ターゲット濃度は低いが")
        
    if not high_t:
        reasons.append("投票率が低いため浮動票の掘り起こしが可能")
        q_label = 'A (重点)' if row['muto_score'] >= med_score else 'D (様子見)'
        q_desc = "ブルーオーシャン"
    else:
        reasons.append("投票率が高く組織票の壁がある")
        q_label = 'B (激戦)' if row['muto_score'] >= med_score else 'C (撤退)'
        q_desc = "レッドオーシャン" if row['muto_score'] >= med_score else "他党地盤"
        
    reason_text = "、".join(reasons) + "。"
    
    # 色分け
    if q_label in ['A (重点)', 'B (激戦)']:
        if row['dom_type'] == 'Family': color = '#FF8C00' # オレンジ
        elif row['dom_type'] == 'Single': color = '#00CED1' # ターコイズ
        else: color = '#FFD700' # 黄色
    elif q_label == 'C (撤退)': color = '#800080'
    else: color = '#808080'

    return pd.Series([q_label, q_desc, reason_text, color])

df_analysis[['quadrant', 'q_desc', 'reason', 'marker_color']] = df_analysis.apply(classify_quadrant, axis=1)

# ---------------------------------------------------------
# 4. メインビュー
# ---------------------------------------------------------
if app_mode == "🗺️ 地図分析":
    col1, col2 = st.columns(2)
    col1.metric("総人口 (補正済)", f"{int(df_pop['total_pop'].sum()):,}人")
    col2.metric("重点エリア数", f"{len(df_analysis[df_analysis['quadrant']=='A (重点)'])}箇所")

    show_votes = st.sidebar.checkbox("投票所マーカー", value=True)
    show_stations = st.sidebar.checkbox("駅アイコン", value=True)
    show_prices = st.sidebar.checkbox("地価ポイント", value=False)

   # 1. 松戸市データの範囲（四隅の座標）を自動取得
    min_lon, min_lat, max_lon, max_lat = gdf_merged.total_bounds

    # 2. 地図の初期化（軽量化と制限を追加）
    m = folium.Map(
        location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2], # 中心をデータから自動計算
        zoom_start=13,
        tiles='CartoDB positron',
        prefer_canvas=True,      # ★重要: これで描画負荷が劇的に下がります（白飛び回避）
        min_zoom=12,             # ズームアウトしすぎないように制限
        max_bounds=True,         # 指定範囲外へのスクロールを禁止
        min_lat=min_lat - 0.03,  # 上下左右に少し余裕を持たせる
        max_lat=max_lat + 0.03,
        min_lon=min_lon - 0.03,
        max_lon=max_lon + 0.03
    )

    # 3. 初期表示を松戸市全体にピタリと合わせる
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
  # ----------------------------------------------------
    # ▼▼▼▼▼▼ 背景地図の描画ロジック修正 ▼▼▼▼▼▼
    # ----------------------------------------------------
    
    # 戦略マップ（Dominant）が選ばれた場合の特殊処理
    if selected_layer_label == "⚔️ [戦略] ターゲット優勢区分 (Dominant)":
        # 1. どちらが優勢かを判定する計算
        mean_f = gdf_merged['tgt_family_base'].mean()
        mean_s = gdf_merged['tgt_single_base'].mean()
        
        def get_dominant_color(row):
            # 平均比でのスコア化
            score_f = row['tgt_family_base'] / mean_f if mean_f > 0 else 0
            score_s = row['tgt_single_base'] / mean_s if mean_s > 0 else 0
            
            # 差が小さい場合は「混合（グレー）」
            if abs(score_f - score_s) < 0.1: return '#dcdde1' # グレー
            # ファミリー優勢
            if score_f > score_s: return '#ff9f43' # オレンジ
            # 独身優勢
            return '#48dbfb' # 水色

        # 2. スタイル関数の定義
        def style_function(feature):
            t_key = feature['properties']['town_key']
            target_row = gdf_merged[gdf_merged['town_key'] == t_key]
            
            if target_row.empty: return {'fillColor': 'gray', 'fillOpacity': 0.1}
            
            color = get_dominant_color(target_row.iloc[0])
            return {
                'fillColor': color,
                'fillOpacity': 0.5, # 少し薄くして下の地図を見えやすく
                'color': 'white',
                'weight': 0.5
            }

        # 3. GeoJsonとして追加
        # (ツールチップはこの後の既存コードで共通表示されるため、ここでは設定不要です)
        folium.GeoJson(
            gdf_merged,
            name="戦略分布マップ",
            style_function=style_function
        ).add_to(m)
        
        # 凡例（HTMLで手作り）
        legend_html_dom = """
        <div style="position: fixed; bottom: 130px; left: 10px; z-index:9999; font-size:12px;
             background:rgba(255,255,255,0.8); padding:5px; border:1px solid #ccc; border-radius:5px;">
             <b>背景色:</b><br>
             <span style="color:#ff9f43">■</span> ファミリー優勢<br>
             <span style="color:#48dbfb">■</span> 独身優勢<br>
             <span style="color:#dcdde1">■</span> 混合・差なし
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html_dom))

    else:
        # 【既存の処理】通常のヒートマップ（赤〜青）
        bins = calculate_bins_diverging(gdf_merged, map_col)
        folium.Choropleth(
            geo_data=gdf_merged, data=gdf_merged, columns=['town_key', map_col],
            key_on='feature.properties.town_key', 
            fill_color='RdYlBu_r', 
            fill_opacity=0.6, line_opacity=0.1,
            bins=bins, legend_name=legend_title
        ).add_to(m)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    # ツールチップ
    folium.GeoJson(
        gdf_merged, name="詳細データ",
        style_function=lambda x: {'color': 'transparent', 'fillColor': 'transparent', 'weight': 0},
        highlight_function=lambda x: {'weight': 2, 'color': '#666', 'fillOpacity': 0.1},
        tooltip=folium.GeoJsonTooltip(
            fields=['S_NAME', 'total_pop', 'pop_0_14', 'tgt_family_base', 'tgt_single_base', 'avg_members', 'land_price'],
            aliases=['町名:', '総人口:', '子供(0-14):', 'ファミリー層:', '独身若手層:', '世帯人員:', '地価:'],
            localize=True
        )
    ).add_to(m)
    
    if show_votes:
        for _, row in df_analysis.iterrows():
            
            # 【修正】メリハリをつけるため係数をアップし、上限も開放
            # sqrt(票数) * 0.8 なので、100票=8px, 400票=16px, 1600票=32px と倍々に増える
            base_radius = math.sqrt(row['potential_votes']) * 0.8
            r = max(8, min(base_radius, 40)) # 最小8px〜最大40px
            
            # メッセージ設定
            if row['dom_type'] == 'Family': msg = "🧒 子育て支援・教育無償化"
            elif row['dom_type'] == 'Single': msg = "💰 手取り増・減税"
            else: msg = "📢 経済・子育て両面訴求"
            
            # ▼▼▼▼▼▼ ポップアップ内容の刷新（係数削除・実数重視） ▼▼▼▼▼▼
            popup_html = f"""
            <div style='width:260px; font-family:sans-serif; font-size:0.95em;'>
                <b style='color:{row['marker_color']}; font-size:1.3em'>{row['quadrant']}</b><br>
                <b style='font-size:1.1em'>{row['name']}</b><hr style='margin:5px 0'>
                
                <b>【攻略ターゲット】</b><br>
                {msg}<br><br>
                
                <b>【票の埋蔵量 (目標投票率60%想定)】</b><br>
                <span style='font-size:1.4em; color:#d63031; font-weight:bold;'>
                    あと {row['potential_votes']:,} 票
                </span><br>
                <span style='font-size:0.85em; color:#666;'>
                    (有権者 {int(row['eligible_voters']):,}人 × 投票率ギャップ {max(0, 60.0 - row['turnout']):.1f}%)
                </span><br><br>
                
                <b>【エリア基礎データ】</b><br>
                ・ファミリー数: {int(row['tgt_family_base']):,} 人<br>
                ・独身若手数: {int(row['tgt_single_base']):,} 人<br>
                ・現在投票率: {row['turnout']}%
            </div>
            """
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # マーカー描画（fill_opacityを上げてクリックしやすく）
            folium.CircleMarker(
                location=[row['lat'], row['lon']], 
                radius=r,
                color='white',
                weight=3,           # 枠線をさらに太くして判定領域を広げる
                fill=True, 
                fill_color=row['marker_color'], 
                fill_opacity=1.0,   # 完全に不透明にして存在感を出す
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"【{row['quadrant']}】{row['name']} (クリックで詳細)" # ツールチップにも誘導を入れる
            ).add_to(m)

            # 文字ラベル（クリック判定を邪魔しないよう pointer-events: none はそのまま）
            label_char = row['quadrant'][0]
            text_html = f"""
            <div style="
                font-family: sans-serif; font-weight: bold; font-size: 11pt; color: black; 
                text-shadow: 2px 0 #fff, -2px 0 #fff, 0 2px #fff, 0 -2px #fff,
                             1px 1px #fff, -1px -1px #fff, 1px -1px #fff, -1px 1px #fff;
                pointer-events: none; text-align: center; width: 20px;
                margin-left: -10px; margin-top: -8px;">
                {label_char}
            </div>
            """
            folium.map.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.DivIcon(html=text_html)
            ).add_to(m)
    
# ▼▼▼▼▼▼ ここから追加（文字ラベル表示） ▼▼▼▼▼▼
            # A/B/C/Dの文字を地図上に直接描画する
            label_char = row['quadrant'][0]  # "A (重点)" の先頭文字 "A" を取得
            
            # 文字色（背景が濃い色なら白、薄いなら黒など調整。今回は白縁取りで見やすく）
            text_html = f"""
            <div style="
                font-family: sans-serif; 
                font-weight: bold; 
                font-size: 10pt; 
                color: black; 
                text-shadow: 
                    1px 1px 0 #fff, -1px -1px 0 #fff,  
                    -1px 1px 0 #fff, 1px -1px 0 #fff;
                pointer-events: none; /* 文字自体はクリック判定を邪魔しない */
                text-align: center;
                width: 20px;
                margin-left: -10px; /* 位置微調整 */
                margin-top: -7px;
            ">
                {label_char}
            </div>
            """
            
            folium.map.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.DivIcon(html=text_html),
                tooltip=f"{row['name']} ({row['quadrant']})" # 文字にカーソルを合わせてもツールチップが出るように
            ).add_to(m)
            # ▲▲▲▲▲▲ ここまで追加 ▲▲▲▲▲▲
    
    if show_stations:
        for i, row in df_stations.iterrows():
            folium.Marker([row['lat'], row['lon']], icon=folium.Icon(color='green', icon='train', prefix='fa'), tooltip=row['name']).add_to(m)
            
    if show_prices:
        for i, row in df_prices_geocoded.iterrows():
            folium.CircleMarker([row['lat'], row['lon']], radius=3, color='#d4af37', fill=True, tooltip=f"地価:{int(row['price']):,}").add_to(m)

    st.markdown("""
    <div style='background-color:white; padding:5px; border-radius:5px; border:1px solid #ddd; font-size:0.9em'>
    <b>マーカー色 (A/Bのみ):</b> <span style='color:#FF8C00'>●ファミリー</span> <span style='color:#00CED1'>●独身</span> <span style='color:#FFD700'>●混合</span>
    </div>
    """, unsafe_allow_html=True)
    
# ▼▼▼▼▼▼ 追加：戦略マトリクス凡例（Legend）の埋め込み ▼▼▼▼▼▼
    
    # 判定基準の閾値（動的に変わる値を取得）
    th_turnout = thresh_turnout  # スライダーの値
    th_score = int(df_analysis['muto_score'].median()) # スコア中央値
    
# 【修正】4象限すべてを表示するコンパクト凡例
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 20px; right: 10px; width: 220px; height: auto; 
        z-index:9999; font-size:11px;
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid #999; border-radius: 5px; padding: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    ">
        <div style="text-align:center; font-weight:bold; margin-bottom:5px;">📊 戦略判定 (基準: 投票率{int(thresh_turnout)}%)</div>
        
        <div style="display: grid; grid-template-columns: 20px 1fr 1fr; gap: 2px; text-align: center;">
            <div style="grid-row:1; grid-column:2; color:#666;">投票率<br><span style="color:red">低</span></div>
            <div style="grid-row:1; grid-column:3; color:#666;">投票率<br><span style="color:blue">高</span></div>
            
            <div style="grid-row:2; grid-column:1; writing-mode: vertical-rl; font-weight:bold; color:#666;">スコア高</div>
            
            <div style="grid-row:2; grid-column:2; background:#FFeee0; border:2px solid #FF8C00; border-radius:4px; padding:2px;">
                <b style="color:#FF8C00; font-size:1.1em">A 重点</b><br>宝の山
            </div>
            <div style="grid-row:2; grid-column:3; background:#e0ffff; border:1px solid #00CED1; border-radius:4px; padding:2px;">
                <b style="color:#008080;">B 激戦</b><br>組織壁
            </div>
            
            <div style="grid-row:3; grid-column:1; writing-mode: vertical-rl; font-weight:bold; color:#666;">スコア低</div>
            
            <div style="grid-row:3; grid-column:2; background:#f9f9f9; border:1px solid #999; border-radius:4px; padding:2px;">
                <b style="color:#555;">D 様子見</b><br>無関心
            </div>
            <div style="grid-row:3; grid-column:3; background:#f0f0f5; border:1px solid #800080; border-radius:4px; padding:2px;">
                <b style="color:#800080;">C 撤退</b><br>高齢者地盤
            </div>
        </div>
    </div>
    """
    
    
    # マップにHTML要素を追加
    m.get_root().html.add_child(folium.Element(legend_html))
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # （この後に st_folium(m, ...) が続きます）

    st_folium(
        m, 
        use_container_width=True, 
        height=750,
        returned_objects=[]  # ★重要: マウス移動やズームでの再リロードを完全に無効化する
    )

elif app_mode == "📊 データリスト":
    st.subheader("💰 埋蔵票ランキング (攻略優先度リスト)")
    st.markdown("現在の投票率と有権者数から、**「あと何票掘り起こせるか（ポテンシャル）」**を算出し、多い順に表示しています。")
    
    tab1, tab2 = st.tabs(["🗳️ 投票所別リスト", "🏘️ 町名別リスト"])
    
    with tab1:
        # 必要な列を抽出し、日本語名にリネーム
        cols = ['name', 'quadrant', 'potential_votes', 'turnout', 'eligible_voters', 'tgt_family_base', 'tgt_single_base']
        
        # 【修正】埋蔵票(potential_votes)の多い順にソート
        df_rank = df_analysis[cols].sort_values('potential_votes', ascending=False).reset_index(drop=True)
        
        df_rank.columns = ['投票所', '判定', '埋蔵票(票)', '現在投票率(%)', '有権者数(人)', 'ファミリー人口', '独身人口']
        
        # データフレーム表示（埋蔵票の列を赤色のバーで強調）
        st.dataframe(
            df_rank.style.bar(subset=['埋蔵票(票)'], color='#ffcccc', vmin=0)
                   .format({'現在投票率(%)': '{:.1f}', '有権者数(人)': '{:,}', '埋蔵票(票)': '{:,}'}),
            use_container_width=True, 
            height=700
        )
        
        st.download_button(
            "📥 このリストをCSVで保存", 
            df_rank.to_csv(index=False).encode('utf-8-sig'), 
            "vote_potential_ranking.csv"
        )
        
    with tab2:
        # 町名リスト（こちらは既存機能の維持だが、列名は修正済みのものを使用）
        st.caption("町名別 基礎データ")
        disp_cols = ['S_NAME', 'total_pop', 'tgt_family_base', 'rate_family', 'tgt_single_base', 'rate_single', 'pop_0_14']
        df_town = gdf_merged[disp_cols].copy()
        df_town = df_town.sort_values('tgt_family_base', ascending=False).reset_index(drop=True)
        # 列名の重複回避（前回の修正適用）
        df_town.columns = ['町名', '総人口', 'ファミリー人口', 'F率%', '独身人口', 'S率%', '子供数']
        
        st.dataframe(
            df_town.style.background_gradient(subset=['ファミリー人口'], cmap='Oranges')
                   .format({'総人口': '{:,}', 'ファミリー人口': '{:,}', '独身人口': '{:,}'}), 
            use_container_width=True, 
            height=600
        )