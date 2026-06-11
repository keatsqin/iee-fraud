import streamlit as st
import webbrowser

# ================== 页面配置 ==================
st.set_page_config(
    page_title="信用卡欺诈智能风控检测可视化平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================== 自定义CSS样式 ==================
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(145deg, #f0f4fc 0%, #e2eaf5 100%);
    }
    
    /* 隐藏默认的Streamlit导航栏和footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主容器居中 */
    .main > div {
        max-width: 1300px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Hero区域 */
    .hero {
        text-align: center;
        margin-bottom: 3rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1A2A3F, #1e3a5f);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(30, 58, 95, 0.12);
        padding: 0.3rem 1rem;
        border-radius: 60px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1e4a76;
    }
    .hero p {
        color: #2c3e50;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0.75rem auto 0;
        opacity: 0.8;
    }
    
    /* 卡片容器 */
    .card-grid {
        display: flex;
        gap: 2rem;
        justify-content: center;
        margin-top: 1rem;
    }
    
    /* 卡片样式 */
    .card {
        flex: 1;
        background: rgba(255, 255, 255, 0.94);
        border-radius: 2rem;
        padding: 2rem 1.8rem 2.2rem;
        transition: transform 0.25s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 28px 40px -14px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 0.98);
    }
    
    /* 风险感知卡片主题色 */
    .card-risk .card-title {
        background: linear-gradient(120deg, #b83b1e, #e26d3c);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
    }
    .card-risk .feature-list li::before {
        background: #ffe1d6;
        color: #c13c1a;
    }
    .card-risk .jump-btn {
        background: linear-gradient(95deg, #bc4a2c, #e0653a);
        box-shadow: 0 8px 18px rgba(188, 74, 44, 0.25);
    }
    
    /* 智能决策卡片主题色 */
    .card-decision .card-title {
        background: linear-gradient(120deg, #1f6392, #3b8fc2);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
    }
    .card-decision .feature-list li::before {
        background: #e1f0fe;
        color: #2a73b3;
    }
    .card-decision .jump-btn {
        background: linear-gradient(95deg, #1f6e9c, #2c86b9);
        box-shadow: 0 8px 18px rgba(31, 110, 156, 0.25);
    }
    
    .icon-area {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .desc {
        color: #2d3e50;
        line-height: 1.5;
        margin: 0.8rem 0 1rem;
        font-size: 1rem;
    }
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0 1.5rem;
        text-align: left;
        display: inline-block;
    }
    .feature-list li {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
        font-size: 0.9rem;
        color: #2c3e4e;
    }
    .feature-list li::before {
        content: "✓";
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    .jump-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.85rem 1.5rem;
        border-radius: 60px;
        text-decoration: none;
        cursor: pointer;
        width: 100%;
        border: none;
        transition: all 0.2s ease;
    }
    .jump-btn:hover {
        transform: scale(0.98);
        filter: brightness(1.05);
    }
    .footer-note {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #4a627a;
        background: rgba(255,255,240,0.5);
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
        padding: 0.5rem 1.2rem;
        border-radius: 40px;
    }
    
    @media (max-width: 768px) {
        .card-grid {
            flex-direction: column;
        }
        .hero h1 {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================== 配置跳转URL ==================
# 请根据实际业务替换为真实的网址
RISK_TARGET_URL = "http://localhost:8050/"      # 风险感知板块网址
DECISION_TARGET_URL = "http://localhost:8501/"  # 智能决策板块网址


# ================== 跳转函数 ==================
def navigate_to(url, block_name):
    """在新标签页中打开URL"""
    if url and (url.startswith('http://') or url.startswith('https://')):
        webbrowser.open_new_tab(url)
    else:
        st.warning(f"无法跳转：{block_name} 的目标地址无效。请检查配置。")


# ================== 页面内容 ==================
# Hero区域
st.markdown("""
<div class="hero">
    <h1>⚡ 信用卡欺诈可视化平台</h1>
    <p>双核引擎驱动：实时风险感知 + 动态智能决策，全面守护支付安全</p>
</div>
""", unsafe_allow_html=True)

# 双卡片布局
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card card-risk">
        <div class="icon-area">🔍📡</div>
        <div class="card-title">风险感知</div>
        <div class="desc">
            多维特征实时捕获，异常行为洞察。基于时序图谱与异常检测算法，主动发现潜在欺诈模式。
        </div>
        <ul class="feature-list">
            <li>设备指纹 &amp; 地理位置跃迁感知</li>
            <li>关联图谱风险传播预警</li>
            <li>自适应阈值动态调优</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚨 进入风险感知模块 →", key="risk_btn", use_container_width=True):
        navigate_to(RISK_TARGET_URL, "风险感知板块")

with col2:
    st.markdown("""
    <div class="card card-decision">
        <div class="icon-area">🧠⚙️</div>
        <div class="card-title">智能决策</div>
        <div class="desc">
            融合规则引擎与强化学习，输出精准处置策略。自动化审批、交易干预、动态额度管控，决策可解释。
        </div>
        <ul class="feature-list">
            <li>图神经网络辅助裁决</li>
            <li>自适应风控规则编排</li>
            <li>事后归因与策略仿真</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🎯 进入智能决策模块 →", key="decision_btn", use_container_width=True):
        navigate_to(DECISION_TARGET_URL, "智能决策板块")

# 底部信息
st.markdown("""
<div class="footer-note">
    ⚡ 双板块独立跳转 | 欺诈防御等级 · 企业级风控解决方案
</div>
""", unsafe_allow_html=True)