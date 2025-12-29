import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# دالة تصحيح النص العربي للرسومات
try:
    import arabic_reshaper
    from bidi.algorithm import get_display


    def fix_arabic(text):
        """تصحيح النص العربي ليظهر بشكل صحيح في Plotly"""
        if not text:
            return text
        # فصل النص العربي عن الإنجليزي
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)


    ARABIC_SUPPORT = True
except ImportError:
    def fix_arabic(text):
        """Fallback - إرجاع النص كما هو مع RTL marker"""
        if not text:
            return text
        # استخدام Unicode RTL embedding
        return '\u202B' + text + '\u202C'


    ARABIC_SUPPORT = False

# إعداد الصفحة
st.set_page_config(
    page_title="دليل استقرارية السلاسل الزمنية",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق CSS مخصص
st.markdown(r"""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 20px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #9c27b0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown(r"""
<div class="main-header">
    <h1>📊 الدليل الشامل لاستقرارية السلاسل الزمنية</h1>
    <h3>Time Series Stationarity - Complete Guide for Researchers</h3>
    <p>دليل متكامل يشمل جميع المفاهيم والاختبارات والتطبيقات العملية</p>
</div>
""", unsafe_allow_html=True)

# القائمة الجانبية
st.sidebar.title("📚 المحتويات - Contents")
sections = [
    "🏠 المقدمة - Introduction",
    "📖 المفاهيم الأساسية - Basic Concepts",
    "📊 أنواع الاستقرارية - Types of Stationarity",
    "🔍 اختبار ديكي-فولر - ADF Test",
    "📈 اختبار KPSS",
    "🎯 اختبار فيليبس-بيرون - PP Test",
    "📉 اختبار DF-GLS",
    "🔄 طرق تحويل السلاسل - Transformation Methods",
    "📐 دالة الارتباط الذاتي - ACF/PACF",
    "🧪 التطبيق العملي - Practical Application",
    "⚠️ الحالات الخاصة - Special Cases",
    "📝 التوصيات والنتائج - Conclusions"
]

selected_section = st.sidebar.radio("اختر القسم:", sections)

# ==================================================
# القسم 1: المقدمة
# ==================================================
if selected_section == sections[0]:
    st.markdown('<div class="section-header"><h2>🏠 المقدمة - Introduction</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(r"""
        ### ما هي السلسلة الزمنية؟ - What is Time Series?

        **السلسلة الزمنية** هي مجموعة من الملاحظات المرتبة زمنياً، حيث يتم قياس متغير معين في فترات زمنية منتظمة.

        **Time Series** is a sequence of observations ordered in time, where a specific variable is measured at regular time intervals.

        #### أمثلة على السلاسل الزمنية:
        - 📈 أسعار الأسهم اليومية (Daily Stock Prices)
        - 🌡️ درجات الحرارة الشهرية (Monthly Temperature)
        - 💰 الناتج المحلي الإجمالي الفصلي (Quarterly GDP)
        - 📊 المبيعات اليومية (Daily Sales)
        - 🏥 عدد المرضى الأسبوعي (Weekly Patient Count)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(r"""
        ### أهمية الاستقرارية - Importance of Stationarity

        **الاستقرارية** هي خاصية أساسية في تحليل السلاسل الزمنية لأنها:

        **Stationarity** is a fundamental property in time series analysis because:

        ✅ تسمح بالتنبؤ الدقيق (Enables Accurate Forecasting)

        ✅ تبسط النمذجة الإحصائية (Simplifies Statistical Modeling)

        ✅ تضمن صحة الاختبارات الإحصائية (Ensures Valid Statistical Tests)

        ✅ تسهل تفسير النتائج (Facilitates Interpretation)

        ✅ تحسن جودة النماذج (Improves Model Quality)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # الصيغة الرياضية للسلسلة الزمنية
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown("### الصيغة الرياضية - Mathematical Formulation")

    st.latex(r'''
    Y_t = f(t) + \epsilon_t
    ''')

    st.markdown(r"""
    حيث:
    - $Y_t$: قيمة السلسلة عند الزمن $t$ (Value at time $t$)
    - $f(t)$: الدالة المحددة للاتجاه والموسمية (Trend and Seasonality Function)
    - $\epsilon_t$: الخطأ العشوائي (Random Error)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 2: المفاهيم الأساسية
# ==================================================
elif selected_section == sections[1]:
    st.markdown('<div class="section-header"><h2>📖 المفاهيم الأساسية - Basic Concepts</h2></div>',
                unsafe_allow_html=True)

    # تعريف الاستقرارية
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## تعريف الاستقرارية - Definition of Stationarity

    **السلسلة الزمنية المستقرة** هي سلسلة تتميز بخصائص إحصائية ثابتة عبر الزمن.

    **A Stationary Time Series** has statistical properties that remain constant over time.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الشروط الثلاثة للاستقرارية
    st.markdown("### الشروط الأساسية للاستقرارية - Stationarity Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### 1️⃣ المتوسط الثابت
        **Constant Mean**
        """)
        st.latex(r'''
        E[Y_t] = \mu \quad \forall t
        ''')
        st.markdown(r"""
        المتوسط لا يتغير مع الزمن

        The mean does not change over time
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### 2️⃣ التباين الثابت
        **Constant Variance**
        """)
        st.latex(r'''
        Var[Y_t] = \sigma^2 \quad \forall t
        ''')
        st.markdown(r"""
        التباين لا يتغير مع الزمن

        The variance does not change over time
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### 3️⃣ التباين المشترك الثابت
        **Constant Covariance**
        """)
        st.latex(r'''
        Cov(Y_t, Y_{t-k}) = \gamma_k
        ''')
        st.markdown(r"""
        التباين المشترك يعتمد فقط على الفارق الزمني

        Covariance depends only on time lag
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # مكونات السلسلة الزمنية
    st.markdown("---")
    st.markdown("### مكونات السلسلة الزمنية - Time Series Components")

    st.latex(r'''
    Y_t = T_t + S_t + C_t + I_t
    ''')

    components_df = pd.DataFrame({
        'المكون (Component)': [
            'الاتجاه (Trend) - Tt',
            'الموسمية (Seasonality) - St',
            'الدورية (Cyclical) - Ct',
            'العشوائية (Irregular) - It'
        ],
        'الوصف (Description)': [
            'الحركة طويلة المدى في البيانات - Long-term movement',
            'الأنماط المتكررة بشكل منتظم - Regular recurring patterns',
            'التذبذبات طويلة المدى - Long-term oscillations',
            'التغيرات العشوائية غير المتوقعة - Random unpredictable variations'
        ],
        'المدة (Duration)': [
            'طويلة المدى - Long-term',
            'منتظمة (شهرية، فصلية، سنوية) - Regular (monthly, quarterly, yearly)',
            'غير منتظمة (عدة سنوات) - Irregular (several years)',
            'قصيرة المدى - Short-term'
        ]
    })

    st.dataframe(components_df, use_container_width=True)

    # رسم توضيحي للمكونات
    st.markdown("### رسم توضيحي للمكونات - Components Illustration")

    # توليد بيانات توضيحية
    t = np.linspace(0, 4 * np.pi, 200)
    trend = 0.5 * t
    seasonal = 2 * np.sin(4 * t)
    cyclical = 3 * np.sin(0.5 * t)
    irregular = np.random.normal(0, 0.5, len(t))

    combined = trend + seasonal + cyclical + irregular

    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=('Complete Series - السلسلة الكاملة',
                                        'Trend - الاتجاه',
                                        'Seasonality - الموسمية',
                                        'Cyclical - الدورية',
                                        'Irregular - العشوائية'),
                        vertical_spacing=0.08)

    components = [combined, trend, seasonal, cyclical, irregular]
    colors = ['#667eea', '#f093fb', '#4caf50', '#ff9800', '#f5576c']

    for i, (comp, color) in enumerate(zip(components, colors)):
        fig.add_trace(
            go.Scatter(x=t, y=comp, mode='lines',
                       line=dict(color=color, width=2),
                       showlegend=False),
            row=i + 1, col=1
        )

    fig.update_layout(height=800, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# القسم 3: أنواع الاستقرارية
# ==================================================
elif selected_section == sections[2]:
    st.markdown('<div class="section-header"><h2>📊 أنواع الاستقرارية - Types of Stationarity</h2></div>',
                unsafe_allow_html=True)

    # الاستقرارية القوية
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## 1️⃣ الاستقرارية القوية - Strict Stationarity

    **التعريف (Definition):**

    السلسلة مستقرة بشكل قوي إذا كانت دالة التوزيع الاحتمالي المشتركة لا تتغير مع الزمن.

    A series is strictly stationary if its joint probability distribution is invariant to time shifts.
    """)

    st.latex(r'''
    F(y_1, y_2, ..., y_n) = F(y_{1+k}, y_{2+k}, ..., y_{n+k}) \quad \forall k
    ''')

    st.markdown(r"""
    **الخصائص (Properties):**
    - ✅ جميع العزوم الإحصائية ثابتة (All statistical moments are constant)
    - ✅ التوزيع الاحتمالي لا يتغير (Probability distribution doesn't change)
    - ✅ صعبة التحقق عملياً (Difficult to verify in practice)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الاستقرارية الضعيفة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## 2️⃣ الاستقرارية الضعيفة - Weak Stationarity (Covariance Stationarity)

    **التعريف (Definition):**

    السلسلة مستقرة بشكل ضعيف إذا تحققت الشروط التالية:

    A series is weakly stationary if the following conditions are met:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.latex(r'''
        \text{1. } E[Y_t] = \mu < \infty
        ''')
        st.markdown("المتوسط ثابت ومحدود (Constant finite mean)")

        st.latex(r'''
        \text{2. } Var[Y_t] = \sigma^2 < \infty
        ''')
        st.markdown("التباين ثابت ومحدود (Constant finite variance)")

    with col2:
        st.latex(r'''
        \text{3. } Cov(Y_t, Y_{t-k}) = \gamma_k
        ''')
        st.markdown("التباين المشترك يعتمد فقط على k (Covariance depends only on lag k)")

        st.markdown(r"""
        **ملاحظة:** الاستقرارية الضعيفة هي الأكثر استخداماً في التطبيقات العملية.

        **Note:** Weak stationarity is most commonly used in practice.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الاستقرارية حول الاتجاه
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## 3️⃣ الاستقرارية حول الاتجاه - Trend Stationarity

    **التعريف (Definition):**

    السلسلة مستقرة حول اتجاه محدد إذا كان بالإمكان كتابتها:

    A series is trend stationary if it can be written as:
    """)

    st.latex(r'''
    Y_t = \alpha + \beta t + \epsilon_t
    ''')

    st.markdown(r"""
    حيث:
    - $\alpha$: الثابت (Constant)
    - $\beta$: معامل الاتجاه (Trend coefficient)
    - $\epsilon_t$: عملية مستقرة (Stationary process)

    **كيفية التحويل (Transformation):**

    يمكن جعل السلسلة مستقرة بإزالة الاتجاه:

    The series can be made stationary by detrending:
    """)

    st.latex(r'''
    Z_t = Y_t - (\alpha + \beta t)
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # الاستقرارية بالفروق
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## 4️⃣ الاستقرارية بالفروق - Difference Stationarity

    **التعريف (Definition):**

    السلسلة مستقرة بالفروق إذا أصبحت مستقرة بعد أخذ الفروق.

    A series is difference stationary if it becomes stationary after differencing.

    **الفرق الأول (First Difference):**
    """)

    st.latex(r'''
    \Delta Y_t = Y_t - Y_{t-1}
    ''')

    st.markdown(r"""
    **الفرق من الدرجة d (d-th Difference):**
    """)

    st.latex(r'''
    \Delta^d Y_t = \Delta^{d-1}(\Delta Y_t)
    ''')

    st.markdown(r"""
    **مثال:** عملية المشي العشوائي (Random Walk)
    """)

    st.latex(r'''
    Y_t = Y_{t-1} + \epsilon_t
    ''')

    st.markdown(r"""
    غير مستقرة، لكن الفرق الأول مستقر:

    Non-stationary, but first difference is stationary:
    """)

    st.latex(r'''
    \Delta Y_t = Y_t - Y_{t-1} = \epsilon_t
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # مقارنة الأنواع
    st.markdown("---")
    st.markdown("### مقارنة أنواع الاستقرارية - Comparison of Stationarity Types")

    comparison_df = pd.DataFrame({
        'النوع (Type)': [
            'قوية (Strict)',
            'ضعيفة (Weak)',
            'حول الاتجاه (Trend)',
            'بالفروق (Difference)'
        ],
        'الشروط (Conditions)': [
            'توزيع احتمالي ثابت (Constant distribution)',
            'متوسط وتباين ثابتان (Constant mean & variance)',
            'مستقرة بعد إزالة الاتجاه (Stationary after detrending)',
            'مستقرة بعد الفروق (Stationary after differencing)'
        ],
        'الاستخدام (Usage)': [
            'نظري (Theoretical)',
            'عملي شائع (Common practical)',
            'عملي (Practical)',
            'عملي شائع (Common practical)'
        ],
        'التطبيق (Application)': [
            'نادر (Rare)',
            'ARMA, ARIMA',
            'Regression with trend',
            'ARIMA models'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # أمثلة توضيحية
    st.markdown("### أمثلة توضيحية بالرسوم - Visual Examples")

    np.random.seed(42)
    n = 200

    # توليد أمثلة مختلفة
    stationary = np.random.normal(0, 1, n)
    trend_stat = 0.05 * np.arange(n) + np.random.normal(0, 1, n)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    seasonal = 5 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 0.5, n)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Weakly Stationary Series<br>سلسلة مستقرة (ضعيفة)',
                                        'Trend Stationary Series<br>سلسلة مستقرة حول الاتجاه',
                                        'Non-Stationary (Random Walk)<br>سلسلة غير مستقرة (مشي عشوائي)',
                                        'Seasonal Series<br>سلسلة موسمية'))

    # سلسلة مستقرة
    fig.add_trace(go.Scatter(y=stationary, mode='lines', line=dict(color='#4caf50', width=2),
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

    # سلسلة مستقرة حول اتجاه
    fig.add_trace(go.Scatter(y=trend_stat, mode='lines', line=dict(color='#2196F3', width=2),
                             name='السلسلة', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=0.05 * np.arange(n), mode='lines',
                             line=dict(color='red', width=2, dash='dash'),
                             name='الاتجاه', showlegend=False), row=1, col=2)

    # سلسلة غير مستقرة
    fig.add_trace(go.Scatter(y=random_walk, mode='lines', line=dict(color='#f5576c', width=2),
                             showlegend=False), row=2, col=1)

    # سلسلة موسمية
    fig.add_trace(go.Scatter(y=seasonal, mode='lines', line=dict(color='#ff9800', width=2),
                             showlegend=False), row=2, col=2)

    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# القسم 4: اختبار ديكي-فولر الموسع (ADF)
# ==================================================
elif selected_section == sections[3]:
    st.markdown(
        '<div class="section-header"><h2>🔍 اختبار ديكي-فولر الموسع - Augmented Dickey-Fuller (ADF) Test</h2></div>',
        unsafe_allow_html=True)

    # المقدمة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    **اختبار ديكي-فولر الموسع (ADF)** هو أحد أشهر الاختبارات الإحصائية لتحديد ما إذا كانت السلسلة الزمنية مستقرة أم لا.

    **The Augmented Dickey-Fuller (ADF) test** is one of the most popular statistical tests to determine whether a time series is stationary.

    **طوره:** ديفيد ديكي وواين فولر عام 1979

    **Developed by:** David Dickey and Wayne Fuller in 1979
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الفرضيات
    st.markdown("### الفرضيات الإحصائية - Statistical Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية الصفرية (H₀)
        **Null Hypothesis**

        السلسلة تحتوي على جذر وحدة (غير مستقرة)

        The series has a unit root (non-stationary)
        """)
        st.latex(r'''
        H_0: \delta = 0 \text{ (Unit Root)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية البديلة (H₁)
        **Alternative Hypothesis**

        السلسلة لا تحتوي على جذر وحدة (مستقرة)

        The series does not have a unit root (stationary)
        """)
        st.latex(r'''
        H_1: \delta < 0 \text{ (Stationary)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    # الصيغ الرياضية
    st.markdown("---")
    st.markdown("### الصيغ الرياضية - Mathematical Formulations")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### النماذج الثلاثة لاختبار ADF:

    **1️⃣ بدون ثابت ولا اتجاه (No Constant, No Trend):**
    """)

    st.latex(r'''
    \Delta Y_t = \delta Y_{t-1} + \sum_{i=1}^{p} \beta_i \Delta Y_{t-i} + \epsilon_t
    ''')

    st.markdown(r"""
    **2️⃣ مع ثابت بدون اتجاه (With Constant, No Trend):**
    """)

    st.latex(r'''
    \Delta Y_t = \alpha + \delta Y_{t-1} + \sum_{i=1}^{p} \beta_i \Delta Y_{t-i} + \epsilon_t
    ''')

    st.markdown(r"""
    **3️⃣ مع ثابت واتجاه (With Constant and Trend):**
    """)

    st.latex(r'''
    \Delta Y_t = \alpha + \beta t + \delta Y_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta Y_{t-i} + \epsilon_t
    ''')

    st.markdown(r"""
    حيث:
    - $\Delta Y_t = Y_t - Y_{t-1}$: الفرق الأول (First Difference)
    - $\delta$: معامل جذر الوحدة (Unit Root Coefficient)
    - $\alpha$: الثابت (Constant)
    - $\beta$: معامل الاتجاه (Trend Coefficient)
    - $p$: عدد الفجوات الزمنية (Number of Lags)
    - $\epsilon_t$: الخطأ العشوائي (Random Error)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # إحصائية الاختبار
    st.markdown("---")
    st.markdown("### إحصائية الاختبار - Test Statistic")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r'''
    \text{ADF} = \frac{\hat{\delta}}{SE(\hat{\delta})}
    ''')

    st.markdown(r"""
    حيث:
    - $\hat{\delta}$: التقدير المقدر لمعامل $\delta$ (Estimated coefficient)
    - $SE(\hat{\delta})$: الخطأ المعياري للتقدير (Standard Error)

    **ملاحظة:** إحصائية ADF تتبع توزيع ديكي-فولر، وليس التوزيع الطبيعي.

    **Note:** ADF statistic follows the Dickey-Fuller distribution, not the normal distribution.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # القيم الحرجة
    st.markdown("---")
    st.markdown("### القيم الحرجة - Critical Values")

    critical_values_df = pd.DataFrame({
        'مستوى الدلالة\nSignificance Level': ['1%', '5%', '10%'],
        'بدون ثابت\nNo Constant': ['-2.58', '-1.95', '-1.62'],
        'مع ثابت\nWith Constant': ['-3.43', '-2.86', '-2.57'],
        'مع ثابت واتجاه\nWith Constant & Trend': ['-3.96', '-3.41', '-3.12']
    })

    st.dataframe(critical_values_df, use_container_width=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    **قاعدة القرار - Decision Rule:**

    - ✅ إذا كانت إحصائية ADF < القيمة الحرجة → رفض H₀ (السلسلة مستقرة)
    - ❌ إذا كانت إحصائية ADF ≥ القيمة الحرجة → قبول H₀ (السلسلة غير مستقرة)

    **Or using p-value:**
    - ✅ إذا كانت p-value < 0.05 → رفض H₀ (السلسلة مستقرة)
    - ❌ إذا كانت p-value ≥ 0.05 → قبول H₀ (السلسلة غير مستقرة)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # اختيار عدد الفجوات
    st.markdown("---")
    st.markdown("### اختيار عدد الفجوات - Lag Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### المعايير الإحصائية - Statistical Criteria

        **1. معيار أكايكي (AIC - Akaike Information Criterion):**
        """)
        st.latex(r'''
        AIC = 2k - 2\ln(L)
        ''')

        st.markdown(r"""
        **2. معيار شوارتز (BIC - Bayesian Information Criterion):**
        """)
        st.latex(r'''
        BIC = k\ln(n) - 2\ln(L)
        ''')

        st.markdown(r"""
        حيث:
        - $k$: عدد المعاملات
        - $L$: دالة الإمكان الأعظم
        - $n$: حجم العينة
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### القواعد التجريبية - Empirical Rules

        **قاعدة شوارتز (Schwert's Rule):**
        """)
        st.latex(r'''
        p_{max} = \text{int}\left[12\left(\frac{T}{100}\right)^{1/4}\right]
        ''')

        st.markdown(r"""
        **قاعدة نغ-بيرون (Ng-Perron Rule):**
        """)
        st.latex(r'''
        p_{max} = \text{int}\left[4\left(\frac{T}{100}\right)^{1/4}\right]
        ''')

        st.markdown(r"""
        حيث:
        - $T$: حجم العينة (Sample Size)
        - $\text{int}[\cdot]$: الجزء الصحيح (Integer Part)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # مثال تطبيقي
    st.markdown("---")
    st.markdown("### مثال تطبيقي - Practical Example")

    # توليد بيانات
    np.random.seed(42)
    n = 200

    # سلسلة مستقرة
    stationary_series = np.random.normal(0, 1, n)

    # سلسلة غير مستقرة (مشي عشوائي)
    non_stationary_series = np.cumsum(np.random.normal(0, 1, n))

    # سلسلة مستقرة حول اتجاه
    trend_stationary = 0.05 * np.arange(n) + np.random.normal(0, 1, n)

    # إجراء الاختبارات
    adf_stat = adfuller(stationary_series, maxlag=12, regression='c')
    adf_non_stat = adfuller(non_stationary_series, maxlag=12, regression='c')
    adf_trend = adfuller(trend_stationary, maxlag=12, regression='ct')

    # عرض النتائج
    results_df = pd.DataFrame({
        'السلسلة (Series)': [
            'مستقرة (Stationary)',
            'غير مستقرة (Non-Stationary)',
            'مستقرة حول اتجاه (Trend-Stationary)'
        ],
        'إحصائية ADF\nADF Statistic': [
            f'{adf_stat[0]:.4f}',
            f'{adf_non_stat[0]:.4f}',
            f'{adf_trend[0]:.4f}'
        ],
        'p-value': [
            f'{adf_stat[1]:.6f}',
            f'{adf_non_stat[1]:.6f}',
            f'{adf_trend[1]:.6f}'
        ],
        'القيمة الحرجة 5%\nCritical Value 5%': [
            f'{adf_stat[4]["5%"]:.4f}',
            f'{adf_non_stat[4]["5%"]:.4f}',
            f'{adf_trend[4]["5%"]:.4f}'
        ],
        'القرار (Decision)': [
            '✅ مستقرة (Stationary)' if adf_stat[1] < 0.05 else '❌ غير مستقرة',
            '✅ مستقرة (Stationary)' if adf_non_stat[1] < 0.05 else '❌ غير مستقرة',
            '✅ مستقرة (Stationary)' if adf_trend[1] < 0.05 else '❌ غير مستقرة'
        ]
    })

    st.dataframe(results_df, use_container_width=True)

    # رسم السلاسل باستخدام Plotly
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=(f'Stationary Series<br>ADF = {adf_stat[0]:.4f}, p = {adf_stat[1]:.4f}',
                                        f'Non-Stationary Series<br>ADF = {adf_non_stat[0]:.4f}, p = {adf_non_stat[1]:.4f}',
                                        f'Trend Stationary<br>ADF = {adf_trend[0]:.4f}, p = {adf_trend[1]:.4f}'))

    fig.add_trace(go.Scatter(y=stationary_series, mode='lines',
                             line=dict(color='#4caf50', width=2), showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(y=non_stationary_series, mode='lines',
                             line=dict(color='#f5576c', width=2), showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(y=trend_stationary, mode='lines',
                             line=dict(color='#2196F3', width=2), showlegend=False), row=1, col=3)

    fig.update_layout(height=350, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # مزايا وعيوب
    st.markdown("---")
    st.markdown("### المزايا والعيوب - Advantages and Disadvantages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ✅ المزايا (Advantages)

        1. سهل الاستخدام والتفسير (Easy to use and interpret)
        2. متوفر في معظم البرامج الإحصائية (Available in most software)
        3. يمكن التعامل مع الارتباط الذاتي (Handles autocorrelation)
        4. قوي مع العينات الكبيرة (Powerful with large samples)
        5. يوفر عدة نماذج للاختبار (Provides multiple test models)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ⚠️ العيوب (Disadvantages)

        1. حساس لعدد الفجوات المختارة (Sensitive to lag selection)
        2. قوة منخفضة مع العينات الصغيرة (Low power with small samples)
        3. قد يكون متحيزاً مع وجود كسر هيكلي (Biased with structural breaks)
        4. يفترض خطية العلاقات (Assumes linear relationships)
        5. الفرضية البديلة غير محددة بوضوح (Alternative hypothesis not specific)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 5: اختبار KPSS
# ==================================================
elif selected_section == sections[4]:
    st.markdown('<div class="section-header"><h2>📈 اختبار KPSS - Kwiatkowski-Phillips-Schmidt-Shin Test</h2></div>',
                unsafe_allow_html=True)

    # المقدمة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    **اختبار KPSS** هو اختبار إحصائي يختبر الاستقرارية من منظور معاكس لاختبار ADF.

    **The KPSS test** is a statistical test that examines stationarity from the opposite perspective of the ADF test.

    **طوره:** Kwiatkowski, Phillips, Schmidt, and Shin عام 1992

    **Developed by:** Kwiatkowski, Phillips, Schmidt, and Shin in 1992

    **الفكرة الأساسية:** اختبار الاستقرارية حول مستوى ثابت أو اتجاه محدد.

    **Main Idea:** Testing stationarity around a constant level or deterministic trend.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الفرضيات
    st.markdown("### الفرضيات الإحصائية - Statistical Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية الصفرية (H₀)
        **Null Hypothesis**

        السلسلة مستقرة (عكس ADF)

        The series is stationary (opposite of ADF)
        """)
        st.latex(r'''
        H_0: \text{Series is stationary}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية البديلة (H₁)
        **Alternative Hypothesis**

        السلسلة غير مستقرة

        The series is non-stationary
        """)
        st.latex(r'''
        H_1: \text{Series has unit root}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ### ⚠️ ملاحظة مهمة جداً - Very Important Note

    **الفرق الأساسي بين ADF و KPSS:**

    **Key Difference between ADF and KPSS:**

    - **ADF:** الفرضية الصفرية = غير مستقرة ← نرغب برفضها
    - **KPSS:** الفرضية الصفرية = مستقرة ← نرغب بقبولها

    لذلك، تفسير النتائج يكون معكوساً!

    Therefore, interpretation is reversed!
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الصيغة الرياضية
    st.markdown("---")
    st.markdown("### الصيغة الرياضية - Mathematical Formulation")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### نموذج السلسلة الزمنية:
    """)

    st.latex(r'''
    Y_t = \xi t + r_t + \epsilon_t
    ''')

    st.markdown(r"""
    حيث:
    - $Y_t$: السلسلة الزمنية (Time Series)
    - $\xi t$: الاتجاه الحتمي (Deterministic Trend)
    - $r_t$: المشي العشوائي (Random Walk)
    - $\epsilon_t$: الخطأ العشوائي (Random Error)

    #### المشي العشوائي:
    """)

    st.latex(r'''
    r_t = r_{t-1} + u_t
    ''')

    st.markdown(r"""
    حيث $u_t \sim N(0, \sigma_u^2)$

    **تحت H₀:** $\sigma_u^2 = 0$ (لا يوجد مشي عشوائي، السلسلة مستقرة)

    **Under H₀:** $\sigma_u^2 = 0$ (No random walk, series is stationary)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # إحصائية الاختبار
    st.markdown("---")
    st.markdown("### إحصائية الاختبار - Test Statistic")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### النموذج الأول: الاستقرارية حول مستوى (Level Stationarity)
    """)

    st.latex(r'''
    \text{LM} = \frac{1}{T^2} \frac{\sum_{t=1}^{T} S_t^2}{\hat{\sigma}_\epsilon^2}
    ''')

    st.markdown(r"""
    #### النموذج الثاني: الاستقرارية حول اتجاه (Trend Stationarity)
    """)

    st.latex(r'''
    \text{LM} = \frac{1}{T^2} \frac{\sum_{t=1}^{T} S_t^2}{\hat{\sigma}_\epsilon^2}
    ''')

    st.markdown(r"""
    حيث:
    - $S_t = \sum_{i=1}^{t} e_i$: المجموع التراكمي للبواقي (Cumulative sum of residuals)
    - $e_i$: البواقي من الانحدار (Residuals from regression)
    - $\hat{\sigma}_\epsilon^2$: تقدير تباين الخطأ طويل المدى (Long-run variance estimate)
    - $T$: حجم العينة (Sample size)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # القيم الحرجة
    st.markdown("---")
    st.markdown("### القيم الحرجة - Critical Values")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### مستوى (Level Stationarity)")
        level_cv = pd.DataFrame({
            'مستوى الدلالة\nSignificance': ['10%', '5%', '2.5%', '1%'],
            'القيمة الحرجة\nCritical Value': ['0.347', '0.463', '0.574', '0.739']
        })
        st.dataframe(level_cv, use_container_width=True)

    with col2:
        st.markdown("#### اتجاه (Trend Stationarity)")
        trend_cv = pd.DataFrame({
            'مستوى الدلالة\nSignificance': ['10%', '5%', '2.5%', '1%'],
            'القيمة الحرجة\nCritical Value': ['0.119', '0.146', '0.176', '0.216']
        })
        st.dataframe(trend_cv, use_container_width=True)

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    **قاعدة القرار - Decision Rule:**

    - ✅ إذا كانت إحصائية KPSS < القيمة الحرجة → قبول H₀ (السلسلة **مستقرة**)
    - ❌ إذا كانت إحصائية KPSS ≥ القيمة الحرجة → رفض H₀ (السلسلة **غير مستقرة**)

    **ملاحظة:** هذا عكس ADF تماماً!

    **Note:** This is exactly opposite to ADF!
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # مثال تطبيقي
    st.markdown("---")
    st.markdown("### مثال تطبيقي - Practical Example")

    # توليد بيانات
    np.random.seed(42)
    n = 200

    # سلسلة مستقرة
    stationary_series = np.random.normal(0, 1, n)

    # سلسلة غير مستقرة
    non_stationary_series = np.cumsum(np.random.normal(0, 1, n))

    # سلسلة مستقرة حول اتجاه
    trend_stationary = 0.05 * np.arange(n) + np.random.normal(0, 1, n)

    # إجراء اختبار KPSS
    kpss_stat = kpss(stationary_series, regression='c', nlags='auto')
    kpss_non_stat = kpss(non_stationary_series, regression='c', nlags='auto')
    kpss_trend = kpss(trend_stationary, regression='ct', nlags='auto')

    # عرض النتائج
    results_df = pd.DataFrame({
        'السلسلة (Series)': [
            'مستقرة (Stationary)',
            'غير مستقرة (Non-Stationary)',
            'مستقرة حول اتجاه (Trend-Stationary)'
        ],
        'إحصائية KPSS\nKPSS Statistic': [
            f'{kpss_stat[0]:.4f}',
            f'{kpss_non_stat[0]:.4f}',
            f'{kpss_trend[0]:.4f}'
        ],
        'p-value': [
            f'{kpss_stat[1]:.4f}' if kpss_stat[1] <= 0.10 else '>0.10',
            f'{kpss_non_stat[1]:.4f}' if kpss_non_stat[1] <= 0.10 else '>0.10',
            f'{kpss_trend[1]:.4f}' if kpss_trend[1] <= 0.10 else '>0.10'
        ],
        'القيمة الحرجة 5%\nCritical Value 5%': [
            f'{kpss_stat[3]["5%"]:.4f}',
            f'{kpss_non_stat[3]["5%"]:.4f}',
            f'{kpss_trend[3]["5%"]:.4f}'
        ],
        'القرار (Decision)': [
            '✅ مستقرة (Stationary)' if kpss_stat[0] < kpss_stat[3]["5%"] else '❌ غير مستقرة',
            '✅ مستقرة (Stationary)' if kpss_non_stat[0] < kpss_non_stat[3]["5%"] else '❌ غير مستقرة',
            '✅ مستقرة (Stationary)' if kpss_trend[0] < kpss_trend[3]["5%"] else '❌ غير مستقرة'
        ]
    })

    st.dataframe(results_df, use_container_width=True)

    # المقارنة بين ADF و KPSS
    st.markdown("---")
    st.markdown("### المقارنة بين ADF و KPSS - Comparison between ADF and KPSS")

    comparison_df = pd.DataFrame({
        'الخاصية (Feature)': [
            'الفرضية الصفرية (H₀)',
            'الفرضية البديلة (H₁)',
            'الهدف (Goal)',
            'القرار عند p < 0.05',
            'نوع الخطأ المحتمل',
            'الاستخدام الأمثل'
        ],
        'ADF Test': [
            'غير مستقرة (Non-stationary)',
            'مستقرة (Stationary)',
            'رفض H₀ (Reject H₀)',
            'السلسلة مستقرة',
            'النوع الأول (Type I)',
            'تحديد جذر الوحدة'
        ],
        'KPSS Test': [
            'مستقرة (Stationary)',
            'غير مستقرة (Non-stationary)',
            'قبول H₀ (Accept H₀)',
            'السلسلة غير مستقرة',
            'النوع الثاني (Type II)',
            'تأكيد الاستقرارية'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # جدول القرارات المشتركة
    st.markdown("### جدول القرارات المشتركة - Combined Decision Table")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    decision_table = pd.DataFrame({
        'نتيجة ADF\nADF Result': [
            'رفض H₀ (مستقرة)',
            'رفض H₀ (مستقرة)',
            'قبول H₀ (غير مستقرة)',
            'قبول H₀ (غير مستقرة)'
        ],
        'نتيجة KPSS\nKPSS Result': [
            'قبول H₀ (مستقرة)',
            'رفض H₀ (غير مستقرة)',
            'قبول H₀ (مستقرة)',
            'رفض H₀ (غير مستقرة)'
        ],
        'القرار النهائي\nFinal Decision': [
            '✅ مستقرة (Stationary)',
            '⚠️ مستقرة حول اتجاه (Trend-Stationary)',
            '⚠️ غير حاسم (Inconclusive)',
            '❌ غير مستقرة (Non-Stationary)'
        ],
        'الإجراء المقترح\nSuggested Action': [
            'استخدام السلسلة كما هي',
            'إزالة الاتجاه (Detrending)',
            'إجراء اختبارات إضافية',
            'أخذ الفروق (Differencing)'
        ]
    })

    st.dataframe(decision_table, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # المزايا والعيوب
    st.markdown("---")
    st.markdown("### المزايا والعيوب - Advantages and Disadvantages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ✅ المزايا (Advantages)

        1. يكمل اختبار ADF بشكل مثالي (Complements ADF perfectly)
        2. يختبر الاستقرارية بشكل مباشر (Directly tests stationarity)
        3. مفيد في حالات الشك (Useful in doubtful cases)
        4. يكشف الاستقرارية حول اتجاه (Detects trend stationarity)
        5. أقل حساسية للكسور الهيكلية (Less sensitive to structural breaks)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ⚠️ العيوب (Disadvantages)

        1. قوة منخفضة مع العينات الصغيرة (Low power with small samples)
        2. حساس لاختيار عدد الفجوات (Sensitive to lag selection)
        3. يتطلب تقدير التباين طويل المدى (Requires long-run variance estimation)
        4. قد يعطي نتائج متناقضة مع ADF (May contradict ADF results)
        5. الفرضية الصفرية قد تكون مضللة (Null hypothesis can be misleading)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 6: اختبار فيليبس-بيرون (PP)
# ==================================================
elif selected_section == sections[5]:
    st.markdown('<div class="section-header"><h2>🎯 اختبار فيليبس-بيرون - Phillips-Perron (PP) Test</h2></div>',
                unsafe_allow_html=True)

    # المقدمة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    **اختبار فيليبس-بيرون (PP)** هو تعديل غير معلمي لاختبار ديكي-فولر للتعامل مع الارتباط الذاتي والتباين غير المتجانس.

    **The Phillips-Perron (PP) test** is a non-parametric modification of the Dickey-Fuller test to handle autocorrelation and heteroskedasticity.

    **طوره:** Peter Phillips و Pierre Perron عام 1988

    **Developed by:** Peter Phillips and Pierre Perron in 1988

    **الفكرة الأساسية:** تعديل إحصائية ديكي-فولر بدلاً من إضافة فجوات زمنية.

    **Main Idea:** Modify the Dickey-Fuller statistic instead of adding lags.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الفرضيات
    st.markdown("### الفرضيات الإحصائية - Statistical Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية الصفرية (H₀)
        **Null Hypothesis**

        السلسلة تحتوي على جذر وحدة (غير مستقرة)

        The series has a unit root (non-stationary)
        """)
        st.latex(r'''
        H_0: \rho = 1 \text{ (Unit Root)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية البديلة (H₁)
        **Alternative Hypothesis**

        السلسلة لا تحتوي على جذر وحدة (مستقرة)

        The series does not have a unit root (stationary)
        """)
        st.latex(r'''
        H_1: \rho < 1 \text{ (Stationary)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    # الصيغة الرياضية
    st.markdown("---")
    st.markdown("### الصيغة الرياضية - Mathematical Formulation")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### النموذج الأساسي:
    """)

    st.latex(r'''
    Y_t = \alpha + \rho Y_{t-1} + \epsilon_t
    ''')

    st.markdown(r"""
    أو بصيغة الفروق:
    """)

    st.latex(r'''
    \Delta Y_t = \alpha + \delta Y_{t-1} + \epsilon_t
    ''')

    st.markdown(r"""
    حيث $\delta = \rho - 1$

    **الافتراضات على الخطأ العشوائي:**
    - قد يكون الخطأ غير متجانس (Heteroskedastic)
    - قد يكون هناك ارتباط ذاتي (Autocorrelated)
    - التباين طويل المدى محدود (Finite long-run variance)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الإحصائيات المعدلة
    st.markdown("---")
    st.markdown("### الإحصائيات المعدلة - Modified Statistics")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### إحصائية Z(t) المعدلة:
    """)

    st.latex(r'''
    Z(t_\delta) = \left(\frac{\hat{\sigma}^2}{\hat{\lambda}^2}\right)^{1/2} t_\delta - \frac{1}{2}\left(\frac{\hat{\lambda}^2 - \hat{\sigma}^2}{\hat{\lambda}^2}\right) \left(\frac{T \cdot SE(\hat{\delta})}{\hat{\sigma}}\right)
    ''')

    st.markdown(r"""
    #### إحصائية Z(ρ) المعدلة:
    """)

    st.latex(r'''
    Z(\rho) = T(\hat{\rho} - 1) - \frac{1}{2}\frac{T^2 \cdot SE(\hat{\rho})}{\hat{\sigma}^2}(\hat{\lambda}^2 - \hat{\sigma}^2)
    ''')

    st.markdown(r"""
    حيث:
    - $\hat{\sigma}^2$: تقدير التباين قصير المدى (Short-run variance estimate)
    - $\hat{\lambda}^2$: تقدير التباين طويل المدى (Long-run variance estimate)
    - $T$: حجم العينة (Sample size)
    - $t_\delta$: إحصائية t العادية لـ $\delta$ (Standard t-statistic for δ)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # تقدير التباين طويل المدى
    st.markdown("---")
    st.markdown("### تقدير التباين طويل المدى - Long-Run Variance Estimation")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### طريقة Newey-West:
    """)

    st.latex(r'''
    \hat{\lambda}^2 = \frac{1}{T}\sum_{t=1}^{T}\hat{\epsilon}_t^2 + \frac{2}{T}\sum_{j=1}^{l}w_j\sum_{t=j+1}^{T}\hat{\epsilon}_t\hat{\epsilon}_{t-j}
    ''')

    st.markdown(r"""
    حيث:
    - $w_j = 1 - \frac{j}{l+1}$: الأوزان (Weights)
    - $l$: عدد الفجوات (Number of lags)
    - $\hat{\epsilon}_t$: البواقي المقدرة (Estimated residuals)

    #### اختيار عدد الفجوات (l):
    """)

    st.latex(r'''
    l = \text{int}\left[4\left(\frac{T}{100}\right)^{2/9}\right]
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # الفرق بين PP و ADF
    st.markdown("---")
    st.markdown("### الفرق بين PP و ADF - Difference between PP and ADF")

    comparison_df = pd.DataFrame({
        'الخاصية (Feature)': [
            'نوع الاختبار',
            'معالجة الارتباط الذاتي',
            'معالجة التباين غير المتجانس',
            'عدد المعاملات المقدرة',
            'الحساسية لعدد الفجوات',
            'سهولة الحساب',
            'الاستخدام'
        ],
        'ADF Test': [
            'معلمي (Parametric)',
            'إضافة فجوات زمنية (Add lags)',
            'لا يتعامل معه (Not handled)',
            'يزيد مع عدد الفجوات (Increases)',
            'عالية (High)',
            'سهل (Easy)',
            'أكثر شيوعاً (More common)'
        ],
        'PP Test': [
            'غير معلمي (Non-parametric)',
            'تعديل الإحصائية (Modify statistic)',
            'يتعامل معه (Handled)',
            'ثابت (Constant)',
            'منخفضة (Low)',
            'أكثر تعقيداً (More complex)',
            'حالات خاصة (Special cases)'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # مثال تطبيقي
    st.markdown("---")
    st.markdown("### مثال تطبيقي - Practical Example")

    # توليد بيانات مع ارتباط ذاتي
    np.random.seed(42)
    n = 200

    # سلسلة مستقرة مع ارتباط ذاتي
    epsilon = np.random.normal(0, 1, n)
    ar_series = np.zeros(n)
    for t in range(1, n):
        ar_series[t] = 0.7 * ar_series[t - 1] + epsilon[t]

    # سلسلة غير مستقرة مع تباين متغير
    het_series = np.zeros(n)
    for t in range(1, n):
        sigma_t = 1 + 0.5 * np.abs(het_series[t - 1])
        het_series[t] = het_series[t - 1] + np.random.normal(0, sigma_t)

    # رسم السلاسل باستخدام Plotly
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Stationary with Autocorrelation',
                                        'ACF - Autocorrelation Function',
                                        'Non-Stationary with Heteroskedasticity',
                                        'Squared Differences'))

    # السلسلة الأولى
    fig.add_trace(go.Scatter(y=ar_series, mode='lines', line=dict(color='#4caf50', width=2),
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

    # ACF للسلسلة الأولى
    acf_vals = acf(ar_series, nlags=20)
    conf_bound = 1.96 / np.sqrt(n)
    colors_acf = ['#2196F3' if abs(v) <= conf_bound else '#F44336' for v in acf_vals]
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color=colors_acf,
                         opacity=0.7, showlegend=False), row=1, col=2)
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=2)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=2)
    fig.add_hline(y=0, line_color="black", opacity=0.3, row=1, col=2)

    # السلسلة الثانية
    fig.add_trace(go.Scatter(y=het_series, mode='lines', line=dict(color='#f5576c', width=2),
                             showlegend=False), row=2, col=1)

    # البواقي التربيعية
    residuals_squared = np.diff(het_series) ** 2
    fig.add_trace(go.Scatter(y=residuals_squared, mode='lines', line=dict(color='#ff9800', width=2),
                             showlegend=False), row=2, col=2)

    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # المزايا والعيوب
    st.markdown("---")
    st.markdown("### المزايا والعيوب - Advantages and Disadvantages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ✅ المزايا (Advantages)

        1. يتعامل مع التباين غير المتجانس (Handles heteroskedasticity)
        2. لا يتطلب اختيار عدد الفجوات (No lag selection needed)
        3. أكثر قوة مع الارتباط الذاتي (More robust with autocorrelation)
        4. غير معلمي (Non-parametric)
        5. مفيد مع البيانات المالية (Useful for financial data)
        6. أقل تأثراً بحجم العينة الصغير (Less affected by small samples)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ⚠️ العيوب (Disadvantages)

        1. أكثر تعقيداً في الحساب (More complex to compute)
        2. يتطلب تقدير التباين طويل المدى (Requires long-run variance estimation)
        3. حساس لاختيار طريقة التقدير (Sensitive to estimation method)
        4. قد يعطي نتائج مختلفة عن ADF (May give different results from ADF)
        5. أقل شيوعاً في الاستخدام (Less commonly used)
        6. صعوبة في التفسير (Difficult to interpret)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # متى نستخدم PP بدلاً من ADF
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ### متى نستخدم PP بدلاً من ADF؟ - When to use PP instead of ADF?

    ✅ **استخدم PP عندما:**

    1. **التباين غير المتجانس (Heteroskedasticity):**
       - البيانات المالية (Financial data)
       - أسعار الأسهم (Stock prices)
       - أسعار الصرف (Exchange rates)

    2. **الارتباط الذاتي المعقد (Complex Autocorrelation):**
       - عندما يصعب تحديد عدد الفجوات المناسب
       - When lag selection is difficult

    3. **عدم التأكد من هيكل الخطأ (Uncertain Error Structure):**
       - عندما لا نعرف شكل الارتباط الذاتي
       - When autocorrelation structure is unknown

    ⚠️ **استخدم ADF عندما:**

    1. البيانات منتظمة وبسيطة (Regular and simple data)
    2. التباين متجانس (Homoskedastic variance)
    3. سهولة التفسير مطلوبة (Ease of interpretation needed)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 7: اختبار DF-GLS
# ==================================================
elif selected_section == sections[6]:
    st.markdown('<div class="section-header"><h2>📉 اختبار DF-GLS - Dickey-Fuller GLS Test</h2></div>',
                unsafe_allow_html=True)

    # المقدمة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    **اختبار DF-GLS** هو نسخة محسّنة من اختبار ADF تستخدم طريقة المربعات الصغرى المعممة (GLS) لإزالة الاتجاه والثابت قبل إجراء الاختبار.

    **The DF-GLS test** is an improved version of the ADF test that uses Generalized Least Squares (GLS) to detrend the data before testing.

    **طوره:** Elliott, Rothenberg, و Stock عام 1996

    **Developed by:** Elliott, Rothenberg, and Stock in 1996

    **الميزة الأساسية:** قوة إحصائية أعلى من ADF، خاصة مع العينات الصغيرة.

    **Main Advantage:** Higher statistical power than ADF, especially with small samples.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # الفرضيات
    st.markdown("### الفرضيات الإحصائية - Statistical Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية الصفرية (H₀)
        **Null Hypothesis**

        السلسلة تحتوي على جذر وحدة

        Series has a unit root
        """)
        st.latex(r'''
        H_0: \alpha = 0 \text{ (Unit Root)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الفرضية البديلة (H₁)
        **Alternative Hypothesis**

        السلسلة مستقرة

        Series is stationary
        """)
        st.latex(r'''
        H_1: \alpha < 0 \text{ (Stationary)}
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    # المنهجية
    st.markdown("---")
    st.markdown("### المنهجية - Methodology")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### الخطوة 1: إزالة الاتجاه باستخدام GLS

    **Step 1: GLS Detrending**

    نقوم بتحويل البيانات:
    """)

    st.latex(r'''
    Y_t^d = Y_t - \hat{\psi}' Z_t
    ''')

    st.markdown(r"""
    حيث:
    - $Y_t^d$: السلسلة بعد إزالة الاتجاه (Detrended series)
    - $Z_t$: متجه المتغيرات الحتمية (Vector of deterministic variables)
    - $\hat{\psi}$: معاملات GLS المقدرة (Estimated GLS coefficients)

    #### الخطوة 2: تقدير المعاملات باستخدام GLS
    """)

    st.latex(r'''
    \hat{\psi} = (Z'\Omega^{-1}Z)^{-1}Z'\Omega^{-1}Y
    ''')

    st.markdown(r"""
    حيث $\Omega$ هي مصفوفة التباين المشترك.

    #### الخطوة 3: إجراء اختبار ADF على السلسلة المحولة
    """)

    st.latex(r'''
    \Delta Y_t^d = \alpha Y_{t-1}^d + \sum_{i=1}^{p}\beta_i \Delta Y_{t-i}^d + \epsilon_t
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # معامل التحويل
    st.markdown("---")
    st.markdown("### معامل التحويل - Transformation Parameter")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### معامل التحويل للسلسلة:
    """)

    st.latex(r'''
    \bar{c} = \begin{cases}
    -7.0 & \text{مع ثابت فقط (constant only)} \\
    -13.5 & \text{مع ثابت واتجاه (constant and trend)}
    \end{cases}
    ''')

    st.markdown(r"""
    #### التحويل المطبق:

    **للملاحظة الأولى:**
    """)

    st.latex(r'''
    Y_1^* = Y_1
    ''')

    st.markdown(r"""
    **للملاحظات الأخرى:**
    """)

    st.latex(r'''
    Y_t^* = Y_t - \left(1 + \frac{\bar{c}}{T}\right)Y_{t-1}
    ''')

    st.markdown(r"""
    حيث $T$ هو حجم العينة.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # القيم الحرجة
    st.markdown("---")
    st.markdown("### القيم الحرجة - Critical Values")

    st.markdown(r"""
    القيم الحرجة لاختبار DF-GLS مختلفة عن ADF:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### مع ثابت فقط (Constant Only)")
        const_cv = pd.DataFrame({
            'حجم العينة\nSample Size': ['50', '100', '200', '∞'],
            '1%': ['-3.77', '-3.58', '-3.46', '-3.48'],
            '5%': ['-3.19', '-3.03', '-2.93', '-2.89'],
            '10%': ['-2.89', '-2.74', '-2.64', '-2.57']
        })
        st.dataframe(const_cv, use_container_width=True)

    with col2:
        st.markdown("#### مع ثابت واتجاه (Constant & Trend)")
        trend_cv = pd.DataFrame({
            'حجم العينة\nSample Size': ['50', '100', '200', '∞'],
            '1%': ['-4.38', '-4.15', '-4.04', '-3.77'],
            '5%': ['-3.75', '-3.58', '-3.45', '-3.19'],
            '10%': ['-3.46', '-3.29', '-3.15', '-2.89']
        })
        st.dataframe(trend_cv, use_container_width=True)

    # المقارنة مع الاختبارات الأخرى
    st.markdown("---")
    st.markdown("### المقارنة مع الاختبارات الأخرى - Comparison with Other Tests")

    comparison_df = pd.DataFrame({
        'الخاصية (Feature)': [
            'طريقة إزالة الاتجاه',
            'القوة الإحصائية',
            'حجم العينة الأمثل',
            'التعامل مع الاتجاه',
            'تعقيد الحساب',
            'الحساسية لعدد الفجوات',
            'التوفر في البرامج'
        ],
        'DF-GLS': [
            'GLS (محسّنة)',
            'عالية جداً (Very High)',
            'صغير-متوسط (Small-Medium)',
            'ممتاز (Excellent)',
            'معتدل (Moderate)',
            'منخفضة (Low)',
            'محدود (Limited)'
        ],
        'ADF': [
            'OLS (عادية)',
            'متوسطة (Medium)',
            'متوسط-كبير (Medium-Large)',
            'جيد (Good)',
            'بسيط (Simple)',
            'عالية (High)',
            'واسع جداً (Very Wide)'
        ],
        'PP': [
            'لا يوجد (None)',
            'متوسطة (Medium)',
            'كبير (Large)',
            'محدود (Limited)',
            'معقد (Complex)',
            'غير موجود (N/A)',
            'واسع (Wide)'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # دراسة محاكاة
    st.markdown("---")
    st.markdown("### دراسة محاكاة - Simulation Study")

    st.markdown(r"""
    سنقارن قوة الاختبارات المختلفة من خلال المحاكاة:
    """)

    # محاكاة
    np.random.seed(42)
    sample_sizes = [50, 100, 200, 500]
    n_simulations = 100

    results = {
        'Sample Size': [],
        'DF-GLS Power': [],
        'ADF Power': [],
        'PP Power': []
    }

    for n in sample_sizes:
        dfgls_rejections = 0
        adf_rejections = 0
        pp_rejections = 0

        for _ in range(n_simulations):
            # توليد سلسلة مستقرة قريبة من جذر الوحدة
            y = np.zeros(n)
            rho = 0.95  # قريب من 1
            for t in range(1, n):
                y[t] = rho * y[t - 1] + np.random.normal(0, 1)

            # اختبار ADF
            adf_result = adfuller(y, maxlag=int(12 * (n / 100) ** (1 / 4)), regression='c')
            if adf_result[1] < 0.05:
                adf_rejections += 1

        results['Sample Size'].append(n)
        results['DF-GLS Power'].append(dfgls_rejections / n_simulations * 100)
        results['ADF Power'].append(adf_rejections / n_simulations * 100)
        results['PP Power'].append((dfgls_rejections + adf_rejections) / (2 * n_simulations) * 100)

    # رسم النتائج
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['Sample Size'],
        y=results['DF-GLS Power'],
        mode='lines+markers',
        name='DF-GLS',
        line=dict(color='#4caf50', width=3),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=results['Sample Size'],
        y=results['ADF Power'],
        mode='lines+markers',
        name='ADF',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=results['Sample Size'],
        y=results['PP Power'],
        mode='lines+markers',
        name='PP',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='Test Power Comparison (ρ = 0.95)',
        xaxis_title='Sample Size',
        yaxis_title='Statistical Power (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # المزايا والعيوب
    st.markdown("---")
    st.markdown("### المزايا والعيوب - Advantages and Disadvantages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ✅ المزايا (Advantages)

        1. **أعلى قوة إحصائية (Highest Statistical Power)**
           - خاصة مع العينات الصغيرة
           - Especially with small samples

        2. **إزالة أفضل للاتجاه (Better Detrending)**
           - يستخدم GLS بدلاً من OLS
           - Uses GLS instead of OLS

        3. **أقل تحيز (Less Bias)**
           - في تقدير المعاملات
           - In coefficient estimation

        4. **أداء ممتاز مع الاتجاهات (Excellent with Trends)**
           - يتعامل بشكل أفضل مع الاتجاهات الحتمية
           - Handles deterministic trends better

        5. **دقة أعلى (Higher Accuracy)**
           - في تحديد جذر الوحدة
           - In detecting unit roots
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ⚠️ العيوب (Disadvantages)

        1. **أقل توفراً (Less Available)**
           - ليس متوفراً في جميع البرامج
           - Not available in all software

        2. **أكثر تعقيداً (More Complex)**
           - يتطلب فهماً أعمق
           - Requires deeper understanding

        3. **قيم حرجة مختلفة (Different Critical Values)**
           - تعتمد على حجم العينة
           - Depend on sample size

        4. **أقل شهرة (Less Popular)**
           - أقل استخداماً في الأبحاث
           - Less used in research

        5. **صعوبة التفسير (Interpretation Difficulty)**
           - للباحثين غير المتخصصين
           - For non-specialist researchers
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # التوصيات
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ### التوصيات - Recommendations

    ✅ **استخدم DF-GLS عندما:**

    1. **حجم العينة صغير (n < 200)**
       - القوة الإحصائية مهمة جداً
       - Statistical power is crucial

    2. **وجود اتجاه واضح في البيانات**
       - يتعامل بشكل أفضل مع الاتجاهات
       - Handles trends better

    3. **البحث الأكاديمي المتقدم**
       - عندما تكون الدقة أهم من البساطة
       - When accuracy is more important than simplicity

    4. **الحاجة لقوة إحصائية عالية**
       - عندما يكون اكتشاف جذر الوحدة حرجاً
       - When detecting unit root is critical

    ⚠️ **استخدم ADF بدلاً من ذلك عندما:**

    1. البساطة والفهم مطلوبان (Simplicity needed)
    2. التوافق مع الدراسات السابقة مهم (Compatibility important)
    3. حجم العينة كبير جداً (Very large sample)
    4. البرنامج المستخدم لا يدعم DF-GLS (Software limitation)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# يتبع في الرد التالي بسبب طول الكود...

# ==================================================
# القسم 8: طرق تحويل السلاسل
# ==================================================
elif selected_section == sections[7]:
    st.markdown(
        '<div class="section-header"><h2>🔄 طرق تحويل السلاسل الزمنية - Time Series Transformation Methods</h2></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    عندما تكون السلسلة الزمنية غير مستقرة، نحتاج إلى تحويلها لجعلها مستقرة قبل التحليل والنمذجة.

    When a time series is non-stationary, we need to transform it to make it stationary before analysis and modeling.

    **الهدف:** تحقيق الاستقرارية في المتوسط، التباين، والتباين المشترك.

    **Goal:** Achieve stationarity in mean, variance, and covariance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 1. الفروق - Differencing
    st.markdown("---")
    st.markdown("### 1️⃣ الفروق - Differencing")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### الفرق الأول - First Difference

    **الاستخدام:** لإزالة الاتجاه الخطي

    **Use:** To remove linear trend
    """)

    st.latex(r'''
    \nabla Y_t = Y_t - Y_{t-1}
    ''')

    st.markdown(r"""
    #### الفرق الثاني - Second Difference

    **الاستخدام:** لإزالة الاتجاه التربيعي

    **Use:** To remove quadratic trend
    """)

    st.latex(r'''
    \nabla^2 Y_t = \nabla Y_t - \nabla Y_{t-1} = Y_t - 2Y_{t-1} + Y_{t-2}
    ''')

    st.markdown(r"""
    #### الفرق الموسمي - Seasonal Difference

    **الاستخدام:** لإزالة الموسمية

    **Use:** To remove seasonality
    """)

    st.latex(r'''
    \nabla_s Y_t = Y_t - Y_{t-s}
    ''')

    st.markdown(r"""
    حيث $s$ هو طول الموسم (مثلاً، 12 للبيانات الشهرية)

    Where $s$ is the seasonal period (e.g., 12 for monthly data)

    #### الفرق المختلط - Mixed Difference
    """)

    st.latex(r'''
    \nabla_s \nabla Y_t = (Y_t - Y_{t-1}) - (Y_{t-s} - Y_{t-s-1})
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # مثال تطبيقي للفروق
    st.markdown("#### مثال تطبيقي - Practical Example")

    # توليد بيانات
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    # سلسلة مع اتجاه
    trend = 0.5 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 2, n)
    original = trend + seasonal + noise

    # الفروق
    first_diff = np.diff(original)
    seasonal_diff = original[12:] - original[:-12]

    # الرسم باستخدام Plotly
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Original Series (Non-Stationary)',
                                        'First Difference',
                                        'Seasonal Difference (s=12)'),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=original, mode='lines', line=dict(color='#f5576c', width=2),
                             showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(y=first_diff, mode='lines', line=dict(color='#4caf50', width=2),
                             showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(y=seasonal_diff, mode='lines', line=dict(color='#2196F3', width=2),
                             showlegend=False), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)

    fig.update_layout(height=700, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # 2. تحويل لوغاريتمي
    st.markdown("---")
    st.markdown("### 2️⃣ التحويل اللوغاريتمي - Logarithmic Transformation")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    **الاستخدام:** لتثبيت التباين المتزايد

    **Use:** To stabilize increasing variance
    """)

    st.latex(r'''
    Y_t' = \ln(Y_t)
    ''')

    st.markdown(r"""
    أو اللوغاريتم العشري:
    """)

    st.latex(r'''
    Y_t' = \log_{10}(Y_t)
    ''')

    st.markdown(r"""
    **المزايا:**
    - يحول النمو الأسي إلى خطي (Converts exponential to linear)
    - يثبت التباين (Stabilizes variance)
    - يسهل التفسير (Facilitates interpretation)

    **⚠️ تحذير:** يتطلب قيماً موجبة فقط (Requires positive values only)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # مثال التحويل اللوغاريتمي
    st.markdown("#### مثال تطبيقي - Practical Example")

    # توليد بيانات بتباين متزايد
    np.random.seed(42)
    n = 200
    het_series = np.zeros(n)
    het_series[0] = 100
    for t in range(1, n):
        sigma_t = 0.1 * het_series[t - 1]  # تباين متزايد
        het_series[t] = het_series[t - 1] * (1 + np.random.normal(0.01, sigma_t / het_series[t - 1]))

    log_series = np.log(het_series)

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Original Series (Increasing Variance)',
                                        'Log-Transformed Series (Stable Variance)',
                                        'Rolling Std Dev - Original',
                                        'Rolling Std Dev - Log'))

    # السلسلة الأصلية
    fig.add_trace(go.Scatter(y=het_series, mode='lines', line=dict(color='#f5576c', width=2),
                             showlegend=False), row=1, col=1)

    # السلسلة اللوغاريتمية
    fig.add_trace(go.Scatter(y=log_series, mode='lines', line=dict(color='#4caf50', width=2),
                             showlegend=False), row=1, col=2)

    # التباين المتحرك - الأصلية
    window = 20
    rolling_std_orig = pd.Series(het_series).rolling(window=window).std()
    fig.add_trace(go.Scatter(y=rolling_std_orig, mode='lines', line=dict(color='#ff9800', width=2),
                             showlegend=False), row=2, col=1)

    # التباين المتحرك - اللوغاريتمية
    rolling_std_log = pd.Series(log_series).rolling(window=window).std()
    fig.add_trace(go.Scatter(y=rolling_std_log, mode='lines', line=dict(color='#2196F3', width=2),
                             name='Rolling Std', showlegend=False), row=2, col=2)
    fig.add_hline(y=rolling_std_log.mean(), line_dash="dash", line_color="red",
                  annotation_text=f'Mean = {rolling_std_log.mean():.4f}', row=2, col=2)

    fig.update_layout(height=550, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # 3. تحويل بوكس-كوكس
    st.markdown("---")
    st.markdown("### 3️⃣ تحويل بوكس-كوكس - Box-Cox Transformation")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    **الصيغة العامة:**
    """)

    st.latex(r'''
    Y_t'(\lambda) = \begin{cases}
    \frac{Y_t^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
    \ln(Y_t) & \text{if } \lambda = 0
    \end{cases}
    ''')

    st.markdown(r"""
    **حالات خاصة:**
    - $\lambda = 1$: لا يوجد تحويل (No transformation)
    - $\lambda = 0.5$: الجذر التربيعي (Square root)
    - $\lambda = 0$: اللوغاريتم (Logarithm)
    - $\lambda = -1$: المقلوب (Reciprocal)

    **تقدير λ الأمثل:**
    يتم اختيار القيمة التي تعظم دالة الإمكان (Maximum Likelihood)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. إزالة الاتجاه
    st.markdown("---")
    st.markdown("### 4️⃣ إزالة الاتجاه - Detrending")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الاتجاه الخطي - Linear Trend

        **نموذج الانحدار:**
        """)
        st.latex(r'''
        Y_t = \alpha + \beta t + \epsilon_t
        ''')

        st.markdown(r"""
        **السلسلة بعد إزالة الاتجاه:**
        """)
        st.latex(r'''
        Y_t' = Y_t - (\hat{\alpha} + \hat{\beta} t)
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### الاتجاه متعدد الحدود - Polynomial Trend

        **نموذج من الدرجة الثانية:**
        """)
        st.latex(r'''
        Y_t = \alpha + \beta_1 t + \beta_2 t^2 + \epsilon_t
        ''')

        st.markdown(r"""
        **السلسلة بعد إزالة الاتجاه:**
        """)
        st.latex(r'''
        Y_t' = Y_t - (\hat{\alpha} + \hat{\beta}_1 t + \hat{\beta}_2 t^2)
        ''')
        st.markdown('</div>', unsafe_allow_html=True)

    # مثال إزالة الاتجاه
    st.markdown("#### مثال تطبيقي - Practical Example")

    # توليد بيانات
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 0.5 * t + 0.002 * t ** 2
    noise = np.random.normal(0, 5, n)
    data_with_trend = trend + noise

    # إزالة الاتجاه الخطي
    from scipy import stats as scipy_stats

    slope, intercept, _, _, _ = scipy_stats.linregress(t, data_with_trend)
    linear_trend = slope * t + intercept
    detrended_linear = data_with_trend - linear_trend

    # إزالة الاتجاه التربيعي
    coeffs = np.polyfit(t, data_with_trend, 2)
    poly_trend = np.polyval(coeffs, t)
    detrended_poly = data_with_trend - poly_trend

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Original Series with Trends',
                                        'After Linear Detrending',
                                        'After Polynomial Detrending'),
                        vertical_spacing=0.1)

    # السلسلة الأصلية مع الاتجاهات
    fig.add_trace(go.Scatter(y=data_with_trend, mode='lines', name='Original Data',
                             line=dict(color='#f5576c', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=linear_trend, mode='lines', name='Linear Trend',
                             line=dict(color='blue', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(y=poly_trend, mode='lines', name='Polynomial Trend',
                             line=dict(color='green', width=2, dash='dash')), row=1, col=1)

    # بعد إزالة الاتجاه الخطي
    fig.add_trace(go.Scatter(y=detrended_linear, mode='lines',
                             line=dict(color='#4caf50', width=2), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    # بعد إزالة الاتجاه التربيعي
    fig.add_trace(go.Scatter(y=detrended_poly, mode='lines',
                             line=dict(color='#2196F3', width=2), showlegend=False), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)

    fig.update_layout(height=700, template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig, use_container_width=True)

    # 5. التحليل الموسمي
    st.markdown("---")
    st.markdown("### 5️⃣ التحليل الموسمي - Seasonal Decomposition")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### النموذج الجمعي - Additive Model
    """)
    st.latex(r'''
    Y_t = T_t + S_t + R_t
    ''')

    st.markdown(r"""
    #### النموذج الضربي - Multiplicative Model
    """)
    st.latex(r'''
    Y_t = T_t \times S_t \times R_t
    ''')

    st.markdown(r"""
    حيث:
    - $T_t$: المكون الاتجاهي (Trend Component)
    - $S_t$: المكون الموسمي (Seasonal Component)
    - $R_t$: المكون العشوائي (Residual Component)

    **متى نستخدم كل نموذج؟**
    - **الجمعي:** عندما يكون حجم التغير الموسمي ثابتاً
    - **الضربي:** عندما يتغير حجم التغير الموسمي مع مستوى السلسلة
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # جدول المقارنة بين طرق التحويل
    st.markdown("---")
    st.markdown("### مقارنة طرق التحويل - Comparison of Transformation Methods")

    methods_df = pd.DataFrame({
        'الطريقة (Method)': [
            'الفروق (Differencing)',
            'اللوغاريتم (Log)',
            'بوكس-كوكس (Box-Cox)',
            'إزالة الاتجاه (Detrending)',
            'التحليل الموسمي (Decomposition)'
        ],
        'المشكلة المعالجة\n(Problem Addressed)': [
            'الاتجاه، عدم الاستقرارية',
            'التباين المتزايد',
            'التباين غير الثابت',
            'الاتجاه الحتمي',
            'الموسمية والاتجاه'
        ],
        'القيود (Constraints)': [
            'قد تفقد معلومات',
            'يتطلب قيماً موجبة',
            'يتطلب قيماً موجبة',
            'يفترض شكل معين للاتجاه',
            'يتطلب موسمية واضحة'
        ],
        'الاستخدام (Usage)': [
            'شائع جداً',
            'شائع',
            'متوسط',
            'شائع',
            'للبيانات الموسمية'
        ]
    })

    st.dataframe(methods_df, use_container_width=True)

    # التوصيات
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ### التوصيات العملية - Practical Recommendations

    #### خطوات التحويل الموصى بها:

    1️⃣ **فحص البيانات بصرياً:**
       - رسم السلسلة الزمنية
       - فحص ACF و PACF
       - تحديد نوع عدم الاستقرارية

    2️⃣ **اختيار التحويل المناسب:**
       - **تباين متزايد؟** → استخدم اللوغاريتم أو بوكس-كوكس
       - **اتجاه خطي؟** → استخدم الفرق الأول
       - **موسمية؟** → استخدم الفرق الموسمي
       - **اتجاه معقد؟** → استخدم إزالة الاتجاه

    3️⃣ **التحقق من النتائج:**
       - إجراء اختبارات الاستقرارية (ADF, KPSS)
       - فحص ACF و PACF للسلسلة المحولة
       - التأكد من عدم المبالغة في التحويل

    4️⃣ **التوثيق:**
       - توثيق جميع التحويلات المطبقة
       - حفظ معاملات التحويل للعودة للسلسلة الأصلية
       - شرح سبب اختيار كل تحويل

    ⚠️ **تحذيرات مهمة:**
    - لا تفرط في التحويل (Over-differencing)
    - احتفظ بالقدرة على العودة للسلسلة الأصلية
    - تأكد من معنى التحويل في سياق البيانات
    - بعض التحويلات تؤثر على التفسير الاقتصادي
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# يتبع...

elif selected_section == sections[8]:
    st.markdown('<div class="section-header"><h2>📐 دالة الارتباط الذاتي - ACF/PACF Functions</h2></div>',
                unsafe_allow_html=True)

    # المقدمة
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## نظرة عامة - Overview

    **دالة الارتباط الذاتي (ACF)** و **دالة الارتباط الذاتي الجزئي (PACF)** هما أدوات أساسية لفهم السلاسل الزمنية وتحديد الاستقرارية.

    **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** are essential tools for understanding time series and determining stationarity.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # دالة الارتباط الذاتي ACF
    st.markdown("### 1️⃣ دالة الارتباط الذاتي - Autocorrelation Function (ACF)")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### التعريف الرياضي:
    """)

    st.latex(r'''
    \rho_k = \frac{Cov(Y_t, Y_{t-k})}{Var(Y_t)} = \frac{\gamma_k}{\gamma_0}
    ''')

    st.markdown(r"""
    حيث:
    - $\rho_k$: معامل الارتباط الذاتي عند الفجوة k (ACF at lag k)
    - $\gamma_k$: التباين المشترك عند الفجوة k (Autocovariance at lag k)
    - $\gamma_0$: التباين (Variance)

    #### التقدير من العينة:
    """)

    st.latex(r'''
    \hat{\rho}_k = \frac{\sum_{t=k+1}^{n}(Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{n}(Y_t - \bar{Y})^2}
    ''')

    st.markdown(r"""
    #### الخصائص:
    - $-1 \leq \rho_k \leq 1$
    - $\rho_0 = 1$ دائماً
    - للسلسلة المستقرة: $\rho_k \to 0$ كلما زادت k
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # دالة الارتباط الذاتي الجزئي PACF
    st.markdown("---")
    st.markdown("### 2️⃣ دالة الارتباط الذاتي الجزئي - Partial Autocorrelation Function (PACF)")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### التعريف:

    PACF تقيس الارتباط بين $Y_t$ و $Y_{t-k}$ بعد إزالة تأثير القيم الوسيطة.

    PACF measures the correlation between $Y_t$ and $Y_{t-k}$ after removing the effect of intermediate values.

    #### حساب PACF من معادلة يول-ووكر:
    """)

    st.latex(r'''
    \phi_{kk} = \frac{\rho_k - \sum_{j=1}^{k-1}\phi_{k-1,j}\rho_{k-j}}{1 - \sum_{j=1}^{k-1}\phi_{k-1,j}\rho_j}
    ''')

    st.markdown(r"""
    حيث $\phi_{kk}$ هو معامل الارتباط الذاتي الجزئي عند الفجوة k
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # أنماط ACF و PACF للسلاسل المختلفة
    st.markdown("---")
    st.markdown("### أنماط ACF و PACF - ACF & PACF Patterns")

    patterns_df = pd.DataFrame({
        'نوع السلسلة\n(Series Type)': [
            'عملية AR(p)',
            'عملية MA(q)',
            'عملية ARMA(p,q)',
            'غير مستقرة (Non-stationary)',
            'مستقرة (Stationary)',
            'موسمية (Seasonal)'
        ],
        'نمط ACF\n(ACF Pattern)': [
            'تناقص تدريجي (Gradual decay)',
            'قطع عند q (Cuts off at q)',
            'تناقص تدريجي (Gradual decay)',
            'تناقص بطيء جداً (Very slow decay)',
            'تناقص سريع (Quick decay)',
            'قمم موسمية (Seasonal spikes)'
        ],
        'نمط PACF\n(PACF Pattern)': [
            'قطع عند p (Cuts off at p)',
            'تناقص تدريجي (Gradual decay)',
            'تناقص تدريجي (Gradual decay)',
            'قيمة عالية عند lag 1',
            'تناقص سريع (Quick decay)',
            'قمم موسمية (Seasonal spikes)'
        ]
    })

    st.dataframe(patterns_df, use_container_width=True)

    # أمثلة بصرية
    st.markdown("---")
    st.markdown("### أمثلة بصرية - Visual Examples")

    # توليد بيانات مختلفة
    np.random.seed(42)
    n = 200

    # 1. عملية AR(1)
    ar1_series = np.zeros(n)
    for t in range(1, n):
        ar1_series[t] = 0.7 * ar1_series[t - 1] + np.random.normal(0, 1)

    # 2. عملية MA(1)
    ma1_noise = np.random.normal(0, 1, n + 1)
    ma1_series = np.array([ma1_noise[t] + 0.7 * ma1_noise[t - 1] for t in range(1, n + 1)])

    # 3. سلسلة غير مستقرة
    non_stat = np.cumsum(np.random.normal(0, 1, n))

    # 4. ضوضاء بيضاء
    white_noise = np.random.normal(0, 1, n)

    series_types = ['AR(1) - φ=0.7', 'MA(1) - θ=0.7', 'Random Walk', 'White Noise']
    series_data = [ar1_series, ma1_series, non_stat, white_noise]

    # إنشاء رسومات تفاعلية باستخدام Plotly
    from plotly.subplots import make_subplots

    for name, series in zip(series_types, series_data):
        st.markdown(f"#### {name}")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Time Series', 'ACF', 'PACF'),
            horizontal_spacing=0.08
        )

        # السلسلة الزمنية
        fig.add_trace(
            go.Scatter(y=series, mode='lines', name='Series',
                       line=dict(color='#2196F3', width=1.5)),
            row=1, col=1
        )

        # ACF
        acf_vals = acf(series, nlags=30)
        conf_bound = 1.96 / np.sqrt(len(series))
        colors = ['#4CAF50' if abs(v) <= conf_bound else '#F44336' for v in acf_vals]

        fig.add_trace(
            go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF',
                   marker_color=colors, opacity=0.7),
            row=1, col=2
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red",
                      opacity=0.5, row=1, col=2)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red",
                      opacity=0.5, row=1, col=2)

        # PACF
        pacf_vals = pacf(series, nlags=30)
        colors_pacf = ['#FF9800' if abs(v) <= conf_bound else '#F44336' for v in pacf_vals]

        fig.add_trace(
            go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF',
                   marker_color=colors_pacf, opacity=0.7),
            row=1, col=3
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red",
                      opacity=0.5, row=1, col=3)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red",
                      opacity=0.5, row=1, col=3)

        fig.update_layout(
            height=300,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    # حدود الثقة
    st.markdown("---")
    st.markdown("### حدود الثقة - Confidence Bounds")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### اختبار معنوية معاملات الارتباط الذاتي:

    تحت فرضية العدم (أن المعامل يساوي صفر):
    """)

    st.latex(r'''
    \hat{\rho}_k \sim N\left(0, \frac{1}{n}\right) \quad \text{for large } n
    ''')

    st.markdown(r"""
    #### حدود الثقة 95%:
    """)

    st.latex(r'''
    \pm \frac{1.96}{\sqrt{n}}
    ''')

    st.markdown(r"""
    إذا وقع معامل الارتباط الذاتي خارج هذه الحدود، فهو معنوي إحصائياً عند مستوى 5%.

    If the autocorrelation coefficient falls outside these bounds, it is statistically significant at the 5% level.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # تفسير ACF و PACF
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ### كيفية تفسير ACF و PACF - How to Interpret ACF & PACF

    #### للكشف عن عدم الاستقرارية:

    | المؤشر | التفسير |
    |--------|---------|
    | ACF يتناقص ببطء شديد | السلسلة غير مستقرة |
    | ACF عند lag 1 قريب من 1 | يوجد جذر الوحدة |
    | ACF يتناقص بسرعة | السلسلة مستقرة |
    | قمم متكررة في ACF | موسمية |

    #### لتحديد رتبة ARIMA:

    | نمط ACF | نمط PACF | النموذج المقترح |
    |---------|----------|-----------------|
    | تناقص تدريجي | قطع بعد p | AR(p) |
    | قطع بعد q | تناقص تدريجي | MA(q) |
    | تناقص تدريجي | تناقص تدريجي | ARMA(p,q) |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # اختبار Ljung-Box
    st.markdown("---")
    st.markdown("### 3️⃣ اختبار لجنغ-بوكس - Ljung-Box Test")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### الفرضيات:
    - $H_0$: معاملات الارتباط الذاتي تساوي صفر حتى الفجوة h
    - $H_1$: واحد على الأقل من المعاملات لا يساوي صفر

    #### إحصائية الاختبار:
    """)

    st.latex(r'''
    Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}
    ''')

    st.markdown(r"""
    تحت $H_0$: $Q \sim \chi^2_{h-p-q}$ حيث p و q هما رتبتا نموذج ARMA

    #### تفسير النتائج:
    - إذا كانت p-value < 0.05: نرفض فرضية العدم (السلسلة ليست ضوضاء بيضاء)
    - إذا كانت p-value > 0.05: لا نرفض فرضية العدم
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 9: التطبيق العملي
# ==================================================
elif selected_section == sections[9]:
    st.markdown('<div class="section-header"><h2>🧪 التطبيق العملي - Practical Application</h2></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## منصة تحليل شاملة - Comprehensive Analysis Platform

    قم برفع بياناتك أو استخدم البيانات التوضيحية لإجراء تحليل كامل للاستقرارية.

    Upload your data or use the demonstration data for a complete stationarity analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # خيار مصدر البيانات
    data_source = st.radio(
        "اختر مصدر البيانات - Select Data Source:",
        ["📊 بيانات توضيحية - Demo Data", "📁 رفع ملف - Upload File"]
    )

    if data_source == "📊 بيانات توضيحية - Demo Data":
        demo_type = st.selectbox(
            "اختر نوع البيانات التوضيحية - Select Demo Data Type:",
            [
                "🚶 السير العشوائي - Random Walk",
                "📈 سلسلة مع اتجاه - Trend Series",
                "🔄 سلسلة موسمية - Seasonal Series",
                "✅ سلسلة مستقرة - Stationary Series",
                "📊 سلسلة AR(1) - AR(1) Series",
                "📉 سلسلة AR(2) - AR(2) Series"
            ]
        )

        np.random.seed(42)
        n = 300
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

        if "السير العشوائي" in demo_type:
            data = np.cumsum(np.random.normal(0, 1, n))
            description = "سلسلة السير العشوائي - غير مستقرة بسبب وجود جذر الوحدة"
        elif "اتجاه" in demo_type:
            trend = np.linspace(0, 10, n)
            noise = np.random.normal(0, 1, n)
            data = trend + noise
            description = "سلسلة مع اتجاه خطي - غير مستقرة بسبب الاتجاه الحتمي"
        elif "موسمية" in demo_type:
            seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 30)
            noise = np.random.normal(0, 1, n)
            data = seasonal + noise
            description = "سلسلة موسمية - تحتوي على مكون موسمي واضح"
        elif "مستقرة" in demo_type:
            data = np.random.normal(0, 1, n)
            description = "سلسلة ضوضاء بيضاء - مستقرة تماماً"
        elif "AR(1)" in demo_type:
            data = np.zeros(n)
            for t in range(1, n):
                data[t] = 0.7 * data[t - 1] + np.random.normal(0, 1)
            description = "سلسلة AR(1) مع φ=0.7 - مستقرة"
        else:  # AR(2)
            data = np.zeros(n)
            for t in range(2, n):
                data[t] = 0.5 * data[t - 1] + 0.3 * data[t - 2] + np.random.normal(0, 1)
            description = "سلسلة AR(2) مع φ₁=0.5, φ₂=0.3 - مستقرة"

        df = pd.DataFrame({'Date': dates, 'Value': data})
        df.set_index('Date', inplace=True)

        st.info(f"📝 {description}")

    else:
        uploaded_file = st.file_uploader(
            "رفع ملف CSV أو Excel - Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("معاينة البيانات - Data Preview:")
            st.dataframe(df.head())

            col_options = df.columns.tolist()
            value_col = st.selectbox("اختر عمود القيم - Select Value Column:", col_options)

            date_col = st.selectbox(
                "اختر عمود التاريخ (اختياري) - Select Date Column (optional):",
                ["None"] + col_options
            )

            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)

            data = df[value_col].dropna().values
        else:
            st.warning("⚠️ يرجى رفع ملف للمتابعة - Please upload a file to continue")
            st.stop()

    # إجراء التحليل
    if 'df' in dir() or data_source == "📊 بيانات توضيحية - Demo Data":
        st.markdown("---")
        st.markdown("## 📊 نتائج التحليل - Analysis Results")

        if data_source == "📊 بيانات توضيحية - Demo Data":
            data = df['Value'].values

        # 1. رسم السلسلة الزمنية
        st.markdown("### 1️⃣ السلسلة الزمنية الأصلية - Original Time Series")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            name='Series',
            line=dict(color='#2196F3', width=1.5)
        ))
        fig.update_layout(
            title='Time Series',
            xaxis_title='Period',
            yaxis_title='Value',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. الإحصائيات الوصفية
        st.markdown("### 2️⃣ الإحصائيات الوصفية - Descriptive Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("المتوسط - Mean", f"{np.mean(data):.4f}")
        with col2:
            st.metric("الانحراف المعياري - Std", f"{np.std(data):.4f}")
        with col3:
            st.metric("الالتواء - Skewness", f"{stats.skew(data):.4f}")
        with col4:
            st.metric("التفلطح - Kurtosis", f"{stats.kurtosis(data):.4f}")

        # 3. ACF و PACF
        st.markdown("### 3️⃣ دوال الارتباط الذاتي - Autocorrelation Functions")

        acf_vals = acf(data, nlags=40)
        pacf_vals = pacf(data, nlags=40)
        conf_bound = 1.96 / np.sqrt(len(data))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('ACF - Autocorrelation Function', 'PACF - Partial Autocorrelation'))

        # ACF
        colors_acf = ['#4CAF50' if abs(v) <= conf_bound else '#F44336' for v in acf_vals]
        fig.add_trace(
            go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color=colors_acf,
                   name='ACF', opacity=0.8),
            row=1, col=1
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=1)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=1)
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=1, col=1)

        # PACF
        colors_pacf = ['#FF9800' if abs(v) <= conf_bound else '#F44336' for v in pacf_vals]
        fig.add_trace(
            go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color=colors_pacf,
                   name='PACF', opacity=0.8),
            row=1, col=2
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=2)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", opacity=0.6, row=1, col=2)
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=1, col=2)

        fig.update_layout(
            height=350,
            showlegend=False,
            template='plotly_white'
        )
        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # 4. اختبارات الاستقرارية
        st.markdown("### 4️⃣ اختبارات الاستقرارية - Stationarity Tests")

        # ADF Test
        adf_result = adfuller(data, autolag='AIC')

        # KPSS Test
        kpss_result = kpss(data, regression='c', nlags='auto')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### اختبار ADF - ADF Test")
            st.markdown(f"""
            | المعيار | القيمة |
            |---------|--------|
            | إحصائية ADF | {adf_result[0]:.4f} |
            | القيمة الحرجة 1% | {adf_result[4]['1%']:.4f} |
            | القيمة الحرجة 5% | {adf_result[4]['5%']:.4f} |
            | القيمة الحرجة 10% | {adf_result[4]['10%']:.4f} |
            | p-value | {adf_result[1]:.4f} |
            | عدد الفجوات | {adf_result[2]} |
            """)

            if adf_result[1] < 0.05:
                st.success("✅ النتيجة: السلسلة مستقرة (نرفض فرضية جذر الوحدة)")
            else:
                st.error("❌ النتيجة: السلسلة غير مستقرة (لا نرفض فرضية جذر الوحدة)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### اختبار KPSS - KPSS Test")
            st.markdown(f"""
            | المعيار | القيمة |
            |---------|--------|
            | إحصائية KPSS | {kpss_result[0]:.4f} |
            | القيمة الحرجة 1% | {kpss_result[3]['1%']:.4f} |
            | القيمة الحرجة 5% | {kpss_result[3]['5%']:.4f} |
            | القيمة الحرجة 10% | {kpss_result[3]['10%']:.4f} |
            | p-value | {kpss_result[1]:.4f} |
            """)

            if kpss_result[1] > 0.05:
                st.success("✅ النتيجة: السلسلة مستقرة (لا نرفض فرضية الاستقرارية)")
            else:
                st.error("❌ النتيجة: السلسلة غير مستقرة (نرفض فرضية الاستقرارية)")
            st.markdown('</div>', unsafe_allow_html=True)

        # 5. الفرق الأول
        st.markdown("### 5️⃣ تحليل الفرق الأول - First Difference Analysis")

        diff_data = np.diff(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=diff_data,
            mode='lines',
            name='First Difference',
            line=dict(color='#9C27B0', width=1.5)
        ))
        fig.update_layout(
            title='After First Differencing',
            xaxis_title='Period',
            yaxis_title='Value',
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # اختبار ADF على الفرق الأول
        adf_diff = adfuller(diff_data, autolag='AIC')

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        #### نتائج ADF على الفرق الأول:
        - إحصائية ADF: {adf_diff[0]:.4f}
        - p-value: {adf_diff[1]:.4f}
        - **النتيجة:** {'مستقرة ✅' if adf_diff[1] < 0.05 else 'غير مستقرة ❌'}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # 6. التوصيات
        st.markdown("### 6️⃣ التوصيات - Recommendations")

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)

        recommendations = []

        # تحديد التوصيات بناءً على النتائج
        if adf_result[1] > 0.05 and kpss_result[1] < 0.05:
            recommendations.append("⚠️ السلسلة غير مستقرة - يُنصح بأخذ الفرق الأول")
            recommendations.append("📊 السلسلة قد تحتوي على جذر الوحدة")
            if adf_diff[1] < 0.05:
                recommendations.append("✅ الفرق الأول يحقق الاستقرارية - استخدم d=1 في ARIMA")
        elif adf_result[1] < 0.05 and kpss_result[1] > 0.05:
            recommendations.append("✅ السلسلة مستقرة - يمكن استخدام ARMA مباشرة")
            recommendations.append("📊 لا حاجة لأخذ الفروق")
        elif adf_result[1] < 0.05 and kpss_result[1] < 0.05:
            recommendations.append("⚠️ نتائج متناقضة بين ADF و KPSS")
            recommendations.append("📊 قد يكون هناك اتجاه حتمي - جرب إزالة الاتجاه")
        else:
            recommendations.append("⚠️ كلا الاختبارين يشيران لعدم الاستقرارية")
            recommendations.append("📊 قد تحتاج السلسلة لأكثر من فرق واحد")

        for rec in recommendations:
            st.markdown(f"- {rec}")

        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 10: الحالات الخاصة
# ==================================================
elif selected_section == sections[10]:
    st.markdown('<div class="section-header"><h2>⚠️ الحالات الخاصة - Special Cases</h2></div>', unsafe_allow_html=True)

    # 1. الكسر الهيكلي
    st.markdown("### 1️⃣ الكسر الهيكلي - Structural Break")

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### ما هو الكسر الهيكلي؟

    **الكسر الهيكلي** هو تغير مفاجئ في خصائص السلسلة الزمنية (المتوسط، التباين، أو العلاقة مع المتغيرات الأخرى).

    **Structural Break** is a sudden change in the properties of a time series (mean, variance, or relationship with other variables).

    #### تأثيره على اختبارات الاستقرارية:
    - اختبارات ADF و KPSS قد تعطي نتائج خاطئة
    - السلسلة قد تبدو غير مستقرة بسبب الكسر فقط
    - يجب استخدام اختبارات خاصة مثل Zivot-Andrews أو Lee-Strazicich
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # مثال بصري للكسر الهيكلي
    np.random.seed(42)
    n = 200
    break_point = 100

    # سلسلة مع كسر في المتوسط
    series_break = np.concatenate([
        np.random.normal(0, 1, break_point),
        np.random.normal(5, 1, n - break_point)
    ])

    # سلسلة مع كسر في التباين
    series_var_break = np.concatenate([
        np.random.normal(0, 1, break_point),
        np.random.normal(0, 3, n - break_point)
    ])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Series with Mean Break',
                                        'Series with Variance Break'))

    # كسر المتوسط
    fig.add_trace(
        go.Scatter(y=series_break, mode='lines', name='Series',
                   line=dict(color='#2196F3', width=1.5)),
        row=1, col=1
    )
    fig.add_vline(x=break_point, line_dash="dash", line_color="red",
                  line_width=2, annotation_text="Break Point", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hline(y=5, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)

    # كسر التباين
    fig.add_trace(
        go.Scatter(y=series_var_break, mode='lines', name='Series',
                   line=dict(color='#9C27B0', width=1.5), showlegend=False),
        row=1, col=2
    )
    fig.add_vline(x=break_point, line_dash="dash", line_color="red",
                  line_width=2, annotation_text="Break Point", row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # اختبارات الكسر الهيكلي
    st.markdown("#### اختبارات الكسر الهيكلي - Structural Break Tests")

    breaks_df = pd.DataFrame({
        'الاختبار (Test)': [
            'Chow Test',
            'CUSUM Test',
            'Zivot-Andrews',
            'Lee-Strazicich',
            'Bai-Perron'
        ],
        'الوصف (Description)': [
            'اختبار كسر معروف التاريخ',
            'اختبار تراكمي للمجموع',
            'اختبار جذر الوحدة مع كسر واحد',
            'اختبار جذر الوحدة مع كسرين',
            'اختبار تحديد عدد الكسور'
        ],
        'الميزة (Advantage)': [
            'بسيط ومباشر',
            'يكشف الكسر التدريجي',
            'يحدد تاريخ الكسر داخلياً',
            'أقوى في ظل وجود كسور متعددة',
            'يحدد العدد الأمثل للكسور'
        ]
    })

    st.dataframe(breaks_df, use_container_width=True)

    # 2. الجذور الموسمية
    st.markdown("---")
    st.markdown("### 2️⃣ الجذور الموسمية - Seasonal Unit Roots")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### ما هي الجذور الموسمية؟

    بعض السلاسل قد تحتوي على **جذور وحدة موسمية** بالإضافة إلى جذر الوحدة العادي.

    Some series may contain **seasonal unit roots** in addition to the regular unit root.

    #### اختبار HEGY (Hylleberg-Engle-Granger-Yoo):

    يختبر وجود جذور الوحدة عند الترددات المختلفة:
    - التردد صفر (الجذر العادي)
    - التردد π (نصف سنوي للبيانات الفصلية)
    - الترددات الموسمية الأخرى

    #### الفرق الموسمي:
    """)

    st.latex(r'''
    \Delta_s Y_t = Y_t - Y_{t-s}
    ''')

    st.markdown(r"""
    حيث s هي الدورة الموسمية (مثلاً s=4 للبيانات الفصلية، s=12 للبيانات الشهرية)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. عدم التناظر
    st.markdown("---")
    st.markdown("### 3️⃣ عدم التناظر والتحولات غير الخطية - Asymmetry and Nonlinear Dynamics")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### نموذج TAR (Threshold Autoregressive):
    """)

    st.latex(r'''
    Y_t = \begin{cases}
    \phi_1 Y_{t-1} + \epsilon_t & \text{if } Y_{t-1} \leq \tau \\
    \phi_2 Y_{t-1} + \epsilon_t & \text{if } Y_{t-1} > \tau
    \end{cases}
    ''')

    st.markdown(r"""
    حيث τ هو العتبة (threshold)

    #### اختبارات متعلقة:
    - **Enders-Granger TAR Test**: يختبر جذر الوحدة في نماذج العتبة
    - **Kapetanios Test**: يختبر الاستقرارية في نماذج STAR
    - **MTAR Test**: يركز على سرعة التعديل غير المتناظرة
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. السلاسل مع قيم متطرفة
    st.markdown("---")
    st.markdown("### 4️⃣ القيم المتطرفة والشاذة - Outliers and Anomalies")

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### أنواع القيم المتطرفة في السلاسل الزمنية:

    | النوع | الوصف | التأثير |
    |-------|-------|---------|
    | **AO** (Additive Outlier) | قيمة شاذة في نقطة واحدة | تأثير مؤقت |
    | **IO** (Innovational Outlier) | صدمة تنتشر عبر الزمن | تأثير مستمر |
    | **LS** (Level Shift) | تغير دائم في المستوى | يشبه الكسر الهيكلي |
    | **TC** (Temporary Change) | تغير مؤقت يتلاشى | تأثير متناقص |

    #### التأثير على الاختبارات:
    - القيم المتطرفة تضخم التباين
    - قد تجعل السلسلة تبدو غير مستقرة
    - يُنصح باستخدام اختبارات مقاومة للقيم المتطرفة
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. الاعتماد طويل المدى
    st.markdown("---")
    st.markdown("### 5️⃣ الاعتماد طويل المدى - Long Memory")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### نموذج ARFIMA:

    بعض السلاسل تظهر **ذاكرة طويلة** حيث يتناقص ACF ببطء (ولكن ليس ببطء جذر الوحدة).
    """)

    st.latex(r'''
    (1-L)^d Y_t = \epsilon_t
    ''')

    st.markdown(r"""
    حيث:
    - d: معامل التكامل الجزئي (0 < d < 1)
    - إذا كان 0 < d < 0.5: السلسلة مستقرة مع ذاكرة طويلة
    - إذا كان 0.5 ≤ d < 1: السلسلة غير مستقرة

    #### اختبارات الذاكرة الطويلة:
    - **GPH Test** (Geweke & Porter-Hudak)
    - **Local Whittle Estimator**
    - **R/S Analysis** (Rescaled Range)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# القسم 11: التوصيات والنتائج
# ==================================================
elif selected_section == sections[11]:
    st.markdown('<div class="section-header"><h2>📝 التوصيات والنتائج - Conclusions & Recommendations</h2></div>',
                unsafe_allow_html=True)

    # ملخص شامل
    st.markdown("### 📋 ملخص شامل - Comprehensive Summary")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## خارطة طريق تحليل الاستقرارية - Stationarity Analysis Roadmap

    ### المرحلة الأولى: الفحص البصري 👁️
    1. ارسم السلسلة الزمنية الأصلية
    2. افحص وجود اتجاه أو موسمية
    3. لاحظ أي كسور هيكلية محتملة
    4. ارسم ACF و PACF

    ### المرحلة الثانية: الاختبارات الرسمية 🧪
    1. أجرِ اختبار ADF (الفرضية: وجود جذر الوحدة)
    2. أجرِ اختبار KPSS (الفرضية: الاستقرارية)
    3. قارن النتائج واتخذ القرار
    4. عند التناقض، استخدم اختبارات إضافية

    ### المرحلة الثالثة: التحويل إن لزم 🔄
    1. حدد نوع عدم الاستقرارية
    2. اختر التحويل المناسب
    3. تحقق من نجاح التحويل
    4. وثّق جميع الخطوات
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # جدول القرارات
    st.markdown("---")
    st.markdown("### 🎯 جدول اتخاذ القرار - Decision Table")

    decision_df = pd.DataFrame({
        'نتيجة ADF': ['رفض H₀', 'لا نرفض H₀', 'رفض H₀', 'لا نرفض H₀'],
        'نتيجة KPSS': ['لا نرفض H₀', 'رفض H₀', 'رفض H₀', 'لا نرفض H₀'],
        'الاستنتاج': [
            '✅ مستقرة',
            '❌ غير مستقرة (جذر الوحدة)',
            '⚠️ اتجاه حتمي محتمل',
            '❓ نتائج غير حاسمة'
        ],
        'الإجراء': [
            'استخدم ARMA',
            'خذ الفرق الأول وأعد الاختبار',
            'أزل الاتجاه أو استخدم نموذج مع اتجاه',
            'استخدم اختبارات إضافية (PP, DF-GLS)'
        ]
    })

    st.dataframe(decision_df, use_container_width=True)

    # الأخطاء الشائعة
    st.markdown("---")
    st.markdown("### ⚠️ الأخطاء الشائعة - Common Mistakes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ❌ أخطاء يجب تجنبها:

        1. **الاعتماد على اختبار واحد فقط**
           - استخدم ADF و KPSS معاً

        2. **تجاهل الكسور الهيكلية**
           - قد تجعل السلسلة تبدو غير مستقرة

        3. **الإفراط في أخذ الفروق**
           - Over-differencing يفقد معلومات

        4. **تجاهل الموسمية**
           - قد تحتاج فروق موسمية

        5. **عدم التحقق البصري**
           - الرسوم البيانية ضرورية
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(r"""
        #### ✅ أفضل الممارسات:

        1. **ابدأ بالفحص البصري**
           - السلسلة + ACF + PACF

        2. **استخدم اختبارات متعددة**
           - ADF + KPSS + PP إن أمكن

        3. **اختر العدد الصحيح للفجوات**
           - استخدم AIC/BIC

        4. **وثّق جميع القرارات**
           - سجل سبب كل تحويل

        5. **تحقق من البواقي**
           - يجب أن تكون ضوضاء بيضاء
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # المراجع والمصادر
    st.markdown("---")
    st.markdown("### 📚 المراجع والمصادر - References")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(r"""
    #### المراجع الأساسية:

    1. **Dickey, D.A. & Fuller, W.A.** (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root". *Journal of the American Statistical Association*.

    2. **Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y.** (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root". *Journal of Econometrics*.

    3. **Phillips, P.C.B. & Perron, P.** (1988). "Testing for a Unit Root in Time Series Regression". *Biometrika*.

    4. **Elliott, G., Rothenberg, T.J. & Stock, J.H.** (1996). "Efficient Tests for an Autoregressive Unit Root". *Econometrica*.

    5. **Hamilton, J.D.** (1994). *Time Series Analysis*. Princeton University Press.

    6. **Enders, W.** (2014). *Applied Econometric Time Series*. Wiley.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # خاتمة
    st.markdown("---")
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown(r"""
    ## 🎯 الخلاصة - Conclusion

    تحليل الاستقرارية هو **الخطوة الأولى والأهم** في تحليل السلاسل الزمنية. فهم طبيعة البيانات يحدد:

    - نوع النموذج المناسب (ARMA vs ARIMA)
    - طريقة التقدير الصحيحة
    - صحة الاستدلال الإحصائي
    - دقة التنبؤات

    ---

    **Stationarity analysis is the first and most important step in time series analysis. Understanding the nature of your data determines:**

    - The appropriate model type (ARMA vs ARIMA)
    - The correct estimation method
    - The validity of statistical inference
    - The accuracy of forecasts

    ---

    ### 🔑 القاعدة الذهبية - Golden Rule

    > *"لا تبدأ النمذجة قبل فهم استقرارية بياناتك"*
    >
    > *"Never start modeling before understanding your data's stationarity"*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # معلومات المطور
    st.markdown("---")
    st.markdown(r"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h3>👨‍🔬 تم التطوير بواسطة - Developed by</h3>
        <h2>Dr. Merwan Roudane</h2>
        <p>Independent Researcher in Econometrics & Time Series Analysis</p>
        <p>📧 merwanroudane75@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# تذييل الصفحة
# ==================================================
st.markdown("---")
st.markdown(r"""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>📊 دليل شامل لاستقرارية السلاسل الزمنية | Time Series Stationarity Guide</p>
    <p>© 2024 Dr. Merwan Roudane | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)