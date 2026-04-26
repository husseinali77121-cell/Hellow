import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

st.set_page_config(page_title="Biobase BK-280 Spline Calibration", layout="wide")
st.title("🧪 حاسبة معاملات المعايرة Spline لجهاز Biobase BK-280")

st.markdown("""
أدخل **5 نقاط معيارية (التركيز - Concentration و الامتصاصية - Absorbance)** ليتم حساب  
`Par A`, `Par B`, `Par C`, `Par D` كما تظهر في شاشة المعايرة.
""")

# ----- إدخال البيانات -----
st.sidebar.header("📥 إدخال النقاط المعيارية")
points = []
for i in range(1, 6):
    col1, col2 = st.sidebar.columns(2)
    conc = col1.number_input(f"التركيز {i}", value=0.0, step=1.0)
    absorb = col2.number_input(f"الامتصاصية {i}", value=0.0, step=0.0001, format="%.4f")
    points.append((conc, absorb))

# تحويل إلى مصفوفتين
conc = np.array([p[0] for p in points])
absorb = np.array([p[1] for p in points])

st.subheader("📊 البيانات المدخلة")
df = pd.DataFrame({"Concentration": conc, "Absorbance": absorb})
st.dataframe(df.style.format({"Absorbance": "{:.6f}"}))

# ----- النموذج الأول: كثير حدود تكعيبي (Cubic Polynomial) -----
st.header("🔷 النموذج 1: كثير حدود تكعيبي (Cubic Polynomial)")
st.markdown("يستخدم هذا النموذج افتراض أن المنحنى هو كثير حدود من الدرجة الثالثة:")

# تجهيز مصفوفة Vandermonde
X = np.vstack([np.ones_like(absorb), absorb, absorb**2, absorb**3]).T
# ملاءمة المربعات الصغرى (إيجاد Par A,B,C,D)
coeff, residuals, rank, s = np.linalg.lstsq(X, conc, rcond=None)
Par_A, Par_B, Par_C, Par_D = coeff

st.latex(r"\text{Concentration} = A + B \cdot \text{Abs} + C \cdot \text{Abs}^2 + D \cdot \text{Abs}^3")
st.markdown(f"""
- **Par A (القطع الثابت)**: `{Par_A:.6f}`  
- **Par B (الحد الخطي)**: `{Par_B:.6f}`  
- **Par C (الحد التربيعي)**: `{Par_C:.6f}`  
- **Par D (الحد التكعيبي)**: `{Par_D:.6f}`
""")

# جودة المطابقة
pred = X @ coeff
ss_res = np.sum((conc - pred)**2)
ss_tot = np.sum((conc - np.mean(conc))**2)
r2 = 1 - ss_res/ss_tot
st.caption(f"معامل التحديد R² = {r2:.4f}")

# ----- النموذج الثاني: شرائح تكعيبية طبيعية (Natural Cubic Spline) -----
st.header("🔶 النموذج 2: شرائح تكعيبية طبيعية (Natural Cubic Spline)")
st.markdown("تفترض هذه الطريقة أن المنحنى مكون من قطع تكعيبية، وتعطي معاملات لكل جزء بين نقطتين.")

# يجب ترتيب البيانات حسب الامتصاصية تصاعديًا للـ spline
sort_idx = np.argsort(absorb)
absorb_sorted = absorb[sort_idx]
conc_sorted = conc[sort_idx]

# إنشاء spline طبيعي
cs = CubicSpline(absorb_sorted, conc_sorted, bc_type='natural')

st.write("**معادلة كل قطعة:**  ")
st.latex(r"C_i = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3")

# عرض المعاملات لكل قطعة
segments = []
for i in range(len(cs.x)-1):
    x_i = cs.x[i]
    a_i = cs.c[3, i] if cs.c.shape[0] > 3 else 0  # CubicSpline.c shape = (4, n-1) للإصدارات الحديثة
    # في scipy، مصفوفة المعاملات cs.c تكون بالشكل:
    # الصفوف: المعاملات لـ (x-x_i)^3 , (x-x_i)^2 , (x-x_i)^1 , الثابت
    # لكن الترتيب قد يختلف حسب الإصدار. سنقرأها بأمان:
    coeff_matrix = cs.c
    # غالباً يكون shape = (4, n-1)، حيث الصف 0 لـ x^3، الصف 1 لـ x^2، الصف 2 لـ x^1، الصف 3 الثابت
    a = coeff_matrix[3][i]  # الثابت (قيمة الدالة عند x_i)
    b = coeff_matrix[2][i]  # معامل الحد الخطي
    c = coeff_matrix[1][i]  # معامل التربيعي
    d = coeff_matrix[0][i]  # معامل التكعيبي

    segments.append({
        "القطعة": f"{i+1} (من {x_i:.6f} إلى {cs.x[i+1]:.6f})",
        "a": a,
        "b": b,
        "c": c,
        "d": d
    })

seg_df = pd.DataFrame(segments)
seg_df.columns = ["القطعة", "Par A (a)", "Par B (b)", "Par C (c)", "Par D (d)"]
st.dataframe(seg_df.style.format({
    "Par A (a)": "{:.6f}",
    "Par B (b)": "{:.6f}",
    "Par C (c)": "{:.6f}",
    "Par D (d)": "{:.6f}"
}))

# ----- رسم بياني توضيحي -----
st.header("📈 الشكل البياني للمعايرة")
import plotly.graph_objects as go

x_plot = np.linspace(absorb.min(), absorb.max(), 200)

# النموذج التكعيبي الكلي
y_poly = Par_A + Par_B * x_plot + Par_C * x_plot**2 + Par_D * x_plot**3

# النموذج الشريحي
y_spline = cs(x_plot)

fig = go.Figure()
fig.add_trace(go.Scatter(x=absorb, y=conc, mode='markers', name='النقاط المعيارية',
                         marker=dict(size=10, color='red')))
fig.add_trace(go.Scatter(x=x_plot, y=y_poly, mode='lines', name='كثير حدود تكعيبي (نموذج 1)',
                         line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=x_plot, y=y_spline, mode='lines', name='شرائح تكعيبية (نموذج 2)',
                         line=dict(dash='dot')))

fig.update_layout(xaxis_title="Absorbance", yaxis_title="Concentration",
                  template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)

# ----- تعليمات النشر -----
st.sidebar.header("🚀 نشر التطبيق")
st.sidebar.markdown("""
1. ارفع الملف `spline_calibrator.py` إلى مستودع GitHub.
2. اذهب إلى [share.streamlit.io](https://share.streamlit.io) واربط المستودع.
3. شغّل التطبيق واستخدمه مباشرة من المتصفح.

المتطلبات في `requirements.txt`:
""")
st.sidebar.code("streamlit\nnumpy\npandas\nscipy\nplotly")

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 إذا كانت معاملات جهازك لا تتطابق مع أحد النموذجين، "
    "فقد يستخدم الجهاز صيغة مختلفة (مثل تحويل لوغاريتمي). "
    "في هذه الحالة يُرجى مراجعة كتيب الجهاز أو تزويدي بالصيغة الدقيقة لأعدل الكود."
)
