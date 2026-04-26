import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

st.set_page_config(page_title="Biobase BK-280 Spline", layout="wide")
st.title("🧪 حاسبة معاملات المعايرة Spline – جهاز Biobase BK-280")

st.markdown("""
أدخل **5 نقاط معيارية** (التركيز Concentration والامتصاصية Absorbance).  
سيتم حساب `Par A, Par B, Par C, Par D` كما تظهر في شاشة المعايرة.
""")

# ----- إدخال البيانات بواسطة data_editor -----
st.subheader("📥 أدخل النقاط الخمس (استعمل الجدول أدناه)")

# إنشاء DataFrame فارغ مكون من 5 صفوف
default_data = pd.DataFrame({
    "Concentration": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "Absorbance": [np.nan, np.nan, np.nan, np.nan, np.nan]
})

edited_df = st.data_editor(
    default_data,
    num_rows="fixed",
    use_container_width=True,
    key="cal_data"
)

# استخراج القيم وإسقاط الصفوف غير المكتملة
valid_df = edited_df.dropna()
if len(valid_df) < 5:
    st.warning("⚠️ يجب إدخال جميع النقاط الخمس (لا تترك خانات فارغة).")
    st.stop()

conc = valid_df["Concentration"].values
absorb = valid_df["Absorbance"].values

# التأكد من أن قيم الامتصاصية متزايدة (شرط أساسي للسلاين)
if not np.all(np.diff(absorb) > 0):
    st.error("❌ يجب أن تكون قيم الامتصاصية (Absorbance) متزايدة بشكل صارم (لا تكرار ولا قيم متناقصة). رتب القيم من الأصغر إلى الأكبر.")
    st.stop()

st.success("✅ البيانات المدخلة صالحة.")
st.dataframe(valid_df.style.format({"Absorbance": "{:.6f}", "Concentration": "{:.6f}"}))

# ----- النموذج الأول: كثير حدود تكعيبي واحد (Cubic Polynomial) -----
st.header("🔷 النموذج 1: كثير حدود تكعيبي (Cubic Polynomial)")
st.markdown("المعادلة:  \n$$Concentration = A + B \\cdot Abs + C \\cdot Abs^2 + D \\cdot Abs^3$$")

X = np.vstack([np.ones_like(absorb), absorb, absorb**2, absorb**3]).T
coeff, _, _, _ = np.linalg.lstsq(X, conc, rcond=None)
Par_A, Par_B, Par_C, Par_D = coeff

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
r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float('nan')
st.caption(f"معامل التحديد R² = {r2:.4f}")

# ----- النموذج الثاني: شرائح تكعيبية طبيعية (Natural Cubic Spline) -----
st.header("🔶 النموذج 2: شرائح تكعيبية طبيعية (Natural Cubic Spline)")
st.markdown("كل قطعة بين نقطتين:  \n$$C_i = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3$$")

cs = CubicSpline(absorb, conc, bc_type='natural')

segments = []
for i in range(len(cs.x)-1):
    a = cs.c[3, i]  # الثابت
    b = cs.c[2, i]  # الحد الخطي
    c = cs.c[1, i]  # الحد التربيعي
    d = cs.c[0, i]  # الحد التكعيبي
    segments.append({
        "القطعة": f"{i+1} (من {cs.x[i]:.6f} إلى {cs.x[i+1]:.6f})",
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

# ----- الرسم البياني -----
st.header("📈 منحنيات المعايرة")
x_plot = np.linspace(absorb.min(), absorb.max(), 200)
y_poly = Par_A + Par_B * x_plot + Par_C * x_plot**2 + Par_D * x_plot**3
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

st.sidebar.info("""
📌 **ترتيب القيم**: يجب إدخال قيم الامتصاصية من الأصغر إلى الأكبر.  
إذا كانت بيانات جهازك تحتوي على قيم مكررة أو غير مرتبة، قم بترتيبها أولاً.
""")
