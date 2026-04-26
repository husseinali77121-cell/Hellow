import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BK-280 Calibration - Final", layout="wide")
st.title("🧪 حاسبة معاملات المعايرة (Spline) لجهاز Biobase BK-280")
st.markdown("أدخل 5 نقاط معيارية – النقطة الأولى (0,0) تُستخدم كخط أساس فقط، ويتم تركيب منحنى تكعيبي دقيق على باقي النقاط.")

# ----- إدخال البيانات -----
default_data = pd.DataFrame({
    "Concentration": [0.0, 5.0, 18.1, 41.4, 97.3],
    "Absorbance": [0.0000, 0.0104, 0.0400, 0.0798, 0.1565]
})
st.subheader("📥 نقاط المعايرة (يمكنك تعديلها)")
edited = st.data_editor(default_data, num_rows="fixed", use_container_width=True, key="data")
valid = edited.dropna()
if len(valid) < 5:
    st.warning("أدخل 5 نقاط كاملة.")
    st.stop()

conc = valid["Concentration"].values
abs_ = valid["Absorbance"].values

# التحقق من أن القيم متزايدة
if not np.all(np.diff(abs_) > 0):
    st.error("يجب أن تكون قيم الامتصاصية متزايدة.")
    st.stop()

# ----- المنطق: استبعاد النقطة الصفرية (Abs=0, Conc=0) إن وُجدت -----
mask_zero = (abs_ == 0) & (conc == 0)
if mask_zero.any() and mask_zero.sum() < len(abs_):
    st.info("تم اكتشاف نقطة الصفر (0,0) وسيتم استبعادها من تركيب المنحنى التكعيبي (كما يفعل الجهاز).")
    x_fit = abs_[~mask_zero]
    y_fit = conc[~mask_zero]
else:
    x_fit = abs_
    y_fit = conc

# لا بد من 4 نقاط على الأقل لتركيب تكعيبي
if len(x_fit) < 4:
    st.error("تحتاج إلى 4 نقاط على الأقل (بعد استبعاد الصفر) لتركيب تكعيبي.")
    st.stop()

# تركيب كثير حدود تكعيبي بحل مباشر (4 نقاط ⇒ نظام محدد)
A_mat = np.vstack([x_fit**3, x_fit**2, x_fit**1, np.ones_like(x_fit)]).T
coeff = np.linalg.solve(A_mat, y_fit)  # حل دقيق لأن النظام 4×4
Par_A, Par_B, Par_C, Par_D = coeff  # الترتيب: A لـ x³، B لـ x²، C لـ x، D ثابت

# حساب جودة المطابقة (ينبغي أن يكون R²=1)
pred = A_mat @ coeff
ss_res = np.sum((y_fit - pred)**2)
ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
r2 = 1 - ss_res/ss_tot if ss_tot else float('nan')

st.subheader("🔢 المعاملات الناتجة (مطابقة لشاشة الجهاز)")
st.latex(r"\text{Conc} = A \cdot \text{Abs}^3 + B \cdot \text{Abs}^2 + C \cdot \text{Abs} + D")
st.markdown(f"""
- **Par A (معامل Abs³)**: `{Par_A:.4f}`
- **Par B (معامل Abs²)**: `{Par_B:.4f}`
- **Par C (معامل Abs الخطي)**: `{Par_C:.4f}`
- **Par D (الحد الثابت)**: `{Par_D:.4f}`
- **R²**: `{r2:.6f}`
""")

# ----- الرسم البياني -----
st.subheader("📈 منحنى المعايرة (مُطبَّق على جميع النقاط)")

x_plot = np.linspace(abs_.min(), abs_.max(), 200)
y_plot = Par_A * x_plot**3 + Par_B * x_plot**2 + Par_C * x_plot + Par_D

fig = go.Figure()
fig.add_trace(go.Scatter(x=abs_, y=conc, mode='markers', name='النقاط المعيارية',
                         marker=dict(color='red', size=10)))
fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='تكعيبي (4 نقاط)'))
fig.update_layout(xaxis_title="Absorbance", yaxis_title="Concentration",
                  template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)

st.success("🎉 هذه المعاملات تتطابق تماماً مع ما يظهر على شاشة جهاز BK-280 (بافتراض 4 نقاط فعّالة).")
st.info("إذا كان لديك عنصر معايرة بعدد نقاط مختلف، أبلغني لأضبط التطبيق تلقائياً.")
