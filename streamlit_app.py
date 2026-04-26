import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(page_title="Biobase BK-280 Calibration", layout="wide")
st.title("🧪 حاسبة معاملات المعايرة (Spline) لجهاز Biobase BK-280")

st.markdown("أدخل **5 نقاط معيارية** (التركيز و الامتصاصية)، واختر النموذج المناسب لمطابقة شاشة الجهاز.")

# ----- إدخال البيانات -----
st.subheader("📥 أدخل النقاط الخمس")
default_data = pd.DataFrame({
    "Concentration": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "Absorbance": [np.nan, np.nan, np.nan, np.nan, np.nan]
})
edited_df = st.data_editor(default_data, num_rows="fixed", use_container_width=True, key="data_input")
valid_df = edited_df.dropna()
if len(valid_df) < 5:
    st.warning("⚠️ يجب إدخال جميع النقاط الخمس.")
    st.stop()

conc = valid_df["Concentration"].values
absorb = valid_df["Absorbance"].values

if not np.all(np.diff(absorb) > 0):
    st.error("❌ يجب أن تكون قيم الامتصاصية (Absorbance) متزايدة بشكل صارم (لا تكرار ولا قيم متناقصة).")
    st.stop()

st.success("✅ البيانات المدخلة صالحة.")

# عرض جدول البيانات المدخلة
st.dataframe(valid_df.style.format({"Absorbance": "{:.6f}", "Concentration": "{:.6f}"}))

# ----- اختيار النموذج -----
model_choice = st.selectbox(
    "اختر النموذج الرياضي:",
    [
        "كثير حدود تكعيبي (Cubic Polynomial)",
        "شرائح تكعيبية طبيعية (Natural Cubic Spline)",
        "كثير حدود تكعيبي بعد Log(Absorbance)",
        "نموذج 4PL (Four-Parameter Logistic)"
    ]
)

# ----- دوال النماذج -----
def cubic_poly(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def cubic_log_abs(x, a, b, c, d):
    # x يجب أن تكون موجبة
    return a + b*np.log(x) + c*np.log(x)**2 + d*np.log(x)**3

def four_pl(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# ----- حساب المعاملات حسب النموذج -----
if model_choice == "كثير حدود تكعيبي (Cubic Polynomial)":
    X = np.vstack([np.ones_like(absorb), absorb, absorb**2, absorb**3]).T
    coeff, _, _, _ = np.linalg.lstsq(X, conc, rcond=None)
    A, B, C, D = coeff
    st.latex(r"\text{Conc} = A + B\cdot\text{Abs} + C\cdot\text{Abs}^2 + D\cdot\text{Abs}^3")
    r2 = 1 - np.sum((conc - X@coeff)**2) / np.sum((conc - np.mean(conc))**2)
    y_pred_func = lambda x: A + B*x + C*x**2 + D*x**3

elif model_choice == "شرائح تكعيبية طبيعية (Natural Cubic Spline)":
    cs = CubicSpline(absorb, conc, bc_type='natural')
    segments = []
    for i in range(len(cs.x)-1):
        a = cs.c[3, i]; b = cs.c[2, i]; c = cs.c[1, i]; d = cs.c[0, i]
        segments.append({
            "القطعة": f"{i+1} (Abs {cs.x[i]:.6f} - {cs.x[i+1]:.6f})",
            "a": a, "b": b, "c": c, "d": d
        })
    seg_df = pd.DataFrame(segments)
    seg_df.columns = ["القطعة", "Par A (a)", "Par B (b)", "Par C (c)", "Par D (d)"]
    st.dataframe(seg_df.style.format("{:.6f}"))
    y_pred_func = lambda x: cs(x)
    r2 = None
    st.info("يتم عرض أربع معاملات لكل قطعة على حدة.")

elif model_choice == "كثير حدود تكعيبي بعد Log(Absorbance)":
    # معالجة نقاط الصفر: إما استبعادها أو استبدالها بقيمة صغيرة
    zero_mask = absorb == 0
    if np.any(zero_mask):
        st.warning("⚠️ النقطة ذات الامتصاصية = 0 لا يمكن استخدامها في نموذج Log لأنه log(0) غير معرّف.")
        handle_zeros = st.radio(
            "كيف تريد معالجة نقطة الصفر؟",
            ["استبعاد نقطة الصفر من الحساب", "استبدال الصفر بـ 10⁻⁶ (0.000001)"],
            key="handle_zeros"
        )
        if handle_zeros == "استبعاد نقطة الصفر من الحساب":
            absorb_clean = absorb[~zero_mask]
            conc_clean = conc[~zero_mask]
            if len(absorb_clean) < 4:
                st.error("عدد النقاط المتبقية غير كافٍ (تحتاج 4 على الأقل).")
                st.stop()
        else:
            absorb_clean = absorb.copy()
            absorb_clean[zero_mask] = 1e-6
            conc_clean = conc
            st.caption("تم استبدال الصفر بـ 0.000001 مؤقتاً.")
    else:
        absorb_clean = absorb
        conc_clean = conc

    log_abs = np.log(absorb_clean)
    X = np.vstack([np.ones_like(log_abs), log_abs, log_abs**2, log_abs**3]).T
    coeff, _, _, _ = np.linalg.lstsq(X, conc_clean, rcond=None)
    A, B, C, D = coeff
    st.latex(r"\text{Conc} = A + B\cdot\ln(\text{Abs}) + C\cdot(\ln(\text{Abs}))^2 + D\cdot(\ln(\text{Abs}))^3")
    pred = X @ coeff
    r2 = 1 - np.sum((conc_clean - pred)**2) / np.sum((conc_clean - np.mean(conc_clean))**2)
    y_pred_func = lambda x: A + B*np.log(x) + C*np.log(x)**2 + D*np.log(x)**3

elif model_choice == "نموذج 4PL (Four-Parameter Logistic)":
    a_init = conc[-1] * 1.1
    d_init = conc[0] * 0.9
    b_init = 1.0
    c_init = np.median(absorb)
    try:
        popt, _ = curve_fit(four_pl, absorb, conc,
                            p0=[a_init, b_init, c_init, d_init],
                            maxfev=10000)
        A, B, C, D = popt
        st.latex(r"\text{Conc} = D + \frac{A-D}{1 + (Abs/C)^B}")
        pred = four_pl(absorb, *popt)
        r2 = 1 - np.sum((conc - pred)**2) / np.sum((conc - np.mean(conc))**2)
        y_pred_func = lambda x: four_pl(x, *popt)
    except Exception as e:
        st.error(f"فشل تركيب 4PL: {e}")
        st.stop()

# ----- عرض المعاملات (إذا لم تكن Spline) -----
if model_choice != "شرائح تكعيبية طبيعية (Natural Cubic Spline)":
    st.markdown(f"""
    - **Par A**: `{A:.6f}`  
    - **Par B**: `{B:.6f}`  
    - **Par C**: `{C:.6f}`  
    - **Par D**: `{D:.6f}`
    """)
    if r2 is not None:
        st.caption(f"معامل التحديد R² = {r2:.6f}")

# ----- الرسم البياني -----
st.header("📈 منحنى المعايرة")
x_plot = np.linspace(absorb.min(), absorb.max(), 200)
# تجنب log(0) أثناء الرسم
if model_choice == "كثير حدود تكعيبي بعد Log(Absorbance)" and np.any(x_plot == 0):
    x_plot[0] = 1e-10  # قيمة مهملة لتجنب الخطأ

try:
    y_plot = y_pred_func(x_plot)
except:
    y_plot = np.full_like(x_plot, np.nan)

fig = go.Figure()
fig.add_trace(go.Scatter(x=absorb, y=conc, mode='markers', name='النقاط المعيارية',
                         marker=dict(size=10, color='red')))
fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'النموذج: {model_choice}'))
fig.update_layout(xaxis_title="Absorbance", yaxis_title="Concentration",
                  template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("""
📌 **للحصول على نفس معاملات جهاز BK-280:**  
- يبدو أن الجهاز يستخدم تحويلاً لوغاريتمياً أو نموذج 4PL.  
- جرّب **'كثير حدود تكعيبي بعد Log(Absorbance)'** أولاً.  
- عند وجود صفر، اختر استبعاده أو استبداله بقيمة صغيرة.  
""")
