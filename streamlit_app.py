import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(page_title="Biobase BK-280 Calibration", layout="wide")
st.title("🧪 حاسبة معاملات المعايرة (Spline) – Biobase BK-280")
st.markdown("أدخل 5 نقاط معيارية. جرّب **النموذج مع تحجيم الامتصاصية** لمطابقة شاشة الجهاز.")

# ----- إدخال البيانات -----
st.subheader("📥 أدخل النقاط المعيارية")
default_data = pd.DataFrame({
    "Concentration": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "Absorbance": [np.nan, np.nan, np.nan, np.nan, np.nan]
})
edited_df = st.data_editor(default_data, num_rows="fixed", use_container_width=True, key="data")
valid_df = edited_df.dropna()
if len(valid_df) < 5:
    st.warning("⚠️ يجب إدخال جميع النقاط الخمس.")
    st.stop()

conc = valid_df["Concentration"].values
absorb = valid_df["Absorbance"].values

if not np.all(np.diff(absorb) > 0):
    st.error("❌ قيم الامتصاصية يجب أن تكون متزايدة بشكل صارم.")
    st.stop()

st.success("✅ البيانات صالحة.")

# ----- اختيار النموذج -----
model_choice = st.selectbox(
    "اختر النموذج الرياضي:",
    [
        "كثير حدود تكعيبي (على الامتصاصية الأصلية)",
        "شرائح تكعيبية طبيعية (Natural Cubic Spline)",
        "كثير حدود تكعيبي بعد Log(Absorbance)",
        "نموذج 4PL (Four-Parameter Logistic)",
        "كثير حدود تكعيبي على Abs×Scale (تجريبي لمطابقة الجهاز)"
    ]
)

# ----- دوال مساعدة -----
def cubic_fit(x, y):
    X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeff

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else float('nan')

# ----- تطبيق النموذج المختار -----
if model_choice == "كثير حدود تكعيبي (على الامتصاصية الأصلية)":
    A, B, C, D = cubic_fit(absorb, conc)
    st.latex(r"\text{Conc} = A + B\cdot\text{Abs} + C\cdot\text{Abs}^2 + D\cdot\text{Abs}^3")
    y_pred = A + B*absorb + C*absorb**2 + D*absorb**3
    r2 = calc_r2(conc, y_pred)
    y_func = lambda x: A + B*x + C*x**2 + D*x**3

elif model_choice == "شرائح تكعيبية طبيعية (Natural Cubic Spline)":
    cs = CubicSpline(absorb, conc, bc_type='natural')
    # عرض المعاملات بشكل آمن
    segments = []
    for i in range(len(cs.x)-1):
        a = float(cs.c[3, i])
        b = float(cs.c[2, i])
        c = float(cs.c[1, i])
        d = float(cs.c[0, i])
        segments.append({
            "القطعة": f"{i+1} (Abs {cs.x[i]:.6f} - {cs.x[i+1]:.6f})",
            "a": a, "b": b, "c": c, "d": d
        })
    seg_df = pd.DataFrame(segments)
    # تنسيق آمن
    def format_seg_df(df):
        return df.style.format({
            "a": "{:.6f}", "b": "{:.6f}", "c": "{:.6f}", "d": "{:.6f}"
        })
    st.dataframe(format_seg_df(seg_df))
    A = B = C = D = None
    r2 = None
    y_func = lambda x: cs(x)
    st.info("يتم عرض أربع معاملات لكل قطعة على حدة.")

elif model_choice == "كثير حدود تكعيبي بعد Log(Absorbance)":
    # معالجة الصفر
    zero_mask = absorb == 0
    if np.any(zero_mask):
        st.warning("⚠️ القيمة صفر لا يمكن استخدامها في اللوغاريتم.")
        handle_zeros = st.radio("اختر طريقة المعالجة:",
                                ["استبعاد نقطة الصفر", "استبدال الصفر بـ 10⁻⁶"],
                                key="log_zero")
        if handle_zeros == "استبعاد نقطة الصفر":
            use_absorb = absorb[~zero_mask]
            use_conc = conc[~zero_mask]
        else:
            use_absorb = absorb.copy()
            use_absorb[zero_mask] = 1e-6
            use_conc = conc
    else:
        use_absorb, use_conc = absorb, conc

    log_abs = np.log(use_absorb)
    A, B, C, D = cubic_fit(log_abs, use_conc)
    st.latex(r"\text{Conc} = A + B\cdot\ln(\text{Abs}) + C\cdot(\ln(\text{Abs}))^2 + D\cdot(\ln(\text{Abs}))^3")
    y_pred = A + B*log_abs + C*log_abs**2 + D*log_abs**3
    r2 = calc_r2(use_conc, y_pred)
    y_func = lambda x: A + B*np.log(x) + C*np.log(x)**2 + D*np.log(x)**3

elif model_choice == "نموذج 4PL (Four-Parameter Logistic)":
    def four_pl(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c)**b)
    try:
        popt, _ = curve_fit(four_pl, absorb, conc,
                            p0=[conc[-1]*1.1, 1, np.median(absorb), conc[0]*0.9],
                            maxfev=10000)
        A, B, C, D = popt
        st.latex(r"\text{Conc} = D + \frac{A-D}{1 + (\text{Abs}/C)^B}")
        y_pred = four_pl(absorb, *popt)
        r2 = calc_r2(conc, y_pred)
        y_func = lambda x: four_pl(x, *popt)
    except Exception as e:
        st.error(f"فشل تركيب 4PL: {e}")
        st.stop()

elif model_choice == "كثير حدود تكعيبي على Abs×Scale (تجريبي لمطابقة الجهاز)":
    scale = st.number_input("عامل التحجيم (عادة 10000 للأجهزة)", value=10000.0, step=1000.0)
    x_scaled = absorb * scale
    A, B, C, D = cubic_fit(x_scaled, conc)
    st.latex(r"\text{Conc} = A + B\cdot(\text{Abs}\times\text{Scale}) + C\cdot(\text{Abs}\times\text{Scale})^2 + D\cdot(\text{Abs}\times\text{Scale})^3")
    st.caption(f"Scale = {scale}")
    y_pred = A + B*x_scaled + C*x_scaled**2 + D*x_scaled**3
    r2 = calc_r2(conc, y_pred)
    y_func = lambda x: A + B*(x*scale) + C*(x*scale)**2 + D*(x*scale)**3

# ----- عرض النتائج (إذا لم تكن Spline) -----
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
if model_choice == "كثير حدود تكعيبي بعد Log(Absorbance)" and x_plot[0] == 0:
    x_plot[0] = 1e-10  # تجنب log(0)
try:
    y_plot = y_func(x_plot)
except:
    y_plot = np.full_like(x_plot, np.nan)

fig = go.Figure()
fig.add_trace(go.Scatter(x=absorb, y=conc, mode='markers', name='النقاط المعيارية',
                         marker=dict(size=10, color='red')))
fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=model_choice))
fig.update_layout(xaxis_title="Absorbance", yaxis_title="Concentration",
                  template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)

# ----- تنبيه خاص للحصول على تطابق الجهاز -----
st.success("""
💡 **للحصول على نفس أرقام جهاز BK-280:**  
1. اختر النموذج: **كثير حدود تكعيبي على Abs×Scale**  
2. استخدم **عامل التحجيم = 10000**  
3. قارن النتائج مع `Par A = -5664.8, Par B = 2796.1, Par C = 313.68, Par D = 1.442`

إذا لم تتطابق تمامًا، جرب زيادة أو تقليل عامل التحجيم قليلًا (مثلاً 9500 أو 10500) حتى تقترب القيم.
""")
