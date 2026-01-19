# =========================
# V2 - Ask Any Excel (CEO-ready)
# - Quick Buttons (No AI)  ⚡ سريع جدًا
# - Free Question (AI -> SQL) 🧠 لأي سؤال
# =========================

import os
import re
import io
import time
import hashlib
import pandas as pd
import streamlit as st
import duckdb
import plotly.express as px

# AI (اختياري)
try:
    import google.generativeai as genai
except Exception:
    genai = None
# -------------------------
#بوابة كلمة المرور
# -------------------------
def require_password():
    pwd = ""
    try:
        pwd = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        pass
    if not pwd:
        pwd = os.getenv("APP_PASSWORD", "")

    if not pwd:
        st.error("APP_PASSWORD غير موجودة.")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("🔒 دخول التجربة")
    entered = st.text_input("Password", type="password")
    if st.button("دخول"):
        if entered == pwd:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("كلمة المرور خطأ.")
    st.stop()
# -------------------------
# إعداد الصفحة
# -------------------------
st.set_page_config(page_title="Ask Any Excel - V2 (CEO)", layout="wide")
st.title("🧠 Ask Any Excel - V2 (CEO-ready)")
st.caption("أزرار جاهزة للأسئلة الشائعة + مربع 'أي سؤال' بالذكاء الاصطناعي (اختياري).")

require_password()
# -------------------------
# Helpers: أسرار / مفاتيح
# -------------------------
def get_secret(name: str, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

GEMINI_API_KEY = (get_secret("GEMINI_API_KEY", "") or "").strip()

# -------------------------
# تهيئة Gemini (كاش)
# -------------------------
@st.cache_resource
def get_model(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# -------------------------
# تنظيف أسماء الأعمدة
# -------------------------
def make_safe_columns(cols):
    safe, used = [], set()
    for c in cols:
        c_str = str(c).strip()
        cleaned = re.sub(r"[^\w]+", "_", c_str, flags=re.UNICODE)
        if cleaned == "" or cleaned == "_":
            cleaned = "col"
        if re.match(r"^\d", cleaned):
            cleaned = "col_" + cleaned
        base = cleaned
        i = 2
        while cleaned.lower() in used:
            cleaned = f"{base}_{i}"
            i += 1
        used.add(cleaned.lower())
        safe.append(cleaned)
    return safe

# -------------------------
# SQL Safety Guard
# -------------------------
def is_safe_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    if ";" in s:
        return False
    forbidden = [
        "insert","update","delete","drop","create","alter","truncate",
        "attach","detach","copy","export","import","pragma","call","grant","revoke"
    ]
    if any(w in s for w in forbidden):
        return False
    # نسمح فقط من جدول data
    if " from " in s and " from data" not in s:
        return False
    return True

def ensure_limit(sql: str, limit: int = 200) -> str:
    s = sql.strip()
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        return s
    return s + f" LIMIT {limit}"

# -------------------------
# رسم تلقائي بسيط
# -------------------------
def autoplot(df_result: pd.DataFrame):
    if df_result is None or df_result.empty:
        st.info("لا توجد بيانات للرسم.")
        return

    numeric_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]
    non_numeric_cols = [c for c in df_result.columns if c not in numeric_cols]

    if not numeric_cols or not non_numeric_cols:
        st.info("النتيجة ليست مناسبة للرسم التلقائي (تحتاج عمود فئة + عمود رقم).")
        return

    x = non_numeric_cols[0]
    y = numeric_cols[0]
    plot_df = df_result[[x, y]].dropna().copy()

    st.subheader("📈 رسم تلقائي")
    fig = px.bar(plot_df, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# قراءة Excel بكاش
# -------------------------
@st.cache_data
def read_excel_cached(file_bytes: bytes, sheet_name: str):
    bio = io.BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet_name)

# -------------------------
# تشغيل SQL وإظهار النتائج (دالة واحدة موحدة)
# -------------------------
def run_sql(con, sql: str, show_sql: bool, want_explain: bool, question_for_explain: str, model=None):
    sql = ensure_limit(sql, limit=200)

    if not is_safe_sql(sql):
        st.error("❌ SQL غير آمن أو يخالف القواعد (لازم SELECT فقط ومن جدول data فقط وبدون ;).")
        st.stop()

    if show_sql:
        st.subheader("🧾 SQL")
        st.code(sql, language="sql")

    st.info("⏳ تنفيذ التحليل...")
    try:
        result = con.execute(sql).fetchdf()
    except Exception as e:
        st.error(f"❌ خطأ في تنفيذ SQL: {e}")
        st.stop()

    st.subheader("✅ النتيجة")
    st.dataframe(result, use_container_width=True)
    autoplot(result)

    if want_explain:
        if model is None:
            st.warning("الشرح العربي يحتاج Gemini API Key. (أضفي GEMINI_API_KEY)")
        else:
            st.subheader("🗣️ شرح عربي")
            st.write(explain_arabic(model, question_for_explain, result))

# -------------------------
# شرح عربي (اختياري)
# -------------------------
def explain_arabic(model, question: str, result_df: pd.DataFrame):
    try:
        snippet = result_df.head(10).to_string(index=False)
    except Exception:
        snippet = str(result_df)

    prompt = f"""
اكتب شرح عربي واضح ومختصر لنتيجة التحليل (بدون ذكر SQL).

هيكل الرد:
- الخلاصة (سطر واحد)
- ماذا يعني هذا؟ (سطرين)
- توصية عملية واحدة

سؤال المستخدم:
{question}

النتيجة:
{snippet}
"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return "تم استخراج النتيجة. (تعذر توليد شرح الآن بسبب حدود الطلبات/الاتصال)."

# -------------------------
# AI: توليد SQL من السؤال (مع كاش بسيط)
# -------------------------
@st.cache_data
def ai_prompt_cached(dataset_sig: str, question: str, cols_safe: list, dtypes_preview: str, rows_preview: str):
    prompt = f"""
حوّل سؤال المستخدم إلى SQL آمن (DuckDB).

قواعد إلزامية:
- اكتب SQL فقط (بدون شرح وبدون ```).
- يبدأ بـ SELECT
- بدون ; نهائياً
- الجدول الوحيد: data
- استخدم فقط هذه الأعمدة: {cols_safe}
- إذا تتوقع نتيجة كبيرة استخدم LIMIT (مثلاً 200)

أنواع الأعمدة (مختصر):
{dtypes_preview}

عينة (3 صفوف):
{rows_preview}

السؤال:
{question}

SQL فقط:
"""
    return prompt

def gen_sql_with_ai(model, dataset_sig: str, question: str, cols_safe: list, df: pd.DataFrame):
    rows_preview = df.head(3).to_string(index=False)
    dtypes_preview = {c: str(df[c].dtype) for c in df.columns[: min(len(df.columns), 30)]}
    prompt = ai_prompt_cached(dataset_sig, question, cols_safe, str(dtypes_preview), rows_preview)

    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "TooManyRequests" in msg or "quota" in msg.lower():
            raise RuntimeError("429_QUOTA")
        raise

# =========================
# رفع الملف
# =========================
uploaded = st.file_uploader("ارفع ملف Excel (xlsx)", type=["xlsx"])
if not uploaded:
    st.info("ارفع ملف Excel للمتابعة.")
    st.stop()

file_bytes = uploaded.getvalue()
dataset_sig = hashlib.sha256(file_bytes).hexdigest()

xls = pd.ExcelFile(io.BytesIO(file_bytes))
sheet = st.selectbox("اختر الـ Sheet", xls.sheet_names)

df_raw = read_excel_cached(file_bytes, sheet)

original_cols = list(df_raw.columns)
safe_cols = make_safe_columns(original_cols)

df = df_raw.copy()
df.columns = safe_cols

col_map = pd.DataFrame({
    "Original Column (الاسم الأصلي)": original_cols,
    "Safe Column (الاسم الداخلي)": safe_cols
})

# تصنيف أعمدة
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in numeric_cols]

with st.expander("🔎 قاموس الأعمدة (اضغط للعرض)", expanded=False):
    st.dataframe(col_map, use_container_width=True)

with st.expander("👀 معاينة البيانات (أول 20 صف)", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)

# DuckDB in-memory
con = duckdb.connect(database=":memory:")
con.register("data", df)

# =========================
# Sidebar Settings
# =========================
st.sidebar.subheader("⚙️ إعدادات")
show_sql = st.sidebar.checkbox("اعرض SQL", value=True)
want_explain = st.sidebar.checkbox("🗣️ شرح عربي (أبطأ)", value=False)

ai_enabled = bool(GEMINI_API_KEY) and (genai is not None)
model_name = st.sidebar.selectbox(
    "Model (للأداء اختاري lite)",
    ["models/gemini-flash-lite-latest", "models/gemini-flash-latest", "models/gemini-2.0-flash", "models/gemini-2.5-flash"],
    index=0
)

model = None
if ai_enabled:
    model = get_model(GEMINI_API_KEY, model_name)

# =========================
# ✅ القسم 1: أزرار جاهزة (سريع بدون AI)
# =========================
st.subheader("⚡ أسئلة جاهزة (سريعة بدون AI)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    btn_rows = st.button("عدد الصفوف", use_container_width=True)
with c2:
    btn_preview = st.button("أول 20 صف", use_container_width=True)
with c3:
    btn_top10 = st.button("أعلى 10 (حسب رقم)", use_container_width=True)
with c4:
    btn_dist = st.button("توزيع حسب فئة", use_container_width=True)

# اختيارات للقوالب
colx, coly, coln = st.columns([1,1,1])
with colx:
    group_col = st.selectbox("عمود فئة (Grouping)", options=cat_cols if cat_cols else ["(لا يوجد)"])
with coly:
    measure_col = st.selectbox("عمود رقمي (Measure)", options=numeric_cols if numeric_cols else ["(لا يوجد)"])
with coln:
    n_val = st.number_input("N", min_value=1, max_value=200, value=10, step=1)

# تنفيذ أزرار جاهزة
if btn_rows:
    st.success("✅ قالب سريع: عدد الصفوف")
    run_sql(con, "SELECT COUNT(*) AS rows_count FROM data", show_sql, want_explain, "كم عدد الصفوف؟", model)

if btn_preview:
    st.success("✅ قالب سريع: أول 20 صف")
    run_sql(con, "SELECT * FROM data LIMIT 20", show_sql, want_explain, "اعرض أول 20 صف", model)

if btn_top10:
    if not numeric_cols:
        st.warning("لا يوجد أعمدة رقمية في الملف لعمل 'أعلى N'.")
    else:
        st.success("✅ قالب سريع: أعلى N حسب عمود رقمي")
        sql = f"SELECT * FROM data ORDER BY {measure_col} DESC LIMIT {int(n_val)}"
        run_sql(con, sql, show_sql, want_explain, f"أعلى {int(n_val)} حسب {measure_col}", model)

if btn_dist:
    if not cat_cols:
        st.warning("لا يوجد أعمدة فئوية (نص/تصنيف) لعمل توزيع.")
    else:
        st.success("✅ قالب سريع: توزيع حسب فئة")
        sql = f"""
        SELECT {group_col} AS group_col, COUNT(*) AS count_value
        FROM data
        GROUP BY {group_col}
        ORDER BY count_value DESC
        LIMIT 100
        """.strip()
        run_sql(con, sql, show_sql, want_explain, f"التوزيع حسب {group_col}", model)

# قالب إضافي: متوسط حسب فئة (زر منفصل)
st.markdown("---")
cA, cB = st.columns([1, 3])
with cA:
    btn_avg_by = st.button("متوسط حسب فئة", use_container_width=True)
with cB:
    st.caption("يحتاج عمود فئة + عمود رقمي.")

if btn_avg_by:
    if not cat_cols or not numeric_cols:
        st.warning("يلزم وجود عمود فئة + عمود رقمي.")
    else:
        st.success("✅ قالب سريع: متوسط حسب فئة")
        sql = f"""
        SELECT {group_col} AS group_col, AVG({measure_col}) AS avg_value
        FROM data
        GROUP BY {group_col}
        ORDER BY avg_value DESC
        LIMIT 50
        """.strip()
        run_sql(con, sql, show_sql, want_explain, f"متوسط {measure_col} حسب {group_col}", model)

# =========================
# ✅ القسم 2: أي سؤال (AI)
# =========================
st.markdown("---")
st.subheader("🧠 أي سؤال (AI)")

if not ai_enabled:
    st.info("ميزة 'أي سؤال' تحتاج GEMINI_API_KEY. القوالب السريعة تعمل بدون AI.")
else:
    question = st.text_input("اكتبي سؤالك (عربي/إنجليزي)", value="ما أعلى 10 عناصر حسب أول عمود رقمي؟")

    run_ai = st.button("اسأل الآن (AI)", type="primary")

    if run_ai:
        if not question.strip():
            st.warning("اكتبي سؤال أولاً.")
            st.stop()

        st.info("⏳ AI يولّد SQL...")
        try:
            sql = gen_sql_with_ai(model, dataset_sig, question, safe_cols, df)
        except RuntimeError as e:
            if str(e) == "429_QUOTA":
                st.error("وصلتِ لحد الطلبات (429). عطّلي الشرح، أو استخدمي lite، أو انتظري قليلًا ثم أعيدي.")
                st.stop()
            raise

        run_sql(con, sql, show_sql, want_explain, question, model)

