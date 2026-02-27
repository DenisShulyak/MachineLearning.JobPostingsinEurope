import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR / "emed_careers_eu.csv"
PROCESSED_FILE = DATA_DIR / "processed_emed_careers_eu_pipeline.csv"


def normalize_token(val) -> str:
    """
    Нормализует текст:
    - NaN -> UNKNOWN
    - убирает лишние пробелы
    - оставляет только буквы/цифры/пробел
    - переводит в UPPER
    """
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return "UNKNOWN"
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9 ]+", "", s)
    s = s.strip().upper()
    return s if s else "UNKNOWN"


def extract_salary_range(text):
    """
    Извлекает информацию о зарплате из текстового описания salary_offered:
    - минимальная / максимальная числовая зарплата
    - флаг 'конкурентной' (competitive / negotiable / not disclosed ...)
    - флаг, найдено ли вообще числовое значение
    """
    if pd.isna(text):
        return (np.nan, np.nan, 0, 0)
    s = str(text).lower().strip()

    competitive_words = ["competitive", "not disclosed", "negotiable", "depending", "doe", "tbd"]
    is_competitive = int(any(w in s for w in competitive_words))

    # 50k -> 50000
    s = re.sub(r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000), s)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)

    nums = re.findall(r"\d[\d,]*", s)
    nums = [n.replace(",", "") for n in nums if n.replace(",", "").isdigit()]
    nums = [int(n) for n in nums]

    if len(nums) >= 2:
        return (nums[0], nums[1], is_competitive, 1)
    if len(nums) == 1:
        return (nums[0], np.nan, is_competitive, 1)
    return (np.nan, np.nan, is_competitive, 0)


def top_n_other(series: pd.Series, n: int = 80) -> pd.Series:
    """
    Оставляет только n самых частых значений, остальные -> OTHER.
    """
    series = series.fillna("UNKNOWN").astype("string")
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), other="OTHER")


def build_pipeline(input_path: Path = RAW_FILE, output_path: Path = PROCESSED_FILE) -> pd.DataFrame:
    print(f"Загружаем исходный файл: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Размер исходного датасета: {df.shape}")

    # Базовая очистка и проверка
    print("Пропуски по столбцам:\n", df.isnull().sum())
    print("Дубликаты строк:", df.duplicated().sum())

    # Удаляем дубликаты
    df = df.drop_duplicates()

    # Стрип пробелов для строковых колонок
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").str.strip()

    # Нормализация company_name и location
    if "company_name" in df.columns:
        df["company_name"] = df["company_name"].apply(normalize_token)
    if "location" in df.columns:
        df["location"] = df["location"].apply(normalize_token)
        df["location"] = df["location"].replace({"UK2": "UK", "U K": "UK"})

    # Обработка дат публикации
    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
        df["post_year"] = df["post_date"].dt.year
        df["post_month"] = df["post_date"].dt.month
        df["post_day"] = df["post_date"].dt.day
        df["post_dayofweek"] = df["post_date"].dt.dayofweek

    # Анализ описаний вакансий
    if "job_description" in df.columns:
        s = df["job_description"].fillna("").astype("string")
        df["desc_len"] = s.str.len()
        df["desc_word_count"] = s.str.split().str.len()

        patterns = {
            "kw_remote": r"\bremote\b|\bwork from home\b|\bwfh\b|\bhybrid\b",
            "kw_bonus": r"\bbonus\b|\bcommission\b|\bperformance\b",
            "kw_senior": r"\bsenior\b|\blead\b|\bstaff\b|\bprincipal\b",
            "kw_junior": r"\bjunior\b|\bentry\b|\bgraduate\b|\bintern\b",
            "kw_manager": r"\bmanager\b|\bhead of\b|\bdirector\b",
            "kw_english": r"\benglish\b",
            "kw_german": r"\bgerman\b|\bdeutsch\b",
            "kw_french": r"\bfrench\b|\bfran[cç]ais\b",
            "kw_visa": r"\bvisa\b|\brelocation\b|\bsponsorship\b",
            "kw_python": r"\bpython\b",
            "kw_sql": r"\bsql\b",
            "kw_ml": r"\bmachine learning\b|\bdeep learning\b|\bartificial intelligence\b|\b ai\b",
            "kw_phd": r"\bphd\b|\bdoctorate\b",
        }

        for feat, pat in patterns.items():
            df[feat] = s.str.contains(pat, case=False, regex=True, na=False).astype(int)

    # Анализ названий должностей
    if "job_title" in df.columns:
        t = df["job_title"].fillna("").astype("string")
        df["title_len"] = t.str.len()
        df["title_word_count"] = t.str.split().str.len()

        df["title_has_senior"] = t.str.contains(
            r"\bsenior\b|\blead\b|\bprincipal\b|\bstaff\b",
            case=False,
            regex=True,
            na=False,
        ).astype(int)
        df["title_has_junior"] = t.str.contains(
            r"\bjunior\b|\bintern\b|\bgraduate\b|\bentry\b",
            case=False,
            regex=True,
            na=False,
        ).astype(int)
        df["title_has_manager"] = t.str.contains(
            r"\bmanager\b|\bdirector\b|\bhead\b",
            case=False,
            regex=True,
            na=False,
        ).astype(int)

    # Извлечение зарплат
    if "salary_offered" in df.columns:
        df["salary_provided"] = df["salary_offered"].notna().astype(int)
        ext = df["salary_offered"].apply(extract_salary_range)
        df["salary_min"] = ext.apply(lambda x: x[0])
        df["salary_max"] = ext.apply(lambda x: x[1])
        df["salary_is_competitive"] = ext.apply(lambda x: x[2])
        df["salary_numeric_found"] = ext.apply(lambda x: x[3])
        df["salary_avg"] = df[["salary_min", "salary_max"]].mean(axis=1)

    # Кодирование категориальных признаков
    TOP_N_COMPANIES = 80
    TOP_N_LOCATIONS = 80

    cat_cols = []
    for col in ["category", "job_type"]:
        if col in df.columns:
            cat_cols.append(col)

    if "company_name" in df.columns:
        df["company_name"] = top_n_other(df["company_name"], n=TOP_N_COMPANIES)
        cat_cols.append("company_name")

    if "location" in df.columns:
        df["location"] = top_n_other(df["location"], n=TOP_N_LOCATIONS)
        cat_cols.append("location")

    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Удаляем исходные текстовые поля, которые уже закодированы / разобраны в фичи
    columns_to_drop = [
        c
        for c in ["post_date", "job_description", "job_title", "salary_offered"]
        if c in df_processed.columns
    ]
    df_processed = df_processed.drop(columns=columns_to_drop, errors="ignore")

    # Приведение bool -> int
    bool_cols = df_processed.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_processed[bool_cols] = df_processed[bool_cols].astype(int)

    # Очищаем salary_avg от экстремальных выбросов по квантилям
    if "salary_avg" in df_processed.columns:
        q_low, q_high = df_processed["salary_avg"].quantile([0.01, 0.99])
        mask_ok = (
            df_processed["salary_avg"].isna()
            | ((df_processed["salary_avg"] >= q_low) & (df_processed["salary_avg"] <= q_high))
        )
        df_processed = df_processed[mask_ok].copy()

        # Восстановление пропусков salary_avg по группам, используя исходный df
        def get_level(row):
            if row.get("title_has_senior", 0) == 1:
                return "senior"
            if row.get("title_has_junior", 0) == 1:
                return "junior"
            if row.get("title_has_manager", 0) == 1:
                return "manager"
            return "other"

        df_processed["title_level"] = df_processed.apply(get_level, axis=1)

        # Используем исходные строки df по индексам df_processed
        base = df.loc[df_processed.index]
        df_processed["loc_group"] = base["location"].fillna("UNKNOWN").astype("string")
        df_processed["cat_group"] = base["category"].fillna("UNKNOWN").astype("string")

        has_salary = df_processed.get("salary_numeric_found", pd.Series(0, index=df_processed.index)) == 1

        group_cols = ["title_level", "loc_group", "cat_group"]
        group_median = (
            df_processed[has_salary]
            .groupby(group_cols)["salary_avg"]
            .median()
            .reset_index()
            .rename(columns={"salary_avg": "group_median"})
        )

        df_processed = df_processed.merge(group_median, on=group_cols, how="left")

        before_nan = df_processed["salary_avg"].isna().sum()

        # 1) сначала заполняем медианой по группе
        df_processed["salary_avg"] = df_processed["salary_avg"].fillna(df_processed["group_median"])

        # 2) затем оставшиеся NaN — глобальной медианой по всем строкам с числовой зарплатой
        global_median = df_processed.loc[df_processed["salary_numeric_found"] == 1, "salary_avg"].median()
        df_processed["salary_avg"] = df_processed["salary_avg"].fillna(global_median)

        after_nan = df_processed["salary_avg"].isna().sum()
        print("Пропусков в salary_avg до заполнения:", before_nan)
        print("Пропусков в salary_avg после заполнения:", after_nan)

        df_processed = df_processed.drop(columns=["group_median", "loc_group", "cat_group"])

    # В конце — удаляем вспомогательные salary_min / salary_max, оставляя целевую salary_avg
    for col in ["salary_min", "salary_max"]:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    print("Итоговый размер df_processed:", df_processed.shape)
    print("Типы данных:\n", df_processed.dtypes.value_counts())

    print(f"Сохраняем обработанный датасет в: {output_path}")

    # Финальная валидация salary_avg
    # Финальная фильтрация нереалистичных зарплат
    if "salary_avg" in df_processed.columns:

        MIN_REASONABLE = 10000

        before_rows = df_processed.shape[0]

        df_processed = df_processed[
            df_processed["salary_avg"].isna() |
            (df_processed["salary_avg"] >= MIN_REASONABLE)
        ].copy()

        after_rows = df_processed.shape[0]

        print("Удалено строк с salary_avg < 10000:", before_rows - after_rows)

        return df_processed

    df_processed.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    build_pipeline()

