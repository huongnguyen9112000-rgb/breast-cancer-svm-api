from pathlib import Path
import json
import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# 1. Nạp dữ liệu Breast Cancer Wisconsin
bundle = load_breast_cancer(as_frame=True)
X = bundle.data
y = bundle.target

# 2. Chia tập dữ liệu (80% train, 20% test, giữ nguyên tỷ lệ lớp với stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. Xây dựng Pipeline chuẩn hóa và SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

# 4. Không gian siêu tham số cần tìm kiếm
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1]
}

# 5. Huấn luyện với Cross-Validation (GridSearchCV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)
search.fit(X_train, y_train)
model = search.best_estimator_

print("Best params:", search.best_params_)

# 6. Lưu artifact và metadata vào thư mục artifacts/
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(model, ARTIFACT_DIR / "breast_cancer_svm.joblib")

metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "feature_names": list(X.columns),
    "class_mapping": {"0": "malignant", "1": "benign"},
    "best_params": search.best_params_,
    "sklearn_version": sklearn.__version__,
    "warning": "Educational use only; not a medical diagnosis."
}

(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("Đã huấn luyện và lưu artifact thành công!")
from pathlib import Path
import json
import joblib

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Lưu mô hình đã huấn luyện
joblib.dump(model, ARTIFACT_DIR / "breast_cancer_svm.joblib")

# Lưu metadata
metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "feature_names": list(X.columns),
    "class_mapping": {"0": "malignant", "1": "benign"},
    "warning": "Educational use only; not a medical diagnosis."
}
(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
