import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import Counter


class PatientModel:
    def __init__(self):
        self.label_encoders = {}
        self.features = [
            "SECTION",
            "HSEX",
            "HFINACL",
            "PRIORITY",
            "BEDDEGREE",
            "ISOLATE_YN",
            "TRAFFIC_YN",
            "DISEASE_YN",
            "DISEASE_TYPE",
            "INLABOR_YN",
            "RSV_TYPE",
            "TURNOUT_FG",
            "ADM_DAYS",
            "GM_YN",
            "COVID19_TYPE",
        ]

        self.clf = xgb.XGBClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=30,
            learning_rate=0.1,
            subsample=0.8,
            tree_method="hist",
            enable_categorical=True,  # 支持分類特徵
            use_label_encoder=False,  # 新版 XGBoost 默认关闭
            max_bin=256,  # 限制直方图分箱数以支持量化
        )

        self.is_model_loaded = False
        self.training_data = None
        self.prediction_data = None
        self.X_pred = None  # 預測資料的特徵矩陣
        self.probability_matrix = None
        self.target_encoder = None
        self.num_patients = None
        self.patient_ids = None

    def load_patient_data(self, data_path, is_training=True):
        try:
            df = pd.read_csv(data_path)

            # 檢查必要的欄位
            required_columns = self.features.copy()
            if is_training:
                required_columns += ["RSV_STATION", "RSV_BED"]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"CSV檔案缺少以下必要欄位：{', '.join(missing_columns)}"

            if is_training:
                self.training_data = df
                return True, "訓練資料載入成功！"
            else:
                self.prediction_data = df
                # 如果模型已載入，立即處理預測資料
                if self.is_model_loaded:
                    self.preprocess_prediction_data()
                return True, "預測資料載入成功！"

        except Exception as e:
            return False, f"載入資料時發生錯誤：{str(e)}"

    def preprocess_prediction_data(self):
        """處理預測資料，將其轉換成模型可預測的 X_pred"""
        if self.prediction_data is None or not self.is_model_loaded:
            return False

        X_pred = pd.DataFrame()
        for feature in self.features:
            if feature not in self.prediction_data.columns:
                continue

            le = self.label_encoders.get(feature)
            if le is not None:
                # 先將 NaN 改成 'MISSING'
                self.prediction_data[feature] = self.prediction_data[feature].fillna(
                    "MISSING"
                )
                col_vals = self.prediction_data[feature].astype(str)
                # 如果出現未知類別，就用 encoder 裏的第一個類別替代
                col_vals = col_vals.apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_pred[feature] = le.transform(col_vals)

        self.X_pred = X_pred
        self.precompute_probabilities()
        return True

    def precompute_probabilities(self):
        """預先計算所有病人的機率矩陣 (用於快速查詢)"""
        if not self.is_model_loaded or self.X_pred is None:
            return False
        print("正在計算機率矩陣...")
        self.probability_matrix = self.predict_parallel(self.X_pred)
        print("機率矩陣計算完成！")
        return True

    def load_model(self, model_file):
        try:
            model_data = joblib.load(model_file)

            self.clf = model_data["classifier"]
            self.label_encoders = model_data["label_encoders"]
            self.target_encoder = model_data["target_encoder"]
            self.features = model_data["features"]

            self.is_model_loaded = True

            # 如果已有預測資料，立即處理
            if self.prediction_data is not None:
                self.preprocess_prediction_data()

            return True

        except Exception as e:
            print(f"載入模型時發生錯誤：{str(e)}")
            return False

    def get_bed_probabilities(self, bed_index):
        """快速獲取特定床位的所有病人機率 (若有 precompute_probabilities)"""
        if not self.is_model_loaded or self.probability_matrix is None:
            return None
        class_index = bed_index % self.probability_matrix.shape[1]
        return self.probability_matrix[:, class_index]

    def predict_parallel(self, X, n_jobs=4):
        """使用並行處理進行預測 (predict_proba)"""
        if not self.is_model_loaded:
            return None
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            batches = np.array_split(X, n_jobs)
            results = list(executor.map(self.predict_batch, batches))
        return np.vstack(results)

    def predict_batch(self, patient_features_batch):
        """批次預測多個病人的機率"""
        if not self.is_model_loaded:
            return None
        return self.clf.predict_proba(patient_features_batch)

    def _encode_data(self, df, fit=False):
        """
        將 DataFrame df 裡的 features & TARGET 轉成數字編碼。
        - fit=True 時，會重新 fit LabelEncoder (訓練階段)
        - fit=False 時，只做 transform (測試/驗證階段)
        回傳: (X_encoded, y_encoded)
        """
        X_encoded = pd.DataFrame()

        if fit:
            # 重新初始化 label_encoders
            self.label_encoders = {}

        # 特徵編碼
        for feature in self.features:
            df[feature] = df[feature].astype(str)
            if fit:
                le = LabelEncoder()
                X_encoded[feature] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
            else:
                le = self.label_encoders.get(feature)
                if le is None:
                    raise ValueError(f"尚未對特徵 {feature} 做 fit，就無法 transform。")
                # 處理未知類別
                col_vals = df[feature].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_encoded[feature] = le.transform(col_vals)

        # 目標編碼
        if fit:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(df["TARGET"])
        else:
            if not self.target_encoder:
                raise ValueError("尚未對 TARGET 做 fit，就無法 transform。")
            valid_classes = set(self.target_encoder.classes_)
            temp_target = df["TARGET"].apply(
                lambda x: x if x in valid_classes else list(valid_classes)[0]
            )
            y_encoded = self.target_encoder.transform(temp_target)

        return X_encoded, y_encoded

    def train(self, data_path):
        """
        讀取 data_path CSV -> 先把所有 NaN => 'MISSING' ->
        -> 強制每個 TARGET 至少有 1 筆進訓練集 -> 分割 train/test -> 編碼 -> 訓練
        -> 回傳 (train_score, test_score)
        """
        # 1) 讀取資料
        df = pd.read_csv(data_path)

        # 2) 先把所有 NaN => "MISSING"
        df = df.fillna("MISSING")

        # 3) 建立 TARGET = RSV_STATION + "_" + df["RSV_BED"]
        df["TARGET"] = df["RSV_STATION"] + "_" + df["RSV_BED"].astype(str)

        # 4) 強制每個類別至少 1 筆樣本進訓練集
        forced_samples = []
        for target_val, group_df in df.groupby("TARGET"):
            # 隨機抽 1 筆(或想要更多可以調整 n=2,3...)
            forced_samples.append(group_df.sample(n=1, random_state=42))
        forced_df = pd.concat(forced_samples)
        forced_indices = forced_df.index

        # 將 forced 的 row 從 df 移除，剩下的才做後續分層切分
        remaining_df = df.drop(index=forced_indices).reset_index(drop=True)

        # 進一步檢查剩餘資料中是否有類別樣本數不足 2 的情況
        remaining_counts = remaining_df["TARGET"].value_counts()
        invalid_classes = remaining_counts[remaining_counts < 2].index

        # 將這些類別樣本全部加入到訓練集
        additional_forced_df = remaining_df[
            remaining_df["TARGET"].isin(invalid_classes)
        ]
        forced_df = pd.concat([forced_df, additional_forced_df], axis=0)

        # 更新剩餘資料，移除這些已加入的類別
        remaining_df = remaining_df[
            ~remaining_df["TARGET"].isin(invalid_classes)
        ].reset_index(drop=True)

        X_train_rem, X_test, y_train_rem, y_test = train_test_split(
            remaining_df[self.features],
            remaining_df["TARGET"],
            test_size=0.2,
            stratify=remaining_df["TARGET"],
            random_state=42,
        )

        # 5) 最終訓練集是 forced_df + train_rem
        train_rem_df = pd.concat([X_train_rem, y_train_rem], axis=1)
        final_train_df = pd.concat([train_rem_df, forced_df], axis=0).reset_index(
            drop=True
        )

        # 6) 若有 HHISNUM / MASKED_HHISNUM，就記錄一下
        if "MASKED_HHISNUM" in final_train_df.columns:
            self.patient_ids = final_train_df["MASKED_HHISNUM"]
        elif "HHISNUM" in final_train_df.columns:
            self.patient_ids = final_train_df["HHISNUM"]
        else:
            self.patient_ids = None

        # 7) 分別對 train/test 做編碼 (train做fit, test只做transform)
        X_train, y_train = self._encode_data(final_train_df, fit=True)
        X_test, y_test = self._encode_data(
            pd.concat([X_test, y_test], axis=1), fit=False
        )

        # 保存資訊
        self.num_patients = len(final_train_df)
        self.training_data = final_train_df  # 保留訓練資料 (原始)

        # 8) 訓練模型
        self.clf.fit(X_train, y_train)

        # 9) 訓練/測試集分數
        train_score = self.clf.score(X_train, y_train)
        test_score = self.clf.score(X_test, y_test)

        # 10) 訓練完成後保存模型
        saved_path = self.save_model()

        return train_score, test_score

    def predict_bed(self, patient_features):
        """
        給單筆病人的 feature dict，輸出 Top N 個最可能的 (病房_床號, probability)。
        """
        if not self.is_model_loaded:
            raise ValueError("模型尚未載入或訓練，無法進行預測。")

        # 對輸入特徵進行編碼
        encoded_features = {}
        for feature in self.features:
            if feature not in patient_features:
                val = "MISSING"
            else:
                val = (
                    str(patient_features[feature])
                    if pd.notna(patient_features[feature])
                    else "MISSING"
                )

            le = self.label_encoders.get(feature)
            if le is not None:
                if val not in le.classes_:
                    val = le.classes_[0]  # 未知類別時，退回到已知的第一個
                encoded_features[feature] = le.transform([val])[0]
            else:
                raise ValueError(f"尚未對特徵 {feature} 進行編碼，無法預測。")

        # 建立特徵 DataFrame
        X = pd.DataFrame([encoded_features])

        # 預測概率
        probs = self.clf.predict_proba(X)[0]

        # 選出前 N 個最可能的床位
        top_n = 10
        top_indices = np.argsort(probs)[-top_n:][::-1]

        predictions = []
        for idx in top_indices:
            bed_str = self.target_encoder.inverse_transform([idx])[0]
            probability = probs[idx]
            predictions.append((bed_str, probability))

        return predictions

    def save_model(self):
        """將整個模型（含編碼器、特徵資訊等）保存到 joblib 檔案。"""
        os.makedirs("model", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f"model/patient_model_{timestamp}.joblib"

        model_data = {
            "classifier": self.clf,
            "label_encoders": self.label_encoders,
            "target_encoder": self.target_encoder,
            "features": self.features,
            "X": getattr(self, "X", None),
            "num_patients": self.num_patients,
            "patient_ids": self.patient_ids,
            "training_data": self.training_data,  # 保存訓練資料
        }

        joblib.dump(model_data, model_path)
        return model_path
