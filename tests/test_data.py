import numpy as np
import pytest

import src.data as data
from src.config import CLASS_LABELS, TARGET
from src.data import file_sha256, split_data, validate_data


def test_valid_data_passes(sample_df):
    validate_data(sample_df)


def test_missing_column_is_rejected(sample_df):
    with pytest.raises(ValueError, match="Missing"):
        validate_data(sample_df.drop(columns="Soil_Moisture"))


def test_missing_values_are_rejected(sample_df):
    sample_df.loc[0, "Temperature_C"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        validate_data(sample_df)


def test_text_in_numeric_column_is_rejected(sample_df):
    sample_df["Humidity"] = sample_df["Humidity"].astype(str)
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_data(sample_df)


def test_unknown_label_is_rejected(sample_df):
    sample_df.loc[0, TARGET] = "Very High"
    with pytest.raises(ValueError, match="Unknown target labels"):
        validate_data(sample_df)


def test_split_is_stratified_and_disjoint(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df)

    assert len(X_train) == 240 and len(X_test) == 60
    assert set(X_train.index).isdisjoint(X_test.index)
    train_share = y_train.value_counts(normalize=True)
    test_share = y_test.value_counts(normalize=True)
    for label in CLASS_LABELS:
        assert abs(train_share[label] - test_share[label]) < 0.02


def test_split_is_reproducible(sample_df):
    first = split_data(sample_df)[1].index
    second = split_data(sample_df)[1].index
    assert first.equals(second)


def test_file_sha256(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(path) == expected


def test_prepare_data_rejects_wrong_checksum(monkeypatch, sample_csv):
    monkeypatch.setattr(data, "DATA_PATH", sample_csv)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.prepare_data()
