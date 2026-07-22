from pathlib import Path

from model_runtime import DeploymentConfig, basic_clean, missing_artifacts


def test_basic_clean_matches_notebook_behavior():
    value = "INI BAGUS 😀 <b>tetapi</b> buka https://example.org"
    assert basic_clean(value) == "Ini Bagus  tetapi buka"


def test_missing_artifacts(tmp_path: Path):
    config = DeploymentConfig()
    assert set(missing_artifacts(tmp_path, config)) == {
        config.model_file,
        config.tokenizer_file,
        config.label_encoder_file,
    }
