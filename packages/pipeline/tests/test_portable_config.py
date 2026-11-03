from pathlib import Path
import pytest
from retrieval.common import load_config
from retrieval.rankers.gripper_ego_ranker import load_config as motion_config
from retrieval.motion.hand_grasp import load_grasp_config
from retrieval.search.droid_to_ego import retrieve_droid_to_ego


def test_file_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    settings = tmp_path / 'settings'
    settings.mkdir()
    file = settings / 'config.yaml'
    file.write_text('artifact_root: ../results\npacked_motion_path: packed.npy\n', encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(file)
    assert Path(cfg['artifact_root']) == tmp_path / 'results'
    assert Path(cfg['packed_motion_path']) == settings / 'packed.npy'
    assert Path(cfg['schema_path']).is_file()
    assert motion_config()['nominal_droid_fps'] > 0
    assert load_grasp_config()['version'] == 'ego-grasp-proxy-v1'


def test_droid_route_rejects_hidden_workspace_dependencies():
    with pytest.raises(ValueError, match='supplied explicitly'):
        retrieve_droid_to_ego('droid_000')
    with pytest.raises(ValueError, match='frozen candidate pool'):
        retrieve_droid_to_ego('droid_000', droid_root='robot', ego_root='ego')
