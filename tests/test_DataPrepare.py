"""
Tests for DataPrepare class and its methods. Run by pytest.
"""
import sys
sys.path.append("..")
import coelurus
import pandas as pd
import numpy as np

def test_imputation1(tmpdir):

    # create mock config
    mock_conf = tmpdir.mkdir('mock_config').join('mock_config.ini')
    mock_conf.write('[data_sources]\ndata_source = local')

    # set up classes and create mock data
    loader = coelurus.Loader(str(tmpdir.join('mock_config', 'mock_config.ini')))

    loader.input_data = pd.DataFrame({'protein_id': ['A', 'B', 'C', 'D', 'E'], 'F1A': [0, 0, 0, 0, 0],
                                      'F2A': [0, 0, 0, 0, 0], 'F3A': [0, 0, 0, 0, 0],
                                      'F4A': [0, 0, 0, 0, 0], 'F5A': [0, 0, 0, 0, 0],
                                      'F6A': [0, 0, 0, 0, 0], 'F7A': [0, 0, 0, 0, 0]},
                                     columns=['protein_id', 'F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A'])

    expected_result = loader.input_data.copy()  # should be unchanged

    expected_result.iloc[:, 1:] = expected_result.iloc[:, 1:].astype(np.float64)

    val = coelurus.Validator(loader)
    dfilter = coelurus.DataPrepare(val)
    imputed = dfilter.impute_missing_values(loader.input_data.copy())

    return pd.testing.assert_frame_equal(imputed, expected_result)


def test_imputation2(tmpdir):

    # create mock config
    mock_conf = tmpdir.mkdir('mock_config').join('mock_config.ini')
    mock_conf.write('[data_sources]\ndata_source = local')

    # set up classes and create mock data
    loader = coelurus.Loader(str(tmpdir.join('mock_config', 'mock_config.ini')))

    loader.input_data = pd.DataFrame({'protein_id': ['A', 'B', 'C', 'D', 'E'], 'F1A': [2, 0, 0, 0, 0],
                                      'F2A': [1, 1, 1, 0, 0], 'F3A': [1, 0, 2, 1, 0],
                                      'F4A': [1, 1, 0, 0, 0], 'F5A': [0, 1, 1, 0, 1],
                                      'F6A': [2, 1, 0, 1, 1], 'F7A': [0, 1, 1, 2, 1]},
                                     columns=['protein_id', 'F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A'])

    expected_result = pd.DataFrame({'protein_id': ['A', 'B', 'C', 'D', 'E'], 'F1A': [2, 0, 0, 0, 0],
                                     'F2A': [1, 1, 1, 0, 0], 'F3A': [1, 1, 2, 1, 0],
                                     'F4A': [1, 1, 1.5, 0, 0], 'F5A': [1.5, 1, 1, 0, 1],
                                     'F6A': [2, 1, 1, 1, 1], 'F7A': [0, 1, 1, 2, 1]},
                                    columns=['protein_id', 'F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A'])
    expected_result.iloc[:, 1:] = expected_result.iloc[:, 1:].astype(np.float64)

    val = coelurus.Validator(loader)
    dfilter = coelurus.DataPrepare(val)
    imputed = dfilter.impute_missing_values(loader.input_data.copy())

    return pd.testing.assert_frame_equal(imputed, expected_result)


def test_filter_missing_profiles1(monkeypatch, tmpdir):

    # create mock config
    mock_conf = tmpdir.mkdir('mock_config').join('mock_config.ini')
    mock_conf.write('[data_sources]\ndata_source = local')

    # set up classes and create mock data/config
    loader = coelurus.Loader(str(tmpdir.join('mock_config', 'mock_config.ini')))
    loader.config.add_section('filter_options')
    loader.config.set('filter_options', 'min_consecutive_fractions', '3')

    loader.input_data = pd.DataFrame({'protein_id': ['A', 'B', 'C', 'D', 'E'], 'F1A': [2, 0, 0, 0, 0],
                                      'F2A': [1, 1, 1, 0, 0], 'F3A': [1, 0, 2, 1, 0],
                                      'F4A': [1, 1, 0, 0, 0], 'F5A': [0, 1, 1, 0, 1],
                                      'F6A': [2, 1, 0, 1, 1], 'F7A': [0, 0, 1, 0, 1]},
                                     columns=['protein_id', 'F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A'])

    expected_result = pd.DataFrame({'protein_id': ['A', 'B', 'C', 'D', 'E'], 'F1A': [2, 0, 0, 0, 0],
                                      'F2A': [1, 1, 1, 0, 0], 'F3A': [1, 0, 2, 1, 0],
                                      'F4A': [1, 1, 0, 0, 0], 'F5A': [0, 1, 1, 0, 1],
                                      'F6A': [2, 1, 0, 1, 1], 'F7A': [0, 0, 1, 0, 1]},
                                     columns=['protein_id', 'F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A'])
    expected_result = expected_result.iloc[[0, 1, 4], :]  # C and D should be removed only

    val = coelurus.Validator(loader)
    dfilter = coelurus.DataPrepare(val)
    filtered = dfilter.filter_missing_profiles(loader.input_data)

    return pd.testing.assert_frame_equal(filtered, expected_result)
