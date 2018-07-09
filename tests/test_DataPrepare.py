"""
Tests for DataPrepare class and its methods. Run by pytest.
"""
import sys
sys.path.append("..")
import coelurus

## use monkey patch to handle the config requirements!
def test_imputation(tmpdir):
    # set up the class
    loader = coelurus.DataPrepare()
    print('foo')
    return 0