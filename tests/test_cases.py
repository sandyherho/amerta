"""Config-manager tests."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from amerta_sv.io.config_manager import ConfigManager

def test_load_parse_types():
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write("# comment\n")
        f.write("name = stoker\n")
        f.write("nx = 400\n")
        f.write("cfl = 0.9\n")
        f.write("save = true\n")
        path = f.name
    c = ConfigManager.load(path)
    assert c['name'] == 'stoker'
    assert c['nx'] == 400 and isinstance(c['nx'], int)
    assert c['cfl'] == 0.9 and isinstance(c['cfl'], float)
    assert c['save'] is True
    os.unlink(path)

def test_validate_defaults():
    c = ConfigManager.validate_config({'h_left': 3.0})
    assert c['h_left'] == 3.0
    assert c['nx'] == 400
    assert c['cfl'] == 0.9
