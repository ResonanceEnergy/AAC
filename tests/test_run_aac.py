import pytest
import sys
from unittest.mock import patch, MagicMock

# Define a fixture to simulate running the script from the correct directory
@pytest.fixture
def mock_path_exists():
    with patch('pathlib.Path.exists', return_value=True):
        yield

# Define a fixture to mock imports of aac_engine and aac_dashboard
@pytest.fixture
def mock_aac_imports():
    with patch.dict('sys.modules', {
        'aac_engine': MagicMock(),
        'aac_engine.AccountingEngine': MagicMock(),
        'aac_dashboard': MagicMock(),
        'aac_dashboard.app': MagicMock()
    }) as mock_modules:
        yield mock_modules

# Test for a successful run with no web dashboard
def test_main_happy_path_no_web(mock_path_exists, mock_aac_imports, capsys):
    mock_accounting_engine = mock_aac_imports['aac_engine.AccountingEngine']
    mock_engine_instance = mock_accounting_engine.return_value
    mock_engine_instance.setup_default_accounts = MagicMock()
    mock_engine_instance.close = MagicMock()

    import run_aac
    run_aac.main()

    captured = capsys.readouterr()
    assert "🚀 AAC - Automated Accounting Center" in captured.out
    assert "✅ Accounting engine initialized successfully" in captured.out
    assert "🎉 AAC system ready!" in captured.out
    assert "🌐 Starting web dashboard..." not in captured.out

# Test for a successful run with web dashboard
def test_main_happy_path_with_web(mock_path_exists, mock_aac_imports, capsys):
    mock_accounting_engine = mock_aac_imports['aac_engine.AccountingEngine']
    mock_engine_instance = mock_accounting_engine.return_value
    mock_engine_instance.setup_default_accounts = MagicMock()
    mock_engine_instance.close = MagicMock()
    mock_app = mock_aac_imports['aac_dashboard.app']

    test_argv = ['run_aac.py', '--web']
    with patch.object(sys, 'argv', test_argv):
        import run_aac
        run_aac.main()

        captured = capsys.readouterr()
        assert "🌐 Starting web dashboard..." in captured.out
        assert "📊 AAC Dashboard available at: http://localhost:5000" in captured.out
        mock_app.run.assert_called_once_with(debug=True, host='0.0.0.0', port=5000)

# Test failure when not in correct directory
def test_main_wrong_directory(mock_aac_imports, capsys):
    with patch('pathlib.Path.exists', return_value=False):
        with pytest.raises(SystemExit):
            import run_aac
            run_aac.main()

        captured = capsys.readouterr()
        assert "❌ Error: Please run this script from the AAC repository root directory" in captured.out

# Test for accounting engine initialization failure
def test_main_accounting_engine_failure(mock_path_exists, capsys):
    with patch('aac_engine.AccountingEngine', side_effect=Exception("Initialization Error")):
        with pytest.raises(SystemExit):
            import run_aac
            run_aac.main()

        captured = capsys.readouterr()
        assert "❌ Accounting engine test failed: Initialization Error" in captured.out

# Test for web dashboard startup failure
def test_main_web_dashboard_failure(mock_path_exists, mock_aac_imports, capsys):
    mock_accounting_engine = mock_aac_imports['aac_engine.AccountingEngine']
    mock_engine_instance = mock_accounting_engine.return_value
    mock_engine_instance.setup_default_accounts = MagicMock()
    mock_engine_instance.close = MagicMock()
    
    mock_app = mock_aac_imports['aac_dashboard.app']
    mock_app.run.side_effect = Exception("Dashboard Startup Error")

    test_argv = ['run_aac.py', '--web']
    with patch.object(sys, 'argv', test_argv):
        with pytest.raises(SystemExit):
            import run_aac
            run_aac.main()

        captured = capsys.readouterr()
        assert "❌ Web dashboard failed to start: Dashboard Startup Error" in captured.out
