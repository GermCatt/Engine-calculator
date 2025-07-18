import sys
import os

# Конфигурация Streamlit (без dev-режима!)
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_BROWSER_AUTO_LAUNCH"] = "true"
os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "localhost"
# НЕ задаём порт — иначе ошибка при devMode=true

# Запуск Streamlit
import streamlit.web.cli as stcli

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)

    script_path = os.path.join(base_path, "EngSite.py")
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())
