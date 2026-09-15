from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watchers.adp_watcher import AdpWatcher


def test_adp_watcher_falls_back_to_spa_report_json(monkeypatch):
    watcher = AdpWatcher("adp_test", "https://adpemploymentreport.com/")
    html = """
    <html><body>
      <div id="root"></div>
      <script type="module" src="./assets/index.js"></script>
    </body></html>
    """
    payload = {
        "reportMonth": "March",
        "reportYear": "2026",
        "reportDownloadLink": "https://adpemploymentreport.com/artifacts/us_ner/20260401/ADP_NER_history.zip",
        "reportPressReleaseLink": "https://adpemploymentreport.com/artifacts/us_ner/20260401/ADP_NATIONAL_EMPLOYMENT_REPORT_Press_Release_2026_03 FINAL.pdf",
        "reportOverview": {
            "title": "Private employers added 62,000 jobs in March",
            "description": "Hiring and pay gains both held steady in March.",
            "cards": [
                {
                    "metricName": "Employment Change",
                    "metricValue": "62,000",
                    "metricDirection": "up",
                }
            ],
        },
    }
    monkeypatch.setattr(watcher, "_fetch_html", lambda: html)
    monkeypatch.setattr(watcher, "_fetch_report_json", lambda: payload)

    items = watcher._fetch_events()

    assert len(items) == 1
    assert items[0].title == "Private employers added 62,000 jobs in March"
    assert items[0].url == "https://adpemploymentreport.com/artifacts/us_ner/20260401/ADP_NATIONAL_EMPLOYMENT_REPORT_Press_Release_2026_03 FINAL.pdf"
    assert items[0].published_at == "March 2026"
    assert items[0].summary == "March 2026 | up 62,000 | Hiring and pay gains both held steady in March."
    assert items[0].raw["json_url"] == "https://adpemploymentreport.com/ner_production.json"
