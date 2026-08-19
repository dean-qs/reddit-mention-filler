"""Every available analysis module, in run order.

To add a new module (sentiment recoding, geolocation, ...):
  1. Write modules/your_module.py implementing AnalysisModule (see base.py).
  2. Import it and add an instance to MODULES below.
Modules run in sequence in a single session: each one's output xlsx becomes
the next one's input (new columns are always appended, never inserted, so
the original 'Query Id' header and required columns stay put between steps).
No changes to app.py are needed.
"""
from .driver_analysis import DriverAnalysisModule
from .geolocation import GeolocationModule
from .mention_filler import MentionFillerModule
from .sentiment import SentimentModule
from .theme_summary import ThemeSummaryModule

MODULES = [
    MentionFillerModule(),
    SentimentModule(),
    GeolocationModule(),
    ThemeSummaryModule(),
    DriverAnalysisModule(),
]
