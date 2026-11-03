"""A tiny stand-in for the `streamlit` module, used only by the smoke tests.

Why it exists: it lets us run every dashboard page from the command line
(and in environments where Streamlit cannot be installed). It records what the
pages draw and it copies Streamlit's strictest rules, so real mistakes still
fail here, for example:
  * a date_input default outside min_value/max_value  -> error
  * a selectbox/radio value that is not one of the options -> error
It is NOT a replacement for looking at the real app in a browser.
"""

import datetime as dt
import os
import types

import matplotlib.pyplot as plt


class FakeStreamlitError(Exception):
    """Raised when the app would have raised a StreamlitAPIException."""


class _Box:
    """Stands for columns / tabs / containers / expanders / the sidebar."""

    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):         # column.markdown(...) == st.markdown(...)
        return getattr(self._st, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.overrides = {}              # widget key -> value to "click"
        self.reset()
        self.sidebar = _Box(self)
        self.column_config = types.SimpleNamespace(
            ProgressColumn=lambda *a, **k: None, NumberColumn=lambda *a, **k: None)

    # ---- bookkeeping ---------------------------------------------------------
    def reset(self):
        self.errors, self.warnings, self.infos = [], [], []
        self.texts = []                  # every piece of text/HTML shown
        self.figures = 0
        self.tables = []
        self.widgets = {}                # key -> value returned
        self.api_calls = []

    def _text(self, *parts):
        self.texts.extend(str(p) for p in parts)

    # ---- display functions -----------------------------------------------------
    def set_page_config(self, **kw): pass
    def markdown(self, body, **kw): self._text(body)
    def write(self, *a, **kw): self._text(*a)
    def caption(self, body, **kw): self._text(body)
    def title(self, body, **kw): self._text(body)
    def header(self, body, **kw): self._text(body)
    def subheader(self, body, **kw): self._text(body)
    def text(self, body, **kw): self._text(body)
    def code(self, body, **kw): self._text(body)
    def divider(self, **kw): pass
    def image(self, *a, **kw): pass
    def progress(self, value, **kw):
        if not 0.0 <= float(value) <= 1.0:
            raise FakeStreamlitError(f"progress value must be 0-1, got {value}")
    def metric(self, label, value, **kw): self._text(label, value)
    def download_button(self, *a, **kw): return False
    def json(self, body, **kw): self._text(str(body))
    def exception(self, error): self.errors.append(f"EXCEPTION {error!r}")

    def info(self, body, **kw): self.infos.append(str(body)); self._text(body)
    def success(self, body, **kw): self._text(body)
    def warning(self, body, **kw): self.warnings.append(str(body)); self._text(body)
    def error(self, body, **kw): self.errors.append(str(body)); self._text(body)

    def dataframe(self, data, **kw):
        self.tables.append(data)
        self._text(data.to_string() if hasattr(data, "to_string") else data)

    def pyplot(self, fig=None, **kw):
        self.figures += 1
        folder = os.environ.get("SMOKE_SAVE_DIR")
        if folder and fig is not None:
            os.makedirs(folder, exist_ok=True)
            fig.savefig(os.path.join(folder, f"{self.current_name}_{self.figures:02d}.png"),
                        dpi=90, bbox_inches="tight")

    # ---- layout ----------------------------------------------------------------
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        if n < 1:
            raise FakeStreamlitError("columns needs at least one column")
        return [_Box(self) for _ in range(n)]

    def tabs(self, labels): return [_Box(self) for _ in labels]
    def container(self, **kw): return _Box(self)
    def expander(self, *a, **kw): return _Box(self)

    # ---- caching: just call the function (data is small) -------------------------
    def _cache(self, func=None, **kw):
        return func if func is not None else (lambda f: f)
    cache_data = _cache
    cache_resource = _cache

    # ---- widgets: return an override, or the widget's default -------------------
    def _value(self, key, default, label):
        value = self.overrides.get(key, default)
        if key is not None:
            self.widgets[key] = value
        return value

    def _choice(self, label, options, index, key, format_func):
        options = list(options)
        for option in options:           # format_func must work for every option
            format_func(option)
        value = self.overrides.get(key, options[index] if options else None)
        if options and value not in options:
            raise FakeStreamlitError(f"{key}: value {value!r} is not one of {options}")
        self.widgets[key] = value
        return value

    def selectbox(self, label, options, index=0, format_func=str, key=None, **kw):
        return self._choice(label, options, index, key, format_func)

    def radio(self, label, options, index=0, format_func=str, key=None, **kw):
        return self._choice(label, options, index, key, format_func)

    def multiselect(self, label, options, default=None, key=None, **kw):
        options = list(options)
        value = self.overrides.get(key, list(default or []))
        bad = [v for v in value if v not in options]
        if bad:
            raise FakeStreamlitError(f"{key}: {bad} not in options")
        self.widgets[key] = value
        return value

    def checkbox(self, label, value=False, key=None, **kw):
        return self._value(key, value, label)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **kw):
        chosen = self.overrides.get(key, value)
        if min_value is not None and chosen < min_value or max_value is not None and chosen > max_value:
            raise FakeStreamlitError(f"{key}: {chosen} is outside {min_value}..{max_value}")
        self.widgets[key] = chosen
        return chosen

    def date_input(self, label, value=None, min_value=None, max_value=None, key=None, **kw):
        chosen = self.overrides.get(key, value)
        if min_value is not None and chosen < min_value:
            raise FakeStreamlitError(f"{key}: {chosen} is before min_value {min_value}")
        if max_value is not None and chosen > max_value:
            raise FakeStreamlitError(f"{key}: {chosen} is after max_value {max_value}")
        self.widgets[key] = chosen
        return chosen

    def button(self, *a, **kw): return False


def install():
    """Put the fake module in place of `streamlit` and return it."""
    import sys
    fake = FakeStreamlit()
    fake.current_name = "page"
    sys.modules["streamlit"] = fake
    return fake
