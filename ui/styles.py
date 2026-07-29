"""Single source of truth for the app's look.

One CSS string, injected once per rerun. Colour values here must stay in sync
with ``.streamlit/config.toml`` — the theme block controls the chrome Streamlit
paints before our CSS lands, so a mismatch shows as a flash on load.
"""

from __future__ import annotations

import streamlit as st

# Palette: near-black canvas, one saturated ember accent, everything else grey.
INK = "#0A0B0D"
SURFACE = "#121316"
SURFACE_HI = "#1A1C21"
BORDER = "#25272E"
TEXT = "#EDEDEF"
MUTED = "#868C96"
FAINT = "#5A606B"
ACCENT = "#FF6A3D"
ACCENT_DIM = "#3A1E14"
DANGER = "#E5484D"

CSS = f"""
<style>
  :root {{
    --ink: {INK};
    --surface: {SURFACE};
    --surface-hi: {SURFACE_HI};
    --border: {BORDER};
    --text: {TEXT};
    --muted: {MUTED};
    --faint: {FAINT};
    --accent: {ACCENT};
    --accent-dim: {ACCENT_DIM};
    --danger: {DANGER};
    --radius: 10px;
  }}

  /* ---------------------------------------------------------------- chrome */
  #MainMenu, footer {{ display: none !important; }}
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
  [data-testid="stToolbar"] {{ display: none !important; }}
  /* Keep the header element itself — it hosts the sidebar expand control, and
     Streamlit unmounts the whole sidebar when that control is unreachable.
     Make it invisible and click-through, but leave its children live. */
  header[data-testid="stHeader"] {{
    background: transparent !important;
    box-shadow: none !important;
    pointer-events: none;
  }}
  header[data-testid="stHeader"] * {{ pointer-events: auto; }}

  .stApp {{ background: var(--ink); }}

  .block-container {{
    padding-top: 1.1rem !important;
    padding-bottom: 7rem !important;
    max-width: 1320px;
  }}

  html, body, [class*="css"] {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}

  /* --------------------------------------------------------------- sidebar */
  [data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
  }}
  [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.4rem; }}
  [data-testid="stSidebarCollapseButton"] button {{ color: var(--faint) !important; }}

  .brand {{
    display: flex; align-items: baseline; gap: 0.5rem;
    padding: 0 0.25rem 0.15rem;
  }}
  .brand-mark {{
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.03em; color: var(--text);
  }}
  .brand-mark span {{ color: var(--accent); }}
  .brand-sub {{
    font-size: 0.66rem; color: var(--faint); letter-spacing: 0.09em;
    text-transform: uppercase; font-weight: 600;
  }}

  .rail-label {{
    font-size: 0.63rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--faint); font-weight: 700; margin: 1.1rem 0.25rem 0.4rem;
  }}

  /* --------------------------------------------------------------- buttons */
  .stButton button, .stDownloadButton button {{
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: 0.84rem;
    font-weight: 500;
    padding: 0.4rem 0.65rem;
    transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
    text-align: left;
    width: 100%;
  }}
  .stButton button:hover {{
    background: var(--surface-hi);
    color: var(--text);
    border-color: var(--border);
  }}
  .stButton button:focus:not(:active) {{ color: var(--text); }}
  .stButton button:active {{ background: var(--surface-hi); }}

  /* primary = the accent action (new chat, save) */
  .stButton button[kind="primary"] {{
    background: var(--accent);
    color: #140A06;
    font-weight: 650;
    border: none;
    text-align: center;
  }}
  .stButton button[kind="primary"]:hover {{
    background: #FF7E56;
    color: #140A06;
  }}

  /* Scoping note: a bare <div> written through st.markdown gets auto-closed and
     wraps nothing, so it cannot scope CSS. Streamlit stamps an ``st-key-<key>``
     class onto every keyed widget and container — that is the hook used below. */

  /* the active conversation in the list */
  .st-key-active_conv_row .stButton button {{
    background: var(--accent-dim);
    color: var(--text);
    border-color: rgba(255,106,61,0.32);
    font-weight: 600;
  }}
  .st-key-active_conv_row .stButton button:hover {{ background: #482519; }}

  /* compact ✎ / ✕ affordances on conversation and memory rows */
  [class*="st-key-ren_"] button,
  [class*="st-key-del_"] button,
  [class*="st-key-medit_"] button,
  [class*="st-key-mdel_"] button {{
    padding: 0.15rem 0.35rem;
    font-size: 0.72rem;
    text-align: center;
    color: var(--faint);
    min-height: 0;
  }}
  [class*="st-key-ren_"] button:hover,
  [class*="st-key-del_"] button:hover,
  [class*="st-key-medit_"] button:hover,
  [class*="st-key-mdel_"] button:hover {{ color: var(--accent); }}

  /* ---------------------------------------------------------------- inputs */
  .stTextInput input, .stTextArea textarea {{
    background: var(--ink) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.85rem !important;
  }}
  .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
    color: var(--faint) !important;
  }}
  .stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,106,61,0.13) !important;
  }}
  [data-testid="stWidgetLabel"] p {{
    font-size: 0.72rem !important; color: var(--muted) !important; font-weight: 600;
  }}

  /* ------------------------------------------------------------ chat input */
  [data-testid="stChatInput"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    transition: border-color 140ms ease, box-shadow 140ms ease;
  }}
  [data-testid="stChatInput"]:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,106,61,0.14), 0 8px 30px rgba(0,0,0,0.45);
  }}
  [data-testid="stChatInput"] textarea {{
    color: var(--text) !important;
    font-size: 0.92rem !important;
  }}
  [data-testid="stChatInput"] textarea::placeholder {{ color: var(--faint) !important; }}
  [data-testid="stChatInput"] button {{ color: var(--accent) !important; }}
  /* Streamlit disables the textarea itself, not the wrapper — match on that. */
  [data-testid="stChatInput"]:has(textarea:disabled) {{
    opacity: 0.5;
    border-color: var(--border) !important;
    box-shadow: none;
  }}
  [data-testid="stBottomBlockContainer"] {{
    background: linear-gradient(to top, var(--ink) 62%, transparent);
    padding-bottom: 1.4rem;
  }}

  /* ---------------------------------------------------------- chat bubbles */
  [data-testid="stChatMessage"] {{
    background: transparent;
    padding: 0.15rem 0 0.9rem 0;
    gap: 0.75rem;
  }}
  [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {{
    background: var(--surface-hi) !important;
    border: 1px solid var(--border);
    color: var(--muted) !important;
    width: 27px; height: 27px; line-height: 27px; font-size: 0.72rem;
  }}
  [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {{
    background: var(--accent-dim) !important;
    border-color: rgba(255,106,61,0.3);
    color: var(--accent) !important;
  }}
  [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p,
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stChatMessage"] li {{
    font-size: 0.925rem; line-height: 1.68; color: #D9DBDF;
  }}
  [data-testid="stChatMessage"] strong {{ color: var(--text); font-weight: 620; }}

  /* user turn reads as a card, assistant turn reads as plain prose */
  .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.6rem 0.9rem;
  }}

  code {{
    background: var(--surface-hi) !important;
    color: #FFB39A !important;
    font-size: 0.83em !important;
    padding: 0.12em 0.36em !important;
    border-radius: 5px;
  }}
  [data-testid="stCode"], pre {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: #0E0F12 !important;
  }}
  [data-testid="stCode"] code {{ background: transparent !important; }}

  /* ---------------------------------------------------------------- header */
  .strip {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; padding: 0 0.15rem 0.85rem;
    border-bottom: 1px solid var(--border); margin-bottom: 1.1rem;
  }}
  .strip-title {{
    font-size: 0.98rem; font-weight: 620; color: var(--text);
    letter-spacing: -0.015em; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }}
  .strip-meta {{
    font-size: 0.7rem; color: var(--faint); margin-top: 0.12rem;
    letter-spacing: 0.02em;
  }}
  .badge {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--accent-dim); color: var(--accent);
    border: 1px solid rgba(255,106,61,0.28);
    border-radius: 999px; padding: 0.26rem 0.66rem;
    font-size: 0.7rem; font-weight: 650; letter-spacing: 0.02em;
    white-space: nowrap;
  }}
  .badge .dot {{
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
  }}

  /* ----------------------------------------------------------- empty state */
  .empty {{ padding: 4.5rem 0 1.6rem; max-width: 620px; }}
  .empty-kicker {{
    font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); font-weight: 700; margin-bottom: 0.7rem;
  }}
  .empty h1 {{
    font-size: 2.05rem; font-weight: 680; letter-spacing: -0.035em;
    color: var(--text); margin: 0 0 0.55rem; line-height: 1.12;
  }}
  .empty p {{
    font-size: 0.94rem; color: var(--muted); line-height: 1.62; margin: 0 0 0.4rem;
  }}
  .starter-label {{
    font-size: 0.63rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--faint); font-weight: 700; margin: 2.2rem 0 0.7rem;
  }}
  [class*="st-key-starter_"] .stButton button {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: #C9CCD2;
    padding: 0.85rem 1rem;
    font-size: 0.86rem;
    line-height: 1.45;
    min-height: 84px;
    white-space: normal;
  }}
  [class*="st-key-starter_"] .stButton button:hover {{
    border-color: rgba(255,106,61,0.45);
    background: var(--surface-hi);
    color: var(--text);
  }}

  /* ---------------------------------------------------------- memory panel */
  /* The panel scrolls inside itself. Otherwise a long fact list makes the page
     taller than the viewport and the chat column scrolls out of view with it. */
  .st-key-memory_scroll {{
    position: sticky;
    top: 0.25rem;
    max-height: calc(100vh - 11rem);
    overflow-y: auto;
    padding-right: 0.55rem;
  }}

  .mem-head {{
    display: flex; align-items: baseline; justify-content: space-between;
    padding-bottom: 0.7rem; margin-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
  }}
  .mem-head h3 {{
    font-size: 0.8rem; font-weight: 700; color: var(--text);
    letter-spacing: 0.05em; text-transform: uppercase; margin: 0;
  }}
  .mem-head span {{ font-size: 0.72rem; color: var(--faint); }}

  [data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface);
    margin-bottom: 0.45rem;
  }}
  [data-testid="stExpander"] summary {{ padding: 0.55rem 0.8rem !important; }}
  [data-testid="stExpander"] summary p {{
    font-size: 0.78rem !important; font-weight: 620 !important; color: #C9CCD2 !important;
  }}
  [data-testid="stExpander"] summary:hover p {{ color: var(--accent) !important; }}
  [data-testid="stExpanderDetails"] {{ padding-top: 0.2rem; }}

  .mem-item {{
    padding: 0.45rem 0 0.2rem;
    border-top: 1px solid rgba(255,255,255,0.045);
  }}
  .mem-item:first-child {{ border-top: none; }}
  .mem-text {{ font-size: 0.83rem; color: #D3D6DB; line-height: 1.5; }}
  .mem-time {{ font-size: 0.68rem; color: var(--faint); margin-top: 0.15rem; }}
  .mem-time em {{ color: var(--accent); font-style: normal; }}

  /* skeleton shown while memories load */
  .skel {{
    height: 34px; border-radius: 8px; margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #17191D 25%, #202329 37%, #17191D 63%);
    background-size: 400% 100%;
    animation: shimmer 1.3s ease-in-out infinite;
  }}
  @keyframes shimmer {{
    0% {{ background-position: 100% 0; }}
    100% {{ background-position: -100% 0; }}
  }}

  /* ----------------------------------------------------------- misc tidying */
  hr, [data-testid="stDivider"] {{ border-color: var(--border) !important; }}
  [data-testid="stCaptionContainer"] p {{ color: var(--faint) !important; font-size: 0.72rem !important; }}
  [data-testid="stToast"] {{
    background: var(--surface-hi) !important;
    border: 1px solid rgba(255,106,61,0.3) !important;
    color: var(--text) !important;
  }}
  [data-testid="stCheckbox"] p {{ font-size: 0.76rem !important; color: var(--muted) !important; }}
  [data-testid="stSpinner"] i {{ border-top-color: var(--accent) !important; }}
  ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: #23262C; border-radius: 5px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #32363E; }}
</style>
"""


def inject() -> None:
    """Write the stylesheet into the page. Call once, early, every rerun."""
    st.markdown(CSS, unsafe_allow_html=True)
