---
name: Browser Use execution
description: Route browser automation through Browser Use Cloud directly instead of requiring the isolated VM.
---

Use Browser Use Cloud and its CDP connection for browser navigation, inspection, and interaction. Do not require InstaVM or VM-installed Playwright for browser tasks. Keep InstaVM for shell, Python, package installation, and filesystem tools. Enforce the configured browser host allowlist when it is set; when no list is configured, allow only public HTTPS hosts and reject private addresses.

**Why:** The user explicitly said browser functionality in this app must work through Browser Use rather than InstaVM.

**How to apply:** Keep Browser Use tools independently available when its key is configured. Apply the configured host list when present, use the public-HTTPS-only policy when absent, and keep CDP credentials server-side.
