---
name: Model reasoning visibility
description: Show detailed provider summaries when raw chain-of-thought is unavailable.
---

The user approved showing the provider's detailed reasoning summary when raw chain-of-thought is unavailable. Label summaries honestly; do not present them as raw reasoning.

**Why:** The configured Meta API keeps raw chain-of-thought private, and the user chose its detailed summary as the visible alternative.

**How to apply:** Request and forward detailed summary fields when available. If a provider exposes raw reasoning instead, preserve the distinction and never fabricate reasoning text.
