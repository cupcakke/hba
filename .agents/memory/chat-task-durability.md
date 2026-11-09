---
name: Chat task durability
description: Preserve active chat work when the user closes the app browser tab.
---

Active chat work must continue when the user closes the app browser tab.

**Why:** The user reported that leaving the app browser tab stops work and expects background tasks to continue.

**How to apply:** Do not bind worker dispatch or execution lifetime to the chat page request or its event stream. Keep task state durable and allow the UI to resume streaming when the user returns.
