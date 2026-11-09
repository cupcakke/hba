---
name: Adaptive cognitive orchestration
description: Keep application-level planning on demand without conflating it with provider reasoning settings.
---

Az elsődleges modell a Muse Spark; az app szokásos válaszait ez kezeli. Egyszerű köszönésre vagy közvetlen, egylépéses kérésre ne induljon tervezés vagy Gemini-hívás. A Gemini csak ritka, elkülönített alügynöki tanácsadóként használható, ha az ügynök ismétlődő végrehajtási elakadásból nem tud továbblépni; ne legyen a Muse API-hibáinak tartalékmodellje. Az alügynöki eszközök ne legyenek elérhetők az elsődleges modell számára. A fő modell `xhigh` reasoning szintet és 27 000-es max output token értéket használjon.

Treat the advisor as optional: on HTTP 429, fail fast to the primary model and suppress additional advisor requests during a short cooldown.

**Why:** A felhasználó kifejezetten azt kérte, hogy a Muse Spark maradjon az elsődleges modell, a Gemini pedig csak ritka, elakadáskezelő alügynökként fusson — ne hívja meg az app egy egyszerű köszönés vagy elsődleges modellhiba miatt. A felhasználó azt is kifejezetten kérte, hogy az `xhigh` reasoning és a 27 000 tokenes korlát maradjon.

**How to apply:** Az alügynöki eszközök maradjanak rejtve az elsődleges modell elől. Tanácsadót csak explicit, ismétlődő ügynök-elakadás esetén hívj; elsődleges modellhibánál állj le világos hibával, és ne indíts újabb fizetős modellt. A költségkontrollt a hívások számának korlátozásával oldd meg, ne az `xhigh` vagy a 27 000 tokenes érték csökkentésével.
