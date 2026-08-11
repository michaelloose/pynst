# Issues und Zuständigkeitsgrenzen

## Erledigt

- `sweep_contract.json` bindet das vollständige, geordnete Sweep-Grid samt
  Parameternamen und Datentypen.
- `expected_result_schema` kann Blocknamen, lokale Indizes, Spalten und
  optional Datentypen vorgeben. Nachträglich beobachtete Datentypen machen eine
  Deklaration ohne `dtypes` beim Resume nicht mehr inkompatibel.
- Der gespeicherte `block_mode` wird beim Resume gegen vorhandene Chunks und
  die nächsten Rückgaben geprüft.
- Bei Mapping-Rückgaben ist die Einfügereihenfolge der Schlüssel irrelevant;
  die Menge und Struktur der benannten Blöcke bleibt verbindlich.
- Bei Listen-Rückgaben bleiben Anzahl und Positionen der Blöcke verbindlich,
  einschließlich dauerhaft leerer (`None`) Positionen.
- Die Resume-Fortschrittsanzeige startet bei der Zahl bereits vollständiger
  Kombinationen und erzeugt dadurch keine künstlich hohe Anfangsrate mehr.
- Die Bereinigung wird als `Cleaning incomplete combinations` bezeichnet.
- Optional kann `provide_previous_result=True` gesetzt werden. Die
  Messfunktion erhält dann als zweites Argument das letzte gültige, bereits mit
  Sweep-Parametern versehene Ergebnis. Beim Resume wird der genaue vollständige
  Vorgänger aus dem im Log referenzierten Chunk geladen. Fehlende oder
  widersprüchliche Daten führen vor dem nächsten Mess-Callback zum Abbruch.

## Bewusst außerhalb von PyNST

- Fachliche Identitäten wie Load-Pull-Plan, Tunermodell, Aufbau,
  Kalibrierzustand oder Referenz-Epoche gehören in die jeweilige Domain-Library
  beziehungsweise deren Mess-Wrapper.
- Instrumentenspezifische Plausibilitätsprüfungen, insbesondere die
  Frequenzachse eines VNA-Ergebnisses, sind nicht Aufgabe der generalistischen
  Sweep-Library.
- PyNST erzwingt die spezifizierte Reihenfolge einzelner Sweep-Punkte. Nur bei
  Mapping-Ergebnisblöcken ist die Reihenfolge der benannten Blöcke irrelevant;
  Listenblöcke bleiben positional.

## Offen

- Falls künftig parallel arbeitende Measurement-Worker eingeführt werden, muss
  die Semantik von `previous_result` neu definiert oder für diesen Modus
  ausdrücklich ausgeschlossen werden. Der aktuelle Manager arbeitet seriell.
