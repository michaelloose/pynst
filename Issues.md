# Issues und Zuständigkeitsgrenzen

## Erledigt

- `sweep_contract.json` bindet das vollständige, geordnete Sweep-Grid samt
  Parameternamen und Datentypen. Kategoriale Ebenen binden zusätzlich den
  vollständigen Kategorienvorrat und dessen Reihenfolge.
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
- `sequence_index` aus der geordneten Contract-Liste ist die einzige
  persistente Punktidentität; Stringdarstellungen von Parameterwerten dienen
  nur noch der Anzeige.
- `run_manifest.json` ist die atomar geschriebene Autorität für Chunks sowie
  vollständige und fehlgeschlagene Sequenzen. Das TSV-Log wird daraus
  rekonstruiert und ist nur diagnostisch.
- Jeder neue Chunk enthält Run-UUID, Contract-Fingerprint, Chunkindex,
  Blockstruktur und Sequenzliste. Resume validiert Hash, Struktur, Dtypes und
  Überlappungsfreiheit, bevor die Messfunktion aufgerufen wird.
- Ein OS-Dateilock schützt einen Run bereits vor Cleanup und bleibt über
  Reconciliation, Messung und Commit hinweg aktiv. Auch Metadatenänderungen und
  Merge-Ziele sind gegen konkurrierende Schreiber geschützt.
- Interrupts versuchen bereits akzeptierte Puffereinträge zu committen und
  werden anschließend erneut ausgelöst. Fehler beim Speichern sind kritische
  Laufabbrüche und keine gewöhnlichen fehlgeschlagenen Messpunkte.
- `run()` meldet nur einen vollständig committeten Sweep als erfolgreich.
  `merge()` verlangt standardmäßig Vollständigkeit; partielle
  Diagnoseartefakte benötigen `require_complete=False`.
- `merge(strategy="fixed" | "streaming")` konsolidiert beide Merge-Pfade.
  `partial_merge()` ist ein deprecated Kompatibilitätswrapper für die
  Streaming-Strategie. `drop_columns` aktualisiert bei beiden Strategien auch
  die Blockmetadaten.
- Der Fixed-Merge prüft vor dem Erzeugen des Zielartefakts konservativ den
  Speicherbedarf des größten Blocks gegen den aktuell verfügbaren physischen
  RAM und verweist bei unzureichendem oder unbekanntem Budget explizit auf die
  Streaming-Strategie.
  Leere Messresultate, instabile Listenpositionen und leere Sweep-Grids werden
  früh abgewiesen.
- `remove_chunks=True` setzt nach einem tief validierten Merge zuerst einen
  persistenten Archivstatus samt Zielhash und entfernt erst danach Chunks.
  Archivierte Runs können nicht erneut aufgenommen werden.
- `GenericSweepDataset.validate_storage(deep=True)` prüft sämtliche Zeilen,
  vollständige Indexeindeutigkeit, Blockabdeckung und gespeicherte Dtypes.
- Unterstützte Python-Version ist nun konsistent `>=3.10` (`X | Y`-Syntax).
- Das alte In-place-Filtern committeter Chunks ist deaktiviert, weil es die
  Manifest- und Hashinvarianten verletzen würde.
- Alte v1/v2-Läufe werden zweiphasig und nach einem Prozessabbruch
  wiederaufnehmbar auf den v3-Contract migriert.
- Merge-Temporärdateien sind kollisionsfrei, Merge-Artefakte werden vor dem
  Veröffentlichen tief validiert und ein Chunk-Archiv darf nicht innerhalb des
  später austauschbaren Run-Verzeichnisses liegen.
- Globale und lokale Index-Dtypes sowie kategoriale Dtype-Semantik bleiben in
  Chunks, Contract, Merge-Artefakt und Dataset-Validierung konsistent.
- Vom pandas-HDF-Backend nicht speicherbare Extension-Dtypes und komplexe
  Indexlevel werden vor der Schemabindung abgewiesen; komplexe Dvars bleiben
  zulässig. Gemischte beziehungsweise nicht-stringförmige globale
  `object`-Ebenen werden vor dem ersten Hardware-Callback, entsprechende
  Ergebnisdaten unmittelbar danach noch vor der Schemabindung abgelehnt.
- `uint64`-Indexlevel werden wegen einer PyTables-Einschränkung ebenfalls früh
  abgewiesen; `uint64` als abhängige Messwertspalte bleibt zulässig.

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
- Exakt-einmalige Wirkung externer Hardware kann PyNST nicht garantieren: Stirbt
  der Prozess nach dem Gerätezugriff, aber vor dem dauerhaften Chunk-Commit,
  wird der Punkt beim Resume erneut ausgeführt. Diese bewusste
  At-least-once-Grenze muss eine Domain-Messfunktion tolerieren.
