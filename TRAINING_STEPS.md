# Train a Model on the Readable MAESTRO Dataset

Open PowerShell and go to the project folder:

```powershell
cd "C:\Users\Surabhi Deshpande\project_1"
```

Train a small test model on the first 100 pieces:

```powershell
python .\train_model.py train --readable-dir .\maestro_readable --output-model .\models\maestro_markov.json --max-pieces 100
```

If your computer uses the WindowsApps Python path, run this instead:

```powershell
& "C:\Users\Surabhi Deshpande\AppData\Local\Microsoft\WindowsApps\python3.13.exe" .\train_model.py train --readable-dir .\maestro_readable --output-model .\models\maestro_markov.json --max-pieces 100
```

Generate a new MIDI sample:

```powershell
python .\train_model.py generate --model .\models\maestro_markov.json --output-dir .\generated --note-count 500
```

Or with the WindowsApps Python path:

```powershell
& "C:\Users\Surabhi Deshpande\AppData\Local\Microsoft\WindowsApps\python3.13.exe" .\train_model.py generate --model .\models\maestro_markov.json --output-dir .\generated --note-count 500
```

Open the generated output:

```powershell
explorer .\generated
```

The generated files will be:

- `generated_notes.csv`
- `generated.mid`

After the small test works, train on the full dataset:

```powershell
python .\train_model.py train --readable-dir .\maestro_readable --output-model .\models\maestro_markov_full.json --max-pieces 1276
```
