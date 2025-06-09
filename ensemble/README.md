This script is used to generate a final output file by performing a voting ensemble on multiple model outputs. The input files are the JSON results from previous inference runs. For example, an output from the MUMC model should be named mumc_output.json, while an output from a Gemini model might be named gemini_<version_number>_test.json.
In our experiments, the best-performing voting pipeline was an ensemble of three separate Gemini 2.5 Flash inference runs.
```bash
python ensemble/voting.py -i <output_file_1>.json <output_file_2>.json <output_file_3>.json ... -o voting.json
```
