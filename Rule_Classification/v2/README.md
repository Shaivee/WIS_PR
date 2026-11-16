# Defect rule classifier

Two pipelines that run independently. Neither imports from or needs the other.

| Folder | Does | Run |
|---|---|---|
| `training/` | builds the class rules from the ISRA data and measures their accuracy | `python build_rules.py` |
| `testing/` | classifies a defect blob with the rules and returns JSON (class and runner-up) | `python test_blob_classifier.py` |

Each folder has its own `README.md` and `requirements.txt`.

## The hand-off

`training/` writes the rule tables to `training/output/`. `testing/` reads its own copy of
`class_rules.csv` from `testing/rules/`. The two folders ship with identical tables, so each works
as it is. After retraining, update the testing side by copying the file:

```bash
cp training/output/class_rules.csv testing/rules/
```

## One duplicated piece

Both pipelines need the range-scoring formula: training uses it to measure accuracy, testing uses
it to classify. So that each folder can run alone, the formula exists in `training/rule_scoring.py`
and in `testing/blob_mask_classifier.py` (`classify_parameters`). The two give the same result and
must stay that way; if you change the rule, change it in both.
