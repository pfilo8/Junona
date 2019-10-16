# Configuration of config file

## Structure
{
  "ClassColumn": "Class",
  "FalsePositiveCost": ...
  "FalseNegativeCost": ...
  "TruePositiveCost": ...
  "TrueNegativeCost": ...
  "DropColumns": []
}

- ClassColumn: str - defines column name of class from dataset.
- *Cost: int/str - int defines constant value of cost for specific classification, str defines column name from dataset which will be example-dependent.
- DropColumns - [str] - defines column names to drop before training model
