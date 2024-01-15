# Example 1: Brain Age prediction

## Requirements

To run this example, please make sure you have the following libraries installed

`julearn`, with the visualization dependencies:

```
pip install julearn[viz]
```

`skrvm`:
```
pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
```

## Running the example

1. Execute `1_get_data.py` to download the required data.

```
python 1_get_data.py
```

2. Run the main CV script:

```
python 2_predict_brain_age.py
```

3. Visualize the scores:

```
panel serve 3_vizualize.py
```