# Optimized model for Bs -> Jpsi Phi time dependent angular fit


## 0. generate toy

```
python gen_toy.py
```


## 1. papering data for fit

```
python create_p4_fast.py
python generate_phsp_fast.py
python cut_phsp_fast.py
python double_phsp_fast.py
python time_int_fast.py
```

## 2. fit model

Some fit script as `fit_{loader,conv,conv2}.py` with model file `config_{loader,conv,conv2}.yml`.

``
python fit_loader.py
``
