# Discontinuous NER with Pointer Network

Codes for the AAAI 2021 paper: [Rethinking Boundaries: End-To-End Recognition of Discontinuous Mentions with Pointer Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17513). 

-------------------


# Requirement
  
```bash
python>=1.6
numpy>=1.13.3
torch>=0.4.0
```

# Datasets

Two benchmark datasets for discontinuous NER.
Download them and put at `./data` folds. 

- [CADEC](https://data.csiro.au/collection/csiro:10948)
- ShARe13


Data format preprocessing.
Please process the annotation as following format:

```bash
Upset stomach and the feeling that I may need to throw up .
0,1 ADR|10,11 ADR
```

See the example data in [./data/examples](data%2Fexamples).



# Experiments

To train the parser, run the following script:

```bash
python ./framework/main.py
```

Change the parameters for training, testing.


# Citation

```
@inproceedings{FeiDisNERAAAI21,
  author = {Hao Fei and Donghong Ji and Bobo Li and
            Yijiang Liu and Yafeng Ren and Fei Li},
  title  = {Rethinking Boundaries: End-To-End Recognition of Discontinuous Mentions with Pointer Networks},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages = {12785--12793},
  year = {2021},
}
```


# License

The code is released under Apache License 2.0 for Non-commercial use only. 