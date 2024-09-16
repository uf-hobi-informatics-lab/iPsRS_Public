# Project Title

iPsRS project --> Link to the paper "A Fair Individualized Polysocial Risk Score for Identifying Increased Social Risk in Type 2 Diabetes"

## Description

Codebase for the iPsRS Fairness project.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

```
./1.tuning_sheduler.sh
./2.parameter_checking_scheduler.sh
./3.bootstrapping_scheduler.sh
./4.assessment_scheduler.sh
./5.5.unfairness_mitigation_scheduler.sh
...
```

## Help

```
Modify the json files in the Settings to build your experiments.
Modify the {#.xx.sh} to run what you want to run.
```

## Authors

Contributors names and contact info
[@Yu](https://yuvisu.github.io/)

## Version History

* 0.1
    * Initial Release
    * Code Structure Reorganization

* 0.2
    * Automated Pipeline, Tuning -> Assessment -> Mitigation

* 0.3
    * Automated Pipeline, Tuning -> Checking -> Bootstraping -> Fairness Assessment -> Fairness Mitigation
    * The framework is more flexiable, the user can DIY the base model (e.g., xgboost, catboost).
    
* 0.3.1
    * Align model output path
    * Reconstruct model.py
    * Reconstruct Settings
    
* 0.4.0
    * add feature importance analysis (SHAP)
    * add Post analysis
    * add causal analysis (double robust learning)
    * The above mentioned functions are implemented in jupyter notebook.
    
## License

TBD

## Acknowledgments

Inspiration, code snippets, etc.
* [AIF-360](https://github.com/Trusted-AI/AIF360)
* [EconML](https://econml.azurewebsites.net/)
* [SHAP](https://shap.readthedocs.io/en/latest/)