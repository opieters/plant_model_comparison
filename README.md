Development of a Quantitative Comparison Tool for Plant Models 
==============================================================

Installation
------------

Run the script in `src/install.sh`.

Generate experimental data
--------------------------

Run `main.py` in `src/experiments/vegetative_stages_1_plant` and
`src/experiments/vegetative_stages_10_plant`.

Analyse the experimental data
-----------------------------

The scripts that analyse the data are located in `src/analyse`.

To run the datapipeline from the presentation, run the `analyse.py` script for
all reservoirs, targets and experiments using the 3 command line arguments.

For instance:

```shell
python3 analyse.py -r 'N' -i 'PARi' -n 1
```

Video presentation
------------------

A video accompanying this research was presented at FSPM2020. A transcript is
is provided at `presentation/transcript.md`. After the conference the video will
be available on online.

Licence
-------

INRAe code: CeCILL-C

Our code: MIT

Presentation: CC-BY-SA-4.0
