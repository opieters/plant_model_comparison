conda create -n FSPM2020 python=3.7 openalea.mtg openalea.plantgl openalea.lpy alinea.caribu alinea.astk coverage nose sphinx statsmodels rpy2 -c conda-forge -c fredboudon -y

conda activate FSPM2020

pip install rpy2

git clone https://github.com/opieters/adel.git
pushd adel
git checkout 90b6505
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/cn-wheat.git
pushd cn-wheat
git checkout 5428dd5
python3 setup.py install
popd

git clone https://github.com/christian34/core.git
pushd core
git checkout e930385
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/elong-wheat.git
pushd elong-wheat
git checkout 353865b
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/farquhar-wheat.git
pushd farquhar-wheat
git checkout 51fe26f
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/growth-wheat.git
pushd growth-wheat
git checkout 349cf7d
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/respi-wheat.git
pushd respi-wheat
git checkout 8a2da9c2fb7811d6432ab5267219015d63da24d7
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/senesc-wheat.git
pushd senesc-wheat
git checkout 049cd6c
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/turgor-growth.git
pushd turgor-growth
git checkout c9180a8
python3 setup.py install
popd

git clone https://github.com/openalea-incubator/fspm-wheat.git
pushd fspm-wheat
git checkout b5ba9b5
python3 setup.py install
popd