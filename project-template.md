# Predict Bike Sharing Demand with AutoGluon Template

## Project: Predict Bike Sharing Demand with AutoGluon
This notebook is a template with each step that you need to complete for the project.

Please fill in your code where there are explicit `?` markers in the notebook. You are welcome to add more cells and code as you see fit.

Once you have completed all the code implementations, please export your notebook as a HTML file so the reviews can view your code. Make sure you have all outputs correctly outputted.

`File-> Export Notebook As... -> Export Notebook as HTML`

There is a writeup to complete as well after all code implememtation is done. Please answer all questions and attach the necessary tables and charts. You can complete the writeup in either markdown or PDF.

Completing the code template and writeup template will cover all of the rubric points for this project.

The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this notebook and also discuss the results in the writeup file.

## Step 1: Create an account with Kaggle

### Create Kaggle Account and download API key
Below is example of steps to get the API username and key. Each student will have their own username and key.

1. Open account settings.
![kaggle1.png](attachment:kaggle1.png)
![kaggle2.png](attachment:kaggle2.png)
2. Scroll down to API and click Create New API Token.
![kaggle3.png](attachment:kaggle3.png)
![kaggle4.png](attachment:kaggle4.png)
3. Open up `kaggle.json` and use the username and key.
![kaggle5.png](attachment:kaggle5.png)

## Step 2: Download the Kaggle dataset using the kaggle python library

### Open up Sagemaker Studio and use starter template

1. Notebook should be using a `ml.t3.medium` instance (2 vCPU + 4 GiB)
2. Notebook should be using kernal: `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`

### Install packages


```python
!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
# Without --no-cache-dir, smaller aws instances may have trouble installing
```

    Requirement already satisfied: pip in /usr/local/lib/python3.7/site-packages (21.3.1)
    Collecting pip
      Using cached pip-22.3.1-py3-none-any.whl (2.1 MB)
    Installing collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 21.3.1
        Uninstalling pip-21.3.1:
          Successfully uninstalled pip-21.3.1
    Successfully installed pip-22.3.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/site-packages (59.4.0)
    Collecting setuptools
      Using cached setuptools-65.6.3-py3-none-any.whl (1.2 MB)
    Collecting wheel
      Using cached wheel-0.38.4-py3-none-any.whl (36 kB)
    Installing collected packages: wheel, setuptools
      Attempting uninstall: setuptools
        Found existing installation: setuptools 59.4.0
        Uninstalling setuptools-59.4.0:
          Successfully uninstalled setuptools-59.4.0
    Successfully installed setuptools-65.6.3 wheel-0.38.4
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting mxnet<2.0.0
      Using cached mxnet-1.9.1-py3-none-manylinux2014_x86_64.whl (49.1 MB)
    Collecting bokeh==2.0.1
      Using cached bokeh-2.0.1-py3-none-any.whl
    Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (5.4.1)
    Requirement already satisfied: numpy>=1.11.3 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (1.19.1)
    Requirement already satisfied: packaging>=16.8 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (21.3)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (2.8.2)
    Requirement already satisfied: tornado>=5 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (6.1)
    Requirement already satisfied: pillow>=4.0 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (8.4.0)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (4.0.1)
    Requirement already satisfied: Jinja2>=2.7 in /usr/local/lib/python3.7/site-packages (from bokeh==2.0.1) (3.0.3)
    Requirement already satisfied: graphviz<0.9.0,>=0.8.1 in /usr/local/lib/python3.7/site-packages (from mxnet<2.0.0) (0.8.4)
    Requirement already satisfied: requests<3,>=2.20.0 in /usr/local/lib/python3.7/site-packages (from mxnet<2.0.0) (2.22.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.7/site-packages (from Jinja2>=2.7->bokeh==2.0.1) (2.0.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/site-packages (from packaging>=16.8->bokeh==2.0.1) (3.0.6)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/site-packages (from python-dateutil>=2.1->bokeh==2.0.1) (1.16.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (1.25.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (2021.10.8)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (2.8)
    Installing collected packages: mxnet, bokeh
      Attempting uninstall: bokeh
        Found existing installation: bokeh 2.4.2
        Uninstalling bokeh-2.4.2:
          Successfully uninstalled bokeh-2.4.2
    Successfully installed bokeh-2.0.1 mxnet-1.9.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting autogluon
      Downloading autogluon-0.6.1-py3-none-any.whl (9.8 kB)
    Collecting autogluon.features==0.6.1
      Downloading autogluon.features-0.6.1-py3-none-any.whl (59 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.0/60.0 kB[0m [31m177.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.multimodal==0.6.1
      Downloading autogluon.multimodal-0.6.1-py3-none-any.whl (289 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m289.7/289.7 kB[0m [31m168.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.text==0.6.1
      Downloading autogluon.text-0.6.1-py3-none-any.whl (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.1/62.1 kB[0m [31m162.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.vision==0.6.1
      Downloading autogluon.vision-0.6.1-py3-none-any.whl (49 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.8/49.8 kB[0m [31m160.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.tabular[all]==0.6.1
      Downloading autogluon.tabular-0.6.1-py3-none-any.whl (286 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m286.0/286.0 kB[0m [31m221.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.core[all]==0.6.1
      Downloading autogluon.core-0.6.1-py3-none-any.whl (226 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m226.6/226.6 kB[0m [31m209.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.timeseries[all]==0.6.1
      Downloading autogluon.timeseries-0.6.1-py3-none-any.whl (103 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m103.0/103.0 kB[0m [31m197.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas!=1.4.0,<1.6,>=1.2.5 in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (1.3.4)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (3.5.0)
    Collecting numpy<1.24,>=1.21
      Downloading numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.7/15.7 MB[0m [31m144.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting scipy<1.10.0,>=1.5.4
      Downloading scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (38.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.1/38.1 MB[0m [31m152.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: tqdm>=4.38.0 in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (4.39.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (2.22.0)
    Collecting distributed<=2021.11.2,>=2021.09.1
      Downloading distributed-2021.11.2-py3-none-any.whl (802 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m802.2/802.2 kB[0m [31m198.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting dask<=2021.11.2,>=2021.09.1
      Downloading dask-2021.11.2-py3-none-any.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m205.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.common==0.6.1
      Downloading autogluon.common-0.6.1-py3-none-any.whl (41 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.5/41.5 kB[0m [31m90.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scikit-learn<1.2,>=1.0.0 in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (1.0.1)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.7/site-packages (from autogluon.core[all]==0.6.1->autogluon) (1.20.17)
    Collecting ray[tune]<2.1,>=2.0
      Downloading ray-2.0.1-cp37-cp37m-manylinux2014_x86_64.whl (60.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.5/60.5 MB[0m [31m161.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting hyperopt<0.2.8,>=0.2.7
      Downloading hyperopt-0.2.7-py2.py3-none-any.whl (1.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m231.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: psutil<6,>=5.7.3 in /usr/local/lib/python3.7/site-packages (from autogluon.features==0.6.1->autogluon) (5.8.0)
    Collecting fairscale<=0.4.6,>=0.4.5
      Downloading fairscale-0.4.6.tar.gz (248 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m248.2/248.2 kB[0m [31m206.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Installing backend dependencies ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hCollecting pytorch-lightning<1.8.0,>=1.7.4
      Downloading pytorch_lightning-1.7.7-py3-none-any.whl (708 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m708.1/708.1 kB[0m [31m236.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torchmetrics<0.9.0,>=0.8.0
      Downloading torchmetrics-0.8.2-py3-none-any.whl (409 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m409.8/409.8 kB[0m [31m159.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting openmim<=0.2.1,>0.1.5
      Downloading openmim-0.2.1-py2.py3-none-any.whl (49 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.7/49.7 kB[0m [31m159.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torchtext<0.14.0
      Downloading torchtext-0.13.1-cp37-cp37m-manylinux1_x86_64.whl (1.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m230.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting text-unidecode<=1.3
      Downloading text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.2/78.2 kB[0m [31m188.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting seqeval<=1.2.2
      Downloading seqeval-1.2.2.tar.gz (43 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.6/43.6 kB[0m [31m152.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting evaluate<=0.3.0
      Downloading evaluate-0.3.0-py3-none-any.whl (72 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.9/72.9 kB[0m [31m161.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nptyping<1.5.0,>=1.4.4
      Downloading nptyping-1.4.4-py3-none-any.whl (31 kB)
    Collecting omegaconf<2.2.0,>=2.1.1
      Downloading omegaconf-2.1.2-py3-none-any.whl (74 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m74.7/74.7 kB[0m [31m185.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pytorch-metric-learning<1.4.0,>=1.3.0
      Downloading pytorch_metric_learning-1.3.2-py3-none-any.whl (109 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m109.4/109.4 kB[0m [31m115.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting albumentations<=1.2.0,>=1.1.0
      Downloading albumentations-1.2.0-py3-none-any.whl (113 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m113.5/113.5 kB[0m [31m163.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting jsonschema<=4.8.0
      Downloading jsonschema-4.8.0-py3-none-any.whl (81 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.4/81.4 kB[0m [31m200.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting defusedxml<=0.7.1,>=0.7.1
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Collecting scikit-image<0.20.0,>=0.19.1
      Downloading scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.5/13.5 MB[0m [31m72.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting smart-open<5.3.0,>=5.2.1
      Downloading smart_open-5.2.1-py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.6/58.6 kB[0m [31m155.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting Pillow<=9.4.0,>=9.3.0
      Downloading Pillow-9.3.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.2/3.2 MB[0m [31m222.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting accelerate<0.14,>=0.9
      Downloading accelerate-0.13.2-py3-none-any.whl (148 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m148.8/148.8 kB[0m [31m210.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting sentencepiece<0.2.0,>=0.1.95
      Downloading sentencepiece-0.1.97-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m204.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nltk<4.0.0,>=3.4.5
      Downloading nltk-3.8-py3-none-any.whl (1.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m239.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch<1.13,>=1.9
      Downloading torch-1.12.1-cp37-cp37m-manylinux1_x86_64.whl (776.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m776.3/776.3 MB[0m [31m170.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting torchvision<0.14.0
      Downloading torchvision-0.13.1-cp37-cp37m-manylinux1_x86_64.whl (19.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m19.1/19.1 MB[0m [31m173.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting timm<0.7.0
      Downloading timm-0.6.12-py3-none-any.whl (549 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m549.1/549.1 kB[0m [31m219.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nlpaug<=1.1.10,>=1.1.10
      Downloading nlpaug-1.1.10-py3-none-any.whl (410 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.8/410.8 kB[0m [31m215.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting transformers<4.24.0,>=4.23.0
      Downloading transformers-4.23.1-py3-none-any.whl (5.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.3/5.3 MB[0m [31m198.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: networkx<3.0,>=2.3 in /usr/local/lib/python3.7/site-packages (from autogluon.tabular[all]==0.6.1->autogluon) (2.6.3)
    Collecting catboost<1.2,>=1.0
      Downloading catboost-1.1.1-cp37-none-manylinux1_x86_64.whl (76.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m76.6/76.6 MB[0m [31m178.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting lightgbm<3.4,>=3.3
      Downloading lightgbm-3.3.3-py3-none-manylinux1_x86_64.whl (2.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m245.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting xgboost<1.8,>=1.6
      Downloading xgboost-1.6.2-py3-none-manylinux2014_x86_64.whl (255.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m255.9/255.9 MB[0m [31m178.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting fastai<2.8,>=2.3.1
      Downloading fastai-2.7.10-py3-none-any.whl (240 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m240.9/240.9 kB[0m [31m113.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting gluonts~=0.11.0
      Downloading gluonts-0.11.6-py3-none-any.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m245.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: joblib~=1.1 in /usr/local/lib/python3.7/site-packages (from autogluon.timeseries[all]==0.6.1->autogluon) (1.1.0)
    Collecting statsmodels~=0.13.0
      Downloading statsmodels-0.13.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.9/9.9 MB[0m [31m167.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting tbats~=1.1
      Downloading tbats-1.1.2-py3-none-any.whl (43 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.8/43.8 kB[0m [31m152.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pmdarima~=1.8.2
      Downloading pmdarima-1.8.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (1.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m228.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting sktime<0.14,>=0.13.1
      Downloading sktime-0.13.4-py3-none-any.whl (7.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.0/7.0 MB[0m [31m169.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting gluoncv<0.10.6,>=0.10.5
      Downloading gluoncv-0.10.5.post0-py2.py3-none-any.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m246.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: setuptools in /usr/local/lib/python3.7/site-packages (from autogluon.common==0.6.1->autogluon.core[all]==0.6.1->autogluon) (65.6.3)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/site-packages (from accelerate<0.14,>=0.9->autogluon.multimodal==0.6.1->autogluon) (5.4.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/site-packages (from accelerate<0.14,>=0.9->autogluon.multimodal==0.6.1->autogluon) (21.3)
    Collecting qudida>=0.0.4
      Downloading qudida-0.0.4-py3-none-any.whl (3.5 kB)
    Collecting opencv-python-headless>=4.1.1
      Downloading opencv_python_headless-4.7.0.68-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 MB[0m [31m155.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting albumentations<=1.2.0,>=1.1.0
      Downloading albumentations-1.1.0-py3-none-any.whl (102 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.4/102.4 kB[0m [31m199.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: graphviz in /usr/local/lib/python3.7/site-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.6.1->autogluon) (0.8.4)
    Requirement already satisfied: six in /usr/local/lib/python3.7/site-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.6.1->autogluon) (1.16.0)
    Requirement already satisfied: plotly in /usr/local/lib/python3.7/site-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.6.1->autogluon) (5.4.0)
    Collecting toolz>=0.8.2
      Downloading toolz-0.12.0-py3-none-any.whl (55 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.8/55.8 kB[0m [31m167.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting partd>=0.3.10
      Downloading partd-1.3.0-py3-none-any.whl (18 kB)
    Requirement already satisfied: cloudpickle>=1.1.1 in /usr/local/lib/python3.7/site-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.6.1->autogluon) (2.0.0)
    Requirement already satisfied: fsspec>=0.6.0 in /usr/local/lib/python3.7/site-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.6.1->autogluon) (2021.11.1)
    Collecting click>=6.6
      Downloading click-8.1.3-py3-none-any.whl (96 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.6/96.6 kB[0m [31m187.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: jinja2 in /usr/local/lib/python3.7/site-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.6.1->autogluon) (3.0.3)
    Collecting zict>=0.1.3
      Downloading zict-2.2.0-py2.py3-none-any.whl (23 kB)
    Collecting msgpack>=0.6.0
      Downloading msgpack-1.0.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (299 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m299.8/299.8 kB[0m [31m211.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tornado>=5 in /usr/local/lib/python3.7/site-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.6.1->autogluon) (6.1)
    Collecting tblib>=1.6.0
      Downloading tblib-1.7.0-py2.py3-none-any.whl (12 kB)
    Collecting sortedcontainers!=2.0.0,!=2.0.1
      Downloading sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
    Collecting huggingface-hub>=0.7.0
      Downloading huggingface_hub-0.11.1-py3-none-any.whl (182 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m182.4/182.4 kB[0m [31m200.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting xxhash
      Downloading xxhash-3.2.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (213 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m213.1/213.1 kB[0m [31m231.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting datasets>=2.0.0
      Downloading datasets-2.8.0-py3-none-any.whl (452 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m452.9/452.9 kB[0m [31m241.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: dill in /usr/local/lib/python3.7/site-packages (from evaluate<=0.3.0->autogluon.multimodal==0.6.1->autogluon) (0.3.4)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.7/site-packages (from evaluate<=0.3.0->autogluon.multimodal==0.6.1->autogluon) (0.70.12.2)
    Collecting responses<0.19
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/site-packages (from evaluate<=0.3.0->autogluon.multimodal==0.6.1->autogluon) (4.8.2)
    Collecting tqdm>=4.38.0
      Downloading tqdm-4.64.1-py2.py3-none-any.whl (78 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.5/78.5 kB[0m [31m171.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting fastprogress>=0.2.4
      Downloading fastprogress-1.0.3-py3-none-any.whl (12 kB)
    Collecting fastdownload<2,>=0.0.5
      Downloading fastdownload-0.0.7-py3-none-any.whl (12 kB)
    Requirement already satisfied: pip in /usr/local/lib/python3.7/site-packages (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.6.1->autogluon) (22.3.1)
    Collecting fastcore<1.6,>=1.4.5
      Downloading fastcore-1.5.27-py3-none-any.whl (67 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.1/67.1 kB[0m [31m179.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting spacy<4
      Downloading spacy-3.4.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.4/6.4 MB[0m [31m167.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting autocfg
      Downloading autocfg-0.0.8-py3-none-any.whl (13 kB)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/site-packages (from gluoncv<0.10.6,>=0.10.5->autogluon.vision==0.6.1->autogluon) (4.5.4.60)
    Collecting yacs
      Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
    Requirement already satisfied: portalocker in /usr/local/lib/python3.7/site-packages (from gluoncv<0.10.6,>=0.10.5->autogluon.vision==0.6.1->autogluon) (2.3.2)
    Collecting pydantic~=1.7
      Downloading pydantic-1.10.3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m188.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.7/site-packages (from gluonts~=0.11.0->autogluon.timeseries[all]==0.6.1->autogluon) (4.0.1)
    Collecting py4j
      Downloading py4j-0.10.9.7-py2.py3-none-any.whl (200 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m200.5/200.5 kB[0m [31m213.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting future
      Downloading future-0.18.2.tar.gz (829 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m829.2/829.2 kB[0m [31m243.2 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting importlib-resources>=1.4.0
      Downloading importlib_resources-5.10.2-py3-none-any.whl (34 kB)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/site-packages (from jsonschema<=4.8.0->autogluon.multimodal==0.6.1->autogluon) (21.2.0)
    Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0
      Downloading pyrsistent-0.19.3-py3-none-any.whl (57 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m148.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel in /usr/local/lib/python3.7/site-packages (from lightgbm<3.4,>=3.3->autogluon.tabular[all]==0.6.1->autogluon) (0.38.4)
    Collecting regex>=2021.8.3
      Downloading regex-2022.10.31-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (757 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m757.1/757.1 kB[0m [31m239.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting typish>=1.7.0
      Downloading typish-1.9.3-py3-none-any.whl (45 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.1/45.1 kB[0m [31m134.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting antlr4-python3-runtime==4.8
      Downloading antlr4-python3-runtime-4.8.tar.gz (112 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m112.4/112.4 kB[0m [31m210.9 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting model-index
      Downloading model_index-0.1.11-py3-none-any.whl (34 kB)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.7/site-packages (from openmim<=0.2.1,>0.1.5->autogluon.multimodal==0.6.1->autogluon) (0.8.9)
    Requirement already satisfied: colorama in /usr/local/lib/python3.7/site-packages (from openmim<=0.2.1,>0.1.5->autogluon.multimodal==0.6.1->autogluon) (0.4.3)
    Collecting rich
      Downloading rich-12.6.0-py3-none-any.whl (237 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m237.5/237.5 kB[0m [31m224.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/site-packages (from pandas!=1.4.0,<1.6,>=1.2.5->autogluon.core[all]==0.6.1->autogluon) (2021.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/site-packages (from pandas!=1.4.0,<1.6,>=1.2.5->autogluon.core[all]==0.6.1->autogluon) (2.8.2)
    Requirement already satisfied: Cython!=0.29.18,>=0.29 in /usr/local/lib/python3.7/site-packages (from pmdarima~=1.8.2->autogluon.timeseries[all]==0.6.1->autogluon) (0.29.24)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/site-packages (from pmdarima~=1.8.2->autogluon.timeseries[all]==0.6.1->autogluon) (1.25.11)
    Collecting tensorboard>=2.9.1
      Downloading tensorboard-2.11.0-py3-none-any.whl (6.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m180.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pyDeprecate>=0.3.1
      Downloading pyDeprecate-0.3.2-py3-none-any.whl (10 kB)
    Collecting aiosignal
      Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Collecting grpcio<=1.43.0,>=1.32.0
      Downloading grpcio-1.43.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.1/4.1 MB[0m [31m176.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting filelock
      Downloading filelock-3.9.0-py3-none-any.whl (9.7 kB)
    Collecting click>=6.6
      Downloading click-8.0.4-py3-none-any.whl (97 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m97.5/97.5 kB[0m [31m192.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting virtualenv
      Downloading virtualenv-20.17.1-py3-none-any.whl (8.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.8/8.8 MB[0m [31m171.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: protobuf<4.0.0,>=3.15.3 in /usr/local/lib/python3.7/site-packages (from ray[tune]<2.1,>=2.0->autogluon.core[all]==0.6.1->autogluon) (3.19.1)
    Collecting frozenlist
      Downloading frozenlist-1.3.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (148 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m148.0/148.0 kB[0m [31m196.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboardX>=1.9
      Downloading tensorboardX-2.5.1-py2.py3-none-any.whl (125 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m125.4/125.4 kB[0m [31m203.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/site-packages (from requests->autogluon.core[all]==0.6.1->autogluon) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.7/site-packages (from requests->autogluon.core[all]==0.6.1->autogluon) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/site-packages (from requests->autogluon.core[all]==0.6.1->autogluon) (2021.10.8)
    Collecting PyWavelets>=1.1.1
      Downloading PyWavelets-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.4/6.4 MB[0m [31m169.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting tifffile>=2019.7.26
      Downloading tifffile-2021.11.2-py3-none-any.whl (178 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.9/178.9 kB[0m [31m201.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.7/site-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.6.1->autogluon) (2.13.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn<1.2,>=1.0.0->autogluon.core[all]==0.6.1->autogluon) (3.0.0)
    Collecting deprecated>=1.2.13
      Downloading Deprecated-1.2.13-py2.py3-none-any.whl (9.6 kB)
    Requirement already satisfied: numba>=0.53 in /usr/local/lib/python3.7/site-packages (from sktime<0.14,>=0.13.1->autogluon.timeseries[all]==0.6.1->autogluon) (0.53.1)
    Collecting patsy>=0.5.2
      Downloading patsy-0.5.3-py2.py3-none-any.whl (233 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m233.8/233.8 kB[0m [31m225.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tokenizers!=0.11.3,<0.14,>=0.11.1
      Downloading tokenizers-0.13.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.6/7.6 MB[0m [31m185.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.7/site-packages (from boto3->autogluon.core[all]==0.6.1->autogluon) (0.10.0)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /usr/local/lib/python3.7/site-packages (from boto3->autogluon.core[all]==0.6.1->autogluon) (0.5.0)
    Requirement already satisfied: botocore<1.24.0,>=1.23.17 in /usr/local/lib/python3.7/site-packages (from boto3->autogluon.core[all]==0.6.1->autogluon) (1.23.17)
    Requirement already satisfied: setuptools-scm>=4 in /usr/local/lib/python3.7/site-packages (from matplotlib->autogluon.core[all]==0.6.1->autogluon) (6.3.2)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.7/site-packages (from matplotlib->autogluon.core[all]==0.6.1->autogluon) (4.28.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/site-packages (from matplotlib->autogluon.core[all]==0.6.1->autogluon) (0.11.0)
    Requirement already satisfied: pyparsing>=2.2.1 in /usr/local/lib/python3.7/site-packages (from matplotlib->autogluon.core[all]==0.6.1->autogluon) (3.0.6)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/site-packages (from matplotlib->autogluon.core[all]==0.6.1->autogluon) (1.3.2)
    Collecting aiohttp
      Downloading aiohttp-3.8.3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (948 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m948.0/948.0 kB[0m [31m141.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.7/site-packages (from datasets>=2.0.0->evaluate<=0.3.0->autogluon.multimodal==0.6.1->autogluon) (6.0.1)
    Collecting wrapt<2,>=1.10
      Downloading wrapt-1.14.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (75 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.2/75.2 kB[0m [31m179.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/site-packages (from importlib-resources>=1.4.0->jsonschema<=4.8.0->autogluon.multimodal==0.6.1->autogluon) (3.6.0)
    Requirement already satisfied: llvmlite<0.37,>=0.36.0rc1 in /usr/local/lib/python3.7/site-packages (from numba>=0.53->sktime<0.14,>=0.13.1->autogluon.timeseries[all]==0.6.1->autogluon) (0.36.0)
    Collecting locket
      Downloading locket-1.0.0-py2.py3-none-any.whl (4.4 kB)
    Collecting typing-extensions~=4.0
      Downloading typing_extensions-4.4.0-py3-none-any.whl (26 kB)
    Requirement already satisfied: tomli>=1.0.0 in /usr/local/lib/python3.7/site-packages (from setuptools-scm>=4->matplotlib->autogluon.core[all]==0.6.1->autogluon) (1.2.2)
    Collecting catalogue<2.1.0,>=2.0.6
      Downloading catalogue-2.0.8-py3-none-any.whl (17 kB)
    Collecting langcodes<4.0.0,>=3.2.0
      Downloading langcodes-3.3.0-py3-none-any.whl (181 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.6/181.6 kB[0m [31m187.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pathy>=0.3.5
      Downloading pathy-0.10.1-py3-none-any.whl (48 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m48.9/48.9 kB[0m [31m106.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting spacy-legacy<3.1.0,>=3.0.10
      Downloading spacy_legacy-3.0.11-py2.py3-none-any.whl (24 kB)
    Collecting spacy-loggers<2.0.0,>=1.0.0
      Downloading spacy_loggers-1.0.4-py3-none-any.whl (11 kB)
    Collecting typing-extensions~=4.0
      Downloading typing_extensions-4.1.1-py3-none-any.whl (26 kB)
    Collecting preshed<3.1.0,>=3.0.2
      Downloading preshed-3.0.8-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (126 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m126.6/126.6 kB[0m [31m181.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting typer<0.8.0,>=0.3.0
      Downloading typer-0.7.0-py3-none-any.whl (38 kB)
    Collecting cymem<2.1.0,>=2.0.2
      Downloading cymem-2.0.7-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (36 kB)
    Collecting thinc<8.2.0,>=8.1.0
      Downloading thinc-8.1.6-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (814 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m814.4/814.4 kB[0m [31m238.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting murmurhash<1.1.0,>=0.28.0
      Downloading murmurhash-1.0.9-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (21 kB)
    Collecting wasabi<1.1.0,>=0.9.1
      Downloading wasabi-0.10.1-py3-none-any.whl (26 kB)
    Collecting srsly<3.0.0,>=2.4.3
      Downloading srsly-2.4.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (490 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m490.0/490.0 kB[0m [31m244.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m235.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth<3,>=1.6.3
      Downloading google_auth-2.15.0-py2.py3-none-any.whl (177 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.0/177.0 kB[0m [31m225.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.7/site-packages (from tensorboard>=2.9.1->pytorch-lightning<1.8.0,>=1.7.4->autogluon.multimodal==0.6.1->autogluon) (2.0.2)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting absl-py>=0.4
      Downloading absl_py-1.3.0-py3-none-any.whl (124 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.6/124.6 kB[0m [31m210.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboard-data-server<0.7.0,>=0.6.0
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m179.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting markdown>=2.6.8
      Downloading Markdown-3.4.1-py3-none-any.whl (93 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.3/93.3 kB[0m [31m175.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting heapdict
      Downloading HeapDict-1.0.1-py3-none-any.whl (3.9 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.7/site-packages (from jinja2->distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.6.1->autogluon) (2.0.1)
    Collecting ordered-set
      Downloading ordered_set-4.1.0-py3-none-any.whl (7.6 kB)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/site-packages (from plotly->catboost<1.2,>=1.0->autogluon.tabular[all]==0.6.1->autogluon) (8.0.1)
    Requirement already satisfied: pygments<3.0.0,>=2.6.0 in /usr/local/lib/python3.7/site-packages (from rich->openmim<=0.2.1,>0.1.5->autogluon.multimodal==0.6.1->autogluon) (2.13.0)
    Collecting commonmark<0.10.0,>=0.9.0
      Downloading commonmark-0.9.1-py2.py3-none-any.whl (51 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m51.1/51.1 kB[0m [31m170.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting importlib-metadata
      Downloading importlib_metadata-5.2.0-py3-none-any.whl (21 kB)
    Collecting platformdirs<3,>=2.4
      Downloading platformdirs-2.6.2-py3-none-any.whl (14 kB)
    Collecting distlib<1,>=0.3.6
      Downloading distlib-0.3.6-py2.py3-none-any.whl (468 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m468.5/468.5 kB[0m [31m231.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard>=2.9.1->pytorch-lightning<1.8.0,>=1.7.4->autogluon.multimodal==0.6.1->autogluon) (4.7.2)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m155.3/155.3 kB[0m [31m172.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.2.0-py3-none-any.whl (9.3 kB)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Collecting platformdirs<3,>=2.4
      Downloading platformdirs-2.6.1-py3-none-any.whl (14 kB)
    Collecting confection<1.0.0,>=0.0.1
      Downloading confection-0.0.3-py3-none-any.whl (32 kB)
    Collecting blis<0.8.0,>=0.7.8
      Downloading blis-0.7.9-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.2/10.2 MB[0m [31m176.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting yarl<2.0,>=1.0
      Downloading yarl-1.8.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (231 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m231.4/231.4 kB[0m [31m222.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting asynctest==0.13.0
      Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)
    Collecting async-timeout<5.0,>=4.0.0a3
      Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
    Collecting charset-normalizer<3.0,>=2.0
      Downloading charset_normalizer-2.1.1-py3-none-any.whl (39 kB)
    Collecting multidict<7.0,>=4.5
      Downloading multidict-6.0.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (94 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m94.8/94.8 kB[0m [31m194.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.9.1->pytorch-lightning<1.8.0,>=1.7.4->autogluon.multimodal==0.6.1->autogluon) (0.4.8)
    Collecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m212.9 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: fairscale, antlr4-python3-runtime, seqeval, future
      Building wheel for fairscale (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for fairscale: filename=fairscale-0.4.6-py3-none-any.whl size=307224 sha256=92104016db37694fc50adc9345f55c89ae93c2aed88520c65b44bb74f52d8b2e
      Stored in directory: /tmp/pip-ephem-wheel-cache-99geygey/wheels/0b/8c/fa/a9e102632bcb86e919561cf25ca1e0dd2ec67476f3a5544653
      Building wheel for antlr4-python3-runtime (setup.py) ... [?25ldone
    [?25h  Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.8-py3-none-any.whl size=141211 sha256=982d83688dd7d15163684b5ad5df6a50dde348bbbb53702bdfa62a4a22992c90
      Stored in directory: /tmp/pip-ephem-wheel-cache-99geygey/wheels/c9/ef/75/1b8c6588a8a8a15d5a9136608a9d65172a226577e7ae89da31
      Building wheel for seqeval (setup.py) ... [?25ldone
    [?25h  Created wheel for seqeval: filename=seqeval-1.2.2-py3-none-any.whl size=16164 sha256=08eec8907079291527d7c59176c84aebed5232a6b790b643cd14fb3cc3b4520f
      Stored in directory: /tmp/pip-ephem-wheel-cache-99geygey/wheels/b2/a1/b7/0d3b008d0c77cd57332d724b92cf7650b4185b493dc785f00a
      Building wheel for future (setup.py) ... [?25ldone
    [?25h  Created wheel for future: filename=future-0.18.2-py3-none-any.whl size=491059 sha256=e3b1371205a6b290d7dcb7ed74d0b31239d26ac43e96a42d6fbcf44a822e5442
      Stored in directory: /tmp/pip-ephem-wheel-cache-99geygey/wheels/3e/3c/b4/7132d27620dd551cf00823f798a7190e7320ae7ffb71d1e989
    Successfully built fairscale antlr4-python3-runtime seqeval future
    Installing collected packages: wasabi, typish, tokenizers, text-unidecode, tensorboard-plugin-wit, sortedcontainers, sentencepiece, py4j, msgpack, heapdict, distlib, cymem, commonmark, antlr4-python3-runtime, zict, yacs, xxhash, wrapt, typing-extensions, tqdm, toolz, tensorboard-data-server, tblib, spacy-loggers, spacy-legacy, smart-open, regex, pyrsistent, pyDeprecate, pyasn1-modules, platformdirs, Pillow, ordered-set, omegaconf, oauthlib, numpy, murmurhash, multidict, locket, langcodes, importlib-resources, grpcio, future, frozenlist, filelock, fastprogress, defusedxml, charset-normalizer, cachetools, autocfg, asynctest, absl-py, yarl, torch, tifffile, tensorboardX, scipy, rich, responses, requests-oauthlib, PyWavelets, pydantic, preshed, patsy, partd, opencv-python-headless, nptyping, importlib-metadata, google-auth, fastcore, deprecated, catalogue, blis, async-timeout, aiosignal, xgboost, virtualenv, torchvision, torchtext, torchmetrics, statsmodels, srsly, scikit-image, nlpaug, markdown, jsonschema, hyperopt, huggingface-hub, google-auth-oauthlib, gluonts, fastdownload, fairscale, dask, click, aiohttp, accelerate, typer, transformers, timm, tensorboard, sktime, seqeval, ray, qudida, pytorch-metric-learning, pmdarima, nltk, model-index, lightgbm, gluoncv, distributed, confection, catboost, thinc, tbats, pytorch-lightning, pathy, openmim, datasets, autogluon.common, albumentations, spacy, evaluate, autogluon.features, autogluon.core, fastai, autogluon.tabular, autogluon.multimodal, autogluon.vision, autogluon.timeseries, autogluon.text, autogluon
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.0.1
        Uninstalling typing_extensions-4.0.1:
          Successfully uninstalled typing_extensions-4.0.1
      Attempting uninstall: tqdm
        Found existing installation: tqdm 4.39.0
        Uninstalling tqdm-4.39.0:
          Successfully uninstalled tqdm-4.39.0
      Attempting uninstall: Pillow
        Found existing installation: Pillow 8.4.0
        Uninstalling Pillow-8.4.0:
          Successfully uninstalled Pillow-8.4.0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.19.1
        Uninstalling numpy-1.19.1:
          Successfully uninstalled numpy-1.19.1
      Attempting uninstall: scipy
        Found existing installation: scipy 1.4.1
        Uninstalling scipy-1.4.1:
          Successfully uninstalled scipy-1.4.1
      Attempting uninstall: importlib-metadata
        Found existing installation: importlib-metadata 4.8.2
        Uninstalling importlib-metadata-4.8.2:
          Successfully uninstalled importlib-metadata-4.8.2
      Attempting uninstall: gluoncv
        Found existing installation: gluoncv 0.8.0
        Uninstalling gluoncv-0.8.0:
          Successfully uninstalled gluoncv-0.8.0
    Successfully installed Pillow-9.3.0 PyWavelets-1.3.0 absl-py-1.3.0 accelerate-0.13.2 aiohttp-3.8.3 aiosignal-1.3.1 albumentations-1.1.0 antlr4-python3-runtime-4.8 async-timeout-4.0.2 asynctest-0.13.0 autocfg-0.0.8 autogluon-0.6.1 autogluon.common-0.6.1 autogluon.core-0.6.1 autogluon.features-0.6.1 autogluon.multimodal-0.6.1 autogluon.tabular-0.6.1 autogluon.text-0.6.1 autogluon.timeseries-0.6.1 autogluon.vision-0.6.1 blis-0.7.9 cachetools-5.2.0 catalogue-2.0.8 catboost-1.1.1 charset-normalizer-2.1.1 click-8.0.4 commonmark-0.9.1 confection-0.0.3 cymem-2.0.7 dask-2021.11.2 datasets-2.8.0 defusedxml-0.7.1 deprecated-1.2.13 distlib-0.3.6 distributed-2021.11.2 evaluate-0.3.0 fairscale-0.4.6 fastai-2.7.10 fastcore-1.5.27 fastdownload-0.0.7 fastprogress-1.0.3 filelock-3.9.0 frozenlist-1.3.3 future-0.18.2 gluoncv-0.10.5.post0 gluonts-0.11.6 google-auth-2.15.0 google-auth-oauthlib-0.4.6 grpcio-1.43.0 heapdict-1.0.1 huggingface-hub-0.11.1 hyperopt-0.2.7 importlib-metadata-5.2.0 importlib-resources-5.10.2 jsonschema-4.8.0 langcodes-3.3.0 lightgbm-3.3.3 locket-1.0.0 markdown-3.4.1 model-index-0.1.11 msgpack-1.0.4 multidict-6.0.4 murmurhash-1.0.9 nlpaug-1.1.10 nltk-3.8 nptyping-1.4.4 numpy-1.21.6 oauthlib-3.2.2 omegaconf-2.1.2 opencv-python-headless-4.7.0.68 openmim-0.2.1 ordered-set-4.1.0 partd-1.3.0 pathy-0.10.1 patsy-0.5.3 platformdirs-2.6.1 pmdarima-1.8.5 preshed-3.0.8 py4j-0.10.9.7 pyDeprecate-0.3.2 pyasn1-modules-0.2.8 pydantic-1.10.3 pyrsistent-0.19.3 pytorch-lightning-1.7.7 pytorch-metric-learning-1.3.2 qudida-0.0.4 ray-2.0.1 regex-2022.10.31 requests-oauthlib-1.3.1 responses-0.18.0 rich-12.6.0 scikit-image-0.19.3 scipy-1.7.3 sentencepiece-0.1.97 seqeval-1.2.2 sktime-0.13.4 smart-open-5.2.1 sortedcontainers-2.4.0 spacy-3.4.4 spacy-legacy-3.0.11 spacy-loggers-1.0.4 srsly-2.4.5 statsmodels-0.13.5 tbats-1.1.2 tblib-1.7.0 tensorboard-2.11.0 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorboardX-2.5.1 text-unidecode-1.3 thinc-8.1.6 tifffile-2021.11.2 timm-0.6.12 tokenizers-0.13.2 toolz-0.12.0 torch-1.12.1 torchmetrics-0.8.2 torchtext-0.13.1 torchvision-0.13.1 tqdm-4.64.1 transformers-4.23.1 typer-0.7.0 typing-extensions-4.1.1 typish-1.9.3 virtualenv-20.17.1 wasabi-0.10.1 wrapt-1.14.1 xgboost-1.6.2 xxhash-3.2.0 yacs-0.1.8 yarl-1.8.2 zict-2.2.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

### Setup Kaggle API Key


```python
# create the .kaggle directory and an empty kaggle.json file
!mkdir -p /root/.kaggle
!touch /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
```


```python
!pip install kaggle
```

    Collecting kaggle
      Using cached kaggle-1.5.12-py3-none-any.whl
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/site-packages (from kaggle) (2021.10.8)
    Collecting python-slugify
      Using cached python_slugify-7.0.0-py2.py3-none-any.whl (9.4 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/site-packages (from kaggle) (4.64.1)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/site-packages (from kaggle) (1.16.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/site-packages (from kaggle) (2.8.2)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/site-packages (from kaggle) (1.25.11)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/site-packages (from kaggle) (2.22.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/site-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.7/site-packages (from requests->kaggle) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/site-packages (from requests->kaggle) (3.0.4)
    Installing collected packages: python-slugify, kaggle
    Successfully installed kaggle-1.5.12 python-slugify-7.0.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
# Fill in your user name and key from creating the kaggle account and API token file
import json
kaggle_username = "ahmed96me"
kaggle_key = "cb87ba677a51421b27776f3146b703ed"

# Save API token the kaggle.json file
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```

### Download and explore dataset

### Go to the [bike sharing demand competition](https://www.kaggle.com/c/bike-sharing-demand) and agree to the terms
![kaggle6.png](attachment:kaggle6.png)


```python
# Download the dataset, it will be in a .zip file so you'll need to unzip it as well.
!kaggle competitions download -c bike-sharing-demand
#kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "Message"
# If you already downloaded it you can use the -o command to overwrite the file
!unzip -o bike-sharing-demand.zip
```

    bike-sharing-demand.zip: Skipping, found more recently modified local copy (use --force to force download)
    Archive:  bike-sharing-demand.zip
      inflating: sampleSubmission.csv    
      inflating: test.csv                
      inflating: train.csv               



```python
import pandas as pd
from autogluon.tabular import TabularPredictor
```

    /usr/local/lib/python3.7/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
# Create the train dataset in pandas by reading the csv
# Set the parsing of the datetime column so you can use some of the `dt` features in pandas later
#train = pd.read_csv('s3://sagemaker-studio-dsy6kv0wc3r/train.csv')
train = pd.read_csv('train.csv')
train.loc[:, "datetime"] = pd.to_datetime(train.loc[:, "datetime"])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Simple output of the train dataset to view some of the min/max/varition of the dataset features.
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.00000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.506614</td>
      <td>0.028569</td>
      <td>0.680875</td>
      <td>1.418427</td>
      <td>20.23086</td>
      <td>23.655084</td>
      <td>61.886460</td>
      <td>12.799395</td>
      <td>36.021955</td>
      <td>155.552177</td>
      <td>191.574132</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.116174</td>
      <td>0.166599</td>
      <td>0.466159</td>
      <td>0.633839</td>
      <td>7.79159</td>
      <td>8.474601</td>
      <td>19.245033</td>
      <td>8.164537</td>
      <td>49.960477</td>
      <td>151.039033</td>
      <td>181.144454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.82000</td>
      <td>0.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.94000</td>
      <td>16.665000</td>
      <td>47.000000</td>
      <td>7.001500</td>
      <td>4.000000</td>
      <td>36.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.50000</td>
      <td>24.240000</td>
      <td>62.000000</td>
      <td>12.998000</td>
      <td>17.000000</td>
      <td>118.000000</td>
      <td>145.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>26.24000</td>
      <td>31.060000</td>
      <td>77.000000</td>
      <td>16.997900</td>
      <td>49.000000</td>
      <td>222.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>41.00000</td>
      <td>45.455000</td>
      <td>100.000000</td>
      <td>56.996900</td>
      <td>367.000000</td>
      <td>886.000000</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create the test pandas dataframe in pandas by reading the csv, remember to parse the datetime!
#test = pd.read_csv('s3://sagemaker-studio-dsy6kv0wc3r/test.csv')
test = pd.read_csv('test.csv')
test.loc[:, "datetime"] = pd.to_datetime(test.loc[:, "datetime"])
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
      <td>6493.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.493300</td>
      <td>0.029108</td>
      <td>0.685815</td>
      <td>1.436778</td>
      <td>20.620607</td>
      <td>24.012865</td>
      <td>64.125212</td>
      <td>12.631157</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.091258</td>
      <td>0.168123</td>
      <td>0.464226</td>
      <td>0.648390</td>
      <td>8.059583</td>
      <td>8.782741</td>
      <td>19.293391</td>
      <td>8.250151</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.820000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.940000</td>
      <td>16.665000</td>
      <td>49.000000</td>
      <td>7.001500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>21.320000</td>
      <td>25.000000</td>
      <td>65.000000</td>
      <td>11.001400</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>27.060000</td>
      <td>31.060000</td>
      <td>81.000000</td>
      <td>16.997900</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>40.180000</td>
      <td>50.000000</td>
      <td>100.000000</td>
      <td>55.998600</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Same thing as train and test dataset
#submission = pd.read_csv('s3://sagemaker-studio-dsy6kv0wc3r/sampleSubmission.csv')
submission = pd.read_csv('sampleSubmission.csv')
submission.loc[:, "datetime"] = pd.to_datetime(submission.loc[:, "datetime"])
submission.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   datetime  6493 non-null   datetime64[ns]
     1   count     6493 non-null   int64         
    dtypes: datetime64[ns](1), int64(1)
    memory usage: 101.6 KB



```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB



```python
train.drop(['casual','registered'], axis=1,inplace=True)
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 10 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(6)
    memory usage: 850.6 KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    6493 non-null   datetime64[ns]
     1   season      6493 non-null   int64         
     2   holiday     6493 non-null   int64         
     3   workingday  6493 non-null   int64         
     4   weather     6493 non-null   int64         
     5   temp        6493 non-null   float64       
     6   atemp       6493 non-null   float64       
     7   humidity    6493 non-null   int64         
     8   windspeed   6493 non-null   float64       
    dtypes: datetime64[ns](1), float64(3), int64(5)
    memory usage: 456.7 KB



```python

mix=pd.merge(test, submission, on="datetime")
mix.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6493 entries, 0 to 6492
    Data columns (total 10 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    6493 non-null   datetime64[ns]
     1   season      6493 non-null   int64         
     2   holiday     6493 non-null   int64         
     3   workingday  6493 non-null   int64         
     4   weather     6493 non-null   int64         
     5   temp        6493 non-null   float64       
     6   atemp       6493 non-null   float64       
     7   humidity    6493 non-null   int64         
     8   windspeed   6493 non-null   float64       
     9   count       6493 non-null   int64         
    dtypes: datetime64[ns](1), float64(3), int64(6)
    memory usage: 558.0 KB


## Step 3: Train a model using AutoGluon’s Tabular Prediction

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.


```python
predictor = TabularPredictor(label='count' , eval_metric="rmse" ).fit(train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221230_075307/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221230_075307/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 9
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    3061.09 MB
    	Train Data (Original)  Memory Usage: 0.78 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['season', 'holiday', 'workingday', 'weather', 'humidity']
    	Types of features in processed data (raw dtype, special dtypes):
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 3 | ['season', 'weather', 'humidity']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.4s = Fit runtime
    	9 features in original data used to generate 13 features in processed data.
    	Train Data (Processed) Memory Usage: 0.98 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.41s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 399.62s of the 599.58s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.05s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 395.8s of the 595.76s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 395.43s of the 595.39s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    2022-12-30 07:53:14,659	ERROR services.py:1365 -- Failed to start the dashboard: Failed to start the dashboard, return code 1
     The last 10 lines of /tmp/ray/session_2022-12-30_07-53-13_154292_41/logs/dashboard.log:
        from ray.dashboard.modules.job.sdk import JobSubmissionClient
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/sdk.py", line 8, in <module>
        from ray.dashboard.modules.job.pydantic_models import (
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/pydantic_models.py", line 4, in <module>
        from pydantic import BaseModel, Field
      File "pydantic/__init__.py", line 2, in init pydantic.__init__
      File "pydantic/dataclasses.py", line 46, in init pydantic.dataclasses
      File "pydantic/main.py", line 121, in init pydantic.main
    TypeError: dataclass_transform() got an unexpected keyword argument 'field_specifiers'
    2022-12-30 07:53:14,663	ERROR services.py:1366 -- Failed to start the dashboard, return code 1
     The last 10 lines of /tmp/ray/session_2022-12-30_07-53-13_154292_41/logs/dashboard.log:
        from ray.dashboard.modules.job.sdk import JobSubmissionClient
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/sdk.py", line 8, in <module>
        from ray.dashboard.modules.job.pydantic_models import (
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/pydantic_models.py", line 4, in <module>
        from pydantic import BaseModel, Field
      File "pydantic/__init__.py", line 2, in init pydantic.__init__
      File "pydantic/dataclasses.py", line 46, in init pydantic.dataclasses
      File "pydantic/main.py", line 121, in init pydantic.main
    TypeError: dataclass_transform() got an unexpected keyword argument 'field_specifiers'
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/ray/_private/services.py", line 1352, in start_dashboard
        raise Exception(err_msg + last_log_str)
    Exception: Failed to start the dashboard, return code 1
     The last 10 lines of /tmp/ray/session_2022-12-30_07-53-13_154292_41/logs/dashboard.log:
        from ray.dashboard.modules.job.sdk import JobSubmissionClient
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/sdk.py", line 8, in <module>
        from ray.dashboard.modules.job.pydantic_models import (
      File "/usr/local/lib/python3.7/site-packages/ray/dashboard/modules/job/pydantic_models.py", line 4, in <module>
        from pydantic import BaseModel, Field
      File "pydantic/__init__.py", line 2, in init pydantic.__init__
      File "pydantic/dataclasses.py", line 46, in init pydantic.dataclasses
      File "pydantic/main.py", line 121, in init pydantic.main
    TypeError: dataclass_transform() got an unexpected keyword argument 'field_specifiers'
    	Warning: Exception caused LightGBMXT_BAG_L1 to fail during training... Skipping this model.
    		The task's local raylet died. Check raylet.out for more information.
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 536, in after_all_folds_scheduled
        raise processed_exception
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 504, in after_all_folds_scheduled
        time_end_fit, predict_time, predict_1_time = self.ray.get(finished)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2282, in get
        raise value
    ray.exceptions.LocalRayletDiedError: The task's local raylet died. Check raylet.out for more information.
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 391.48s of the 591.44s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 391.41s of the 591.37s of remaining time.
    	-116.5443	 = Validation score   (-root_mean_squared_error)
    	10.15s	 = Training   runtime
    	0.54s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 377.96s of the 577.92s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 377.9s of the 577.86s of remaining time.
    	-124.5881	 = Validation score   (-root_mean_squared_error)
    	4.64s	 = Training   runtime
    	0.51s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 366.59s of the 566.55s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetFastAI_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: XGBoost_BAG_L1 ... Training model for up to 366.52s of the 566.48s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused XGBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 366.45s of the 566.41s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetTorch_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBMLarge_BAG_L1 ... Training model for up to 366.38s of the 566.34s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMLarge_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 566.21s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.34s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 565.8s of the 565.79s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMXT_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 565.73s of the 565.72s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 565.66s of the 565.65s of remaining time.
    	-53.2203	 = Validation score   (-root_mean_squared_error)
    	19.41s	 = Training   runtime
    	0.69s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 543.01s of the 543.0s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTreesMSE_BAG_L2 ... Training model for up to 542.95s of the 542.93s of remaining time.
    	-53.1571	 = Validation score   (-root_mean_squared_error)
    	6.96s	 = Training   runtime
    	0.59s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L2 ... Training model for up to 532.79s of the 532.78s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetFastAI_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: XGBoost_BAG_L2 ... Training model for up to 532.71s of the 532.7s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused XGBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 532.59s of the 532.58s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetTorch_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBMLarge_BAG_L2 ... Training model for up to 532.51s of the 532.5s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMLarge_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 532.38s of remaining time.
    	-52.5564	 = Validation score   (-root_mean_squared_error)
    	0.22s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 68.05s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221230_075307/")


### Review AutoGluon's training run with ranking of models that did the best.


```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                        model   score_val  pred_time_val   fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0     WeightedEnsemble_L3  -52.556431       2.549327  41.458223                0.000902           0.216571            3       True          8
    1    ExtraTreesMSE_BAG_L2  -53.157140       1.855971  21.832484                0.593848           6.959294            2       True          7
    2  RandomForestMSE_BAG_L2  -53.220266       1.954576  34.282358                0.692454          19.409167            2       True          6
    3   KNeighborsDist_BAG_L1  -84.125061       0.103515   0.028849                0.103515           0.028849            1       True          2
    4     WeightedEnsemble_L2  -84.125061       0.104251   0.367054                0.000736           0.338206            2       True          5
    5   KNeighborsUnif_BAG_L1 -101.546199       0.103862   0.054713                0.103862           0.054713            1       True          1
    6  RandomForestMSE_BAG_L1 -116.544294       0.541335  10.149008                0.541335          10.149008            1       True          3
    7    ExtraTreesMSE_BAG_L1 -124.588053       0.513411   4.640622                0.513411           4.640622            1       True          4
    Number of models trained: 8
    Types of models trained:
    {'StackerEnsembleModel_KNN', 'WeightedEnsembleModel', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_RF'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 3 | ['season', 'weather', 'humidity']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20221230_075307/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'ExtraTreesMSE_BAG_L2': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'RandomForestMSE_BAG_L1': -116.54429428704391,
      'ExtraTreesMSE_BAG_L1': -124.58805258915959,
      'WeightedEnsemble_L2': -84.12506123181602,
      'RandomForestMSE_BAG_L2': -53.22026566077818,
      'ExtraTreesMSE_BAG_L2': -53.157139677682935,
      'WeightedEnsemble_L3': -52.55643086621058},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20221230_075307/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20221230_075307/models/KNeighborsDist_BAG_L1/',
      'RandomForestMSE_BAG_L1': 'AutogluonModels/ag-20221230_075307/models/RandomForestMSE_BAG_L1/',
      'ExtraTreesMSE_BAG_L1': 'AutogluonModels/ag-20221230_075307/models/ExtraTreesMSE_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20221230_075307/models/WeightedEnsemble_L2/',
      'RandomForestMSE_BAG_L2': 'AutogluonModels/ag-20221230_075307/models/RandomForestMSE_BAG_L2/',
      'ExtraTreesMSE_BAG_L2': 'AutogluonModels/ag-20221230_075307/models/ExtraTreesMSE_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20221230_075307/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.054712772369384766,
      'KNeighborsDist_BAG_L1': 0.028848648071289062,
      'RandomForestMSE_BAG_L1': 10.149007558822632,
      'ExtraTreesMSE_BAG_L1': 4.640621662139893,
      'WeightedEnsemble_L2': 0.33820557594299316,
      'RandomForestMSE_BAG_L2': 19.409167289733887,
      'ExtraTreesMSE_BAG_L2': 6.959293842315674,
      'WeightedEnsemble_L3': 0.21657133102416992},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.10386157035827637,
      'KNeighborsDist_BAG_L1': 0.1035149097442627,
      'RandomForestMSE_BAG_L1': 0.5413353443145752,
      'ExtraTreesMSE_BAG_L1': 0.5134110450744629,
      'WeightedEnsemble_L2': 0.0007357597351074219,
      'RandomForestMSE_BAG_L2': 0.6924536228179932,
      'ExtraTreesMSE_BAG_L2': 0.593848466873169,
      'WeightedEnsemble_L3': 0.0009024143218994141},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                     model   score_val  pred_time_val   fit_time  \
     0     WeightedEnsemble_L3  -52.556431       2.549327  41.458223   
     1    ExtraTreesMSE_BAG_L2  -53.157140       1.855971  21.832484   
     2  RandomForestMSE_BAG_L2  -53.220266       1.954576  34.282358   
     3   KNeighborsDist_BAG_L1  -84.125061       0.103515   0.028849   
     4     WeightedEnsemble_L2  -84.125061       0.104251   0.367054   
     5   KNeighborsUnif_BAG_L1 -101.546199       0.103862   0.054713   
     6  RandomForestMSE_BAG_L1 -116.544294       0.541335  10.149008   
     7    ExtraTreesMSE_BAG_L1 -124.588053       0.513411   4.640622   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000902           0.216571            3       True   
     1                0.593848           6.959294            2       True   
     2                0.692454          19.409167            2       True   
     3                0.103515           0.028849            1       True   
     4                0.000736           0.338206            2       True   
     5                0.103862           0.054713            1       True   
     6                0.541335          10.149008            1       True   
     7                0.513411           4.640622            1       True   
     
        fit_order  
     0          8  
     1          7  
     2          6  
     3          2  
     4          5  
     5          1  
     6          3  
     7          4  }



### Create predictions from test dataset


```python
predictions = predictor.predict(test)
predictions.head()
```




    0    22.213421
    1    40.962395
    2    44.579826
    3    47.908035
    4    51.491283
    Name: count, dtype: float32



#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0.


```python
# Describe the `predictions` series to see if there are any negative values
predictions.describe()
```




    count    6493.000000
    mean       99.999901
    std        90.366615
    min         2.169402
    25%        18.610256
    50%        63.247437
    75%       169.233765
    max       361.222229
    Name: count, dtype: float64




```python
# How many negative values do we have?

```


```python
# Set them to zero

```

### Set predictions to submission dataframe, save, and submit


```python
submission["count"] =predictions
submission.to_csv("submission.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "first raw submission"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 367kB/s]
    Successfully submitted to Bike Sharing Demand

#### View submission via the command line or in the web browser under the competition's page - `My Submissions`


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                        status    publicScore  privateScore  
    ---------------------------  -------------------  ---------------------------------  --------  -----------  ------------  
    submission.csv               2022-12-30 07:54:24  first raw submission               complete  1.84484      1.84484       
    submission_new_hpo.csv       2022-12-22 10:14:53  new features with hyperparameters  complete  0.46822      0.46822       
    submission_new_hpo.csv       2022-12-22 10:02:33  new features with hyperparameters  complete  0.54460      0.54460       
    submission_new_hpo.csv       2022-12-22 09:49:48  new features with hyperparameters  complete  1.86283      1.86283       


#### Initial score of `?`

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.


```python
# Create a histogram of all features to show the distribution of each one relative to the data. This is part of the exploritory data analysis
train.hist()
```




    array([[<AxesSubplot:title={'center':'datetime'}>,
            <AxesSubplot:title={'center':'season'}>,
            <AxesSubplot:title={'center':'holiday'}>],
           [<AxesSubplot:title={'center':'workingday'}>,
            <AxesSubplot:title={'center':'weather'}>,
            <AxesSubplot:title={'center':'temp'}>],
           [<AxesSubplot:title={'center':'atemp'}>,
            <AxesSubplot:title={'center':'humidity'}>,
            <AxesSubplot:title={'center':'windspeed'}>],
           [<AxesSubplot:title={'center':'count'}>, <AxesSubplot:>,
            <AxesSubplot:>]], dtype=object)




    
![png](output_47_1.png)
    



```python
# create a new feature
train_new_features=train
train_new_features["month"] = train_new_features.datetime.dt.month
train_new_features["day"] = train_new_features.datetime.dt.day
train_new_features["hour"] = train_new_features.datetime.dt.hour
test_new_features=test
test_new_features["month"] = test_new_features.datetime.dt.month
test_new_features["day"] = test_new_features.datetime.dt.day
test_new_features["hour"] = test_new_features.datetime.dt.hour
```


```python
submission_new_features=submission
```


```python
#submission_new_features["month"] = submission_new_features.datetime.dt.month
#submission_new_features["day"] = submission_new_features.datetime.dt.day
#submission_new_features["hour"] = submission_new_features.datetime.dt.hour
```

## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.


```python
train_new_features.loc[:, "season"] = train_new_features["season"].astype("category")
train_new_features.loc[:, "weather"] = train_new_features["weather"].astype("category")
test_new_features.loc[:, "season"] = test_new_features["season"].astype("category")
test_new_features.loc[:, "weather"] = test_new_features["weather"].astype("category")

```


```python
# View are new feature
train_new_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>count</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_new_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>1</td>
      <td>20</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_new_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   datetime  6493 non-null   datetime64[ns]
     1   count     6493 non-null   float32       
    dtypes: datetime64[ns](1), float32(1)
    memory usage: 76.2 KB



```python
train_new_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 13 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  category      
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  category      
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   count       10886 non-null  int64         
     10  month       10886 non-null  int64         
     11  day         10886 non-null  int64         
     12  hour        10886 non-null  int64         
    dtypes: category(2), datetime64[ns](1), float64(3), int64(7)
    memory usage: 957.3 KB



```python
test_new_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    6493 non-null   datetime64[ns]
     1   season      6493 non-null   category      
     2   holiday     6493 non-null   int64         
     3   workingday  6493 non-null   int64         
     4   weather     6493 non-null   category      
     5   temp        6493 non-null   float64       
     6   atemp       6493 non-null   float64       
     7   humidity    6493 non-null   int64         
     8   windspeed   6493 non-null   float64       
     9   month       6493 non-null   int64         
     10  day         6493 non-null   int64         
     11  hour        6493 non-null   int64         
    dtypes: category(2), datetime64[ns](1), float64(3), int64(6)
    memory usage: 520.5 KB



```python
# View histogram of all features again now with the hour feature
train_new_features.hist()
```




    array([[<AxesSubplot:title={'center':'datetime'}>,
            <AxesSubplot:title={'center':'holiday'}>,
            <AxesSubplot:title={'center':'workingday'}>],
           [<AxesSubplot:title={'center':'temp'}>,
            <AxesSubplot:title={'center':'atemp'}>,
            <AxesSubplot:title={'center':'humidity'}>],
           [<AxesSubplot:title={'center':'windspeed'}>,
            <AxesSubplot:title={'center':'count'}>,
            <AxesSubplot:title={'center':'month'}>],
           [<AxesSubplot:title={'center':'day'}>,
            <AxesSubplot:title={'center':'hour'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](output_58_1.png)
    



```python
train_new_features.isnull().sum().sum()
```




    0




```python
test_new_features.isnull().sum().sum()
```




    0




```python
submission_new_features.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>22.213421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>40.962395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>44.579826</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>47.908035</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>51.491283</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-01-20 05:00:00</td>
      <td>51.681370</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-01-20 06:00:00</td>
      <td>51.855644</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2011-01-20 07:00:00</td>
      <td>53.202908</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011-01-20 08:00:00</td>
      <td>56.181454</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2011-01-20 09:00:00</td>
      <td>59.535728</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2011-01-20 10:00:00</td>
      <td>59.645302</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2011-01-20 11:00:00</td>
      <td>60.441795</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2011-01-20 12:00:00</td>
      <td>60.463760</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2011-01-20 13:00:00</td>
      <td>60.763252</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2011-01-20 14:00:00</td>
      <td>60.571030</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2011-01-20 15:00:00</td>
      <td>62.070168</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2011-01-20 16:00:00</td>
      <td>61.264107</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2011-01-20 17:00:00</td>
      <td>61.363167</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2011-01-20 18:00:00</td>
      <td>64.355560</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2011-01-20 19:00:00</td>
      <td>61.799404</td>
    </tr>
  </tbody>
</table>
</div>



## Step 5: Rerun the model with the same settings as before, just with more features


```python
predictor_new_features = TabularPredictor(label='count' , eval_metric="rmse" ).fit(train_data= train_new_features ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221230_075431/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221230_075431/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2305.64 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.2s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.27s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 399.72s of the 599.73s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 399.35s of the 599.36s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 398.99s of the 599.0s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMXT_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 398.92s of the 598.93s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 398.85s of the 598.86s of remaining time.
    	-38.3878	 = Validation score   (-root_mean_squared_error)
    	13.06s	 = Training   runtime
    	0.57s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 382.7s of the 582.71s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 382.62s of the 582.63s of remaining time.
    	-38.3981	 = Validation score   (-root_mean_squared_error)
    	5.55s	 = Training   runtime
    	0.54s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 373.98s of the 573.99s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetFastAI_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: XGBoost_BAG_L1 ... Training model for up to 373.92s of the 573.93s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused XGBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 373.85s of the 573.86s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetTorch_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBMLarge_BAG_L1 ... Training model for up to 373.79s of the 573.79s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMLarge_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 573.68s of remaining time.
    	-37.0541	 = Validation score   (-root_mean_squared_error)
    	0.33s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 573.27s of the 573.25s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMXT_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 573.18s of the 573.17s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 573.11s of the 573.1s of remaining time.
    	-34.2916	 = Validation score   (-root_mean_squared_error)
    	22.48s	 = Training   runtime
    	0.61s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 547.58s of the 547.57s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTreesMSE_BAG_L2 ... Training model for up to 547.48s of the 547.47s of remaining time.
    	-33.744	 = Validation score   (-root_mean_squared_error)
    	7.55s	 = Training   runtime
    	0.57s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L2 ... Training model for up to 536.84s of the 536.83s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetFastAI_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: XGBoost_BAG_L2 ... Training model for up to 536.76s of the 536.75s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused XGBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 536.69s of the 536.68s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused NeuralNetTorch_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: LightGBMLarge_BAG_L2 ... Training model for up to 536.62s of the 536.61s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBMLarge_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 536.48s of remaining time.
    	-33.7024	 = Validation score   (-root_mean_squared_error)
    	0.19s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 63.9s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221230_075431/")



```python
predictor_new_features.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                        model   score_val  pred_time_val   fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0     WeightedEnsemble_L3  -33.702449       2.497470  48.900362                0.000731           0.188344            3       True          8
    1    ExtraTreesMSE_BAG_L2  -33.744046       1.888228  26.228769                0.574589           7.550521            2       True          7
    2  RandomForestMSE_BAG_L2  -34.291597       1.922151  41.161496                0.608511          22.483248            2       True          6
    3     WeightedEnsemble_L2  -37.054135       1.211599  18.973604                0.000758           0.327929            2       True          5
    4  RandomForestMSE_BAG_L1  -38.387847       0.570624  13.064379                0.570624          13.064379            1       True          3
    5    ExtraTreesMSE_BAG_L1  -38.398087       0.536531   5.549037                0.536531           5.549037            1       True          4
    6   KNeighborsDist_BAG_L1  -84.125061       0.103687   0.032259                0.103687           0.032259            1       True          2
    7   KNeighborsUnif_BAG_L1 -101.546199       0.102798   0.032573                0.102798           0.032573            1       True          1
    Number of models trained: 8
    Types of models trained:
    {'StackerEnsembleModel_KNN', 'WeightedEnsembleModel', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_RF'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])             : 2 | ['season', 'weather']
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20221230_075431/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'ExtraTreesMSE_BAG_L2': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'RandomForestMSE_BAG_L1': -38.387846615896144,
      'ExtraTreesMSE_BAG_L1': -38.39808749149808,
      'WeightedEnsemble_L2': -37.05413463194309,
      'RandomForestMSE_BAG_L2': -34.29159696679107,
      'ExtraTreesMSE_BAG_L2': -33.74404640596902,
      'WeightedEnsemble_L3': -33.7024492496171},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20221230_075431/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20221230_075431/models/KNeighborsDist_BAG_L1/',
      'RandomForestMSE_BAG_L1': 'AutogluonModels/ag-20221230_075431/models/RandomForestMSE_BAG_L1/',
      'ExtraTreesMSE_BAG_L1': 'AutogluonModels/ag-20221230_075431/models/ExtraTreesMSE_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20221230_075431/models/WeightedEnsemble_L2/',
      'RandomForestMSE_BAG_L2': 'AutogluonModels/ag-20221230_075431/models/RandomForestMSE_BAG_L2/',
      'ExtraTreesMSE_BAG_L2': 'AutogluonModels/ag-20221230_075431/models/ExtraTreesMSE_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20221230_075431/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.03257298469543457,
      'KNeighborsDist_BAG_L1': 0.032259225845336914,
      'RandomForestMSE_BAG_L1': 13.06437873840332,
      'ExtraTreesMSE_BAG_L1': 5.549036741256714,
      'WeightedEnsemble_L2': 0.3279290199279785,
      'RandomForestMSE_BAG_L2': 22.483248472213745,
      'ExtraTreesMSE_BAG_L2': 7.550521373748779,
      'WeightedEnsemble_L3': 0.18834400177001953},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.1027984619140625,
      'KNeighborsDist_BAG_L1': 0.10368680953979492,
      'RandomForestMSE_BAG_L1': 0.5706236362457275,
      'ExtraTreesMSE_BAG_L1': 0.5365309715270996,
      'WeightedEnsemble_L2': 0.0007579326629638672,
      'RandomForestMSE_BAG_L2': 0.6085109710693359,
      'ExtraTreesMSE_BAG_L2': 0.5745885372161865,
      'WeightedEnsemble_L3': 0.0007305145263671875},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'ExtraTreesMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                     model   score_val  pred_time_val   fit_time  \
     0     WeightedEnsemble_L3  -33.702449       2.497470  48.900362   
     1    ExtraTreesMSE_BAG_L2  -33.744046       1.888228  26.228769   
     2  RandomForestMSE_BAG_L2  -34.291597       1.922151  41.161496   
     3     WeightedEnsemble_L2  -37.054135       1.211599  18.973604   
     4  RandomForestMSE_BAG_L1  -38.387847       0.570624  13.064379   
     5    ExtraTreesMSE_BAG_L1  -38.398087       0.536531   5.549037   
     6   KNeighborsDist_BAG_L1  -84.125061       0.103687   0.032259   
     7   KNeighborsUnif_BAG_L1 -101.546199       0.102798   0.032573   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000731           0.188344            3       True   
     1                0.574589           7.550521            2       True   
     2                0.608511          22.483248            2       True   
     3                0.000758           0.327929            2       True   
     4                0.570624          13.064379            1       True   
     5                0.536531           5.549037            1       True   
     6                0.103687           0.032259            1       True   
     7                0.102798           0.032573            1       True   
     
        fit_order  
     0          8  
     1          7  
     2          6  
     3          5  
     4          3  
     5          4  
     6          2  
     7          1  }




```python
# Remember to set all negative values to zero
prediction_new_features = predictor_new_features.predict(test_new_features)
prediction_new_features.head()
```




    0    17.280388
    1    13.691551
    2    11.677364
    3     4.801706
    4     3.156667
    Name: count, dtype: float32




```python
prediction_new_features.describe()
```




    count    6493.000000
    mean      166.698441
    std       146.093704
    min         2.200698
    25%        50.898293
    50%       133.619858
    75%       238.238068
    max       845.413879
    Name: count, dtype: float64




```python
# Same submitting predictions
submission_new_features["count"] = prediction_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "new features"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 298kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                        status    publicScore  privateScore  
    ---------------------------  -------------------  ---------------------------------  --------  -----------  ------------  
    submission_new_features.csv  2022-12-30 07:55:40  new features                       complete  0.65113      0.65113       
    submission.csv               2022-12-30 07:54:24  first raw submission               complete  1.84484      1.84484       
    submission_new_hpo.csv       2022-12-22 10:14:53  new features with hyperparameters  complete  0.46822      0.46822       
    submission_new_hpo.csv       2022-12-22 10:02:33  new features with hyperparameters  complete  0.54460      0.54460       


#### New Score of `?`

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.


```python
predictor_new_hpo1 = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'GBM': {'num_boost_round': 10000}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221222_092509/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221222_092509/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2086.92 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.4s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.5s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.57s of the 599.5s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-33.9169	 = Validation score   (-root_mean_squared_error)
    	45.25s	 = Training   runtime
    	2.97s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 351.26s of the 551.19s of remaining time.
    	Warning: Exception caused LightGBM_BAG_L1 to fail during training... Skipping this model.
    		'NoneType' object has no attribute '_user_params_aux'
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 688, in fit
        kwargs = self._preprocess_fit_args(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 428, in _preprocess_fit_args
        kwargs = self._preprocess_fit_resources(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 539, in _preprocess_fit_resources
        k_fold=k_fold
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 488, in _process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble
        user_specified_model_level_resource = self.model_base._user_params_aux.get(resource_type, None)
    AttributeError: 'NoneType' object has no attribute '_user_params_aux'
    Completed 2/20 k-fold bagging repeats ...
    No base models to train on, skipping auxiliary stack level 2...
    No base models to train on, skipping stack level 2...
    No base models to train on, skipping auxiliary stack level 3...
    AutoGluon training complete, total runtime = 48.99s ... Best model: "None"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221222_092509/")



```python
predictor_new_hpo1_ = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'GBM': {'num_boost_round': 10000}}, train_data= train ,time_limit=600 ,presets="best_quality")
```


```python
predictor_new_hpo2 = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'CAT': {'iterations': 10000}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221222_093331/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221222_093331/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    1976.48 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.2s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.26s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 1 L1 models ...
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 399.73s of the 599.74s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-33.5198	 = Validation score   (-root_mean_squared_error)
    	333.68s	 = Training   runtime
    	0.25s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 261.87s of remaining time.
    	-33.5198	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 1 L2 models ...
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 261.78s of the 261.78s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-33.7082	 = Validation score   (-root_mean_squared_error)
    	43.55s	 = Training   runtime
    	0.1s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 213.4s of the 213.39s of remaining time.
    	Warning: Exception caused CatBoost_BAG_L2 to fail during training... Skipping this model.
    		'NoneType' object has no attribute '_user_params_aux'
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 688, in fit
        kwargs = self._preprocess_fit_args(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 428, in _preprocess_fit_args
        kwargs = self._preprocess_fit_resources(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 539, in _preprocess_fit_resources
        k_fold=k_fold
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 488, in _process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble
        user_specified_model_level_resource = self.model_base._user_params_aux.get(resource_type, None)
    AttributeError: 'NoneType' object has no attribute '_user_params_aux'
    Completed 2/20 k-fold bagging repeats ...
    No base models to train on, skipping auxiliary stack level 3...
    AutoGluon training complete, total runtime = 386.77s ... Best model: "WeightedEnsemble_L2"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221222_093331/")



```python
predictor_new_hpo2_ = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'CAT': {'iterations': 10000}}, train_data= train ,time_limit=600 ,presets="best_quality")
```


```python
predictor_new_hpo3 = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'RF': {'n_estimators': 300}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221222_094318/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221222_094318/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2009.01 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.3s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.39s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 1 L1 models ...
    Fitting model: RandomForest_BAG_L1 ... Training model for up to 399.64s of the 599.6s of remaining time.
    	-38.3878	 = Validation score   (-root_mean_squared_error)
    	13.39s	 = Training   runtime
    	0.57s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 583.09s of remaining time.
    	-38.3878	 = Validation score   (-root_mean_squared_error)
    	0.0s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 1 L2 models ...
    Fitting model: RandomForest_BAG_L2 ... Training model for up to 582.99s of the 582.98s of remaining time.
    	-39.5809	 = Validation score   (-root_mean_squared_error)
    	17.41s	 = Training   runtime
    	0.6s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 562.48s of remaining time.
    	-39.5809	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 37.73s ... Best model: "WeightedEnsemble_L2"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221222_094318/")



```python
predictor_new_hpo3_ = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'RF': {'n_estimators': 300}}, train_data= train ,time_limit=600 ,presets="best_quality")
```


```python
predictor_new_hpo4 = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'XT': {'n_estimators': 300}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221222_094554/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221222_094554/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2006.27 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.3s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.4s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 1 L1 models ...
    Fitting model: ExtraTrees_BAG_L1 ... Training model for up to 399.63s of the 599.59s of remaining time.
    	-38.3981	 = Validation score   (-root_mean_squared_error)
    	6.01s	 = Training   runtime
    	0.64s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 590.03s of remaining time.
    	-38.3981	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 1 L2 models ...
    Fitting model: ExtraTrees_BAG_L2 ... Training model for up to 589.95s of the 589.94s of remaining time.
    	-37.695	 = Validation score   (-root_mean_squared_error)
    	6.99s	 = Training   runtime
    	0.6s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 579.73s of remaining time.
    	-37.695	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 20.49s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221222_094554/")



```python
#bad one
predictor_new_hpo = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'KNN': {}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221222_094931/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221222_094931/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2015.72 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.3s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.35s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 1 L1 models ...
    Fitting model: KNeighbors_BAG_L1 ... Training model for up to 399.67s of the 599.64s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.06s	 = Training   runtime
    	0.1s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 599.17s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.0s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 0 L2 models ...
    Completed 1/20 k-fold bagging repeats ...
    No base models to train on, skipping auxiliary stack level 3...
    AutoGluon training complete, total runtime = 1.06s ... Best model: "WeightedEnsemble_L2"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221222_094931/")



```python
predictor_new_hpo5 = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'GBM': {'num_boost_round': 10000},'CAT': {'iterations': 10000},'RF': {'n_estimators': 300},'XT': {'n_estimators': 300}}, train_data= train ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221230_075542/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221230_075542/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2363.97 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.2s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.21s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 4 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.76s of the 599.79s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForest_BAG_L1 ... Training model for up to 399.68s of the 599.71s of remaining time.
    	-38.3878	 = Validation score   (-root_mean_squared_error)
    	13.06s	 = Training   runtime
    	0.55s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 383.52s of the 583.55s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTrees_BAG_L1 ... Training model for up to 383.45s of the 583.48s of remaining time.
    	-38.3981	 = Validation score   (-root_mean_squared_error)
    	6.07s	 = Training   runtime
    	0.53s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 574.28s of remaining time.
    	-37.1766	 = Validation score   (-root_mean_squared_error)
    	0.22s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 4 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 573.99s of the 573.98s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForest_BAG_L2 ... Training model for up to 573.92s of the 573.91s of remaining time.
    	-37.4105	 = Validation score   (-root_mean_squared_error)
    	18.88s	 = Training   runtime
    	0.65s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 551.71s of the 551.7s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTrees_BAG_L2 ... Training model for up to 551.63s of the 551.62s of remaining time.
    	-36.8272	 = Validation score   (-root_mean_squared_error)
    	6.96s	 = Training   runtime
    	0.58s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 541.4s of remaining time.
    	-36.791	 = Validation score   (-root_mean_squared_error)
    	0.23s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 59.03s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221230_075542/")



```python
predictor_new_hpo5 = predictor_new_hpo5.predict(test)
predictor_new_hpo5.head()
```




    0    11.225268
    1     5.968387
    2     5.144086
    3     2.621613
    4     2.661075
    Name: count, dtype: float32




```python
predictor_new_hpo5.describe()
```




    count    6493.000000
    mean      193.610519
    std       174.406296
    min         1.999462
    25%        48.744518
    50%       152.201828
    75%       282.017944
    max       908.991821
    Name: count, dtype: float64




```python
submission_new_hpo["count"] = predictor_new_hpo5
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
```


```python
predictor_new_hpo5_ = TabularPredictor(label='count', eval_metric="rmse" ).fit(hyperparameters={'GBM': {'num_boost_round': 10000},'CAT': {'iterations': 10000},'RF': {'n_estimators': 300},'XT': {'n_estimators': 300}}, train_data= train_new_features ,time_limit=600 ,presets="best_quality")
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20221230_075904/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20221230_075904/"
    AutoGluon Version:  0.6.1
    Python Version:     3.7.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 26 20:36:53 UTC 2022
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == int and many unique label-values observed).
    	Label info (max, min, mean, stddev): (977, 1, 191.57413, 181.14445)
    	If 'regression' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2358.91 MB
    	Train Data (Original)  Memory Usage: 0.89 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    /usr/local/lib/python3.7/site-packages/autogluon/features/generators/datetime.py:59: FutureWarning: casting datetime64[ns, UTC] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
      good_rows = series[~series.isin(bad_rows)].astype(np.int64)
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 6 | ['holiday', 'workingday', 'humidity', 'month', 'day', ...]
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.2s = Fit runtime
    	12 features in original data used to generate 16 features in processed data.
    	Train Data (Processed) Memory Usage: 1.09 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.22s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 4 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.75s of the 599.77s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForest_BAG_L1 ... Training model for up to 399.67s of the 599.69s of remaining time.
    	-38.3878	 = Validation score   (-root_mean_squared_error)
    	12.97s	 = Training   runtime
    	0.56s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 383.54s of the 583.56s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L1 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTrees_BAG_L1 ... Training model for up to 383.44s of the 583.47s of remaining time.
    	-38.3981	 = Validation score   (-root_mean_squared_error)
    	5.56s	 = Training   runtime
    	0.53s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 574.76s of remaining time.
    	-37.1766	 = Validation score   (-root_mean_squared_error)
    	0.19s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 4 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 574.49s of the 574.48s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused LightGBM_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: RandomForest_BAG_L2 ... Training model for up to 574.43s of the 574.42s of remaining time.
    	-37.4105	 = Validation score   (-root_mean_squared_error)
    	18.66s	 = Training   runtime
    	0.6s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 552.73s of the 552.73s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	Warning: Exception caused CatBoost_BAG_L2 to fail during training... Skipping this model.
    		System error: Broken pipe
    Detailed Traceback:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1422, in _train_and_save
        model = self._train_single(X, y, model, X_val, y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1367, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, total_resources=total_resources, **model_fit_kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/abstract/abstract_model.py", line 696, in fit
        out = self._fit(**kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 154, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 238, in _fit
        n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 515, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 487, in after_all_folds_scheduled
        X, y, X_pseudo, y_pseudo = self._prepare_data()
      File "/usr/local/lib/python3.7/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 656, in _prepare_data
        X = self.ray.put(self.X)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
        return func(*args, **kwargs)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 2345, in put
        object_ref = worker.put_object(value, owner_address=serialize_owner_address)
      File "/usr/local/lib/python3.7/site-packages/ray/_private/worker.py", line 621, in put_object
        serialized_value, object_ref=object_ref, owner_address=owner_address
      File "python/ray/_raylet.pyx", line 1364, in ray._raylet.CoreWorker.put_serialized_object_and_increment_local_ref
      File "python/ray/_raylet.pyx", line 1254, in ray._raylet.CoreWorker._create_put_buffer
      File "python/ray/_raylet.pyx", line 179, in ray._raylet.check_status
    ray.exceptions.RaySystemError: System error: Broken pipe
    Fitting model: ExtraTrees_BAG_L2 ... Training model for up to 552.65s of the 552.64s of remaining time.
    	-36.8272	 = Validation score   (-root_mean_squared_error)
    	6.9s	 = Training   runtime
    	0.58s	 = Validation runtime
    Repeating k-fold bagging: 2/20
    Repeating k-fold bagging: 3/20
    Repeating k-fold bagging: 4/20
    Repeating k-fold bagging: 5/20
    Repeating k-fold bagging: 6/20
    Repeating k-fold bagging: 7/20
    Repeating k-fold bagging: 8/20
    Repeating k-fold bagging: 9/20
    Repeating k-fold bagging: 10/20
    Repeating k-fold bagging: 11/20
    Repeating k-fold bagging: 12/20
    Repeating k-fold bagging: 13/20
    Repeating k-fold bagging: 14/20
    Repeating k-fold bagging: 15/20
    Repeating k-fold bagging: 16/20
    Repeating k-fold bagging: 17/20
    Repeating k-fold bagging: 18/20
    Repeating k-fold bagging: 19/20
    Repeating k-fold bagging: 20/20
    Completed 20/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 542.37s of remaining time.
    	-36.791	 = Validation score   (-root_mean_squared_error)
    	0.99s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 59.02s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20221230_075904/")



```python
predictor_new_hpo5_ = predictor_new_hpo5_.predict(test_new_features)
predictor_new_hpo5_.head()
```




    0    11.225268
    1     5.968387
    2     5.144086
    3     2.621613
    4     2.661075
    Name: count, dtype: float32




```python
predictor_new_hpo5_.describe()
```




    count    6493.000000
    mean      193.610519
    std       174.406296
    min         1.999462
    25%        48.744518
    50%       152.201828
    75%       282.017944
    max       908.991821
    Name: count, dtype: float64




```python
submission_new_hpo["count"] = predictor_new_hpo5_
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
```


```python
#, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
# 'NN': {'num_epochs': 500}
```


```python
predictor_new_hpo.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                     model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0  WeightedEnsemble_L3 -32.526630       5.684208  446.529177                0.000877           0.362089            3       True         10
    1  WeightedEnsemble_L2 -32.583721       4.197559  345.795672                0.001168           0.361895            2       True          5
    2      CatBoost_BAG_L2 -32.836159       4.305158  390.063782                0.108768          44.630004            2       True          8
    3    ExtraTrees_BAG_L2 -32.884504       4.813185  354.125036                0.616795           8.691259            2       True          9
    4      LightGBM_BAG_L2 -33.038311       4.320831  367.921482                0.124441          22.487705            2       True          6
    5  RandomForest_BAG_L2 -33.270670       4.833328  370.358121                0.636938          24.924343            2       True          7
    6      CatBoost_BAG_L1 -33.637541       0.176255  279.086344                0.176255         279.086344            1       True          3
    7      LightGBM_BAG_L1 -33.916921       2.846511   46.185931                2.846511          46.185931            1       True          1
    8  RandomForest_BAG_L1 -38.387847       0.599941   13.899373                0.599941          13.899373            1       True          2
    9    ExtraTrees_BAG_L1 -38.398087       0.573683    6.262129                0.573683           6.262129            1       True          4
    Number of models trained: 10
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_RF', 'StackerEnsembleModel_LGB'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])             : 2 | ['season', 'weather']
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 4 | ['humidity', 'month', 'day', 'hour']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20221222_100313/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForest_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTrees_BAG_L1': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForest_BAG_L2': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L2': 'StackerEnsembleModel_CatBoost',
      'ExtraTrees_BAG_L2': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'LightGBM_BAG_L1': -33.91692104371548,
      'RandomForest_BAG_L1': -38.387846615896144,
      'CatBoost_BAG_L1': -33.63754134027187,
      'ExtraTrees_BAG_L1': -38.39808749149808,
      'WeightedEnsemble_L2': -32.58372115426817,
      'LightGBM_BAG_L2': -33.038310883890325,
      'RandomForest_BAG_L2': -33.27067047830258,
      'CatBoost_BAG_L2': -32.836158681865214,
      'ExtraTrees_BAG_L2': -32.88450419638176,
      'WeightedEnsemble_L3': -32.52663032242893},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20221222_100313/models/LightGBM_BAG_L1/',
      'RandomForest_BAG_L1': 'AutogluonModels/ag-20221222_100313/models/RandomForest_BAG_L1/',
      'CatBoost_BAG_L1': 'AutogluonModels/ag-20221222_100313/models/CatBoost_BAG_L1/',
      'ExtraTrees_BAG_L1': 'AutogluonModels/ag-20221222_100313/models/ExtraTrees_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20221222_100313/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20221222_100313/models/LightGBM_BAG_L2/',
      'RandomForest_BAG_L2': 'AutogluonModels/ag-20221222_100313/models/RandomForest_BAG_L2/',
      'CatBoost_BAG_L2': 'AutogluonModels/ag-20221222_100313/models/CatBoost_BAG_L2/',
      'ExtraTrees_BAG_L2': 'AutogluonModels/ag-20221222_100313/models/ExtraTrees_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20221222_100313/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'LightGBM_BAG_L1': 46.18593120574951,
      'RandomForest_BAG_L1': 13.899373292922974,
      'CatBoost_BAG_L1': 279.08634400367737,
      'ExtraTrees_BAG_L1': 6.262129306793213,
      'WeightedEnsemble_L2': 0.3618946075439453,
      'LightGBM_BAG_L2': 22.487704515457153,
      'RandomForest_BAG_L2': 24.92434287071228,
      'CatBoost_BAG_L2': 44.630003929138184,
      'ExtraTrees_BAG_L2': 8.691258668899536,
      'WeightedEnsemble_L3': 0.3620893955230713},
     'model_pred_times': {'LightGBM_BAG_L1': 2.846510648727417,
      'RandomForest_BAG_L1': 0.5999414920806885,
      'CatBoost_BAG_L1': 0.1762552261352539,
      'ExtraTrees_BAG_L1': 0.5736827850341797,
      'WeightedEnsemble_L2': 0.0011684894561767578,
      'LightGBM_BAG_L2': 0.12444067001342773,
      'RandomForest_BAG_L2': 0.6369378566741943,
      'CatBoost_BAG_L2': 0.10876822471618652,
      'ExtraTrees_BAG_L2': 0.6167945861816406,
      'WeightedEnsemble_L3': 0.0008769035339355469},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForest_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTrees_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForest_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTrees_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                  model  score_val  pred_time_val    fit_time  \
     0  WeightedEnsemble_L3 -32.526630       5.684208  446.529177   
     1  WeightedEnsemble_L2 -32.583721       4.197559  345.795672   
     2      CatBoost_BAG_L2 -32.836159       4.305158  390.063782   
     3    ExtraTrees_BAG_L2 -32.884504       4.813185  354.125036   
     4      LightGBM_BAG_L2 -33.038311       4.320831  367.921482   
     5  RandomForest_BAG_L2 -33.270670       4.833328  370.358121   
     6      CatBoost_BAG_L1 -33.637541       0.176255  279.086344   
     7      LightGBM_BAG_L1 -33.916921       2.846511   46.185931   
     8  RandomForest_BAG_L1 -38.387847       0.599941   13.899373   
     9    ExtraTrees_BAG_L1 -38.398087       0.573683    6.262129   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000877           0.362089            3       True   
     1                0.001168           0.361895            2       True   
     2                0.108768          44.630004            2       True   
     3                0.616795           8.691259            2       True   
     4                0.124441          22.487705            2       True   
     5                0.636938          24.924343            2       True   
     6                0.176255         279.086344            1       True   
     7                2.846511          46.185931            1       True   
     8                0.599941          13.899373            1       True   
     9                0.573683           6.262129            1       True   
     
        fit_order  
     0         10  
     1          5  
     2          8  
     3          9  
     4          6  
     5          7  
     6          3  
     7          1  
     8          2  
     9          4  }




```python
predictor_new_hpo = predictor_new_hpo.predict(test_new_features)
predictor_new_hpo.head()
```




    0    13.070760
    1     6.590817
    2     6.123107
    3     5.006030
    4     5.061359
    Name: count, dtype: float32




```python
submission_new_hpo = submission
```


```python
# Remember to set all negative values to zero
predictor_new_hpo.describe()
```




    count    6493.000000
    mean      191.732910
    std       173.544250
    min         3.812018
    25%        47.191841
    50%       148.530106
    75%       285.792297
    max       887.144592
    Name: count, dtype: float64




```python
# Same submitting predictions
submission_new_hpo["count"] = predictor_new_hpo
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hpo.csv -m "new features with hyperparameters"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 365kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                        status    publicScore  privateScore  
    ---------------------------  -------------------  ---------------------------------  --------  -----------  ------------  
    submission_new_hpo.csv       2022-12-30 08:01:02  new features with hyperparameters  complete  0.47099      0.47099       
    submission_new_hpo.csv       2022-12-30 07:58:36  new features with hyperparameters  complete  0.46822      0.46822       
    submission_new_hpo.csv       2022-12-30 07:56:58  new features with hyperparameters  complete  0.46822      0.46822       
    submission_new_features.csv  2022-12-30 07:55:40  new features                       complete  0.65113      0.65113       


#### New Score of `?`

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report


```python
# Taking the top model score from each training run and creating a line plot to show improvement
# You can create these in the notebook and save them to PNG or use some other tool (e.g. google sheets, excel)
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score": [?, ?, ?]
    }
).plot(x="model", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_train_score.png')
```


```python
# Take the 3 kaggle scores and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.79693, 0.65521, 0.46822]
    }
).plot(x="test_eval", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_test_score.png')
```


    
![png](output_99_0.png)
    


### Hyperparameter table


```python
# The 3 hyperparameters we tuned with the kaggle score as the result
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": [?, ?, ?],
    "hpo2": [?, ?, ?],
    "hpo3": [?, ?, ?],
    "score": [?, ?, ?]
})
```
