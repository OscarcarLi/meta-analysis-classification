How to setup the datasets
- tieredImagenet-base
    - cd data/
    - download tieredImagenet-base.tar into the folder data/
    gdown --id 1wUvTVEhZorao9DHgCQZySejHGhI9JdZw -O tieredImagenet-base.tar
    - tar xf tieredImagenet-base.tar
    - this will extract out a tieredImagenet-base folder
    - cd tieredImagenet-base
    - python3 make_json_base_val_novel_base_test_novel_large.py

- miniImagenet-base
    - cd data/
    - download miniImagenet-base.tar into the folder data/
        gdown --id 1oAvvVIemrTrljJ6Qd5LFGGe0MXRLdtFU -O miniImagenet-base.tar
    - tar xf miniImagenet-base.tar
    - cd miniImagenet-base
    - python3 make_json_base_val_novel_base_test_novel_large.py

- FC100-base
    - cd data/
    - download FC100-base.tar into the folder data/
        gdown --id 11pRhOK9HFZFjbdqnJYNzfRM4-FSLYS1g -O FC100-base.tar
    - tar xf FC100-base.tar
    - cd FC100-base
    - python3 process.py
    - python3 make_json_base_val_novel_base_test.py

- cifar-fs-base
    - cd data
    - download cifar-fs-base.tar into the folder data/
        gdown --id 12tRzxlnWSMd3j-9D3w35VjXYfmVYenmR -O cifar-fs-base.tar
    - tar xf cifar-fs-base.tar
    - cd cifar-fs-base
    - bash make_cifar-fs-base.sh