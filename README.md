# Stability and Generalization in Free Adversarial Training

The code in this repository accompanies the experiments performed in the paper Stability and Generalization in Free Adversarial Training by Xiwei Cheng, Kexin Fu, and Farzan Farnia. 


## Getting started

To adversarially train a ResNet-18 model against $L_2$-norm attack by the free-AT algorithm, run: 
```
python train.py   --data cifar10  --method free  --attack L2  --eps 128.0  --model res18  --save_path cifar10_l2_free
```


