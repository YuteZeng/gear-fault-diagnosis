# gear-fault-diagnosis
this repo is the code implementation for course "Advanced Machine Learning" report - HUST

## Environment Setup
The project is built on PyTorch and Conda.
1. Conda env setup (replace 'gear' with your_env_name)  
`conda create -n gear python=3.10 -y`  
2. Conda env activate  
`conda activate gear`  
3. Install PyTorch (torch2.1.0+cu118 prebuilt wheel recommanded)  
`pip install torch==2.1.0+cu118 torchvision -f https://download.pytorch.org/whl/torch_stable.html`
4. Install other packages using the requirements.txt  
`pip install -r requirements.txt`

## Run Test Instances
1. Go to the project directory  
`cd gear-fault-diagnosis`
2. Run test instances, basically evaluating samples from one of the datasets  
`python Test.py`
3. Results of the test  
`Begin Time: 2025-05-28 15:17:56.205341  <br>
测试样本中的前十个样例标签如下： tensor([4, 6, 5, 4, 1, 1, 3, 3, 2, 1], device='cuda:0')  <br>
Testing_accuracy: 96.57142857142857  <br>
Test_macro_precision: 96.9387755102041 Test_macro_recall: 96.57142857142857 Test_macro_f1_score: 96.54639947452725  <br>
End Time: 2025-05-28 15:17:56.610228` 

## Run Model Training
1. Go to the project directory  
`cd gear-fault-diagnosis`
2. Run model training  
`python Train.py`
3. You would expect output like this  
`Begin Time: 2025-05-28 15:31:51.484444  
Epoch: 0  
100%|█████████████████| 6/6 [00:01<00:00,  4.96it/s]  
Training_loss: 0.016741958515984673  
Training_accuracy: 14.285714285714286  
Train_macro_precision: 2.857142857142857 Train_macro_recall: 14.285714285714285 Train_macro_f1_score: 4.761904761904762  
Validation_loss: 0.022370311873299735  
Validation_accuracy: 14.285714285714286  
Valid_macro_precision: 1.82370820668693 Valid_macro_recall: 14.285714285714285 Valid_macro_f1_score: 3.234501347708895`  
